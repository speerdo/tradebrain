"""
TradeBrain — Main Agent Loop (Coinbase Advanced Perps Edition)

Entry point. Startup sequence:
1. Load config
2. Connect to DB
3. Verify Coinbase API auth + futures provisioned
4. Run initial screener
5. Start FastAPI
6. Start position monitor
7. Enter signal loop
"""

import asyncio
import os
import signal

import uvicorn
from loguru import logger

import config
from agent import llm_router
from agent.api import app, set_agent_state
from agent.burt import Burt
from agent.coinbase_client import CoinbaseClient
from agent.database import get_db
from agent.derivatives import DerivativesContext
from agent.executor import Executor
from agent.indicator_engine import compute_indicators, compute_4h_indicators, aggregate_candles
from agent.memory_engine import MemoryEngine
from agent.notifier import Notifier
from agent.position_monitor import PositionMonitor
from agent.regime import RegimeEngine
from agent.risk_manager import RiskManager
from agent.screener import Screener
from agent.sentiment import SentimentContext
from agent.signal_engine import SignalEngine
from strategies import STRATEGIES


class TradeBrainAgent:

    def __init__(self):
        self.cfg = config.get_config()
        self.db = None
        self.cb = CoinbaseClient()
        self.executor = Executor(self.cb)
        self.risk = RiskManager()
        self.screener = Screener(self.cb)
        self.signal_engine = SignalEngine()
        self.regime_engine = RegimeEngine(self.cb)
        self.memory_engine = MemoryEngine(self.signal_engine)
        self.signal_engine.set_memory_engine(self.memory_engine)
        self.derivatives = DerivativesContext()
        self.sentiment = SentimentContext(cryptopanic_token=os.environ.get("CRYPTOPANIC_TOKEN", ""))
        self.monitor = PositionMonitor(self.executor, self.cb, self.risk)
        self.burt = Burt(None, self.executor, self.risk, self.screener)
        self.notifier = Notifier(self.burt)
        self.monitor.set_notifier(self.notifier)
        self.executor.set_notifier(self.notifier)
        # P0: positions closed on the exchange between monitor ticks never
        # reach PositionMonitor._handle_exit — the executor books them itself.
        self.executor.set_risk_manager(self.risk)
        self.watchlist: list[str] = []
        self._shutdown = asyncio.Event()
        self._api_task = None
        self._burt_task = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        logger.info("╔══════════════════════════════════════╗")
        logger.info("║      TradeBrain Starting [CB]      ║")
        logger.info("╚══════════════════════════════════════╝")

        self.db = await get_db()
        logger.info("✅ Database connected")

        # P3: enforce MODELS.md §2 — the run plane pays per token. A seat
        # subscription on the run plane degrades silently within hours.
        routing_errors = llm_router.validate_run_plane_config(self.cfg)
        if routing_errors:
            for err in routing_errors:
                logger.error(f"Run-plane config violation: {err}")
            raise RuntimeError("Run-plane LLM routing violates MODELS.md §2 — refusing to start")
        llm_router.log_routing_table(self.cfg)

        # Wire Burt after DB is ready
        self.burt.db = self.db
        self._burt_task = asyncio.create_task(self.burt.start())

        ok = await self.cb.verify_auth()
        if not ok:
            logger.warning("⚠️  Coinbase auth failed — check COINBASE_API_KEY + SECRET")

        await self.cb.verify_futures_provisioned()

        # P0: live positions are reconciled from the exchange before anything
        # evaluates — portfolio caps and re-entry suppression must see reality.
        if not self.cfg.paper_trading:
            live = await self.executor.reconcile_live_positions()
            logger.info(f"Live reconciliation: {len(live)} open position(s) on exchange")

        logger.info("Running initial screener...")
        self.watchlist = await self.screener.run()
        logger.info(f"Watchlist ({len(self.watchlist)}): {self.watchlist}")

        set_agent_state(self.executor, self.risk, self.screener)
        self._api_task = asyncio.create_task(self._run_api())
        self.monitor.start()
        logger.info("✅ FastAPI + position monitor started")

    async def _run_api(self) -> None:
        cfg = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
        server = uvicorn.Server(cfg)
        await server.serve()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.startup()
        try:
            await self._loop()
        except asyncio.CancelledError:
            logger.info("Loop cancelled")
        finally:
            await self.shutdown()

    async def _loop(self) -> None:
        screener_counter = 0

        while not self._shutdown.is_set():
            try:
                await self.db.sync_config()
                await self.risk.sync()

                # Read hot-reload values fresh each iteration so UI/Burt edits
                # take effect on the next tick instead of requiring a restart.
                interval = self.cfg.signal_interval
                screener_interval = max(1, 4 * 3600 // interval)

                screener_counter += 1
                if screener_counter >= screener_interval:
                    screener_counter = 0
                    self.watchlist = await self.screener.run()

                # Fetch market regime once per loop (C1) — injected into prompts
                # and used to mechanically gate strategies.
                regime_ctx = await self.regime_engine.get_context()
                current_regime = regime_ctx.get("regime", "unknown")

                # Fetch Fear & Greed once per loop (C4) — shared across all symbols
                fng = await self.sentiment.get_fear_greed()

                # Regime gate: skip strategy if it's not compatible with the current
                # regime. "unknown" (regime fetch failed) fails OPEN — a transient
                # API error must not silently stop all trading.
                strategy = STRATEGIES.get(self.cfg.strategy)
                if (strategy and strategy.compatible_regimes is not None
                        and current_regime != "unknown"):
                    if current_regime not in strategy.compatible_regimes:
                        logger.info(
                            f"Regime gate: strategy '{strategy.name}' disabled in "
                            f"'{current_regime}' regime (compatible: {strategy.compatible_regimes})"
                        )
                    else:
                        for symbol in self.watchlist:
                            await self._evaluate(symbol, regime_ctx, fng)
                else:
                    for symbol in self.watchlist:
                        await self._evaluate(symbol, regime_ctx, fng)

                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error(f"Loop error: {exc}")
                await asyncio.sleep(5)

    async def _evaluate(self, product_id: str, regime_ctx: dict | None = None,
                        fng: dict | None = None) -> None:
        if self.executor.has_position(product_id):
            return

        signal = type("Sig", (), {"direction": "none", "confidence": 0.0})()
        skip = self.risk.check_trade_allowed(signal, product_id)
        if skip and ("Manual pause" in skip or "Circuit breaker" in skip):
            return

        try:
            candles_15m = await self.cb.get_candles(product_id, "FIFTEEN_MINUTE")
            candles_1h  = await self.cb.get_candles(product_id, "ONE_HOUR")
            candles_2h  = await self.cb.get_candles(product_id, "TWO_HOUR")

            if len(candles_15m) < 30 or len(candles_1h) < 20:
                return

            indicators = compute_indicators(
                self._candles_to_df(candles_15m),
                self._candles_to_df(candles_1h),
            )
            # 4h context for cascading multi-timeframe prompt (C2)
            # Coinbase doesn't offer 4h directly — aggregate 2h candles.
            if candles_2h and len(candles_2h) >= 20:
                df_2h = self._candles_to_df(candles_2h)
                df_4h = aggregate_candles(df_2h, factor=2)
                indicators["4h"] = compute_4h_indicators(df_4h)

            strategy = STRATEGIES.get(self.cfg.strategy)
            if not strategy:
                return

            # C3: Fetch derivatives context (funding + OI deltas)
            deriv_ctx = await self.derivatives.get_context(product_id)

            # C4: Fetch news for this symbol (best-effort, may be empty)
            # Extract a currency code from the product_id for CryptoPanic filter
            currency = self._extract_currency(product_id)
            news = await self.sentiment.get_news(currency)

            # Build extra context string for the prompt
            extra_parts = []
            if deriv_ctx:
                extra_parts.append(self.derivatives.format_prompt_block(deriv_ctx))
            if fng:
                extra_parts.append(self.sentiment.format_prompt_block(fng, news, currency))
            extra_context = "".join(extra_parts)

            sig = await self.signal_engine.evaluate(
                product_id, strategy, indicators,
                regime=regime_ctx,
                extra_context=extra_context,
            )
            open_positions = self.executor.get_open_positions()
            skip = self.risk.check_trade_allowed(sig, product_id, open_positions)
            if skip:
                logger.info(f"Skip {product_id}: {skip}")
                return

            # C3: Server-side funding-cost rule
            if deriv_ctx:
                fund_skip = self.derivatives.check_funding_rule(sig.direction, deriv_ctx)
                if fund_skip:
                    logger.info(f"Skip {product_id}: {fund_skip}")
                    return

            # C4: News veto — skip entries during high-panic news for this asset
            if news and self.sentiment.has_high_panic_news(news):
                logger.info(f"Skip {product_id}: high-panic news veto")
                return

            entry = sig.entry_price or indicators["15m"]["price"]
            sl, tp, notional, margin = self.risk.calculate_trade_params(
                sig.direction, entry, indicators["15m"]["atr"]
            )
            if notional <= 0 or margin <= 0:
                return

            result = await self.executor.enter_position(
                symbol=product_id,
                direction=sig.direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                size_usdc=notional,
                margin_usdc=margin,
                leverage=self.risk.state.leverage,
                risk_usdc=self.risk.state.balance_usdc * self.risk.state.risk_per_trade_pct,
                strategy=strategy.name,
                confidence=sig.confidence,
                reasoning=sig.reasoning,
            )
            if result.success:
                self.risk.protections.record_entry()
                # get_position only tracks paper positions — None in live mode
                pos = self.executor.get_position(product_id)
                if pos is not None:
                    self.monitor.record_entry(pos, indicators["15m"]["atr"])
                logger.info(f"✅ Position opened: {product_id} {sig.direction}")
            else:
                logger.warning(f"Failed to open {product_id}: {result.error}")

        except Exception as exc:
            logger.error(f"Error evaluating {product_id}: {exc}")

    @staticmethod
    def _extract_currency(product_id: str) -> str:
        """Extract a currency code from a Coinbase FCM product_id for CryptoPanic.
        E.g. 'BIP-20DEC30-CDE' -> 'BTC', 'ETP-20DEC30-CDE' -> 'ETH'.
        Falls back to the first 3 chars."""
        # FCM product_ids for perps: {ASSET}-{EXPIRY}-CDE
        # The prefix before the first '-' maps: BIP->BTC, ETP->ETH, etc.
        prefix = product_id.split("-")[0] if product_id else ""
        mapping = {"BIP": "BTC", "ETP": "ETH", "SOP": "SOL", "XRP": "XRP",
                   "DOP": "DOGE", "LAP": "LTC", "AVP": "AVAX", "LIP": "LINK"}
        return mapping.get(prefix, prefix[:3] if prefix else "BTC")

    @staticmethod
    def _candles_to_df(candles: list):
        import pandas as pd
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame([
            {"time": c.start, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").reset_index(drop=True)

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        self._shutdown.set()
        self.monitor.stop()
        self.burt.stop()

        # Cancel AND await the background tasks. Without the await, uvicorn's
        # lifespan task is left mid-queue.get() when the event loop closes,
        # which raises "Event loop is closed" during GC.
        pending = [t for t in (self._burt_task, self._api_task) if t]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        await self.signal_engine.close()
        await self.sentiment.close()
        await self.cb.close()
        if self.db:
            await self.db.close()
        logger.info("Shutdown complete")

    def _sigint(self) -> None:
        self._shutdown.set()


def main() -> None:
    agent = TradeBrainAgent()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent._sigint)
    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
