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
import sys
import time

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
from agent.maintenance import MaintenanceWindow
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
        # The screener needs the balance to drop products whose single
        # contract can't be afforded (whole-contract CFM sizing).
        self.screener = Screener(self.cb, risk=self.risk)
        self.maintenance = MaintenanceWindow()
        self.signal_engine = SignalEngine()
        self.regime_engine = RegimeEngine(self.cb)
        self.memory_engine = MemoryEngine(self.signal_engine)
        self.signal_engine.set_memory_engine(self.memory_engine)
        self.derivatives = DerivativesContext()
        self.sentiment = SentimentContext(cryptopanic_token=os.environ.get("CRYPTOPANIC_TOKEN", ""))
        self.monitor = PositionMonitor(self.executor, self.cb, self.risk)
        self.burt = Burt(None, self.executor, self.risk, self.screener)
        self.notifier = Notifier(self.burt)
        self.executor.set_notifier(self.notifier)
        # P0: positions closed on the exchange between monitor ticks never
        # reach PositionMonitor._handle_exit — the executor books them itself.
        self.executor.set_risk_manager(self.risk)
        self.risk.set_client(self.cb)
        self.risk.set_notifier(self.notifier)
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

        # Balance before anything reads it. risk.sync() would fix this on the
        # first loop tick, but the API/UI and Burt come up before that and would
        # report the 100k placeholder as the account size.
        await self.db.sync_config()
        await self.risk.sync()

        # Paper positions are in-memory only — restore open ones from the
        # trades table before the monitor, screener, or risk caps read them,
        # or a restart would silently forget every open paper trade (rows
        # stuck 'open' forever, no stop/TP ever fires).
        restored = await self.executor.restore_paper_positions()
        if restored:
            logger.info(f"Paper restore: {len(restored)} position(s) back under management")

        logger.info(
            f"Account: ${self.risk.state.balance_usdc:,.2f} "
            f"({'PAPER' if self.cfg.paper_trading else 'LIVE'}) | "
            f"risk/trade {self.risk.state.risk_per_trade_pct:.1%} | "
            f"daily stop ${self.risk.state.balance_usdc * self.risk.state.daily_loss_limit_pct:,.2f}"
        )

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

    # Per-tick plumbing (config sync, screener, regime, sentiment) previously
    # ran with no timeout of its own. On 2026-09-11 one of these calls hung
    # (no exception, no log line) and froze the whole loop for 3+ days
    # straight — Burt kept adjusting min_confidence/risk/strategy over Discord
    # the entire time, but nothing ever consumed those changes because the
    # loop that reads them had already died. _evaluate_guarded already
    # watchdogs the per-symbol call; these tick-level calls need the same
    # belt-and-suspenders treatment so a single stuck HTTP/DB call can never
    # again take down days of trading silently.
    _TICK_STEP_TIMEOUT_SEC = 45

    async def _tick_step(self, name: str, coro, default=None):
        try:
            return await asyncio.wait_for(coro, timeout=self._TICK_STEP_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.error(
                f"Watchdog: '{name}' exceeded {self._TICK_STEP_TIMEOUT_SEC}s — "
                f"using fallback and moving on"
            )
            return default

    # Floor guarantee: "at least one trade per 24h" is a hard requirement, not
    # an aspiration. Under the single configured strategy, most ticks resolve
    # to "No directional signal" by design (donchian_breakout requires an
    # actual 20-bar breakout — rare by construction) which is fine on average
    # but gives no floor. If nothing has traded in _DROUGHT_HOURS, broaden to
    # every OTHER strategy that's still compatible with the current regime
    # (each already self-declares its regime fit — chop-only bollinger never
    # runs in a trend regime and vice versa) instead of relying on just one.
    # Scoped to the drought window only, so normal-day LLM cost is unchanged.
    # Threshold comes from config.drought_guard_hours (hot-reloadable);
    # 0 disables the guard entirely.

    async def _last_trade_age_hours(self) -> float:
        try:
            val = await self.db.fetchval(
                "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 3600.0 "
                "FROM trades WHERE is_paper = $1",
                self.cfg.paper_trading,
            )
            return float(val) if val is not None else 999.0
        except Exception as exc:
            logger.warning(f"Drought check failed: {exc}")
            return 0.0  # fail closed — don't broaden strategy selection on a DB hiccup

    async def _loop(self) -> None:
        screener_counter = 0

        while not self._shutdown.is_set():
            tick_started = time.monotonic()
            try:
                await self._tick_step("db.sync_config", self.db.sync_config())
                await self._tick_step("risk.sync", self.risk.sync())

                # Read hot-reload values fresh each iteration so UI/Burt edits
                # take effect on the next tick instead of requiring a restart.
                interval = self.cfg.signal_interval
                screener_hours = float(getattr(self.cfg, "screener_interval_h", 4.0) or 4.0)
                screener_interval = max(1, int(screener_hours * 3600 // interval))

                screener_counter += 1
                if screener_counter >= screener_interval:
                    screener_counter = 0
                    new_watchlist = await self._tick_step(
                        "screener.run", self.screener.run(), default=None
                    )
                    if new_watchlist is not None:
                        self.watchlist = new_watchlist

                # Fetch market regime once per loop (C1) — injected into prompts
                # and used to mechanically gate strategies. Falls open to
                # "unknown" on timeout, same as regime.py's own internal
                # exception handling.
                regime_ctx = await self._tick_step(
                    "regime.get_context", self.regime_engine.get_context(),
                    default={"btc_dominance": 0.5, "regime": "unknown", "context_tickers": []},
                )

                # Fetch Fear & Greed once per loop (C4) — shared across all symbols
                fng = await self._tick_step(
                    "sentiment.get_fear_greed", self.sentiment.get_fear_greed(), default={}
                )

                # Regime gate: only strategies compatible with the current regime
                # run. "unknown" (regime fetch failed) fails OPEN — a transient
                # API error must not silently stop all trading.
                current_regime = regime_ctx.get("regime", "unknown")

                def _regime_ok(s) -> bool:
                    return (s.compatible_regimes is None or current_regime == "unknown"
                            or current_regime in s.compatible_regimes)

                primary = STRATEGIES.get(self.cfg.strategy)
                candidates = [primary] if primary and _regime_ok(primary) else []

                drought_limit = float(self.cfg.drought_guard_hours or 0)
                if drought_limit > 0:
                    drought_hours = await self._tick_step(
                        "drought_check", self._last_trade_age_hours(), default=0.0
                    )
                    if drought_hours >= drought_limit:
                        fallback = [s for s in STRATEGIES.values()
                                    if s not in candidates and _regime_ok(s)]
                        if fallback:
                            logger.warning(
                                f"Drought guard: {drought_hours:.1f}h since last trade "
                                f"(≥{drought_limit}h) — also trying "
                                f"{[s.name for s in fallback]} this tick"
                            )
                            candidates += fallback

                if not candidates:
                    logger.info(
                        f"Regime gate: no strategy compatible with '{current_regime}' regime"
                    )
                elif not self.maintenance.is_open():
                    # Fri 5-6pm ET: CFM is closed. Documented in
                    # FCM_TRADING_HOURS.md as wired here, but never was — a
                    # live entry attempt in the window just errors out.
                    logger.info("Maintenance window — no entries this tick")
                else:
                    for symbol in self.watchlist:
                        await self._evaluate_guarded(symbol, candidates, regime_ctx, fng)

                logger.info(
                    f"Tick complete: {len(self.watchlist)} symbol(s) scanned, "
                    f"regime={current_regime}, took {time.monotonic() - tick_started:.1f}s"
                )

                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error(f"Loop error: {exc}")
                await asyncio.sleep(5)

    # Hard ceiling on a single symbol's evaluation. Every external call inside
    # _evaluate (candles, derivatives, sentiment, LLM, embeddings) already
    # carries its own timeout, but a gap in that chain (e.g. a DB query or a
    # connection-pool wait with no timeout) can still hang forever and freeze
    # the whole loop — as happened on 2026-09-03, where the bot sat idle for
    # 13+ hours after stalling mid-watchlist. This belt-and-suspenders timeout
    # guarantees the loop always moves on to the next symbol/tick.
    _EVALUATE_TIMEOUT_SEC = 180

    async def _evaluate_guarded(self, product_id: str, strategies: list,
                                 regime_ctx: dict | None = None,
                                 fng: dict | None = None) -> None:
        try:
            await asyncio.wait_for(
                self._evaluate(product_id, strategies, regime_ctx, fng),
                timeout=self._EVALUATE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Watchdog: {product_id} evaluation exceeded "
                f"{self._EVALUATE_TIMEOUT_SEC}s — abandoning and moving on"
            )

    # Skip reasons that mean "this particular signal wasn't good enough" —
    # worth retrying with the next candidate strategy. Everything else
    # (cooldowns, circuit breaker, portfolio caps) is symbol/account-level
    # and would fail identically for every strategy, so it stops the loop.
    _RETRYABLE_SKIP_PREFIXES = ("No directional signal", "Confidence ")

    async def _evaluate(self, product_id: str, strategies: list,
                        regime_ctx: dict | None = None,
                        fng: dict | None = None) -> None:
        if self.executor.has_position(product_id):
            return
        if not strategies:
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

            # C3: Fetch derivatives context (funding + OI deltas) — symbol-level,
            # shared across every candidate strategy this tick.
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

            # C4: News veto applies to the symbol regardless of strategy —
            # check once up front rather than per candidate.
            if news and self.sentiment.has_high_panic_news(news):
                logger.info(f"Skip {product_id}: high-panic news veto")
                return

            for strategy in strategies:
                sig = await self.signal_engine.evaluate(
                    product_id, strategy, indicators,
                    regime=regime_ctx,
                    extra_context=extra_context,
                )
                open_positions = self.executor.get_open_positions()
                skip = self.risk.check_trade_allowed(sig, product_id, open_positions)
                if skip:
                    logger.info(f"Skip {product_id} [{strategy.name}]: {skip}")
                    if skip.startswith(self._RETRYABLE_SKIP_PREFIXES):
                        continue
                    return

                # C3: Server-side funding-cost rule
                if deriv_ctx:
                    fund_skip = self.derivatives.check_funding_rule(sig.direction, deriv_ctx)
                    if fund_skip:
                        logger.info(f"Skip {product_id} [{strategy.name}]: {fund_skip}")
                        continue

                # Per-symbol trend filter (log-driven, 2026-09-01..10): every
                # NEAR/HYPE/LINK loss was a long entered while price sat below
                # its 1h EMA50 — the global regime gate says nothing about an
                # individual symbol's trend. Mechanically block counter-trend
                # entries; the signal's entry_price may be a level the LLM wants
                # to see filled, so compare the CURRENT price instead.
                i_1h = indicators.get("1h", {})
                trend_price = i_1h.get("price")
                trend_ema = i_1h.get("ema50")
                if trend_price and trend_ema:
                    if sig.direction == "long" and trend_price < trend_ema:
                        logger.info(
                            f"Skip {product_id} [{strategy.name}]: trend filter — long "
                            f"blocked, 1h price {trend_price:.4f} < EMA50 {trend_ema:.4f}"
                        )
                        continue
                    if sig.direction == "short" and trend_price > trend_ema:
                        logger.info(
                            f"Skip {product_id} [{strategy.name}]: trend filter — short "
                            f"blocked, 1h price {trend_price:.4f} > EMA50 {trend_ema:.4f}"
                        )
                        continue

                # Hard 4h-bias gate. Every prompt's decision tree starts with
                # "4H bias must agree — SKIP otherwise", but only the 1h EMA50
                # was enforced server-side; the LLM was free to ignore the
                # 4h step. Backtested 2026-09-18 (see backtest/engine.py
                # require_4h_bias) — enforce it mechanically.
                i_4h = indicators.get("4h", {})
                if self.cfg.require_4h_bias and i_4h.get("price_vs_ema50"):
                    if sig.direction == "long" and i_4h["price_vs_ema50"] != "above":
                        logger.info(f"Skip {product_id} [{strategy.name}]: 4h bias filter — long blocked, 4h below EMA50")
                        continue
                    if sig.direction == "short" and i_4h["price_vs_ema50"] != "below":
                        logger.info(f"Skip {product_id} [{strategy.name}]: 4h bias filter — short blocked, 4h above EMA50")
                        continue

                entry = sig.entry_price or indicators["15m"]["price"]
                spec = self.screener.specs.get(product_id)
                if spec is None:
                    # No contract economics for this product — refuse rather
                    # than size continuously; the exchange fills whole
                    # contracts and a continuous size is fiction in both modes.
                    logger.warning(
                        f"Skip {product_id} [{strategy.name}]: no contract spec from screener"
                    )
                    return
                sl, tp, notional, margin, risk_usdc, contracts = self.risk.calculate_trade_params(
                    sig.direction, entry, indicators["15m"]["atr"], spec=spec,
                )
                if notional <= 0 or margin <= 0 or contracts < 1:
                    continue

                result = await self.executor.enter_position(
                    symbol=product_id,
                    display_name=self.screener.display_names.get(product_id, product_id),
                    direction=sig.direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    size_usdc=notional,
                    margin_usdc=margin,
                    leverage=self.risk.state.leverage,
                    risk_usdc=risk_usdc,
                    strategy=strategy.name,
                    confidence=sig.confidence,
                    reasoning=sig.reasoning,
                    contracts=contracts,
                )
                if result.success:
                    self.risk.protections.record_entry()
                    # get_position only tracks paper positions — None in live mode
                    pos = self.executor.get_position(product_id)
                    if pos is not None:
                        self.monitor.record_entry(pos, indicators["15m"]["atr"])
                    logger.info(f"✅ Position opened: {product_id} {sig.direction} [{strategy.name}]")
                else:
                    logger.warning(f"Failed to open {product_id} [{strategy.name}]: {result.error}")
                return  # entry attempted (success or hard failure) — done with this symbol

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


def _setup_logging() -> None:
    """Durable file sink. Previously logs only went wherever stdout happened
    to be redirected at launch time — if the process was started without a
    shell redirect (e.g. a bare `python -m agent.main` in a terminal), the
    log file silently went stale while the process kept running, hiding a
    later hang for hours."""
    logger.add(
        "logs/agent.log",
        rotation="20 MB",
        retention=5,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def main() -> None:
    _setup_logging()
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
