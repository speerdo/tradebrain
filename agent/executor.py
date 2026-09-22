"""
Executor — Order placement + paper trading

Paper mode: simulate in-memory positions against Coinbase mark prices.
Live mode: place real CFM futures orders via /api/v3/brokerage/orders.

Paper and live share one set of accessors (get_open_positions, has_position,
get_position, close_position) but are otherwise kept strictly apart:

  - every entry and partial is mode-guarded (agent/trading_mode.py): a
    position stamped is_paper=True can never be acted on while the agent is
    LIVE, and vice versa. Closing is exempt — flattening must always work.
  - paper state restores from the trades table (is_paper = TRUE); live state
    restores from the trades table AND is then reconciled against the
    exchange, which wins on every disagreement.
  - live orders record their real exchange identities (client_order_id,
    order_id, stop/exit order ids) and their actual fills, so every row in
    `trades` can be tied back to Coinbase's own order history.
"""

import asyncio
import math
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

import config
from agent import trading_mode
from agent.coinbase_client import CoinbaseClient
from agent.database import get_db


@dataclass
class PaperPosition:
    product_id: str
    display_name: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size_usdc: float
    margin_usdc: float
    leverage: int
    risk_usdc: float
    opened_at: float = field(default_factory=time.time)
    strategy: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    is_paper: bool = True
    status: str = "open"
    exit_price: float | None = None
    pnl_usdc: float = 0.0
    realized_partial: float = 0.0  # PnL already booked by partial take-profits
    fees_usdc: float = 0.0  # Cumulative taker fees across entry + partial + exit legs
    # Whole contracts held (0 = sized continuously, pre-contract-aware rows).
    # Paper tracks this too so partial closes round to whole contracts
    # exactly as live must.
    contracts: int = 0
    tax_treatment: str = "1256"
    product_type: str = "perp"
    # --- Audit identity ---------------------------------------------------
    # The trades row this position owns. Closes used to find their row with
    # "WHERE product_id = ? AND status = 'open' ORDER BY created_at DESC" —
    # which, with paper and live rows in one table, could update the OTHER
    # mode's open row for the same product. Carrying the id removes the
    # guesswork entirely.
    trade_id: int = 0
    # Exchange order identities (live only; empty in paper).
    client_order_id: str = ""     # ours — survives a timed-out request
    entry_order_id: str = ""
    exit_order_id: str = ""

    @property
    def symbol(self) -> str:
        """Backward-compat alias — `product_id` is the canonical exchange identifier."""
        return self.product_id

    @property
    def mode(self) -> str:
        return trading_mode.mode_of(self.is_paper)


@dataclass
class OrderResult:
    success: bool
    order_id: str | None = None
    error: str = ""
    filled_price: float | None = None


class Executor:

    def __init__(self, cb: CoinbaseClient):
        self.cfg = config.get_config()
        self.cb = cb
        self.paper_positions: dict[str, PaperPosition] = {}
        self.live_positions: dict[str, PaperPosition] = {}
        # Exchange-side metadata per live product: contracts, contract_size,
        # unrealized_pnl, protective stop order id, etc.
        self.live_meta: dict[str, dict] = {}
        self._contract_sizes: dict[str, float] = {}
        self._notifier: Any = None
        self._risk: Any = None
        # Fill audits run detached from the order path (the fills endpoint
        # lags the order ack by a beat, and an entry must not block on it).
        # Held in a set so the loop keeps a strong reference — a bare
        # create_task can be garbage-collected mid-flight.
        self._audit_tasks: set[asyncio.Task] = set()

    def set_notifier(self, notifier: Any) -> None:
        self._notifier = notifier

    def set_risk_manager(self, risk: Any) -> None:
        """Wired by main.py so externally-closed live positions still reach the
        circuit breaker and the protections (see _report_external_close)."""
        self._risk = risk

    async def enter_position(
        self, symbol: str, direction: str, entry_price: float,
        stop_loss: float, take_profit: float, size_usdc: float,
        margin_usdc: float, leverage: int, risk_usdc: float,
        strategy: str = "", confidence: float = 0.0, reasoning: str = "",
        display_name: str = "", product_type: str = "perp",
        contracts: int = 0,
    ) -> OrderResult:
        # The hard mode check, at the one place every entry passes through.
        # `paper` is read ONCE here and carried into the branch, so a config
        # hot-reload landing mid-entry cannot route to the live exchange and
        # then stamp the row is_paper=True (or the reverse).
        paper = bool(self.cfg.paper_trading)
        drift = trading_mode.check_boot_drift(self.cfg)
        if drift:
            logger.error(f"🚫 Refusing entry for {symbol} — {drift}")
            return OrderResult(success=False, error=f"Mode drift: {drift}")
        if paper:
            return await self._enter_paper(
                symbol, display_name or symbol, direction, entry_price,
                stop_loss, take_profit, size_usdc, margin_usdc, leverage,
                risk_usdc, strategy, confidence, reasoning, contracts,
            )
        return await self._enter_live(
            symbol, display_name or symbol, direction, entry_price,
            stop_loss, take_profit, size_usdc, margin_usdc, leverage,
            risk_usdc, strategy, confidence, reasoning, contracts,
        )

    def _guard(self, pos_is_paper: bool, action: str) -> str:
        """Mode guard for an action on a position. Returns "" or the refusal."""
        reason = trading_mode.guard_trade(pos_is_paper, action, self.cfg)
        if reason:
            logger.error(f"🚫 {reason}")
        return reason

    async def _enter_paper(self, product_id: str, display_name: str, direction: str,
                            entry_price: float, stop_loss: float, take_profit: float,
                            size_usdc: float, margin_usdc: float, leverage: int,
                            risk_usdc: float, strategy: str, confidence: float,
                            reasoning: str, contracts: int = 0) -> OrderResult:
        # is_paper=True is about to be stamped on this position — refuse if
        # the agent is actually LIVE. This is the mismatch that made a real
        # account report simulated trades.
        guard = self._guard(True, f"open a PAPER position in {product_id}")
        if guard:
            return OrderResult(success=False, error=guard)
        if product_id in self.paper_positions:
            return OrderResult(success=False, error=f"Already open: {product_id}")
        entry_fee = self.fee_for_leg(size_usdc)
        pos = PaperPosition(
            product_id=product_id, display_name=display_name, direction=direction,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size_usdc=size_usdc, margin_usdc=margin_usdc, leverage=leverage,
            risk_usdc=risk_usdc, strategy=strategy, confidence=confidence,
            reasoning=reasoning, is_paper=True, fees_usdc=entry_fee,
            contracts=contracts,
        )
        self.paper_positions[product_id] = pos
        logger.info(
            f"📄 PAPER ENTRY: {direction.upper()} {display_name} @ {entry_price:.2f} "
            f"{contracts} contract(s) ~${size_usdc:,.0f} margin ${margin_usdc:.2f} "
            f"risk ${risk_usdc:.2f} (entry fee ${entry_fee:.2f})"
        )
        pos.trade_id = await self._log_trade(pos, order_id="")
        if self._notifier:
            try:
                await self._notifier.notify_trade_opened(
                    display_name, direction, entry_price, size_usdc,
                    leverage, stop_loss, take_profit,
                )
            except Exception as exc:
                logger.warning(f"Entry notification failed: {exc}")
        return OrderResult(success=True, order_id=f"paper-{product_id}")

    async def close_position(self, product_id: str, exit_price: float | None = None) -> OrderResult:
        """
        Flatten a position in either book.

        Deliberately NOT mode-guarded: if a mismatch ever does put a position
        in the wrong book, refusing to close it would strand it. The
        mismatch is logged instead, loudly, and the close proceeds.
        """
        pos = self.paper_positions.get(product_id)
        if pos is not None:
            if pos.is_paper is not bool(self.cfg.paper_trading):
                logger.warning(
                    f"Closing {product_id} from the PAPER book while the agent is "
                    f"{trading_mode.current_mode(self.cfg)} — mode mismatch, "
                    "closing anyway so nothing is stranded"
                )
            return await self._close_paper(pos, exit_price)
        pos = self.live_positions.get(product_id)
        if pos is not None:
            if pos.is_paper is not bool(self.cfg.paper_trading):
                logger.warning(
                    f"Closing {product_id} from the LIVE book while the agent is "
                    f"{trading_mode.current_mode(self.cfg)} — mode mismatch, "
                    "closing anyway so nothing is stranded"
                )
            return await self._close_live(pos, exit_price)
        return OrderResult(success=False, error=f"No position: {product_id}")

    async def _close_paper(self, pos: PaperPosition, exit_price: float | None) -> OrderResult:
        if exit_price is None:
            exit_price = pos.entry_price
        # Final PnL = PnL on the remaining size + anything already banked
        # by partial take-profits, net of every fee leg paid so far
        # (entry + any partials) plus this exit fill.
        pos.fees_usdc += self.fee_for_leg(pos.size_usdc)
        pnl = self._calc_pnl(pos, exit_price) + pos.realized_partial - pos.fees_usdc
        pos.exit_price = exit_price
        pos.pnl_usdc = pnl
        pos.status = "closed"
        del self.paper_positions[pos.product_id]
        logger.info(
            f"📄 PAPER CLOSE: {pos.display_name} @ {exit_price:.2f} "
            f"P&L=${pnl:+.2f} (fees ${pos.fees_usdc:.2f})"
        )
        await self._update_trade_close(pos)
        await self._apply_paper_pnl(pnl)
        if self._notifier:
            try:
                await self._notifier.notify_trade_closed(
                    pos.display_name, pos.direction, pos.entry_price,
                    exit_price, pnl,
                )
            except Exception as exc:
                logger.warning(f"Close notification failed: {exc}")
        return OrderResult(success=True, filled_price=exit_price)

    async def _close_live(self, pos: PaperPosition, exit_price: float | None = None) -> OrderResult:
        """Close a live CFM position via the dedicated close-position endpoint."""
        try:
            meta = self.live_meta.get(pos.product_id, {})
            exit_client_order_id = secrets.token_hex(16)
            result = await self.cb.close_futures_position(
                pos.product_id, client_order_id=exit_client_order_id,
            )
            if not result.get("success", False):
                err = str(result.get("error_response", {}))[:300]
                logger.error(
                    f"Close rejected for {pos.product_id} "
                    f"(client_order_id={exit_client_order_id}): {err}"
                )
                return OrderResult(success=False, error=f"Exchange close failed: {err}")
            pos.exit_order_id = (result.get("success_response", {}) or {}).get("order_id", "")
            # The protective stop is NOT reduce-only. Left open after the
            # position is flat it would, on trigger, OPEN a new position in the
            # opposite direction. Cancel it before anything else.
            await self._cancel_protective_stop(pos.product_id)
            # Prefer the caller's exit price (PositionMonitor passes the mark or
            # the SL/TP level it fired on), then the last reconciled mark. Only
            # fall back to entry_price as a last resort — that yields PnL≈0 and
            # would hide a real loss from the circuit breaker.
            exit_price = (
                exit_price
                or meta.get("current_price")
                or pos.entry_price
            )
            pos.fees_usdc += self.fee_for_leg(pos.size_usdc)
            pnl = self._calc_pnl(pos, exit_price) + pos.realized_partial - pos.fees_usdc
            pos.exit_price = exit_price
            pos.pnl_usdc = pnl
            pos.status = "closed"
            self.live_positions.pop(pos.product_id, None)
            self.live_meta.pop(pos.product_id, None)
            logger.info(
                f"💰 LIVE CLOSE: {pos.display_name} @ ~{exit_price:.2f} "
                f"P&L≈${pnl:+.2f} (fees ${pos.fees_usdc:.2f}) "
                f"exit_order_id={pos.exit_order_id or 'unknown'}"
            )
            await self._update_trade_close(
                pos, exit_client_order_id=exit_client_order_id,
            )
            # The modeled exit price above is an estimate; the fill audit
            # replaces it with what the exchange actually filled at, and
            # books the real commission onto the row.
            self._audit_order_later(pos, pos.exit_order_id, leg="exit")
            if self._notifier:
                try:
                    await self._notifier.notify_trade_closed(
                        pos.display_name, pos.direction, pos.entry_price,
                        exit_price, pnl,
                    )
                except Exception as exc:
                    logger.warning(f"Close notification failed: {exc}")
            return OrderResult(success=True, filled_price=exit_price)
        except Exception as exc:
            logger.error(f"Live close failed for {pos.product_id}: {exc}")
            return OrderResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Live entry (CFM futures)
    # ------------------------------------------------------------------

    async def _get_contract_size(self, product_id: str) -> float:
        """Contract multiplier (e.g. 0.01 BTC per contract) — cached per product."""
        if product_id not in self._contract_sizes:
            details = await self.cb.hydrate_product_details(product_id)
            cs = details.get("contract_size")
            if not cs or cs <= 0:
                raise ValueError(f"No contract_size for {product_id} — refusing live entry")
            self._contract_sizes[product_id] = cs
        return self._contract_sizes[product_id]

    async def _enter_live(
        self, product_id: str, display_name: str, direction: str,
        entry_price: float, stop_loss: float, take_profit: float,
        size_usdc: float, margin_usdc: float, leverage: int,
        risk_usdc: float, strategy: str, confidence: float,
        reasoning: str, contracts: int = 0,
    ) -> OrderResult:
        """Place a real CFM market order + exchange-native protective stop."""
        # is_paper=False is about to be stamped — refuse if the agent is in
        # PAPER. Real money must never move because a stale flag said LIVE.
        guard = self._guard(False, f"open a LIVE position in {product_id}")
        if guard:
            return OrderResult(success=False, error=guard)
        if product_id in self.live_positions or product_id in self.paper_positions:
            return OrderResult(success=False, error=f"Already open: {product_id}")
        try:
            contract_size = await self._get_contract_size(product_id)
            if contracts < 1:
                # Legacy continuous sizing — round DOWN to whole contracts so
                # live notional never exceeds the risk-sized notional.
                contracts = math.floor(size_usdc / (entry_price * contract_size))
            if contracts < 1:
                return OrderResult(
                    success=False,
                    error=(
                        f"Size too small for {product_id}: ${size_usdc:.2f} < 1 contract "
                        f"(${entry_price * contract_size:,.2f})"
                    ),
                )
            # Whatever the caller estimated, the exchange fills whole contracts:
            # derive notional AND $-at-risk from the contract count so every
            # downstream R-multiple uses the real position.
            live_notional = contracts * entry_price * contract_size
            if size_usdc > 0 and abs(live_notional - size_usdc) / size_usdc > 1e-6:
                risk_usdc = risk_usdc * live_notional / size_usdc
                margin_usdc = margin_usdc * live_notional / size_usdc

            side = "BUY" if direction == "long" else "SELL"
            # Our own id, minted before the request. If the call times out
            # after the exchange accepted the order, this is the only handle
            # that can find it again in Coinbase's history.
            client_order_id = secrets.token_hex(16)
            entry = await self.cb.place_futures_market_order(
                product_id, side, contracts, leverage=leverage, margin_type="ISOLATED",
                client_order_id=client_order_id,
            )
            if not entry.get("success", False):
                err = str(entry.get("error_response", {}))[:300]
                logger.error(
                    f"Entry order rejected for {product_id} "
                    f"(client_order_id={client_order_id}): {err}"
                )
                return OrderResult(success=False, error=f"Entry order rejected: {err}")
            success_response = entry.get("success_response", {})
            entry_id = success_response.get("order_id", "")
            logger.info(
                f"💰 LIVE ENTRY: {direction.upper()} {display_name} "
                f"{contracts} contract(s) (~${contracts * entry_price * contract_size:,.0f}) "
                f"order_id={entry_id} client_order_id={client_order_id}"
            )

            stop_order_id = None
            if stop_loss and stop_loss > 0:
                stop_order_id = await self._place_protective_stop(
                    product_id, direction, contracts, stop_loss,
                )

            pos = PaperPosition(
                product_id=product_id, display_name=display_name,
                direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                size_usdc=live_notional,
                margin_usdc=margin_usdc, leverage=leverage,
                risk_usdc=risk_usdc, strategy=strategy,
                confidence=confidence, reasoning=reasoning,
                is_paper=False, fees_usdc=self.fee_for_leg(live_notional),
                contracts=contracts,
                client_order_id=client_order_id, entry_order_id=entry_id,
            )
            self.live_positions[product_id] = pos
            self.live_meta[product_id] = {
                "contracts": contracts,
                "contract_size": contract_size,
                "entry_order_id": entry_id,
                "stop_order_id": stop_order_id,
            }
            pos.trade_id = await self._log_trade(
                pos, order_id=entry_id, stop_order_id=stop_order_id,
            )
            # Pull the real fills (price, size, commission) onto the row in
            # the background — the fills endpoint lags the order ack, and an
            # entry must not sit waiting on an audit record.
            self._audit_order_later(pos, entry_id, leg="entry")
            if self._notifier:
                try:
                    await self._notifier.notify_trade_opened(
                        display_name, direction, entry_price, pos.size_usdc,
                        leverage, stop_loss, take_profit,
                    )
                except Exception as exc:
                    logger.warning(f"Entry notification failed: {exc}")
            return OrderResult(success=True, order_id=entry_id, filled_price=entry_price)
        except Exception as exc:
            logger.error(f"Live entry failed for {product_id}: {exc}")
            return OrderResult(success=False, error=str(exc))

    async def _place_protective_stop(
        self, product_id: str, direction: str, contracts: int, stop_loss: float,
    ) -> str | None:
        """
        Exchange-native stop so the position is protected if the bot dies.
        Best-effort: failure is alerted loudly but does not abort the entry —
        the PositionMonitor still manages the stop locally.
        """
        long = direction == "long"
        stop_direction = "STOP_DIRECTION_STOP_DOWN" if long else "STOP_DIRECTION_STOP_UP"
        # Give the stop-limit a 0.5% price band in the trigger direction
        limit_price = stop_loss * (0.995 if long else 1.005)
        try:
            res = await self.cb.place_futures_stop_order(
                product_id,
                side="SELL" if long else "BUY",
                contracts=contracts,
                stop_price=stop_loss,
                limit_price=limit_price,
                stop_direction=stop_direction,
            )
            if res.get("success", False):
                order_id = res.get("success_response", {}).get("order_id")
                logger.info(f"🛡️  Exchange stop placed: {product_id} @ {stop_loss:.2f} ({order_id})")
                return order_id
            logger.error(
                f"🛡️  EXCHANGE STOP REJECTED for {product_id}: "
                f"{str(res.get('error_response', {}))[:300]}"
            )
            await self._alert(
                f"⚠️ Live entry without exchange stop: {product_id} @ SL {stop_loss:.2f} "
                "— position protected only while the bot runs."
            )
        except Exception as exc:
            logger.error(f"Protective stop failed for {product_id}: {exc}")
            await self._alert(
                f"⚠️ Live entry without exchange stop: {product_id} ({exc})"
            )
        return None

    async def _cancel_protective_stop(self, product_id: str) -> None:
        """
        Cancel the exchange-native stop for a product, if we placed one.

        This MUST run whenever the position goes flat or changes size. The stop
        is a plain stop-limit, not reduce-only: an orphaned one that triggers
        opens a brand-new position in the opposite direction, and an oversized
        one left after a partial close does the same with the excess contracts.
        """
        meta = self.live_meta.get(product_id, {})
        order_id = meta.get("stop_order_id")
        if not order_id:
            return
        try:
            await self.cb.cancel_orders([order_id])
            logger.info(f"🛡️  Cancelled exchange stop {order_id} for {product_id}")
        except Exception as exc:
            # Loud: an uncancelled stop is a latent reverse-position risk.
            logger.error(f"Failed to cancel exchange stop {order_id} for {product_id}: {exc}")
            await self._alert(
                f"Could not cancel the exchange stop for {product_id} (order {order_id}). "
                "Cancel it manually — if it triggers it will OPEN a reversed position."
            )
        finally:
            meta.pop("stop_order_id", None)

    async def sync_live_stop(self, pos: PaperPosition) -> None:
        """
        Re-place the exchange stop after the bot ratchets `pos.stop_loss`
        (breakeven move / trailing stop). Cancel-then-replace: the exchange has
        no amend for stop-limit, and leaving the old one would double the size
        protected.
        """
        if pos.is_paper or pos.stop_loss <= 0:
            return
        meta = self.live_meta.get(pos.product_id, {})
        contracts = int(meta.get("contracts") or 0)
        if contracts < 1:
            return
        await self._cancel_protective_stop(pos.product_id)
        new_id = await self._place_protective_stop(
            pos.product_id, pos.direction, contracts, pos.stop_loss,
        )
        meta["stop_order_id"] = new_id
        # Keep the audit row pointing at the stop that is actually resting on
        # the exchange, so a stop-fill can be traced back to this trade.
        if new_id and pos.trade_id:
            try:
                db = await get_db()
                await db.update_trade_orders(pos.trade_id, stop_order_id=new_id)
            except Exception as exc:
                logger.warning(f"Could not record replacement stop order id: {exc}")

    async def _alert(self, message: str) -> None:
        if self._notifier:
            try:
                await self._notifier.notify_alert("Live trading alert", message)
            except Exception as exc:
                logger.warning(f"Alert notification failed: {exc}")

    # ------------------------------------------------------------------
    # Reconciliation — the exchange is the source of truth
    # ------------------------------------------------------------------

    async def reconcile_live_positions(self) -> list[PaperPosition]:
        """
        Sync local live positions with get_futures_positions().

        - Exchange position we don't know about → adopt it (manual trade,
          restart with lost state) with stop_loss=0 (conservative in risk math)
        - Local live position missing on exchange → it closed externally;
          drop it, close the DB row, alert
        """
        try:
            exchange = await self.cb.get_futures_positions()
        except Exception as exc:
            logger.error(f"Live reconciliation failed — keeping last known state: {exc}")
            return list(self.live_positions.values())

        by_product: dict[str, dict] = {}
        for raw in exchange:
            side = (raw.get("side") or "").upper()
            contracts = float(raw.get("number_of_contracts") or 0)
            if side in ("LONG", "SHORT") and contracts != 0:
                by_product[raw["product_id"]] = raw

        adopted: list[PaperPosition] = []
        # 1. Exchange → local: adopt unknowns, refresh knowns
        for product_id, raw in by_product.items():
            pos = self.live_positions.get(product_id)
            if pos is None:
                pos = await self._adopt_live_position(raw)
                adopted.append(pos)
            meta = self.live_meta.setdefault(product_id, {
                "contracts": float(raw.get("number_of_contracts") or 0),
                "contract_size": self._contract_sizes.get(product_id, 1.0),
            })
            meta.update({
                "current_price": float(raw.get("current_price") or 0) or None,
                "unrealized_pnl": float(raw.get("unrealized_pnl") or 0),
                "daily_realized_pnl": float(raw.get("daily_realized_pnl") or 0),
            })
            # Adopt the exchange's view of entry price and size. _enter_live
            # records the pre-trade *estimate*; a market order slips, and
            # without this the stop distance, R-multiples, and every PnL for
            # the life of the position stay anchored to a price that never
            # traded. The exchange is the source of truth.
            actual_entry = float(raw.get("avg_entry_price") or 0)
            if actual_entry > 0 and abs(actual_entry - pos.entry_price) > 1e-9:
                if pos.entry_price > 0:
                    slip_pct = (actual_entry - pos.entry_price) / pos.entry_price * 100
                    logger.info(
                        f"Reconciled {product_id} entry {pos.entry_price:.2f} → "
                        f"{actual_entry:.2f} ({slip_pct:+.3f}% slippage)"
                    )
                pos.entry_price = actual_entry
            actual_contracts = float(raw.get("number_of_contracts") or 0)
            cs = meta.get("contract_size") or self._contract_sizes.get(product_id)
            if actual_contracts and cs:
                meta["contracts"] = actual_contracts
                pos.size_usdc = actual_contracts * cs * pos.entry_price
        # 2. Local → exchange: positions we think are open but aren't
        for product_id in list(self.live_positions):
            if product_id not in by_product:
                pos = self.live_positions[product_id]
                meta = self.live_meta.get(product_id, {})
                # A stop that fired is the most likely reason the position
                # vanished, but we can't tell — cancel any stop we still own
                # so a stale one can't reopen the trade in reverse.
                await self._cancel_protective_stop(product_id)
                self.live_positions.pop(product_id, None)
                self.live_meta.pop(product_id, None)
                # Estimate the exit from the last reconciled mark. Booking $0
                # here would hide a real loss from the circuit breaker and from
                # the protections' R-tracking.
                exit_price = meta.get("current_price") or pos.stop_loss or pos.entry_price
                pos.fees_usdc += self.fee_for_leg(pos.size_usdc)
                pnl = self._calc_pnl(pos, exit_price) + pos.realized_partial - pos.fees_usdc
                pos.exit_price = exit_price
                pos.pnl_usdc = pnl
                pos.status = "closed"
                logger.warning(
                    f"⚠️ RECONCILIATION: {product_id} open locally but not on the "
                    f"exchange — dropping local state (external close?). "
                    f"Estimated exit ~{exit_price:.2f}, P&L≈${pnl:+.2f}"
                )
                await self._alert(
                    f"Position {pos.display_name} disappeared from the exchange "
                    f"(stop filled, or closed manually). Estimated P&L ≈ ${pnl:+.2f} "
                    "— booked from the last known mark, not an actual fill."
                )
                await self._update_trade_close(pos)
                self._report_external_close(pos, pnl)
        return list(self.live_positions.values())

    def _report_external_close(self, pos: PaperPosition, pnl: float) -> None:
        """
        Feed an externally-closed position into risk tracking.

        PositionMonitor._handle_exit normally does this, but it never sees a
        position that vanished between ticks — without this the daily loss
        never accumulates and the circuit breaker stays silent through a
        losing streak that the exchange executed on our behalf.
        """
        if self._risk is None:
            return
        stop_distance = abs(pos.entry_price - pos.stop_loss)
        risk_usd = (stop_distance / pos.entry_price) * pos.size_usdc if pos.entry_price else 0
        pnl_r = pnl / risk_usd if risk_usd > 0 else 0.0
        try:
            self._risk.apply_loss(pnl, symbol=pos.product_id, pnl_r=pnl_r,
                                  is_paper=pos.is_paper)
        except Exception as exc:
            logger.warning(f"Risk accounting for external close failed: {exc}")

    async def _adopt_live_position(self, raw: dict) -> PaperPosition:
        """Map an FCMPosition we didn't open into our shared position shape."""
        product_id = raw["product_id"]
        side = (raw.get("side") or "").upper()
        contracts = float(raw.get("number_of_contracts") or 0)
        entry = float(raw.get("avg_entry_price") or 0)
        try:
            contract_size = await self._get_contract_size(product_id)
        except Exception:
            contract_size = 0.0
        notional = contracts * contract_size * entry if contract_size else 0.0
        pos = PaperPosition(
            product_id=product_id, display_name=product_id,
            direction="long" if side == "LONG" else "short",
            entry_price=entry,
            # Unknown stop → 0.0: risk math reads max risk per position
            # (full notional at risk), which blocks new entries rather than
            # understating exposure.
            stop_loss=0.0, take_profit=0.0,
            size_usdc=notional, margin_usdc=0.0, leverage=1,
            risk_usdc=notional, strategy="adopted", confidence=0.0,
            reasoning="Reconciled from exchange — not opened by this bot",
            is_paper=False,
        )
        self.live_positions[product_id] = pos
        self.live_meta[product_id] = {
            "contracts": contracts,
            "contract_size": contract_size,
        }
        # Give it a trades row. An adopted position that closes later still
        # books P&L into the circuit breaker, so it needs somewhere to land —
        # and a live position with no row is a hole in the audit trail
        # against Coinbase's history.
        pos.contracts = int(contracts)
        pos.trade_id = await self._log_trade(pos, order_id="")
        logger.warning(
            f"⚠️ RECONCILIATION: adopted unknown exchange position {product_id} "
            f"{'LONG' if side == 'LONG' else 'SHORT'} {contracts} contracts "
            f"@ {entry:.2f} (~${notional:,.0f})"
        )
        await self._alert(
            f"Adopted position not opened by this bot: {product_id} "
            f"{'LONG' if side == 'LONG' else 'SHORT'} {contracts} contracts @ {entry:.2f}.\n"
            "It has no tracked stop, so it counts as FULL NOTIONAL at risk — that will "
            "exceed the portfolio risk cap and **block all new entries** until it is "
            "closed. It is also monitor-only: the bot will not stop it out or take "
            "profit on it. Close it manually or via Burt to resume trading."
        )
        return pos

    async def reduce_position(self, product_id: str, exit_price: float,
                              fraction: float) -> OrderResult:
        """
        Close `fraction` (0..1) of a position at exit_price — partial take-profit.

        Books the realized PnL on the closed portion into `realized_partial`
        (included in the final PnL when the remainder closes) and shrinks
        size/margin proportionally. Paper mode simulates; live issues a
        reduce-only close for the fraction of contracts.
        """
        fraction = max(0.0, min(1.0, fraction))
        if fraction <= 0:
            return OrderResult(success=False, error="Fraction must be > 0")

        pos = self.paper_positions.get(product_id) or self.live_positions.get(product_id)
        if pos is not None:
            # Same hard check as an entry: a partial changes size and
            # re-places the protective stop, so it must not run against a
            # position belonging to the other mode.
            guard = self._guard(pos.is_paper, f"reduce {product_id}")
            if guard:
                return OrderResult(success=False, error=guard)

        pos = self.paper_positions.get(product_id)
        if pos is not None:
            # Whole-contract parity: a 1-contract paper position cannot be
            # split any more than a live one can.
            if pos.contracts > 0:
                close_contracts = math.floor(pos.contracts * fraction)
                if close_contracts < 1:
                    return OrderResult(success=False, error="Fraction rounds to 0 contracts")
                fraction = close_contracts / pos.contracts
                pos.contracts -= close_contracts
            closed_notional = pos.size_usdc * fraction
            if pos.direction == "long":
                realized = (exit_price - pos.entry_price) / pos.entry_price * closed_notional
            else:
                realized = (pos.entry_price - exit_price) / pos.entry_price * closed_notional

            partial_fee = self.fee_for_leg(closed_notional)
            pos.realized_partial += realized
            pos.fees_usdc += partial_fee
            pos.size_usdc -= closed_notional
            pos.margin_usdc *= (1 - fraction)
            logger.info(
                f"📄 PAPER PARTIAL: closed {fraction:.0%} of {pos.display_name} "
                f"@ {exit_price:.2f} banked ${realized:+.2f} gross (fee ${partial_fee:.2f}) "
                f"(remaining ${pos.size_usdc:.2f})"
            )
            return OrderResult(success=True, filled_price=exit_price)

        pos = self.live_positions.get(product_id)
        if pos is not None:
            meta = self.live_meta.get(product_id, {})
            contracts = int(meta.get("contracts") or 0)
            close_contracts = math.floor(contracts * fraction)
            if close_contracts < 1:
                return OrderResult(success=False, error="Fraction rounds to 0 contracts")
            try:
                partial_client_order_id = secrets.token_hex(16)
                result = await self.cb.close_futures_position(
                    product_id, size=str(close_contracts),
                    client_order_id=partial_client_order_id,
                )
                if not result.get("success", False):
                    err = str(result.get("error_response", {}))[:300]
                    return OrderResult(success=False, error=f"Exchange partial close failed: {err}")
                partial_order_id = (result.get("success_response", {}) or {}).get("order_id", "")
                closed_notional = pos.size_usdc * fraction
                if pos.direction == "long":
                    realized = (exit_price - pos.entry_price) / pos.entry_price * closed_notional
                else:
                    realized = (pos.entry_price - exit_price) / pos.entry_price * closed_notional
                partial_fee = self.fee_for_leg(closed_notional)
                pos.realized_partial += realized
                pos.fees_usdc += partial_fee
                pos.size_usdc -= closed_notional
                pos.margin_usdc *= (1 - fraction)
                meta["contracts"] = contracts - close_contracts
                pos.contracts = contracts - close_contracts
                logger.info(
                    f"💰 LIVE PARTIAL: closed {close_contracts} contract(s) of "
                    f"{pos.display_name} @ ~{exit_price:.2f} banked ${realized:+.2f} gross "
                    f"(fee ${partial_fee:.2f}) order_id={partial_order_id or 'unknown'}"
                )
                self._audit_order_later(pos, partial_order_id, leg="partial")
                # The old stop still covers the pre-partial contract count.
                # Resize it, or on trigger it closes the remainder and opens a
                # reversed position with the excess.
                await self.sync_live_stop(pos)
                return OrderResult(success=True, filled_price=exit_price)
            except Exception as exc:
                logger.error(f"Live partial close failed for {product_id}: {exc}")
                return OrderResult(success=False, error=str(exc))

        return OrderResult(success=False, error=f"No position: {product_id}")

    @staticmethod
    def _calc_pnl(pos: PaperPosition, current: float) -> float:
        if pos.direction == "long":
            return (current - pos.entry_price) / pos.entry_price * pos.size_usdc
        return (pos.entry_price - current) / pos.entry_price * pos.size_usdc

    def fee_for_leg(self, notional_usdc: float) -> float:
        """Modeled taker fee for one fill (market orders only — entries and
        exits are both taker). Coinbase's retail rate is a percentage with a
        per-transaction minimum; at our position sizes the minimum is what
        actually bites, not the percentage."""
        return max(abs(notional_usdc) * self.cfg.taker_fee_pct, self.cfg.min_fee_usdc)

    def get_open_positions(self) -> list[PaperPosition]:
        """Open positions in BOTH modes — one accessor, paper + live."""
        return list(self.paper_positions.values()) + list(self.live_positions.values())

    async def restore_paper_positions(self) -> list[PaperPosition]:
        """
        Rebuild in-memory paper positions from the trades table.

        Paper positions otherwise live only in memory, so every restart
        silently forgets open trades — their DB rows stay 'open' forever and
        nothing ever stops them out or takes profit. The trades row is written
        at entry (log_trade) and updated at close, so it is a complete source
        of truth for open paper positions.

        Refuses to run in LIVE mode: restoring simulated positions into a
        live session would have them block real entries (has_position), get
        "managed" by the position monitor, and report as open exposure the
        exchange knows nothing about.
        """
        if not self.cfg.paper_trading:
            logger.info(
                "Skipping paper-position restore — agent is LIVE "
                "(paper rows stay open in the DB and are ignored)"
            )
            return []
        try:
            db = await get_db()
            rows = await db.fetch(
                """
                SELECT * FROM trades
                WHERE status = 'open' AND is_paper = TRUE
                  AND created_at >= NOW() - INTERVAL '7 days'
                ORDER BY created_at ASC
                """
            )
        except Exception as exc:
            logger.error(f"Paper position restore failed: {exc}")
            return []

        restored: list[PaperPosition] = []
        for row in rows:
            # Stale test rows (e.g. the legacy ONDO entries with absurd
            # margins) must not resurrect as managed positions. Anything
            # without a usable product_id is unadoptable; the row can be
            # closed manually or via SQL.
            product_id = row["product_id"] or ""
            if not product_id:
                logger.warning(
                    f"⚠️ RESTORE: trade {row['id']} ({row['symbol']}) has no "
                    "product_id — cannot restore, leaving row open"
                )
                continue
            if product_id in self.paper_positions:
                continue  # duplicate row for a product we already hold
            pos = PaperPosition(
                product_id=product_id,
                display_name=row["display_name"] or row["symbol"],
                direction=row["direction"],
                entry_price=float(row["entry_price"]),
                stop_loss=float(row["stop_loss"] or 0.0),
                take_profit=float(row["take_profit"] or 0.0),
                size_usdc=float(row["size_usdc"]),
                margin_usdc=float(row["margin_usdc"] or 0.0),
                leverage=int(row["leverage"] or 1),
                risk_usdc=float(row["risk_usdc"] or 0.0),
                opened_at=row["created_at"].timestamp() if row["created_at"] else time.time(),
                strategy=row["strategy"] or "",
                confidence=float(row["confidence"] or 0.0),
                reasoning=row["reasoning"] or "Restored from DB on restart",
                is_paper=True,
                realized_partial=float(row["realized_partial"] or 0.0)
                    if "realized_partial" in row.keys() else 0.0,
                fees_usdc=float(row["fees_usdc"] or 0.0)
                    if "fees_usdc" in row.keys() else self.fee_for_leg(float(row["size_usdc"])),
                contracts=int(row["contracts"] or 0) if "contracts" in row.keys() else 0,
                trade_id=int(row["id"]),
            )
            self.paper_positions[product_id] = pos
            restored.append(pos)
        if restored:
            logger.info(
                f"♻️  Restored {len(restored)} open paper position(s) from DB: "
                + ", ".join(f"{p.display_name} {p.direction} @ {p.entry_price:.2f}" for p in restored)
            )
        return restored

    async def restore_live_positions(self) -> list[PaperPosition]:
        """
        Rebuild live positions from the trades table, BEFORE reconciling.

        Without this, a restart lost every live position's stop, take-profit,
        strategy and risk — `reconcile_live_positions` saw each one as an
        unknown exchange position and "adopted" it with stop_loss=0, which
        marks it monitor-only, counts it as full notional at risk, and
        therefore blocks all new entries until it is closed by hand. The DB
        row already holds the real levels; load them first and let
        reconciliation correct the details against the exchange.

        The exchange still wins: anything restored here that is NOT open on
        the exchange is closed out by the reconcile pass immediately after.
        """
        if self.cfg.paper_trading:
            return []
        try:
            db = await get_db()
            rows = await db.fetch(
                """
                SELECT * FROM trades
                WHERE status = 'open' AND is_paper = FALSE
                  AND created_at >= NOW() - INTERVAL '30 days'
                ORDER BY created_at ASC
                """
            )
        except Exception as exc:
            logger.error(f"Live position restore failed: {exc}")
            return []

        restored: list[PaperPosition] = []
        for row in rows:
            product_id = row["product_id"] or ""
            if not product_id or product_id in self.live_positions:
                continue
            pos = PaperPosition(
                product_id=product_id,
                display_name=row["display_name"] or row["symbol"],
                direction=row["direction"],
                entry_price=float(row["entry_price"]),
                stop_loss=float(row["stop_loss"] or 0.0),
                take_profit=float(row["take_profit"] or 0.0),
                size_usdc=float(row["size_usdc"]),
                margin_usdc=float(row["margin_usdc"] or 0.0),
                leverage=int(row["leverage"] or 1),
                risk_usdc=float(row["risk_usdc"] or 0.0),
                opened_at=row["created_at"].timestamp() if row["created_at"] else time.time(),
                strategy=row["strategy"] or "",
                confidence=float(row["confidence"] or 0.0),
                reasoning=row["reasoning"] or "Restored from DB on restart",
                is_paper=False,
                realized_partial=float(row["realized_partial"] or 0.0),
                fees_usdc=float(row["fees_usdc"] or 0.0),
                contracts=int(row["contracts"] or 0),
                trade_id=int(row["id"]),
                client_order_id=row["client_order_id"] or "",
                entry_order_id=row["order_id"] or "",
            )
            # Hydrate the contract multiplier now (cached, one call per
            # product). reconcile_live_positions only adopts the exchange's
            # contract count when it knows the multiplier; without this a
            # restored position would keep the DB's contract count forever
            # instead of self-correcting against the exchange — and
            # sync_live_stop would re-place the stop for the wrong size.
            try:
                contract_size = await self._get_contract_size(product_id)
            except Exception as exc:
                contract_size = 0.0
                logger.warning(
                    f"No contract size for restored {product_id} ({exc}) — "
                    "its size will not reconcile against the exchange"
                )
            self.live_positions[product_id] = pos
            self.live_meta[product_id] = {
                "contracts": float(pos.contracts or 0),
                "contract_size": contract_size,
                "entry_order_id": pos.entry_order_id,
                "stop_order_id": row["stop_order_id"],
            }
            restored.append(pos)
        if restored:
            logger.info(
                f"♻️  Restored {len(restored)} open LIVE position(s) from DB "
                "(levels kept; the exchange reconcile pass runs next): "
                + ", ".join(
                    f"{p.display_name} {p.direction} @ {p.entry_price:.2f} "
                    f"SL {p.stop_loss:.2f}" for p in restored
                )
            )
        return restored

    def has_position(self, product_id: str) -> bool:
        return product_id in self.paper_positions or product_id in self.live_positions

    def get_position(self, product_id: str) -> PaperPosition | None:
        return self.paper_positions.get(product_id) or self.live_positions.get(product_id)

    async def _log_trade(self, pos: PaperPosition, order_id: str,
                         stop_order_id: str | None = None) -> int:
        """Write the opening trade row. Returns its id (0 on failure).

        The id is the position's audit handle for the rest of its life —
        every later update (fills, order ids, the close) targets that exact
        row instead of re-finding it by product_id.
        """
        try:
            db = await get_db()
            return await db.log_trade({
                "symbol": pos.display_name,
                "direction": pos.direction,
                "strategy": pos.strategy,
                "confidence": pos.confidence,
                "entry_price": pos.entry_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "size_usdc": pos.size_usdc,
                "margin_usdc": pos.margin_usdc,
                "leverage": pos.leverage,
                "risk_usdc": pos.risk_usdc,
                "is_paper": pos.is_paper,
                "status": "open",
                "reasoning": pos.reasoning,
                "order_id": order_id,
                "product_id": pos.product_id,
                "display_name": pos.display_name,
                "tax_treatment": pos.tax_treatment,
                "product_type": pos.product_type,
                "fees_usdc": pos.fees_usdc,
                "contracts": pos.contracts,
                "client_order_id": pos.client_order_id or None,
                "stop_order_id": stop_order_id,
            })
        except Exception as exc:
            logger.warning(f"Failed to log trade: {exc}")
            return 0

    async def _resolve_trade_id(self, pos: PaperPosition) -> int:
        """
        The trades row this position owns.

        Prefers the id captured at entry. The fallback lookup is scoped to
        the position's OWN mode: paper and live rows share the table, so an
        unscoped "latest open row for this product" could close the other
        mode's trade — booking a live P&L onto a simulated row, or worse.
        """
        if pos.trade_id:
            return pos.trade_id
        try:
            db = await get_db()
            row = await db.fetchrow(
                """
                SELECT id FROM trades
                WHERE product_id = $1 AND status = 'open' AND is_paper = $2
                ORDER BY created_at DESC LIMIT 1
                """,
                pos.product_id, pos.is_paper,
            )
            if row:
                pos.trade_id = int(row["id"])
        except Exception as exc:
            logger.warning(f"Could not resolve trade row for {pos.product_id}: {exc}")
        return pos.trade_id

    async def _update_trade_close(self, pos: PaperPosition,
                                  exit_client_order_id: str | None = None) -> None:
        try:
            db = await get_db()
            trade_id = await self._resolve_trade_id(pos)
            if not trade_id:
                logger.error(
                    f"No trades row found for {pos.product_id} ({pos.mode}) — "
                    f"close not persisted (P&L ${pos.pnl_usdc:+.2f})"
                )
                return
            await db.close_trade(
                trade_id, pos.exit_price or 0, pos.pnl_usdc, "closed",
                realized_partial=pos.realized_partial, fees_usdc=pos.fees_usdc,
                exit_order_id=pos.exit_order_id or None,
                exit_client_order_id=exit_client_order_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to update close: {exc}")

    # ------------------------------------------------------------------
    # Fill audit — tie every live order to Coinbase's own history
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fill_time(raw: Any) -> datetime | None:
        """Coinbase timestamps are RFC3339 with a trailing Z."""
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _leg_for_order(trade_row: Any, order_id: str) -> str | None:
        """Which leg of a trade an order id represents."""
        if order_id and order_id == (trade_row["order_id"] or ""):
            return "entry"
        if order_id and order_id == (trade_row["exit_order_id"] or ""):
            return "exit"
        if order_id and order_id == (trade_row["stop_order_id"] or ""):
            return "stop"
        return None

    def _audit_order_later(self, pos: PaperPosition, order_id: str, leg: str) -> None:
        """Schedule a fill audit without blocking the order path."""
        if pos.is_paper or not self.cfg.sync_fills or not order_id:
            return
        task = asyncio.create_task(
            self._audit_order(pos.trade_id, pos.product_id, order_id, leg),
            name=f"fill-audit-{leg}-{pos.product_id}",
        )
        self._audit_tasks.add(task)
        task.add_done_callback(self._audit_tasks.discard)

    async def _audit_order(self, trade_id: int, product_id: str,
                           order_id: str, leg: str) -> None:
        """
        Fetch an order's fills and record them against the trade.

        Polls a few times: /orders/historical/fills lags the order
        acknowledgement, so the first read of a just-placed market order is
        usually empty even though it filled instantly.
        """
        attempts = max(1, int(self.cfg.fill_sync_attempts or 1))
        for attempt in range(attempts):
            await asyncio.sleep(1.0 + attempt)
            try:
                fills = await self.cb.get_fills(order_ids=[order_id])
            except Exception as exc:
                logger.warning(f"Fill lookup failed for order {order_id}: {exc}")
                continue
            if not fills:
                continue
            try:
                await self._record_fills(fills, trade_id=trade_id, leg=leg)
            except Exception as exc:
                logger.warning(f"Could not record fills for order {order_id}: {exc}")
            return
        logger.warning(
            f"No fills returned for {leg} order {order_id} ({product_id}) after "
            f"{attempts} attempts — the periodic fill sync will pick them up"
        )

    async def _record_fills(self, raw_fills: list[dict], trade_id: int | None = None,
                            leg: str | None = None) -> int:
        """
        Persist exchange fills and roll their real numbers onto the trade row.

        The aggregate that lands on `trades` is what makes the row auditable:
        `filled_entry_price` / `filled_exit_price` are volume-weighted actual
        fill prices (not our pre-trade estimate) and `exchange_fees_usdc` is
        the commission Coinbase actually charged (not the modeled 0.14%).
        """
        db = await get_db()
        rows: list[dict] = []
        for f in raw_fills:
            fill_id = f.get("trade_id") or f.get("entry_id")
            if not fill_id:
                continue
            order_id = f.get("order_id", "")
            row_trade_id = trade_id
            row_leg = leg
            if row_trade_id is None:
                # Backfill path: work out which trade (and which leg) this
                # fill belongs to from the order ids already on the row.
                match = await db.find_trade_by_order(order_id)
                if match:
                    row_trade_id = int(match["id"])
                    row_leg = self._leg_for_order(match, order_id)
            rows.append({
                "trade_id": row_trade_id or None,
                "fill_id": str(fill_id),
                "order_id": order_id,
                "client_order_id": f.get("client_order_id"),
                "product_id": f.get("product_id", ""),
                "side": f.get("side"),
                "leg": row_leg,
                "price": float(f.get("price") or 0) or None,
                "size": float(f.get("size") or 0) or None,
                "commission_usdc": float(f.get("commission") or 0),
                "liquidity_indicator": f.get("liquidity_indicator"),
                "filled_at": self._parse_fill_time(f.get("trade_time")
                                                   or f.get("sequence_timestamp")),
                "is_paper": False,
                "raw": f,
            })
        if not rows:
            return 0
        written = await db.record_fills(rows)

        if trade_id and leg in ("entry", "exit"):
            priced = [(r["price"], r["size"]) for r in rows if r["price"] and r["size"]]
            total_size = sum(sz for _, sz in priced)
            vwap = (sum(px * sz for px, sz in priced) / total_size) if total_size else None
            commission = sum(r["commission_usdc"] or 0.0 for r in rows)
            patch: dict[str, Any] = {
                "fills_synced_at": datetime.now(timezone.utc),
                "exchange_fees_usdc": round(commission, 6) or None,
            }
            if vwap:
                patch["filled_entry_price" if leg == "entry" else "filled_exit_price"] = vwap
            if leg == "entry" and total_size:
                patch["filled_contracts"] = total_size
            await db.update_trade_orders(trade_id, **patch)
            detail = (
                f", {total_size:g} contract(s) @ {vwap:.4f} avg, "
                f"commission ${commission:.4f}"
            ) if vwap else ""
            logger.info(
                f"🧾 {leg.upper()} fills recorded for trade {trade_id}: "
                f"{written} fill(s){detail}"
            )
        return written

    async def _audited_product_ids(self) -> list[str]:
        """
        Products the sweep should pull fills for: whatever is open now, plus
        whatever this bot has traded live recently.

        Unscoped, /orders/historical/fills returns the WHOLE account history
        — the first sweep on this account dragged in spot fills from 2024,
        which have nothing to do with the bot and only make the audit table
        harder to read.
        """
        products = set(self.live_positions)
        try:
            db = await get_db()
            rows = await db.fetch(
                """
                SELECT DISTINCT product_id FROM trades
                WHERE is_paper = FALSE AND product_id <> ''
                  AND created_at >= NOW() - INTERVAL '30 days'
                """
            )
            products.update(r["product_id"] for r in rows if r["product_id"])
        except Exception as exc:
            logger.warning(f"Could not list products for fill sync: {exc}")
        return sorted(products)

    async def sync_recent_fills(self, limit: int = 50) -> int:
        """
        Periodic catch-up sweep over the exchange's recent fills.

        Covers everything the per-order audit cannot see: a protective stop
        that filled on its own, a position closed from the Coinbase app, and
        any order whose fills arrived after the audit gave up. Fills are
        upserted on `fill_id`, so re-reading the same window is free of
        duplicates.
        """
        if self.cfg.paper_trading or not self.cfg.sync_fills:
            return 0
        products = await self._audited_product_ids()
        if not products:
            return 0
        try:
            fills = await self.cb.get_fills(limit=limit, product_ids=products)
        except Exception as exc:
            logger.warning(f"Recent-fill sync failed: {exc}")
            return 0
        if not fills:
            return 0
        try:
            db = await get_db()
            known = await db.get_known_fill_ids(product_ids=products,
                                                limit=max(limit * 4, 200))
            fresh = [f for f in fills
                     if str(f.get("trade_id") or f.get("entry_id") or "") not in known]
            if not fresh:
                return 0
            written = await self._record_fills(fresh)
            if written:
                logger.info(f"🧾 Fill sync: {written} new exchange fill(s) recorded")
            return written
        except Exception as exc:
            logger.warning(f"Recent-fill sync could not be persisted: {exc}")
            return 0

    async def _apply_paper_pnl(self, pnl: float) -> None:
        """
        Bank realized paper PnL into the persistent balance.

        `paper_balance` in agent_config is the paper account's source of truth
        (risk.sync() pushes it into risk.state.balance_usdc each loop tick, and
        Burt / the UI report from that). Without this, paper PnL evaporates —
        the balance only ever changes via manual UI edits, so sizing, daily
        loss limits and Burt's account reports all run off a static number.
        """
        if not self.cfg.paper_trading:
            return
        try:
            db = await get_db()
            bal = await db.get_config_value("paper_balance")
            current = float(bal) if bal is not None else float(self.cfg.paper_balance or 0)
            new_bal = round(current + pnl, 2)
            await db.set_config("paper_balance", str(new_bal))
            config.set_config_key("paper_balance", new_bal)
            logger.info(f"💰 Paper balance: ${current:,.2f} → ${new_bal:,.2f} (P&L ${pnl:+.2f})")
        except Exception as exc:
            logger.warning(f"Failed to persist paper balance: {exc}")
