"""
Executor — Order placement + paper trading

Paper mode: simulate in-memory positions against Coinbase mark prices.
Live mode: place real CFM futures orders via /api/v3/brokerage/orders.

Paper and live share one set of accessors (get_open_positions, has_position,
get_position, close_position). Live positions are reconciled from the exchange
(get_futures_positions) on startup and on every position-monitor tick — the
exchange is the source of truth, local state is a cache.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import config
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
    tax_treatment: str = "1256"
    product_type: str = "perp"

    @property
    def symbol(self) -> str:
        """Backward-compat alias — `product_id` is the canonical exchange identifier."""
        return self.product_id


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
    ) -> OrderResult:
        if self.cfg.paper_trading:
            return await self._enter_paper(
                symbol, display_name or symbol, direction, entry_price,
                stop_loss, take_profit, size_usdc, margin_usdc, leverage,
                risk_usdc, strategy, confidence, reasoning
            )
        return await self._enter_live(
            symbol, display_name or symbol, direction, entry_price,
            stop_loss, take_profit, size_usdc, margin_usdc, leverage,
            risk_usdc, strategy, confidence, reasoning,
        )

    async def _enter_paper(self, product_id: str, display_name: str, direction: str,
                            entry_price: float, stop_loss: float, take_profit: float,
                            size_usdc: float, margin_usdc: float, leverage: int,
                            risk_usdc: float, strategy: str, confidence: float,
                            reasoning: str) -> OrderResult:
        if product_id in self.paper_positions:
            return OrderResult(success=False, error=f"Already open: {product_id}")
        entry_fee = self.fee_for_leg(size_usdc)
        pos = PaperPosition(
            product_id=product_id, display_name=display_name, direction=direction,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size_usdc=size_usdc, margin_usdc=margin_usdc, leverage=leverage,
            risk_usdc=risk_usdc, strategy=strategy, confidence=confidence,
            reasoning=reasoning, is_paper=True, fees_usdc=entry_fee,
        )
        self.paper_positions[product_id] = pos
        logger.info(
            f"📄 PAPER ENTRY: {direction.upper()} {display_name} @ {entry_price:.2f} "
            f"(entry fee ${entry_fee:.2f})"
        )
        await self._log_trade(pos, order_id="")
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
        pos = self.paper_positions.get(product_id)
        if pos is not None:
            return await self._close_paper(pos, exit_price)
        pos = self.live_positions.get(product_id)
        if pos is not None:
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
            result = await self.cb.close_futures_position(pos.product_id)
            if not result.get("success", False):
                err = str(result.get("error_response", {}))[:300]
                return OrderResult(success=False, error=f"Exchange close failed: {err}")
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
                f"P&L≈${pnl:+.2f} (fees ${pos.fees_usdc:.2f})"
            )
            await self._update_trade_close(pos)
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
        reasoning: str,
    ) -> OrderResult:
        """Place a real CFM market order + exchange-native protective stop."""
        if product_id in self.live_positions or product_id in self.paper_positions:
            return OrderResult(success=False, error=f"Already open: {product_id}")
        try:
            contract_size = await self._get_contract_size(product_id)
            # Round DOWN to whole contracts so live notional never exceeds
            # the risk-sized notional.
            contracts = math.floor(size_usdc / (entry_price * contract_size))
            if contracts < 1:
                return OrderResult(
                    success=False,
                    error=(
                        f"Size too small for {product_id}: ${size_usdc:.2f} < 1 contract "
                        f"(${entry_price * contract_size:,.2f})"
                    ),
                )

            side = "BUY" if direction == "long" else "SELL"
            entry = await self.cb.place_futures_market_order(
                product_id, side, contracts, leverage=leverage, margin_type="ISOLATED",
            )
            if not entry.get("success", False):
                err = str(entry.get("error_response", {}))[:300]
                return OrderResult(success=False, error=f"Entry order rejected: {err}")
            success_response = entry.get("success_response", {})
            entry_id = success_response.get("order_id", "")
            logger.info(
                f"💰 LIVE ENTRY: {direction.upper()} {display_name} "
                f"{contracts} contract(s) (~${contracts * entry_price * contract_size:,.0f}) "
                f"order_id={entry_id}"
            )

            stop_order_id = None
            if stop_loss and stop_loss > 0:
                stop_order_id = await self._place_protective_stop(
                    product_id, direction, contracts, stop_loss,
                )

            live_notional = contracts * entry_price * contract_size
            pos = PaperPosition(
                product_id=product_id, display_name=display_name,
                direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                size_usdc=live_notional,
                margin_usdc=margin_usdc, leverage=leverage,
                risk_usdc=risk_usdc, strategy=strategy,
                confidence=confidence, reasoning=reasoning,
                is_paper=False, fees_usdc=self.fee_for_leg(live_notional),
            )
            self.live_positions[product_id] = pos
            self.live_meta[product_id] = {
                "contracts": contracts,
                "contract_size": contract_size,
                "entry_order_id": entry_id,
                "stop_order_id": stop_order_id,
            }
            await self._log_trade(pos, order_id=entry_id)
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
            self._risk.apply_loss(pnl, symbol=pos.product_id, pnl_r=pnl_r)
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

        pos = self.paper_positions.get(product_id)
        if pos is not None:
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
                result = await self.cb.close_futures_position(
                    product_id, size=str(close_contracts),
                )
                if not result.get("success", False):
                    err = str(result.get("error_response", {}))[:300]
                    return OrderResult(success=False, error=f"Exchange partial close failed: {err}")
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
                logger.info(
                    f"💰 LIVE PARTIAL: closed {close_contracts} contract(s) of "
                    f"{pos.display_name} @ ~{exit_price:.2f} banked ${realized:+.2f} gross "
                    f"(fee ${partial_fee:.2f})"
                )
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
        """
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
            )
            self.paper_positions[product_id] = pos
            restored.append(pos)
        if restored:
            logger.info(
                f"♻️  Restored {len(restored)} open paper position(s) from DB: "
                + ", ".join(f"{p.display_name} {p.direction} @ {p.entry_price:.2f}" for p in restored)
            )
        return restored

    def has_position(self, product_id: str) -> bool:
        return product_id in self.paper_positions or product_id in self.live_positions

    def get_position(self, product_id: str) -> PaperPosition | None:
        return self.paper_positions.get(product_id) or self.live_positions.get(product_id)

    async def _log_trade(self, pos: PaperPosition, order_id: str) -> None:
        try:
            db = await get_db()
            await db.log_trade({
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
            })
        except Exception as exc:
            logger.warning(f"Failed to log trade: {exc}")

    async def _update_trade_close(self, pos: PaperPosition) -> None:
        try:
            db = await get_db()
            row = await db.fetchrow(
                "SELECT id FROM trades WHERE product_id = $1 AND status = 'open' ORDER BY created_at DESC LIMIT 1",
                pos.product_id,
            )
            if row:
                await db.close_trade(row["id"], pos.exit_price or 0, pos.pnl_usdc, "closed",
                                     realized_partial=pos.realized_partial, fees_usdc=pos.fees_usdc)
        except Exception as exc:
            logger.warning(f"Failed to update close: {exc}")

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
