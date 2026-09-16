"""
Position Monitor — tracks open positions and handles exits + management.

Checks every 30 seconds:
- Paper positions: compare to current mark price
- Live positions: sync with /cfm/positions

Exit management (B2):
- Breakeven move: at +1.5R (same trigger as the partial), move stop to
  entry (+fees) on the remaining size
- Trailing stop: after +1.5R, trail by ATR × multiplier; ratchets toward profit
- Time-based exit: positions open > max_hold_h get closed IF not yet in
  meaningful profit (< +0.5R) — winners run on their trailed stop
- Partial TP: 30% at +1.5R — handled via reduce-only order amendment (live)
  / notional split (paper)
"""

import asyncio
import time
from dataclasses import dataclass

from loguru import logger

from agent.executor import Executor, PaperPosition
from agent.coinbase_client import CoinbaseClient
from agent.database import get_db
from agent.risk_manager import RiskManager
import config


# Exit management defaults (live-tunable via config if desired)
# BREAKEVEN_AT_R used to fire at +1.0R — a full half-R before the partial's
# +1.5R. That meant a trade only had to tag +1R once (routine noise, not a
# real move) to get its stop yanked all the way up to flat, with zero profit
# banked yet. 2026-09-16 (NEAR PERP, trade #20): 40 minutes grinding to
# +$2.12 (~1.01R) tripped breakeven, then a normal pullback round-tripped it
# to a dead-flat $0.00 close after fees — never got a chance at the +1.5R
# partial. Aligning breakeven with the partial means a trade that reverses
# before +1.5R rides its ORIGINAL stop (a real, but bounded, -1R loss) instead
# of being guaranteed a wash; a trade that reaches +1.5R banks 30% AND arms
# breakeven on the runner in the same tick, which is the outcome that
# actually justifies paying the round-trip fee twice.
BREAKEVEN_AT_R = 1.5
TRAILING_ACTIVATE_R = 1.5
TRAILING_ATR_MULT = 2.0
MAX_HOLD_H = 12.0
# Log-driven (2026-09-01..10, 15 closed trades): every loss was a full -1R
# stop-out while every winner was capped at ~+0.5R by the partial at +1R
# banking HALF the position — the runner never reached the 2R+ target and
# the 12h time exit closed the rest flat. Avg win $0.63 vs avg loss $1.36
# required a 68% win rate to break even. Partial later (1.5R) and smaller
# (30%) so the runner is 70% of the book.
PARTIAL_TP_AT_R = 1.5
PARTIAL_TP_PCT = 0.3
ENABLE_PARTIAL_TP = True
# Force-close at MAX_HOLD_H only when the position hasn't paid — closing
# winners at the deadline was converting +1R runners into 0R time exits.
TIME_EXIT_MIN_R = 0.5


@dataclass
class PositionSnapshot:
    product_id: str
    direction: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    status: str


class PositionMonitor:

    CHECK_INTERVAL = 30

    def __init__(self, executor: Executor, cb: CoinbaseClient, risk: RiskManager):
        self.executor = executor
        self.cb = cb
        self.risk = risk
        self._task: asyncio.Task | None = None
        self._running = False
        # Track per-position ATR at entry + original stop for R-multiple + trailing
        self._pos_meta: dict[str, dict] = {}  # product_id -> {atr, original_stop, original_size, partial_done}

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="position_monitor")
        logger.info("PositionMonitor started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def record_entry(self, pos: PaperPosition, atr: float | None) -> None:
        """Called when a new position opens — store meta for exit management."""
        self._pos_meta[pos.product_id] = {
            "atr": atr or 0.0,
            "original_stop": pos.stop_loss,
            "original_size": pos.size_usdc,
            "partial_done": False,
        }

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._check()
            except Exception as exc:
                logger.error(f"PositionMonitor error: {exc}")
            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check(self) -> None:
        # Live mode: reconcile from the exchange every tick — it is the
        # source of truth for what is actually open.
        if not self.executor.cfg.paper_trading:
            await self.executor.reconcile_live_positions()

        positions = self.executor.get_open_positions()
        if not positions:
            return

        prices: dict[str, float] = {}
        for pos in positions:
            try:
                details = await self.cb.hydrate_product_details(pos.product_id)
                prices[pos.product_id] = details.get("mark_price") or pos.entry_price
            except Exception:
                prices[pos.product_id] = pos.entry_price

        for pos in positions:
            price = prices.get(pos.product_id)
            if not price:
                continue

            # Adopted positions (reconciled from the exchange, not opened by
            # this bot) carry stop_loss=0/take_profit=0 as a deliberate
            # "unknown levels" marker. Running them through _evaluate would
            # compare price against 0 and instantly report a stop-out (shorts)
            # or take-profit (longs), market-closing a position we did not
            # open and booking a full-notional fake PnL. Report only.
            if pos.stop_loss <= 0 and pos.take_profit <= 0:
                logger.info(
                    f"👁️  {pos.display_name} adopted/unmanaged — monitoring only "
                    f"@ {price:.2f} (no stop or TP known; close it manually or "
                    f"via Burt)"
                )
                continue

            # --- Exit management (B2): update stops before evaluating ---
            prev_stop = pos.stop_loss
            await self._manage_exits(pos, price)
            # Live: keep the exchange-native crash stop aligned with the stop
            # the bot just ratcheted, or the protection left on the exchange
            # is the stale (wider) level if the bot dies.
            if not pos.is_paper and pos.stop_loss != prev_stop:
                await self.executor.sync_live_stop(pos)

            snap = self._evaluate(pos, price)
            if snap.status != "open":
                await self._handle_exit(pos, snap)
            else:
                logger.debug(f"{pos.display_name} open @ {pos.entry_price:.2f} "
                            f"current={price:.2f} uP&L=${snap.unrealized_pnl:+.2f}")

    async def _manage_exits(self, pos: PaperPosition, price: float) -> None:
        """Apply partial take-profit + breakeven + trailing stop ratchets (B2)."""
        meta = self._pos_meta.get(pos.product_id)
        if meta is None:
            # Position opened before monitor was aware (e.g. restart) — bootstrap
            meta = {
                "atr": abs(pos.entry_price - pos.stop_loss) / 1.5 if pos.stop_loss else 0.0,
                "original_stop": pos.stop_loss,
                "original_size": pos.size_usdc,
                "partial_done": True,  # skip partial TP for bootstrap positions
            }
            self._pos_meta[pos.product_id] = meta

        atr = meta["atr"]
        original_stop = meta["original_stop"]
        stop_distance = abs(pos.entry_price - original_stop)
        if stop_distance <= 0:
            return

        # R-multiple: how far in profit
        if pos.direction == "long":
            r_mult = (price - pos.entry_price) / stop_distance
        else:
            r_mult = (pos.entry_price - price) / stop_distance

        # Partial take-profit: bank PARTIAL_TP_PCT (30%) of the position at
        # +PARTIAL_TP_AT_R (1.5R). Combined with the breakeven move below,
        # the remainder becomes a risk-free runner.
        # Runs in BOTH modes. Gating this on `pos.is_paper` made paper and live
        # take structurally different exits, which silently invalidates every
        # paper result as evidence about live behaviour — the same divergence
        # class as the P0 position-tracking bug. If partials are a bad idea,
        # ENABLE_PARTIAL_TP turns them off for both modes; never just one.
        #
        # A partial adds a third fee leg (entry + partial-exit + final-exit).
        # At small position sizes the per-transaction fee minimum dominates,
        # so skip the partial when the TOTAL fee burden across all three legs
        # (entry already paid + this partial + the eventual final exit on
        # what's left) would eat too much of the trade's $-at-risk — banking
        # early isn't worth it if fees alone are a double-digit percentage of
        # the R they're supposed to protect.
        if (ENABLE_PARTIAL_TP and not meta.get("partial_done")
                and r_mult >= PARTIAL_TP_AT_R):
            partial_notional = pos.size_usdc * PARTIAL_TP_PCT
            remaining_notional = pos.size_usdc - partial_notional
            partial_fee = self.executor.fee_for_leg(partial_notional)
            est_final_fee = self.executor.fee_for_leg(remaining_notional)
            total_fee_if_partial = pos.fees_usdc + partial_fee + est_final_fee
            fee_budget = self.executor.cfg.fee_budget_pct_of_risk * pos.risk_usdc
            if pos.risk_usdc > 0 and total_fee_if_partial > fee_budget:
                meta["partial_done"] = True  # decision is final — risk_usdc doesn't change
                logger.info(
                    f"⏭️  {pos.display_name} skipping partial TP @ +{r_mult:.2f}R — "
                    f"3-leg fee burden ${total_fee_if_partial:.2f} > budget ${fee_budget:.2f} "
                    f"({self.executor.cfg.fee_budget_pct_of_risk:.0%} of ${pos.risk_usdc:.2f} risk)"
                )
            else:
                result = await self.executor.reduce_position(
                    pos.product_id, price, PARTIAL_TP_PCT
                )
                # Marked done either way: a live position too small to split (1
                # contract) must not be retried on every 30s tick.
                meta["partial_done"] = True
                if result.success:
                    logger.info(
                        f"💰 {pos.display_name} partial TP at +{r_mult:.2f}R — "
                        f"banked {PARTIAL_TP_PCT:.0%} @ {price:.2f}"
                    )
                else:
                    logger.warning(f"Partial TP failed for {pos.display_name}: {result.error}")

        # Breakeven move: at +1R, move stop to entry PLUS a buffer covering
        # the fees already paid (entry, any partial) and the fee this exit
        # will itself cost. Without the buffer, a stop hit exactly at entry
        # is a guaranteed net loser once real fees apply — "breakeven" in
        # name only.
        if r_mult >= BREAKEVEN_AT_R and pos.size_usdc > 0:
            fee_buffer_usdc = pos.fees_usdc + self.executor.fee_for_leg(pos.size_usdc)
            fee_buffer_price = fee_buffer_usdc / pos.size_usdc * pos.entry_price
            if pos.direction == "long":
                be_price = pos.entry_price + fee_buffer_price
                new_stop = max(pos.stop_loss, be_price)
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📈 {pos.display_name} BE stop → {new_stop:.2f} (fee buffer ${fee_buffer_usdc:.2f})")
            else:
                be_price = pos.entry_price - fee_buffer_price
                new_stop = min(pos.stop_loss, be_price)
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📉 {pos.display_name} BE stop → {new_stop:.2f} (fee buffer ${fee_buffer_usdc:.2f})")

        # Trailing stop: after +1.5R, trail by ATR × mult, but never looser
        # than the position's own original stop distance. TRAILING_ATR_MULT
        # (2.0) is wider than the default atr_multiplier (1.5) used to set
        # the initial stop, so an uncapped trail could give back MORE than
        # the trade's entire planned risk before firing — a runner at +1.9R
        # (NEAR PERP, trade #21, 2026-09-16) round-tripped to +0.3R because
        # the trail sat 2.0×ATR behind price while the trade was only ever
        # risking 1.5×ATR. Capping at stop_distance keeps the trail at least
        # as tight as the trade's own risk once it's protecting gains.
        if r_mult >= TRAILING_ACTIVATE_R and atr > 0:
            trail_dist = min(atr * TRAILING_ATR_MULT, stop_distance)
            if pos.direction == "long":
                new_stop = price - trail_dist
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.debug(f"📈 {pos.display_name} trail → {new_stop:.2f}")
            else:
                new_stop = price + trail_dist
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.debug(f"📉 {pos.display_name} trail → {new_stop:.2f}")

    def _evaluate(self, pos: PaperPosition, price: float) -> PositionSnapshot:
        if pos.direction == "long":
            pnl = (price - pos.entry_price) / pos.entry_price * pos.size_usdc
            status = "stopped" if price <= pos.stop_loss else \
                     "taken_profit" if price >= pos.take_profit else "open"
        else:
            pnl = (pos.entry_price - price) / pos.entry_price * pos.size_usdc
            status = "stopped" if price >= pos.stop_loss else \
                     "taken_profit" if price <= pos.take_profit else "open"

        # --- Time-based exit (B2) ---
        # Only applies to positions that haven't paid yet (< TIME_EXIT_MIN_R).
        # A winner past the deadline keeps running — its trailed stop is the
        # exit, not the clock. R is measured against the ORIGINAL stop (meta),
        # not the current (breakeven/trailed) stop.
        if status == "open":
            hold_h = (time.time() - pos.opened_at) / 3600
            if hold_h >= MAX_HOLD_H:
                meta = self._pos_meta.get(pos.product_id, {})
                original_stop = meta.get("original_stop", pos.stop_loss)
                stop_distance = abs(pos.entry_price - original_stop)
                if stop_distance <= 0:
                    r_mult = 0.0
                elif pos.direction == "long":
                    r_mult = (price - pos.entry_price) / stop_distance
                else:
                    r_mult = (pos.entry_price - price) / stop_distance
                if r_mult < TIME_EXIT_MIN_R:
                    status = "time_exit"
                else:
                    logger.debug(
                        f"{pos.display_name} past {MAX_HOLD_H:.0f}h at "
                        f"+{r_mult:.2f}R — letting the runner continue"
                    )

        return PositionSnapshot(
            product_id=pos.product_id, direction=pos.direction,
            entry_price=pos.entry_price, current_price=price,
            unrealized_pnl=pnl, stop_loss=pos.stop_loss,
            take_profit=pos.take_profit, status=status,
        )

    async def _handle_exit(self, pos: PaperPosition, snap: PositionSnapshot) -> None:
        exit_price = snap.current_price
        # For time exits, close at market (current price); for SL/TP, use the level
        if snap.status == "stopped":
            exit_price = pos.stop_loss
        elif snap.status == "taken_profit":
            exit_price = pos.take_profit

        result = await self.executor.close_position(pos.product_id, exit_price)
        if not result.success:
            logger.error(f"Failed to close {pos.product_id}: {result.error}")
            return

        # Use the PnL the executor just computed and persisted, rather than
        # recomputing from price — it's already net of every fee leg (entry,
        # partial(s), this exit) and includes anything banked by partial
        # take-profits. Recomputing here previously dropped fees, so risk
        # tracking (circuit breaker, R-multiples) and the notification both
        # ran on gross PnL while the DB recorded net — two different numbers
        # for the same trade.
        exit_pnl = pos.pnl_usdc
        snap.current_price = exit_price
        snap.unrealized_pnl = exit_pnl

        # Compute R-multiple for protections tracking. Risk is measured
        # against the ORIGINAL size — pos.size_usdc shrinks after a partial.
        meta = self._pos_meta.pop(pos.product_id, {})
        original_stop = meta.get("original_stop", pos.stop_loss)
        original_size = meta.get("original_size", pos.size_usdc)
        stop_distance = abs(pos.entry_price - original_stop)
        risk_usd = (stop_distance / pos.entry_price) * original_size if pos.entry_price else 0
        pnl_r = exit_pnl / risk_usd if risk_usd > 0 else 0.0

        self.risk.apply_loss(exit_pnl, symbol=pos.product_id, pnl_r=pnl_r)
        await self._notify(pos, snap)

    async def _notify(self, pos: PaperPosition, snap: PositionSnapshot) -> None:
        # Log-only. executor.close_position() (called just above in
        # _handle_exit) already sent the Discord/Burt "trade closed"
        # notification for this exact close — calling the notifier again
        # here duplicated every single close message (visible in the logs as
        # two identical "CLOSED ..." Burt messages per trade).
        mode = "PAPER" if pos.is_paper else "LIVE"
        emoji = "🟢" if snap.unrealized_pnl >= 0 else "🔴"
        logger.info(f"{emoji} {mode} CLOSE: {pos.display_name} "
                   f"P&L=${snap.unrealized_pnl:+.2f}")
