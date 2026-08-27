"""
Position Monitor — tracks open positions and handles exits + management.

Checks every 30 seconds:
- Paper positions: compare to current mark price
- Live positions: sync with /cfm/positions

Exit management (B2):
- Breakeven move: at +1R, move stop to entry (+fees)
- Trailing stop: after +1.5R, trail by ATR × multiplier; ratchets toward profit
- Time-based exit: positions open > max_hold_h get closed
- Partial TP: handled via reduce-only order amendment (live) / notional split (paper)
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from agent.executor import Executor, PaperPosition
from agent.coinbase_client import CoinbaseClient
from agent.database import get_db
from agent.risk_manager import RiskManager
import config


# Exit management defaults (live-tunable via config if desired)
BREAKEVEN_AT_R = 1.0
TRAILING_ACTIVATE_R = 1.5
TRAILING_ATR_MULT = 2.0
MAX_HOLD_H = 12.0
PARTIAL_TP_AT_R = 1.0      # bank half at +1R, let the rest run
PARTIAL_TP_PCT = 0.5
ENABLE_PARTIAL_TP = True


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
        self._notifier: Any = None
        # Track per-position ATR at entry + original stop for R-multiple + trailing
        self._pos_meta: dict[str, dict] = {}  # product_id -> {atr, original_stop, original_size, partial_done}

    def set_notifier(self, notifier: Any) -> None:
        self._notifier = notifier

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

        # Partial take-profit: bank PARTIAL_TP_PCT of the position at +1R.
        # Combined with the breakeven move below, the remainder becomes a
        # risk-free runner.
        # Runs in BOTH modes. Gating this on `pos.is_paper` made paper and live
        # take structurally different exits, which silently invalidates every
        # paper result as evidence about live behaviour — the same divergence
        # class as the P0 position-tracking bug. If partials are a bad idea,
        # ENABLE_PARTIAL_TP turns them off for both modes; never just one.
        if (ENABLE_PARTIAL_TP and not meta.get("partial_done")
                and r_mult >= PARTIAL_TP_AT_R):
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

        # Breakeven move: at +1R, move stop to entry
        if r_mult >= BREAKEVEN_AT_R:
            if pos.direction == "long":
                new_stop = max(pos.stop_loss, pos.entry_price)
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📈 {pos.display_name} BE stop → {new_stop:.2f}")
            else:
                new_stop = min(pos.stop_loss, pos.entry_price)
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📉 {pos.display_name} BE stop → {new_stop:.2f}")

        # Trailing stop: after +1.5R, trail by ATR × mult
        if r_mult >= TRAILING_ACTIVATE_R and atr > 0:
            trail_dist = atr * TRAILING_ATR_MULT
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
        if status == "open":
            hold_h = (time.time() - pos.opened_at) / 3600
            if hold_h >= MAX_HOLD_H:
                status = "time_exit"

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

        # Recompute PnL at the actual exit price so risk tracking and the
        # notification match the fill logged by the executor (snap PnL was
        # marked at the observed price, not the SL/TP level). Includes any
        # PnL already banked by partial take-profits.
        if pos.direction == "long":
            exit_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size_usdc
        else:
            exit_pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size_usdc
        exit_pnl += pos.realized_partial
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
        mode = "PAPER" if pos.is_paper else "LIVE"
        emoji = "🟢" if snap.unrealized_pnl >= 0 else "🔴"
        logger.info(f"{emoji} {mode} CLOSE: {pos.display_name} "
                   f"P&L=${snap.unrealized_pnl:+.2f}")
        if self._notifier:
            try:
                await self._notifier.notify_trade_closed(
                    pos.display_name, pos.direction, pos.entry_price,
                    snap.current_price, snap.unrealized_pnl,
                )
            except Exception as exc:
                logger.warning(f"Close notification failed: {exc}")
