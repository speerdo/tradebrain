"""
Backtest Engine — bar-by-bar replay through the production code paths.

Discipline (Jesse-inspired):
- Indicators are computed ONCE on the full frame, but decisions at bar i may
  only inspect rows up to and including i. `compute_indicators_at` enforces this.
- Entry uses the same `strategy.check_entry()` + `risk_manager.compute_stops` +
  `compute_position_size` code that live trading uses.
- Exit logic mirrors `position_monitor.py` (SL/TP hit on bar high/low).
- Fees, slippage, and funding are modeled.

Usage:
    from backtest.engine import BacktestEngine, BacktestConfig
    engine = BacktestEngine(strategy, cfg)
    result = engine.run(df_15m, df_1h)
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from agent.indicator_engine import _compute_indicator_df, compute_indicators_at
from agent.risk_manager import compute_position_size, compute_stops
from strategies.base import BaseStrategy, SignalResult


@dataclass
class BacktestConfig:
    """Backtest parameters — mirrors live RiskManager.state + fee model."""
    balance_usdc: float = 100_000.0
    leverage: int = 3
    risk_per_trade_pct: float = 0.01
    atr_multiplier: float = 1.5
    take_profit_rr: float = 2.0
    fixed_stop_pct: float = 0.02
    stop_loss_method: str = "atr"
    min_confidence: float = 0.0          # backtest uses deterministic signals — no confidence gate by default
    fee_taker_pct: float = 0.0005        # 5 bps taker
    fee_maker_pct: float = 0.0002        # 2 bps maker (unused for market entries — kept for future)
    slippage_pct: float = 0.0005         # 5 bps slippage on entry/exit
    funding_rate_8h: float = 0.0001      # 1 bps per 8h — long pays when positive
    max_concurrent_positions: int = 3
    max_total_risk_pct: float = 0.03     # cap total open risk at 3% of balance
    # Exit management (Phase B2 in the proposal — included here so the
    # backtester can measure their impact before they ship to live)
    enable_breakeven: bool = True
    breakeven_at_r: float = 1.0
    enable_trailing: bool = True
    trailing_activate_r: float = 1.5
    trailing_atr_mult: float = 2.0
    enable_partial_tp: bool = True   # mirrors live position_monitor default
    partial_tp_r: float = 1.0
    partial_tp_pct: float = 0.5
    enable_time_exit: bool = True
    max_hold_bars: int = 48              # 48 * 15m = 12h


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    size_usdc: float
    margin_usdc: float
    pnl_usdc: float
    pnl_r: float
    fees_usdc: float
    funding_usdc: float
    bars_held: int
    exit_reason: str
    confidence: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"n_trades": 0, "final_equity": self.config.balance_usdc}
        pnls = [t.pnl_usdc for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        equity = self.config.balance_usdc + sum(pnls)
        # Sharpe-ish: per-trade return
        rets = np.array(pnls) / self.config.balance_usdc
        sharpe = (rets.mean() / rets.std() * np.sqrt(252 * 24 * 4)) if rets.std() else 0.0
        # Max drawdown from equity curve
        eq = [e for _, e in self.equity_curve]
        peak = -np.inf
        max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return {
            "n_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": sum(pnls),
            "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
            "final_equity": equity,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_annualized": sharpe,
            "avg_pnl": np.mean(pnls),
            "avg_bars_held": np.mean([t.bars_held for t in self.trades]),
            "total_fees": sum(t.fees_usdc for t in self.trades),
            "total_funding": sum(t.funding_usdc for t in self.trades),
        }


class _OpenPosition:
    """Internal mutable position state for the engine."""
    def __init__(self, trade: BacktestTrade, atr_at_entry: float, stop_distance: float):
        self.trade = trade
        self.atr_at_entry = atr_at_entry
        self.stop_distance = stop_distance
        self.original_stop = trade.stop_loss
        self.original_tp = trade.take_profit
        self.partial_filled = False
        self.remaining_size = trade.size_usdc
        self.realized_partial = 0.0  # gross PnL banked by partial take-profit
        self.bars_held = 0


class BacktestEngine:
    """
    Replays candles through strategy.check_entry + risk rules + exit management.
    """

    def __init__(self, strategy: BaseStrategy, config: BacktestConfig):
        self.strategy = strategy
        self.cfg = config

    def run(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        symbol: str = "BACKTEST",
    ) -> BacktestResult:
        # Pre-compute indicators on the full frame (look-ahead-safe slicing happens in compute_indicators_at)
        df_15m_e, df_1h_e = _compute_indicator_df(df_15m, df_1h)

        # Map each 15m bar to the most recent CLOSED 1h bar index.
        # A 1h bar starting at HH:00 closes at HH+1:00, so it is only usable
        # once the decision time (close of the 15m bar = bar start + 15m) has
        # reached HH+1:00. Concretely: only the HH:45 15m bar may use the
        # HH:00 1h bar; earlier bars in the hour must use the prior hour.
        # Using the in-progress 1h bar would be look-ahead bias.
        one_hour_ts = df_1h_e["time"].dt.floor("h").values
        bar_to_1h = {}
        for i, ts in enumerate(df_15m_e["time"]):
            decision_time = ts + pd.Timedelta(minutes=15)
            latest_closed_start = decision_time - pd.Timedelta(hours=1)
            idx = np.searchsorted(one_hour_ts, np.datetime64(latest_closed_start), side="right") - 1
            if idx < 0:
                idx = 0
            bar_to_1h[i] = int(idx)

        result = BacktestResult(config=self.cfg)
        open_positions: list[_OpenPosition] = []
        balance = self.cfg.balance_usdc
        warmup = 60  # need enough bars for indicators to be valid

        for i in range(warmup, len(df_15m_e)):
            current_bar = df_15m_e.iloc[i]
            current_price = float(current_bar["close"])
            bar_high = float(current_bar["high"])
            bar_low = float(current_bar["low"])
            bar_time = current_bar["time"]

            # --- Exit check first (intra-bar: SL/TP hit if high/low crosses) ---
            for op in list(open_positions):
                op.bars_held += 1
                t = op.trade
                exit_price, exit_reason = self._check_exit(op, bar_high, bar_low, current_price)
                if exit_reason:
                    filled = exit_price
                    # Apply slippage on exit
                    slip = filled * self.cfg.slippage_pct
                    if t.direction == "long":
                        filled -= slip
                    else:
                        filled += slip
                    # Update trade
                    t.exit_price = filled
                    t.exit_time = bar_time
                    t.exit_reason = exit_reason
                    t.bars_held = op.bars_held
                    gross_pnl = self._pnl(t.direction, t.entry_price, filled, op.remaining_size)
                    # remaining_size is already USD notional — fee is notional * rate
                    exit_fee = op.remaining_size * self.cfg.fee_taker_pct
                    t.fees_usdc += exit_fee
                    t.pnl_usdc = op.realized_partial + gross_pnl - t.fees_usdc - t.funding_usdc
                    t.pnl_r = self._pnl_r(op)
                    balance += t.pnl_usdc
                    result.trades.append(t)
                    open_positions.remove(op)

            # --- Time-based exit ---
            if self.cfg.enable_time_exit:
                for op in list(open_positions):
                    if op.bars_held >= self.cfg.max_hold_bars:
                        t = op.trade
                        filled = current_price
                        slip = filled * self.cfg.slippage_pct
                        if t.direction == "long":
                            filled -= slip
                        else:
                            filled += slip
                        t.exit_price = filled
                        t.exit_time = bar_time
                        t.exit_reason = "time_exit"
                        t.bars_held = op.bars_held
                        gross_pnl = self._pnl(t.direction, t.entry_price, filled, op.remaining_size)
                        exit_fee = op.remaining_size * self.cfg.fee_taker_pct
                        t.fees_usdc += exit_fee
                        t.pnl_usdc = op.realized_partial + gross_pnl - t.fees_usdc - t.funding_usdc
                        t.pnl_r = self._pnl_r(op)
                        balance += t.pnl_usdc
                        result.trades.append(t)
                        open_positions.remove(op)

            # --- Partial take-profit (intra-bar: fires if high/low reaches +partial_tp_r) ---
            if self.cfg.enable_partial_tp:
                for op in open_positions:
                    if op.partial_filled or op.stop_distance <= 0:
                        continue
                    t = op.trade
                    if t.direction == "long":
                        trigger = t.entry_price + op.stop_distance * self.cfg.partial_tp_r
                        hit = bar_high >= trigger
                    else:
                        trigger = t.entry_price - op.stop_distance * self.cfg.partial_tp_r
                        hit = bar_low <= trigger
                    if not hit:
                        continue
                    fill = trigger
                    slip = fill * self.cfg.slippage_pct
                    fill = fill - slip if t.direction == "long" else fill + slip
                    closed = op.remaining_size * self.cfg.partial_tp_pct
                    op.realized_partial += self._pnl(t.direction, t.entry_price, fill, closed)
                    t.fees_usdc += closed * self.cfg.fee_taker_pct
                    op.remaining_size -= closed
                    op.partial_filled = True

            # --- Funding cost (every 8h = 32 bars on 15m) ---
            for op in open_positions:
                if op.bars_held % 32 == 0 and op.bars_held > 0:
                    notional = op.remaining_size
                    # Longs pay positive funding, shorts receive (pay negative)
                    funding = notional * self.cfg.funding_rate_8h
                    if op.trade.direction == "short":
                        funding = -funding
                    op.trade.funding_usdc += funding

            # --- Trailing stop / breakeven / partial TP updates (using bar close) ---
            for op in open_positions:
                self._manage_exits(op, current_price)

            # --- Entry check ---
            if len(open_positions) < self.cfg.max_concurrent_positions:
                total_risk = sum(
                    o.trade.size_usdc * abs(o.trade.entry_price - o.original_stop) / o.trade.entry_price
                    for o in open_positions
                )
                if total_risk < balance * self.cfg.max_total_risk_pct:
                    indicators = compute_indicators_at(df_15m_e, df_1h_e, i, bar_to_1h[i])
                    if indicators:
                        sig = self.strategy.check_entry(indicators)
                        if sig.direction in ("long", "short") and sig.confidence >= self.cfg.min_confidence:
                            entry = indicators["15m"]["price"]
                            atr = indicators["15m"]["atr"]
                            sl, tp = compute_stops(
                                entry, atr,
                                fixed_pct=self.cfg.fixed_stop_pct,
                                atr_mult=self.cfg.atr_multiplier,
                                method=self.cfg.stop_loss_method,
                                rr=self.cfg.take_profit_rr,
                                direction=sig.direction,
                            )
                            notional, margin, risk = compute_position_size(
                                entry, sl, balance,
                                self.cfg.risk_per_trade_pct, self.cfg.leverage,
                            )
                            if notional > 0 and margin > 0:
                                # Apply slippage on entry
                                slip = entry * self.cfg.slippage_pct
                                if sig.direction == "long":
                                    fill_price = entry + slip
                                else:
                                    fill_price = entry - slip
                                entry_fee = notional * self.cfg.fee_taker_pct
                                stop_distance = abs(fill_price - sl)
                                trade = BacktestTrade(
                                    entry_time=bar_time,
                                    exit_time=bar_time,
                                    symbol=symbol,
                                    direction=sig.direction,
                                    entry_price=fill_price,
                                    exit_price=0.0,
                                    stop_loss=sl,
                                    take_profit=tp,
                                    size_usdc=notional,
                                    margin_usdc=margin,
                                    pnl_usdc=0.0,
                                    pnl_r=0.0,
                                    fees_usdc=entry_fee,
                                    funding_usdc=0.0,
                                    bars_held=0,
                                    exit_reason="",
                                    confidence=sig.confidence,
                                )
                                op = _OpenPosition(trade=trade, atr_at_entry=atr or 0.0, stop_distance=stop_distance)
                                open_positions.append(op)

            # --- Equity curve snapshot (mark-to-market) ---
            unrealized = 0.0
            for op in open_positions:
                u = self._pnl(op.trade.direction, op.trade.entry_price, current_price, op.remaining_size)
                unrealized += op.realized_partial + u - op.trade.fees_usdc - op.trade.funding_usdc
            result.equity_curve.append((bar_time, balance + unrealized))

        # --- Close any remaining positions at the last bar ---
        last_price = float(df_15m_e.iloc[-1]["close"])
        last_time = df_15m_e.iloc[-1]["time"]
        for op in open_positions:
            t = op.trade
            slip = last_price * self.cfg.slippage_pct
            filled = last_price - slip if t.direction == "long" else last_price + slip
            t.exit_price = filled
            t.exit_time = last_time
            t.bars_held = op.bars_held
            gross_pnl = self._pnl(t.direction, t.entry_price, filled, op.remaining_size)
            t.fees_usdc += op.remaining_size * self.cfg.fee_taker_pct
            t.pnl_usdc = op.realized_partial + gross_pnl - t.fees_usdc - t.funding_usdc
            t.pnl_r = self._pnl_r(op)
            t.exit_reason = "end_of_data"
            balance += t.pnl_usdc
            result.trades.append(t)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_exit(self, op: _OpenPosition, bar_high: float, bar_low: float, bar_close: float) -> tuple[float, str]:
        """Intra-bar exit check. Returns (exit_price, reason) or (0, "")."""
        t = op.trade
        if t.direction == "long":
            if bar_low <= t.stop_loss:
                return t.stop_loss, "stop_loss"
            if bar_high >= t.take_profit:
                return t.take_profit, "take_profit"
        else:
            if bar_high >= t.stop_loss:
                return t.stop_loss, "stop_loss"
            if bar_low <= t.take_profit:
                return t.take_profit, "take_profit"
        return 0.0, ""

    def _manage_exits(self, op: _OpenPosition, current_price: float) -> None:
        """Update trailing stop, breakeven, partial TP — stop only ratchets."""
        t = op.trade
        r_multiple = self._r_multiple(op, current_price)

        # Breakeven move
        if self.cfg.enable_breakeven and r_multiple >= self.cfg.breakeven_at_r:
            if t.direction == "long":
                new_stop = max(t.stop_loss, t.entry_price)
                if new_stop > t.stop_loss:
                    t.stop_loss = new_stop
            else:
                new_stop = min(t.stop_loss, t.entry_price)
                if new_stop < t.stop_loss:
                    t.stop_loss = new_stop

        # Trailing stop
        if self.cfg.enable_trailing and r_multiple >= self.cfg.trailing_activate_r and op.atr_at_entry > 0:
            trail_dist = op.atr_at_entry * self.cfg.trailing_atr_mult
            if t.direction == "long":
                new_stop = current_price - trail_dist
                if new_stop > t.stop_loss:
                    t.stop_loss = new_stop
            else:
                new_stop = current_price + trail_dist
                if new_stop < t.stop_loss:
                    t.stop_loss = new_stop

        # (Partial take-profit is handled intra-bar in the main loop, before
        # funding accrual — see the enable_partial_tp block in run().)

    def _r_multiple(self, op: _OpenPosition, price: float) -> float:
        if op.stop_distance <= 0:
            return 0.0
        t = op.trade
        if t.direction == "long":
            return (price - t.entry_price) / op.stop_distance
        return (t.entry_price - price) / op.stop_distance

    @staticmethod
    def _pnl_r(op: _OpenPosition) -> float:
        """PnL in R-multiples, measured against the ORIGINAL stop distance.

        Must not use the current (possibly breakeven/trailed) stop — after a
        breakeven move entry == stop and the risk denominator would be zero.
        """
        t = op.trade
        if op.stop_distance <= 0 or t.entry_price <= 0:
            return 0.0
        risk_usd = op.stop_distance / t.entry_price * t.size_usdc
        return t.pnl_usdc / risk_usd if risk_usd > 0 else 0.0

    @staticmethod
    def _pnl(direction: str, entry: float, exit_price: float, size_usdc: float) -> float:
        if direction == "long":
            return (exit_price - entry) / entry * size_usdc
        return (entry - exit_price) / entry * size_usdc