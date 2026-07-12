"""
Risk Manager — Position sizing, stops, circuit breaker, protections, portfolio risk.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import config


@dataclass
class RiskParams:
    """Dynamic risk state (updated from DB hot-reload)."""
    balance_usdc: float = 100_000.0  # default dummy balance (updated in real run)
    leverage: int = 3
    risk_per_trade_pct: float = 0.01
    daily_loss_limit_pct: float = 0.05
    atr_multiplier: float = 1.5
    take_profit_rr: float = 2.0
    fixed_stop_pct: float = 0.02
    stop_loss_method: str = "atr"
    min_confidence: float = 0.65
    circuit_breaker_active: bool = False
    daily_loss_usdc: float = 0.0
    manual_pause: bool = False
    # --- Portfolio-level risk (B3) ---
    max_concurrent_positions: int = 3
    max_total_risk_pct: float = 0.03         # cap total open $-at-risk at 3% of balance
    max_correlated_directions: int = 2       # max same-direction bets (alts ≈ BTC beta)
    drawdown_scale_threshold_pct: float = 0.5  # when trailing 7d DD >= 50% of daily limit, scale risk
    drawdown_scale_factor: float = 0.5       # scale risk-per-trade by this when in drawdown
    # --- Protections (B1) ---
    stoploss_guard_count: int = 3            # ≥N stop-outs in window → cooldown
    stoploss_guard_window_h: int = 4         # lookback window in hours
    stoploss_guard_cooldown_h: int = 2       # global cooldown after trip
    per_symbol_cooldown_h: float = 1.0       # lockout after closing a position in a symbol
    losing_symbol_window_d: int = 7          # LosingSymbolLock lookback
    losing_symbol_threshold_r: float = -2.0  # ≤ -2R cumulative → bench
    churn_limit_per_day: int = 10            # max N new positions per day


def compute_position_size(entry_price: float, stop_price: float,
                          balance: float, risk_pct: float, leverage: int) -> tuple[float, float, float]:
    """
    Returns (notional_size, margin_required, risk_usdc).
    Safety cap: margin <= 20% of balance.
    """
    risk_dollars = balance * risk_pct
    stop_distance_pct = abs(entry_price - stop_price) / entry_price
    if stop_distance_pct <= 0:
        logger.warning("Stop distance is zero — cannot compute position size")
        return 0.0, 0.0, 0.0

    notional_size = risk_dollars / stop_distance_pct
    margin_required = notional_size / leverage

    # Safety cap: max 20% of balance
    max_margin = balance * 0.20
    if margin_required > max_margin:
        scale = max_margin / margin_required
        notional_size *= scale
        risk_dollars *= scale
        margin_required = notional_size / leverage
        logger.warning(f"Position scaled down to fit 20% margin cap: margin=${margin_required:.2f}")

    return notional_size, margin_required, risk_dollars


def compute_stops(entry_price: float, atr: float | None,
                  fixed_pct: float = 0.02,
                  atr_mult: float = 1.5,
                  method: str = "atr",
                  rr: float = 2.0,
                  direction: str = "long") -> tuple[float, float]:
    """
    Returns (stop_loss_price, take_profit_price).
    """
    if method == "atr" and atr is not None and atr > 0:
        stop_distance = atr * atr_mult
    else:
        stop_distance = entry_price * fixed_pct

    if direction == "long":
        sl = entry_price - stop_distance
        tp = entry_price + (stop_distance * rr)
    else:
        sl = entry_price + stop_distance
        tp = entry_price - (stop_distance * rr)

    return sl, tp


# =====================================================================
# Protections (B1) — Freqtrade-style trade-outcome guards
# =====================================================================

class Protections:
    """
    Tracks stop-outs, per-symbol cooldowns, churn, and benched symbols.

    All state is in-memory and resets on restart — the protections are meant
    to be fast, cheap guards on top of the persistent DB audit trail.
    """

    def __init__(self):
        # StoplossGuard: deque of (timestamp, symbol) for recent stop-outs
        self._stopouts: deque[tuple[float, str]] = deque(maxlen=50)
        self._global_cooldown_until: float = 0.0
        # Per-symbol cooldown: symbol -> expiry epoch
        self._symbol_cooldown: dict[str, float] = {}
        # LosingSymbolLock: symbol -> cumulative R over trailing 7d
        self._symbol_cum_r: dict[str, list[tuple[float, float]]] = {}
        self._benched: set[str] = set()
        # Churn: deque of timestamps of new positions opened today
        self._entries_today: deque[float] = deque(maxlen=100)
        self._last_day = int(time.time() / 86400)

    def _maybe_reset_day(self) -> None:
        today = int(time.time() / 86400)
        if today > self._last_day:
            self._entries_today.clear()
            self._last_day = today

    def record_stopout(self, symbol: str) -> None:
        now = time.time()
        self._stopouts.append((now, symbol))

    def record_close(self, symbol: str, pnl_r: float, cooldown_h: float) -> None:
        """Called whenever a position closes. pnl_r = pnl / initial $-at-risk."""
        now = time.time()
        # Per-symbol cooldown
        self._symbol_cooldown[symbol] = now + cooldown_h * 3600
        # Stoploss tracking — only meaningful losses count toward the guard.
        # A -0.05R time exit is not a stop-out; without this threshold a few
        # near-flat closes would trip the global cooldown.
        if pnl_r <= -0.5:
            self.record_stopout(symbol)
        # LosingSymbolLock cumulative R
        bucket = self._symbol_cum_r.setdefault(symbol, [])
        bucket.append((now, pnl_r))
        # Prune to window
        cutoff = now - 7 * 86400
        self._symbol_cum_r[symbol] = [(t, r) for t, r in bucket if t >= cutoff]

    def record_entry(self) -> None:
        self._maybe_reset_day()
        self._entries_today.append(time.time())

    def check(self, symbol: str, params: RiskParams) -> str:
        """
        Returns "" if allowed, or a skip reason string.
        """
        now = time.time()

        # 1. Global StoplossGuard cooldown
        if now < self._global_cooldown_until:
            remaining = (self._global_cooldown_until - now) / 3600
            return f"StoplossGuard cooldown active ({remaining:.1f}h left)"

        # 2. Per-symbol cooldown
        until = self._symbol_cooldown.get(symbol, 0)
        if now < until:
            remaining = (until - now) / 60
            return f"Symbol cooldown ({remaining:.0f}m left)"

        # 3. LosingSymbolLock — benched symbols
        if symbol in self._benched:
            return "Symbol benched (LosingSymbolLock) until weekly review"

        # 4. StoplossGuard trip — count stop-outs in lookback window
        window_s = params.stoploss_guard_window_h * 3600
        recent = [(t, s) for t, s in self._stopouts if t >= now - window_s]
        if len(recent) >= params.stoploss_guard_count:
            self._global_cooldown_until = now + params.stoploss_guard_cooldown_h * 3600
            logger.warning(
                f"StoplossGuard tripped: {len(recent)} stop-outs in "
                f"{params.stoploss_guard_window_h}h — cooldown {params.stoploss_guard_cooldown_h}h"
            )
            return "StoplossGuard tripped — entering cooldown"

        # 5. LosingSymbolLock threshold check (≤ -2R cumulative → bench)
        cum_r = sum(r for _, r in self._symbol_cum_r.get(symbol, []))
        if cum_r <= params.losing_symbol_threshold_r:
            self._benched.add(symbol)
            logger.warning(f"LosingSymbolLock: {symbol} cumR={cum_r:.2f} → benched")
            return f"Symbol benched (cumR {cum_r:.2f} ≤ {params.losing_symbol_threshold_r})"

        # 6. Churn limit
        self._maybe_reset_day()
        if len(self._entries_today) >= params.churn_limit_per_day:
            return f"Churn limit reached ({params.churn_limit_per_day} entries today)"

        return ""

    def unbench(self, symbol: str) -> None:
        self._benched.discard(symbol)
        self._symbol_cum_r.pop(symbol, None)
        logger.info(f"Unbenched {symbol}")

    def status(self) -> dict:
        return {
            "global_cooldown_until": self._global_cooldown_until,
            "symbol_cooldowns": dict(self._symbol_cooldown),
            "benched_symbols": list(self._benched),
            "entries_today": len(self._entries_today),
            "recent_stopouts": len(self._stopouts),
        }


class RiskManager:
    """Tracks daily loss, circuit breaker, portfolio risk, and validates trades."""

    def __init__(self):
        self.cfg = config.get_config()
        self.state = RiskParams(
            leverage=self.cfg.default_leverage,
            risk_per_trade_pct=self.cfg.default_risk_per_trade,
            daily_loss_limit_pct=self.cfg.default_daily_loss_limit,
        )
        self.protections = Protections()
        self._last_reset_day = int(time.time() / 86400)
        # Trailing closed-trade PnLs for drawdown-scaled sizing (B3)
        self._recent_pnl: deque[tuple[float, float]] = deque(maxlen=200)  # (ts, pnl)

    # ------------------------------------------------------------------
    # Sync from DB / UI
    # ------------------------------------------------------------------

    async def sync(self) -> None:
        """Called at top of each signal loop iteration."""
        from agent.database import get_db
        # DB key → RiskParams field. Several params have a `_pct` suffix on the
        # state object that the DB key omits — map them explicitly so the UI
        # actually moves the right knob.
        key_to_field = {
            "leverage": "leverage",
            "risk_per_trade": "risk_per_trade_pct",
            "daily_loss_limit": "daily_loss_limit_pct",
            "atr_multiplier": "atr_multiplier",
            "take_profit_rr": "take_profit_rr",
            "fixed_stop_pct": "fixed_stop_pct",
            "stop_loss_method": "stop_loss_method",
            "min_confidence": "min_confidence",
        }
        try:
            db = await get_db()
            for key, field_name in key_to_field.items():
                val = await db.get_config_value(key)
                if val is not None:
                    setattr(self.state, field_name, self._coerce(key, val))
        except Exception as exc:
            logger.warning(f"RiskManager sync failed: {exc}")

        # Midnight UTC circuit breaker reset
        current_day = int(time.time() / 86400)
        if current_day > self._last_reset_day:
            self.state.daily_loss_usdc = 0.0
            self.state.circuit_breaker_active = False
            self._last_reset_day = current_day
            logger.info("Circuit breaker auto-reset (midnight UTC)")

    @staticmethod
    def _coerce(key: str, val: str) -> Any:
        if key in ("leverage",):
            return int(val)
        if key in ("risk_per_trade", "daily_loss_limit", "atr_multiplier",
                   "take_profit_rr", "min_confidence", "fixed_stop_pct"):
            return float(val)
        return val

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def check_trade_allowed(self, signal: Any, symbol: str,
                            open_positions: list | None = None) -> str:
        """
        Returns empty string if allowed, otherwise returns skip reason.

        `open_positions` is an optional list of open position objects with
        `.product_id`, `.direction`, `.entry_price`, `.stop_loss`, `.size_usdc`
        attributes — used for portfolio-level risk checks (B3).
        """
        if self.state.manual_pause:
            return "Manual pause is active"
        if self.state.circuit_breaker_active:
            return "Circuit breaker active"
        if signal.direction == "none":
            return "No directional signal"
        if signal.confidence < self.state.min_confidence:
            return f"Confidence {signal.confidence:.2f} < {self.state.min_confidence}"

        # --- Protections (B1) ---
        prot_skip = self.protections.check(symbol, self.state)
        if prot_skip:
            return prot_skip

        # --- Portfolio-level risk (B3) ---
        if open_positions is not None:
            port_skip = self.check_portfolio_risk(signal, symbol, open_positions)
            if port_skip:
                return port_skip

        return ""

    def check_portfolio_risk(self, signal: Any, symbol: str,
                             open_positions: list) -> str:
        """Portfolio-level exposure checks (B3)."""
        # Max concurrent positions
        if len(open_positions) >= self.state.max_concurrent_positions:
            return f"Max concurrent positions ({self.state.max_concurrent_positions})"

        # Max total open risk
        total_risk = 0.0
        same_dir_count = 0
        for pos in open_positions:
            stop_dist = abs(pos.entry_price - pos.stop_loss)
            risk_usd = (stop_dist / pos.entry_price) * getattr(pos, "size_usdc", 0)
            total_risk += risk_usd
            if getattr(pos, "direction", "") == signal.direction:
                same_dir_count += 1

        limit_usd = self.state.balance_usdc * self.state.max_total_risk_pct
        if total_risk >= limit_usd:
            return f"Max total open risk (${total_risk:.2f} ≥ ${limit_usd:.2f})"

        # Correlation cap — treat alts as correlated (≈1 BTC beta)
        if same_dir_count >= self.state.max_correlated_directions:
            return f"Max same-direction exposure ({self.state.max_correlated_directions})"

        return ""

    def get_drawdown_scale(self) -> float:
        """
        Returns a multiplier (0..1) for risk-per-trade based on trailing 7d drawdown.
        When DD >= drawdown_scale_threshold_pct of daily_loss_limit, scale down.
        """
        now = time.time()
        cutoff = now - 7 * 86400
        recent = [p for t, p in self._recent_pnl if t >= cutoff]
        if not recent or self.state.balance_usdc <= 0:
            return 1.0
        # Walk the cumulative PnL over the window and measure the largest
        # peak-to-trough giveback, expressed as a fraction of balance.
        cum = 0.0
        peak = 0.0
        max_dd_usd = 0.0
        for pnl in recent:
            cum += pnl
            peak = max(peak, cum)
            max_dd_usd = max(max_dd_usd, peak - cum)
        max_dd = max_dd_usd / self.state.balance_usdc
        threshold = self.state.daily_loss_limit_pct * self.state.drawdown_scale_threshold_pct
        if max_dd >= threshold:
            return self.state.drawdown_scale_factor
        return 1.0

    # ------------------------------------------------------------------
    # Circuit breaker + outcome tracking
    # ------------------------------------------------------------------

    def apply_loss(self, pnl_usdc: float, symbol: str = "",
                   pnl_r: float = 0.0) -> None:
        """
        Record a closed trade's PnL. Updates circuit breaker, protections,
        and drawdown tracking.

        `pnl_r` is the PnL in R-multiples (pnl / initial $-at-risk) — used by
        the LosingSymbolLock. Pass 0.0 if unknown (no R-based protection trip).
        """
        # Track for drawdown-scaled sizing (B3)
        self._recent_pnl.append((time.time(), pnl_usdc))

        # Protections (B1) — record the outcome
        if symbol:
            self.protections.record_close(
                symbol, pnl_r, self.state.per_symbol_cooldown_h
            )

        if pnl_usdc >= 0:
            return
        self.state.daily_loss_usdc += abs(pnl_usdc)
        limit = self.state.balance_usdc * self.state.daily_loss_limit_pct
        if self.state.daily_loss_usdc >= limit:
            self.state.circuit_breaker_active = True
            logger.error(
                f"╔══════════════════════════════════════╗\n"
                f"║   CIRCUIT BREAKER TRIGGERED          ║\n"
                f"║   Daily loss: ${self.state.daily_loss_usdc:.2f} >= ${limit:.2f}   ║\n"
                f"╚══════════════════════════════════════╝"
            )

    def reset_circuit_breaker(self) -> None:
        self.state.daily_loss_usdc = 0.0
        self.state.circuit_breaker_active = False
        logger.info("Circuit breaker MANUALLY reset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def calculate_trade_params(self, direction: str, entry_price: float,
                               atr: float | None) -> tuple[float, float, float, float]:
        """Returns (stop_loss, take_profit, notional_size, margin_required)."""
        sl, tp = compute_stops(
            entry_price, atr,
            atr_mult=self.state.atr_multiplier,
            fixed_pct=self.state.fixed_stop_pct,
            method=self.state.stop_loss_method,
            rr=self.state.take_profit_rr,
            direction=direction,
        )
        # Drawdown-scaled sizing (B3): reduce risk-per-trade in losing streaks
        scaled_risk = self.state.risk_per_trade_pct * self.get_drawdown_scale()
        notional, margin, risk = compute_position_size(
            entry_price, sl,
            self.state.balance_usdc,
            scaled_risk,
            self.state.leverage,
        )
        return sl, tp, notional, margin
