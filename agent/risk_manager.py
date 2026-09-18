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
    # Whole-contract sizing ceilings (config.max_risk_per_trade / max_margin_pct)
    max_risk_per_trade_pct: float = 0.04
    max_margin_pct: float = 0.50
    min_stop_pct: float = 0.015
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
    # Log-driven (2026-09-01..10): NEAR lost 3 straight full-R stop-outs
    # (-$2.27, -$2.18, -$2.31) hours apart — the 1h cooldown let the bot
    # re-enter the same falling market and buy each successive rally.
    losing_close_cooldown_h: float = 4.0      # replaces the base cooldown after a loss
    losing_symbol_window_d: int = 7          # LosingSymbolLock lookback
    losing_symbol_threshold_r: float = -2.0  # ≤ -2R cumulative → bench
    churn_limit_per_day: int = 10            # max N new positions per day


def compute_position_size(entry_price: float, stop_price: float,
                          balance: float, risk_pct: float, leverage: int,
                          taker_fee_pct: float = 0.0, min_fee_usdc: float = 0.0,
                          entry_fee_budget_pct: float | None = None,
                          ) -> tuple[float, float, float]:
    """
    Returns (notional_size, margin_required, risk_usdc) — all zero if the
    trade is rejected.

    Safety cap: margin <= 20% of balance. A tight stop wants more notional
    than the cap allows, so `risk_usdc` (the 3rd return value) is the ACTUAL
    dollar risk after any cap-scaling, not the nominal `balance * risk_pct`
    the caller asked for — callers must use this value, not recompute their
    own, or every R-multiple and fee-ratio calc downstream silently uses the
    wrong denominator.

    If `entry_fee_budget_pct` is set, also rejects (returns all zeros) when
    the round-trip taker fee on the capped notional would exceed that
    fraction of the actual risk — a stop distance small enough that even a
    max-margin position can't earn back its own entry+exit cost.
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

    if entry_fee_budget_pct is not None and risk_dollars > 0:
        round_trip_fee = 2 * max(notional_size * taker_fee_pct, min_fee_usdc)
        fee_budget = entry_fee_budget_pct * risk_dollars
        if round_trip_fee > fee_budget:
            logger.info(
                f"Skipping entry — stop too tight for this account: round-trip fee "
                f"${round_trip_fee:.2f} > {entry_fee_budget_pct:.0%} of ${risk_dollars:.2f} "
                f"risk (would size to ${notional_size:.2f} notional)"
            )
            return 0.0, 0.0, 0.0

    return notional_size, margin_required, risk_dollars


@dataclass
class ContractSpec:
    """Whole-contract economics for one CFM product (from the screener's
    hydrated product list). `margin_rate_*` are the exchange's OVERNIGHT
    rates — what it actually charges outside the 8am-4pm ET window."""
    contract_size: float                 # base units per contract (0.1 ETH, 500 NEAR ...)
    margin_rate_long: float
    margin_rate_short: float


@dataclass
class ContractSizing:
    contracts: int
    notional: float
    margin: float
    risk_usdc: float
    reason: str = ""     # non-empty => rejected (contracts == 0)


def compute_contract_position(entry_price: float, stop_price: float,
                              direction: str, balance: float, risk_pct: float,
                              spec: ContractSpec,
                              max_risk_pct: float, max_margin_pct: float,
                              taker_fee_pct: float = 0.0, min_fee_usdc: float = 0.0,
                              entry_fee_budget_pct: float | None = None,
                              ) -> ContractSizing:
    """
    Whole-contract position sizing — the live-truth counterpart of
    `compute_position_size`. CFM fills are whole contracts and the exchange
    charges margin at its own rate, so "$381 notional at 5x" is not a thing
    the exchange can execute; "1 contract of ETH PERP, $64 margin" is.

    Rules, in order:
      1. contracts = floor(target_risk / risk_per_contract). If that is 0
         (one contract already risks more than risk_pct wants — the normal
         case on a $200 account), round UP to 1 contract provided its
         actual $-at-risk <= balance * max_risk_pct.
      2. Margin (notional * exchange rate for this side) <= balance * max_margin_pct.
      3. Round-trip fee <= entry_fee_budget_pct * actual risk (same rule as
         the continuous sizer).
    Returns contracts == 0 with `reason` set on rejection.
    """
    if entry_price <= 0 or spec.contract_size <= 0:
        return ContractSizing(0, 0.0, 0.0, 0.0, "bad entry price / contract size")
    stop_pct = abs(entry_price - stop_price) / entry_price
    if stop_pct <= 0:
        return ContractSizing(0, 0.0, 0.0, 0.0, "zero stop distance")

    contract_notional = entry_price * spec.contract_size
    risk_per_contract = contract_notional * stop_pct
    target_risk = balance * risk_pct
    hard_max_risk = balance * max_risk_pct

    contracts = int(target_risk // risk_per_contract)
    if contracts < 1:
        if risk_per_contract <= hard_max_risk:
            contracts = 1
        else:
            return ContractSizing(
                0, contract_notional, 0.0, risk_per_contract,
                f"1 contract risks ${risk_per_contract:.2f} > hard max ${hard_max_risk:.2f} "
                f"({max_risk_pct:.0%} of ${balance:,.2f}) at a {stop_pct:.2%} stop",
            )

    rate = spec.margin_rate_long if direction == "long" else spec.margin_rate_short
    max_margin = balance * max_margin_pct
    while contracts >= 1:
        notional = contracts * contract_notional
        margin = notional * rate
        if margin <= max_margin:
            break
        contracts -= 1
    if contracts < 1:
        return ContractSizing(
            0, contract_notional, contract_notional * rate, risk_per_contract,
            f"1 contract needs ${contract_notional * rate:.2f} margin ({rate:.0%} overnight rate) "
            f"> cap ${max_margin:.2f} ({max_margin_pct:.0%} of ${balance:,.2f})",
        )

    notional = contracts * contract_notional
    margin = notional * rate
    risk_usdc = contracts * risk_per_contract
    if entry_fee_budget_pct is not None and risk_usdc > 0:
        round_trip_fee = 2 * max(notional * taker_fee_pct, min_fee_usdc)
        if round_trip_fee > entry_fee_budget_pct * risk_usdc:
            return ContractSizing(
                0, notional, margin, risk_usdc,
                f"round-trip fee ${round_trip_fee:.2f} > {entry_fee_budget_pct:.0%} of "
                f"${risk_usdc:.2f} risk — stop too tight to pay for itself",
            )
    return ContractSizing(contracts, notional, margin, risk_usdc)


def compute_stops(entry_price: float, atr: float | None,
                  fixed_pct: float = 0.02,
                  atr_mult: float = 1.5,
                  method: str = "atr",
                  rr: float = 2.0,
                  direction: str = "long",
                  min_stop_pct: float = 0.0) -> tuple[float, float]:
    """
    Returns (stop_loss_price, take_profit_price).

    `min_stop_pct` floors the stop distance (fraction of entry) — an ATR
    stop in a quiet market can land inside the fee's reach.
    """
    if method == "atr" and atr is not None and atr > 0:
        stop_distance = atr * atr_mult
    else:
        stop_distance = entry_price * fixed_pct
    stop_distance = max(stop_distance, entry_price * min_stop_pct)

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

    def record_close(self, symbol: str, pnl_r: float, cooldown_h: float,
                     losing_cooldown_h: float | None = None) -> None:
        """Called whenever a position closes. pnl_r = pnl / initial $-at-risk."""
        now = time.time()
        # Per-symbol cooldown — extended after a losing close. Re-entering a
        # symbol that just stopped you out is how one bad trend becomes three
        # losses in a day (NEAR, 2026-09-03..04).
        if pnl_r < 0 and losing_cooldown_h is not None:
            cooldown_h = max(cooldown_h, losing_cooldown_h)
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

    def set_notifier(self, notifier: Any) -> None:
        """Wire notifications so the circuit breaker actually reaches Discord."""
        self._notifier = notifier

    def set_client(self, cb: Any) -> None:
        """Wire the exchange client so live mode can read the real balance."""
        self._cb = cb

    async def _sync_balance(self) -> None:
        """
        Keep balance_usdc real.

        Every position size, the total-risk cap, the drawdown scale, and the
        circuit-breaker threshold are percentages of this number. It used to be
        hardcoded at 100_000 with a comment claiming it was "updated in a real
        run" — nothing updated it, so a $200 paper account sized trades for a
        $100k one and put the daily loss limit at $5,000.
        """
        if self.cfg.paper_trading:
            bal = float(self.cfg.paper_balance or 0)
            if bal > 0 and bal != self.state.balance_usdc:
                logger.info(f"Paper balance set to ${bal:,.2f}")
                self.state.balance_usdc = bal
            return
        cb = getattr(self, "_cb", None)
        if cb is None:
            return
        try:
            summary = await cb.get_futures_balance_summary()
            bs = summary.get("balance_summary", summary) or {}
            for key in ("cfm_usd_balance", "total_usd_balance", "futures_buying_power"):
                raw = bs.get(key)
                val = float(raw.get("value")) if isinstance(raw, dict) else float(raw or 0)
                if val > 0:
                    if abs(val - self.state.balance_usdc) > 0.01:
                        logger.info(f"Live balance synced from exchange ({key}): ${val:,.2f}")
                    self.state.balance_usdc = val
                    return
        except Exception as exc:
            logger.warning(f"Balance sync failed — keeping ${self.state.balance_usdc:,.2f}: {exc}")

    async def sync(self) -> None:
        """Called at top of each signal loop iteration, right after
        db.sync_config() — which already did ONE query to pull every row out
        of agent_config into the shared config singleton (self.cfg is that
        same object, not a copy). This used to re-fetch each of these 9 keys
        with its own separate `SELECT ... WHERE key = $1` — 9 extra pool
        round-trips every tick for data already sitting in self.cfg. On Neon
        that was regularly enough to blow the 45s tick-step watchdog (2026-09-14
        log: 'risk.sync' timed out on ~40% of ticks), eating away the interval
        between symbol scans. Read the already-synced singleton instead.
        """
        await self._sync_balance()
        # cfg attr → RiskParams field. Several params have a `_pct` suffix on
        # the state object that the cfg attr omits — map them explicitly so
        # the UI actually moves the right knob.
        attr_to_field = {
            "leverage": "leverage",
            "risk_per_trade": "risk_per_trade_pct",
            "daily_loss_limit": "daily_loss_limit_pct",
            "atr_multiplier": "atr_multiplier",
            "take_profit_rr": "take_profit_rr",
            "fixed_stop_pct": "fixed_stop_pct",
            "stop_loss_method": "stop_loss_method",
            "min_confidence": "min_confidence",
            "max_risk_per_trade": "max_risk_per_trade_pct",
            "max_margin_pct": "max_margin_pct",
            "min_stop_pct": "min_stop_pct",
        }
        if self.cfg.paper_trading:
            attr_to_field["paper_balance"] = "balance_usdc"
        for attr, field_name in attr_to_field.items():
            val = getattr(self.cfg, attr, None)
            if val is not None:
                setattr(self.state, field_name, val)

        # Midnight UTC circuit breaker reset
        current_day = int(time.time() / 86400)
        if current_day > self._last_reset_day:
            self.state.daily_loss_usdc = 0.0
            self.state.circuit_breaker_active = False
            self._last_reset_day = current_day
            logger.info("Circuit breaker auto-reset (midnight UTC)")

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
                symbol, pnl_r, self.state.per_symbol_cooldown_h,
                losing_cooldown_h=self.state.losing_close_cooldown_h,
            )

        if pnl_usdc >= 0:
            return
        self.state.daily_loss_usdc += abs(pnl_usdc)
        limit = self.state.balance_usdc * self.state.daily_loss_limit_pct
        if self.state.daily_loss_usdc >= limit:
            self.state.circuit_breaker_active = True
            # The breaker halts all trading — it is the single most important
            # thing to be told about, and it had no notification path at all.
            self._notify_circuit_breaker(limit)
            logger.error(
                f"╔══════════════════════════════════════╗\n"
                f"║   CIRCUIT BREAKER TRIGGERED          ║\n"
                f"║   Daily loss: ${self.state.daily_loss_usdc:.2f} >= ${limit:.2f}   ║\n"
                f"╚══════════════════════════════════════╝"
            )

    def _notify_circuit_breaker(self, limit: float) -> None:
        """Fire-and-forget Discord alert. apply_loss is sync and is called from
        inside the running loop, so schedule rather than await."""
        notifier = getattr(self, "_notifier", None)
        if notifier is None:
            return
        import asyncio
        try:
            asyncio.get_running_loop().create_task(
                notifier.notify_circuit_breaker(self.state.daily_loss_usdc, limit)
            )
        except RuntimeError:
            logger.warning("Circuit breaker alert not sent — no running event loop")

    def reset_circuit_breaker(self) -> None:
        self.state.daily_loss_usdc = 0.0
        self.state.circuit_breaker_active = False
        logger.info("Circuit breaker MANUALLY reset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def calculate_trade_params(self, direction: str, entry_price: float,
                               atr: float | None,
                               spec: ContractSpec | None = None,
                               ) -> tuple[float, float, float, float, float, int]:
        """Returns (stop_loss, take_profit, notional_size, margin_required,
        risk_usdc, contracts).

        With a `ContractSpec` (the normal path — the screener hydrates one for
        every product it selects) sizing is in WHOLE CONTRACTS at the
        exchange's own margin rate, in paper and live alike, so paper fills
        are the same lot sizes live would get. Without one (spec unknown) it
        falls back to the continuous sizer and contracts == 0.

        `risk_usdc` is the ACTUAL dollar risk of the sized position — use it,
        don't recompute `balance * risk_pct`: with whole contracts the real
        risk is routinely 1.5-2x the nominal target on a small account.
        """
        sl, tp = compute_stops(
            entry_price, atr,
            atr_mult=self.state.atr_multiplier,
            fixed_pct=self.state.fixed_stop_pct,
            method=self.state.stop_loss_method,
            rr=self.state.take_profit_rr,
            direction=direction,
            min_stop_pct=self.state.min_stop_pct,
        )
        # Drawdown-scaled sizing (B3): reduce risk-per-trade in losing streaks
        scaled_risk = self.state.risk_per_trade_pct * self.get_drawdown_scale()
        if spec is not None:
            sizing = compute_contract_position(
                entry_price, sl, direction,
                self.state.balance_usdc, scaled_risk, spec,
                max_risk_pct=self.state.max_risk_per_trade_pct,
                max_margin_pct=self.state.max_margin_pct,
                taker_fee_pct=self.cfg.taker_fee_pct,
                min_fee_usdc=self.cfg.min_fee_usdc,
                entry_fee_budget_pct=self.cfg.entry_fee_budget_pct_of_risk,
            )
            if sizing.contracts < 1:
                logger.info(f"Skipping entry — {sizing.reason}")
                return sl, tp, 0.0, 0.0, 0.0, 0
            return sl, tp, sizing.notional, sizing.margin, sizing.risk_usdc, sizing.contracts
        notional, margin, risk = compute_position_size(
            entry_price, sl,
            self.state.balance_usdc,
            scaled_risk,
            self.state.leverage,
            taker_fee_pct=self.cfg.taker_fee_pct,
            min_fee_usdc=self.cfg.min_fee_usdc,
            entry_fee_budget_pct=self.cfg.entry_fee_budget_pct_of_risk,
        )
        return sl, tp, notional, margin, risk, 0
