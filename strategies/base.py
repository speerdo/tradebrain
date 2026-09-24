"""
Base strategy ABC + shared dataclasses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SignalResult:
    """Output of strategy.parse_response() — a single signal evaluation."""
    direction: str            # "long" | "short" | "none"
    confidence: float = 0.0   # 0.0 - 1.0
    reasoning: str = ""       # max 150 chars
    entry_price: float | None = None
    invalidation: str = ""    # what would cancel the setup
    parse_failed: bool = False  # LLM output was malformed — "none" is NOT a real no-signal
    raw_response_snippet: str = ""  # first 200 chars of unparseable / errored LLM output
    # Start (epoch seconds) of the closed higher-timeframe bar the signal was
    # read from. Bar-based strategies (trend_4h) fire once per bar: without
    # this, a breakout bar would re-signal on every 2-minute tick for 4 hours.
    bar_time: float | None = None
    # signals-table row id, so the outcome (entered / why skipped) can be
    # written back once risk and execution have had their say.
    signal_id: int | None = None
    skip_reason: str = ""


@dataclass(frozen=True)
class TradePolicy:
    """
    Per-strategy stop, target and exit management.

    The 15m strategies share one set of exit rules (breakeven at +1.5R, a
    trail capped at the original risk, a 12h time exit) that were tuned for
    trades lasting hours. A 4h trend trade lasts days and makes its money on
    the rare 10R+ runner, so each of those rules would cut exactly the trades
    it needs. A strategy with `policy = None` keeps the shared rules
    (position_monitor's module constants and the config's atr_multiplier /
    take_profit_rr); one that sets a policy gets these instead, in the
    position monitor and the backtester alike.
    """
    stop_atr_mult: float | None = None     # None -> config.atr_multiplier
    atr_timeframe: str = "15m"             # which ATR the stop and trail use: "15m" | "4h"
    take_profit_rr: float | None = None    # None -> config.take_profit_rr
    breakeven_at_r: float | None = 1.5     # None disables the breakeven move
    trail_activate_r: float = 1.5
    trail_atr_mult: float = 2.0
    trail_capped_at_risk: bool = True      # never trail looser than the original stop distance
    # "tick": ratchet on every monitor tick. "4h": ratchet once per closed 4h
    # bar off the peak since entry — the chandelier exit as backtested, and
    # at most six exchange stop replacements a day.
    trail_update: str = "tick"
    max_hold_h: float | None = 12.0        # None disables the time exit
    time_exit_min_r: float = 0.5
    # The 1h-EMA50 trend filter and 4h-EMA50 bias gate in main.py were added
    # for the 15m strategies; a strategy with its own trend definition opts out.
    legacy_trend_filters: bool = True
    # LosingSymbolLock benches a symbol at -2R over 7 days. At a ~36% win
    # rate that is two ordinary losses, and ETH is the only symbol a small
    # account can trade — it would bench the strategy most weeks.
    losing_symbol_lock: bool = True


class BaseStrategy(ABC):
    """
    Abstract base for all TradeBrain strategies.

    A strategy defines:
      - check_entry(indicators) -> SignalResult  (deterministic rules — backtestable)
      - build_prompt(indicators) -> str           (injected into the LLM prompt)
      - parse_response(resp, indicators) -> SignalResult  (gates LLM output through hard rules)

    `check_entry` is pure-Python, no LLM. It encodes the same entry conditions
    described in the prompt so the backtester can replay them bar-by-bar without
    API calls. `parse_response` remains the live path: it takes the LLM's
    candidate signal and re-checks the hard gates so the model can't hallucinate
    a trade the rules don't allow.

    Regime compatibility:
      - `compatible_regimes` lists the regimes where this strategy is active.
        The main loop skips strategies in incompatible regimes (server-side check,
        not just a prompt hint). None means "runs in any regime".
    """

    name: str = ""
    description: str = ""
    compatible_regimes: set[str] | None = None  # None = all regimes OK
    policy: TradePolicy | None = None  # None = the shared 15m exit rules
    # Set when check_entry reads indicators["4h_trend"] — main.py then fetches
    # enough 2h history to build it (one request is only 150 4h bars).
    needs_4h_history: bool = False
    # Walk-forward grid over BacktestConfig fields; None = the default grid.
    walk_forward_grid: dict[str, list[float]] | None = None

    @abstractmethod
    def build_prompt(self, indicators: dict, symbol: str,
                     regime: dict | None = None) -> str:
        """Return the strategy-specific prompt fragment for Kimi K2.6."""
        ...

    @abstractmethod
    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        """Parse the LLM JSON response into a SignalResult."""
        ...

    @abstractmethod
    def check_entry(self, indicators: dict) -> SignalResult:
        """
        Deterministic entry check — pure function of indicators, no LLM.

        Mandatory: M7/G1 gates the live LLM call on this method and the
        backtester replays it bar-by-bar. A strategy without an override is
        silently gated to zero trades forever — fail at import instead.
        """
        ...

    def fallback_signal(self) -> SignalResult:
        """Return a default "none" signal (used on parse failure)."""
        return SignalResult(
            direction="none",
            confidence=0.0,
            reasoning="Parse failure or no clear setup",
            invalidation="",
        )


# Some LLMs return categorical confidence ("low" / "medium" / "high") despite
# the prompt asking for a number. Map them to reasonable midpoints instead of
# crashing on float("low").
_CATEGORICAL_CONFIDENCE = {
    "none": 0.0, "very_low": 0.15, "very low": 0.15,
    "low": 0.30, "medium_low": 0.40, "medium-low": 0.40,
    "medium": 0.55, "med": 0.55,
    "medium_high": 0.70, "medium-high": 0.70,
    "high": 0.80, "very_high": 0.92, "very high": 0.92,
}


def coerce_confidence(raw: Any) -> float:
    """Coerce an LLM-returned confidence value into a 0.0-1.0 float."""
    if raw is None:
        return 0.0
    if isinstance(raw, bool):  # bool is a subclass of int — treat False/True as 0/1
        return float(raw)
    if isinstance(raw, (int, float)):
        v = float(raw)
    else:
        s = str(raw).strip().lower().rstrip("%")
        try:
            v = float(s)
        except ValueError:
            return _CATEGORICAL_CONFIDENCE.get(s, 0.0)
    # Allow models to return 0-100 instead of 0-1. Use a threshold (>=2) so that
    # near-miss values like 1.2 clamp to 1.0 rather than getting divided down to
    # 0.012 — much closer to model intent.
    if v >= 2.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))
