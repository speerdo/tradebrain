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

    @abstractmethod
    def build_prompt(self, indicators: dict, symbol: str,
                     regime: dict | None = None) -> str:
        """Return the strategy-specific prompt fragment for Kimi K2.6."""
        ...

    @abstractmethod
    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        """Parse the LLM JSON response into a SignalResult."""
        ...

    def check_entry(self, indicators: dict) -> SignalResult:
        """
        Deterministic entry check — pure function of indicators, no LLM.

        Default implementation returns "none". Override in subclasses to
        encode the strategy's mechanical rules. Used by the backtester and
        as a server-side gate in live trading.
        """
        return SignalResult(direction="none", confidence=0.0)

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
