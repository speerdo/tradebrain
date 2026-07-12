"""
Strategy 2: Bollinger Band Mean Reversion

Use case: Ranging markets. Fades overextension back to the mean.
"""

from strategies.base import BaseStrategy, SignalResult, coerce_confidence


class BollingerStrategy(BaseStrategy):
    name = "bollinger"
    description = "Bollinger Band mean reversion with RSI confirmation"
    compatible_regimes = {"chop", "mixed"}  # mean-reversion — disabled in strong trends

    def build_prompt(self, indicators: dict, symbol: str,
                     regime: dict | None = None) -> str:
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})
        i4h = indicators.get("4h", {})
        regime_block = ""
        if regime:
            regime_block = (
                f"\nMARKET REGIME: {regime.get('regime', 'unknown')} "
                f"(BTC dominance: {regime.get('btc_dominance', 'n/a')})\n"
            )
        return f"""
STRATEGY: Bollinger Band Mean Reversion (Cascading Multi-Timeframe)
Asset: {symbol}{regime_block}
DECISION TREE — require alignment from top down:

STEP 1 — 4H BIAS (the macro tide):
  4H price vs EMA50: {i4h.get('price_vs_ema50', 'n/a')} (price={i4h.get('price')}, ema50={i4h.get('ema50')})
  4H RSI: {i4h.get('rsi')}
  → Mean reversion works best in RANGING markets. If 4H is in a strong trend
    (price far from EMA50, RSI > 65 or < 35), the fade is dangerous — SKIP.

STEP 2 — 1H STRUCTURE (range or trend?):
  1H price vs EMA50: {i1h.get('price_vs_ema50')} (price={i1h.get('price')}, ema50={i1h.get('ema50')})
  1H EMA20: {i1h.get('ema20')}
  → If 1H price is near EMA50 (within 1 ATR), market is ranging — OK to fade.
  → If 1H price is far from EMA50, trend is strong — SKIP the fade.

STEP 3 — 15m TRIGGER (the band touch):
  15m Price: {i15.get('price')} (high: {i15.get('high')}, low: {i15.get('low')})
  15m BB: lower={i15.get('bb_lower')} middle={i15.get('bb_middle')} upper={i15.get('bb_upper')}
  15m RSI: {i15.get('rsi')}
  BB width: {i15.get('bb_width')}
  → LONG trigger: wick touches/crosses below BB lower AND RSI < 35 AND BB width > 1%
  → SHORT trigger: wick touches/crosses above BB upper AND RSI > 65 AND BB width > 1%
  Target: Middle band (BB basis / 20 EMA).
  Stop: Beyond the wick that touched the band.

ALL THREE STEPS MUST ALIGN. A 15m band touch in a strong 4H trend is a SKIP.
15m ATR: {i15.get('atr')}

Return ONLY valid JSON with keys: direction, confidence, reasoning, entry_price, invalidation.
"""

    def check_entry(self, indicators: dict) -> SignalResult:
        """Deterministic Bollinger mean-reversion check (no LLM)."""
        i15 = indicators.get("15m", {})
        price = i15.get("price")
        low = i15.get("low")
        high = i15.get("high")
        bb_lower = i15.get("bb_lower")
        bb_upper = i15.get("bb_upper")
        bb_width = i15.get("bb_width")
        rsi = i15.get("rsi")

        if bb_width is not None and bb_width <= 0.01:
            return SignalResult(direction="none", confidence=0.0, entry_price=price)

        if low is not None and bb_lower is not None and low <= bb_lower and rsi is not None and rsi < 35:
            return SignalResult(
                direction="long", confidence=0.7,
                reasoning="BB long: wick touched lower band, RSI<35, width>1%",
                entry_price=price,
            )
        if high is not None and bb_upper is not None and high >= bb_upper and rsi is not None and rsi > 65:
            return SignalResult(
                direction="short", confidence=0.7,
                reasoning="BB short: wick touched upper band, RSI>65, width>1%",
                entry_price=price,
            )
        return SignalResult(direction="none", confidence=0.0, entry_price=price)

    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        i15 = indicators.get("15m", {})
        direction = response.get("direction", "none")
        confidence = coerce_confidence(response.get("confidence"))

        price = i15.get("price")
        low = i15.get("low")
        high = i15.get("high")
        bb_lower = i15.get("bb_lower")
        bb_upper = i15.get("bb_upper")
        bb_width = i15.get("bb_width")
        rsi = i15.get("rsi")

        # Hard gate: require BB width > 1%
        if bb_width is not None and bb_width <= 0.01:
            return self.fallback_signal()

        if direction == "long":
            conditions = [
                low is not None and bb_lower is not None and low <= bb_lower,
                rsi is not None and rsi < 35,
            ]
            if not all(conditions):
                confidence = min(confidence, 0.5)
                direction = "none"

        elif direction == "short":
            conditions = [
                high is not None and bb_upper is not None and high >= bb_upper,
                rsi is not None and rsi > 65,
            ]
            if not all(conditions):
                confidence = min(confidence, 0.5)
                direction = "none"

        return SignalResult(
            direction=direction,
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            entry_price=price,
            invalidation=response.get("invalidation", ""),
        )
