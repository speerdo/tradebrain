"""
Strategy 4: Donchian Channel Breakout (Trend-Following)

Use case: volatility-expansion / trending markets. Catches confirmed breakouts
of the prior 20-bar range. Best fit for leveraged trading because asymmetric
R:R amplifies favorably and the strategy naturally sidesteps chop (the worst
regime for leverage). Donchian — not Bollinger — is the right tool here:
Bollinger uses standard deviation (mean-reversion-flavoured); Donchian uses
raw highs/lows (the actual definition of a breakout).
"""

from strategies.base import BaseStrategy, SignalResult, coerce_confidence


class DonchianBreakoutStrategy(BaseStrategy):
    name = "donchian_breakout"
    description = "Donchian 20-bar breakout with volume + volatility-expansion confirmation"
    compatible_regimes = {"risk-on", "risk-off", "mixed"}  # breakout — disabled in chop

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
STRATEGY: Donchian Channel Breakout (Cascading Multi-Timeframe)
Asset: {symbol}{regime_block}
DECISION TREE — require alignment from top down:

STEP 1 — 4H BIAS (the macro tide):
  4H price vs EMA50: {i4h.get('price_vs_ema50', 'n/a')} (price={i4h.get('price')}, ema50={i4h.get('ema50')})
  4H RSI: {i4h.get('rsi')}
  → LONG bias: 4H price above EMA50 (macro uptrend supports long breakouts)
  → SHORT bias: 4H price below EMA50 (macro downtrend supports short breakouts)
  → Counter-trend breakouts (long in 4H downtrend) are fakeout-prone — SKIP.

STEP 2 — 1H STRUCTURE (trend confirmation):
  1H price vs EMA50: {i1h.get('price_vs_ema50')}
  1H RSI: {i1h.get('rsi')}
  → LONG requires 1H price above EMA50
  → SHORT requires 1H price below EMA50
  → 4H and 1H must agree.

STEP 3 — 15m TRIGGER (the breakout):
  15m close: {i15.get('price')}
  15m PRIOR 20-bar range: upper_prev={i15.get('dc_upper_prev')} lower_prev={i15.get('dc_lower_prev')}
  15m volume ratio: {i15.get('vol_ratio')} (1.0 = average)
  15m BB width: {i15.get('bb_width')} | 20-bar avg: {i15.get('bb_width_avg20')}
  15m RSI: {i15.get('rsi')}
  → LONG trigger: close strictly ABOVE dc_upper_prev, vol_ratio >= 1.2, BB width expanding, RSI 50-75
  → SHORT trigger: close strictly BELOW dc_lower_prev, vol_ratio >= 1.2, BB width expanding, RSI 25-50

ALL THREE STEPS MUST ALIGN. A 15m breakout against the 4H/1H trend is a fakeout — SKIP.
AVOID: marginal break on low volume, contracting BB width, RSI already extreme (>78 or <22).
15m ATR: {i15.get('atr')}

For invalidation, the natural stop is the OPPOSITE Donchian boundary
(longs: dc_lower; shorts: dc_upper) or 1.5 ATR — whichever is closer.

Return ONLY valid JSON with keys: direction, confidence, reasoning, entry_price, invalidation.
"""

    def check_entry(self, indicators: dict) -> SignalResult:
        """Deterministic Donchian breakout check (no LLM)."""
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})

        price = i15.get("price")
        dc_upper_prev = i15.get("dc_upper_prev")
        dc_lower_prev = i15.get("dc_lower_prev")
        vol_ratio = i15.get("vol_ratio")
        bb_width = i15.get("bb_width")
        bb_width_avg = i15.get("bb_width_avg20")
        rsi_15 = i15.get("rsi")

        if price is None:
            return SignalResult(direction="none", confidence=0.0)

        # Long breakout
        if dc_upper_prev is not None and price > dc_upper_prev:
            if i1h.get("price_vs_ema50") == "above":
                conf = 0.75
                if vol_ratio is not None and vol_ratio < 1.0:
                    conf = min(conf, 0.4)
                if bb_width is not None and bb_width_avg is not None and bb_width <= bb_width_avg:
                    conf = min(conf, 0.45)
                if rsi_15 is not None and rsi_15 > 78:
                    conf = min(conf, 0.5)
                if conf >= 0.5:
                    return SignalResult(
                        direction="long", confidence=conf,
                        reasoning=f"Donchian long: close {price} > prior high {dc_upper_prev}",
                        entry_price=price,
                    )

        # Short breakout
        if dc_lower_prev is not None and price < dc_lower_prev:
            if i1h.get("price_vs_ema50") == "below":
                conf = 0.75
                if vol_ratio is not None and vol_ratio < 1.0:
                    conf = min(conf, 0.4)
                if bb_width is not None and bb_width_avg is not None and bb_width <= bb_width_avg:
                    conf = min(conf, 0.45)
                if rsi_15 is not None and rsi_15 < 22:
                    conf = min(conf, 0.5)
                if conf >= 0.5:
                    return SignalResult(
                        direction="short", confidence=conf,
                        reasoning=f"Donchian short: close {price} < prior low {dc_lower_prev}",
                        entry_price=price,
                    )

        return SignalResult(direction="none", confidence=0.0, entry_price=price)

    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})

        direction = response.get("direction", "none")
        confidence = coerce_confidence(response.get("confidence"))

        price = i15.get("price")
        dc_upper_prev = i15.get("dc_upper_prev")
        dc_lower_prev = i15.get("dc_lower_prev")
        vol_ratio = i15.get("vol_ratio")
        bb_width = i15.get("bb_width")
        bb_width_avg = i15.get("bb_width_avg20")
        rsi_15 = i15.get("rsi")
        ema_filter_long = i1h.get("price_vs_ema50") == "above"
        ema_filter_short = i1h.get("price_vs_ema50") == "below"

        # Hard gate: actual breakout must have happened. The LLM can hallucinate;
        # we re-check the math here so a "long" signal without a real upper-band
        # break gets killed.
        if direction == "long":
            if price is None or dc_upper_prev is None or price <= dc_upper_prev:
                direction = "none"
            else:
                # Soft penalties — fold weak setups into "skip" via min_confidence
                if vol_ratio is not None and vol_ratio < 1.0:
                    confidence = min(confidence, 0.4)
                if bb_width is not None and bb_width_avg is not None and bb_width <= bb_width_avg:
                    confidence = min(confidence, 0.45)
                if not ema_filter_long:
                    confidence = min(confidence, 0.5)
                if rsi_15 is not None and rsi_15 > 78:
                    confidence = min(confidence, 0.5)

        elif direction == "short":
            if price is None or dc_lower_prev is None or price >= dc_lower_prev:
                direction = "none"
            else:
                if vol_ratio is not None and vol_ratio < 1.0:
                    confidence = min(confidence, 0.4)
                if bb_width is not None and bb_width_avg is not None and bb_width <= bb_width_avg:
                    confidence = min(confidence, 0.45)
                if not ema_filter_short:
                    confidence = min(confidence, 0.5)
                if rsi_15 is not None and rsi_15 < 22:
                    confidence = min(confidence, 0.5)

        return SignalResult(
            direction=direction,
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            entry_price=response.get("entry_price"),
            invalidation=response.get("invalidation", ""),
        )
