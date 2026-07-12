"""
Strategy 3: EMA Trend + Pullback

Use case: Strong trending markets. Buys/sells pullbacks to the 20 EMA.
"""

from strategies.base import BaseStrategy, SignalResult, coerce_confidence


class EmaPullbackStrategy(BaseStrategy):
    name = "ema_pullback"
    description = "EMA trend with pullback entry on 15m"
    compatible_regimes = {"risk-on", "risk-off", "mixed"}  # trend-following — disabled in chop

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
STRATEGY: EMA Trend + Pullback (Cascading Multi-Timeframe)
Asset: {symbol}{regime_block}
DECISION TREE — require alignment from top down:

STEP 1 — 4H BIAS (the macro tide):
  4H price vs EMA50: {i4h.get('price_vs_ema50', 'n/a')} (price={i4h.get('price')}, ema50={i4h.get('ema50')})
  4H EMA20: {i4h.get('ema20')}
  → LONG bias: 4H price above EMA50 AND EMA20 above EMA50 (clear uptrend)
  → SHORT bias: 4H price below EMA50 AND EMA20 below EMA50 (clear downtrend)
  → If 4H EMAs are tangled, there's no trend to pull back into — SKIP.

STEP 2 — 1H STRUCTURE (the trend we're trading):
  1H: price={i1h.get('price')} ema20={i1h.get('ema20')} ema50={i1h.get('ema50')} (above 50 EMA: {i1h.get('price_vs_ema50')})
  → LONG requires 1H price above EMA20 AND EMA20 above EMA50
  → SHORT requires 1H price below EMA20 AND EMA20 below EMA50
  → 4H and 1H must agree on direction.

STEP 3 — 15m TRIGGER (the pullback entry):
  15m: price={i15.get('price')} high={i15.get('high')} low={i15.get('low')}
  15m RSI: {i15.get('rsi')}
  → LONG trigger: price pulls back to within 0.5% of 1H EMA20, RSI 40-60 (pulled back, not exhausted)
  → SHORT trigger: inverse

ALL THREE STEPS MUST ALIGN. A 15m pullback against the 4H/1H trend is a SKIP.
15m ATR: {i15.get('atr')}

Return ONLY valid JSON with keys: direction, confidence, reasoning, entry_price, invalidation.
"""

    def check_entry(self, indicators: dict) -> SignalResult:
        """Deterministic EMA trend + pullback check (no LLM)."""
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})

        price_1h = i1h.get("price")
        ema20_1h = i1h.get("ema20")
        ema50_1h = i1h.get("ema50")
        price_15m = i15.get("price")
        rsi_15m = i15.get("rsi")

        if ema20_1h is None or ema50_1h is None or price_1h is None:
            return SignalResult(direction="none", confidence=0.0, entry_price=price_15m)

        uptrend = price_1h > ema20_1h and ema20_1h > ema50_1h
        downtrend = price_1h < ema20_1h and ema20_1h < ema50_1h

        if uptrend and price_15m and ema20_1h:
            away_pct = abs(price_15m - ema20_1h) / ema20_1h
            if away_pct <= 0.005 and rsi_15m is not None and 40 <= rsi_15m <= 60:
                return SignalResult(
                    direction="long", confidence=0.7,
                    reasoning="EMA pullback long: uptrend, price near 20EMA, RSI 40-60",
                    entry_price=price_15m,
                )

        if downtrend and price_15m and ema20_1h:
            away_pct = abs(price_15m - ema20_1h) / ema20_1h
            if away_pct <= 0.005 and rsi_15m is not None and 40 <= rsi_15m <= 60:
                return SignalResult(
                    direction="short", confidence=0.7,
                    reasoning="EMA pullback short: downtrend, price near 20EMA, RSI 40-60",
                    entry_price=price_15m,
                )

        return SignalResult(direction="none", confidence=0.0, entry_price=price_15m)

    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})
        direction = response.get("direction", "none")
        confidence = coerce_confidence(response.get("confidence"))

        price_1h = i1h.get("price")
        ema20_1h = i1h.get("ema20")
        ema50_1h = i1h.get("ema50")
        price_15m = i15.get("price")
        rsi_15m = i15.get("rsi")

        # 1. Trend clarity
        if ema20_1h is None or ema50_1h is None or price_1h is None:
            return self.fallback_signal()

        uptrend = price_1h > ema20_1h and ema20_1h > ema50_1h
        downtrend = price_1h < ema20_1h and ema20_1h < ema50_1h

        if direction == "long":
            if not uptrend:
                return self.fallback_signal()
            # Pullback within 0.5% of 20 EMA
            if ema20_1h and price_15m:
                away_pct = abs(price_15m - ema20_1h) / ema20_1h
                if away_pct > 0.005:
                    confidence = min(confidence, 0.5)
                    direction = "none"
            # RSI 40-60
            if rsi_15m is not None and not (40 <= rsi_15m <= 60):
                confidence = min(confidence, 0.5)
                direction = "none"

        elif direction == "short":
            if not downtrend:
                return self.fallback_signal()
            if ema20_1h and price_15m:
                away_pct = abs(price_15m - ema20_1h) / ema20_1h
                if away_pct > 0.005:
                    confidence = min(confidence, 0.5)
                    direction = "none"
            if rsi_15m is not None and not (40 <= rsi_15m <= 60):
                confidence = min(confidence, 0.5)
                direction = "none"

        return SignalResult(
            direction=direction,
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            entry_price=price_15m,
            invalidation=response.get("invalidation", ""),
        )
