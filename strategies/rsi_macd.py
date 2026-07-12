"""
Strategy 1: RSI + MACD Momentum (Default)

Use case: Trending markets. Catches momentum reversals on 15m chart.
"""

from strategies.base import BaseStrategy, SignalResult, coerce_confidence


class RsiMacdStrategy(BaseStrategy):
    name = "rsi_macd"
    description = "RSI(14) + MACD momentum with 1h EMA trend filter"
    compatible_regimes = {"risk-on", "risk-off", "mixed"}  # momentum — disabled in chop

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
STRATEGY: RSI + MACD Momentum (Cascading Multi-Timeframe)
Asset: {symbol}{regime_block}
DECISION TREE — require alignment from top down:

STEP 1 — 4H BIAS (the macro tide):
  4H price vs EMA50: {i4h.get('price_vs_ema50', 'n/a')} (price={i4h.get('price')}, ema50={i4h.get('ema50')})
  4H RSI: {i4h.get('rsi')}
  → LONG bias: 4H price above EMA50, RSI > 45
  → SHORT bias: 4H price below EMA50, RSI < 55
  → If 4H bias conflicts with the signal direction, SKIP.

STEP 2 — 1H STRUCTURE (the trend filter):
  1H price vs EMA50: {i1h.get('price_vs_ema50')} (price={i1h.get('price')}, ema50={i1h.get('ema50')})
  1H EMA20: {i1h.get('ema20')}
  → LONG requires 1H price ABOVE EMA50 (uptrend confirmed)
  → SHORT requires 1H price BELOW EMA50 (downtrend confirmed)

STEP 3 — 15m TRIGGER (the entry):
  15m RSI: {i15.get('rsi')} (prev: {i15.get('rsi_prev')})
  15m MACD line: {i15.get('macd_line')} | signal: {i15.get('macd_signal')} | hist: {i15.get('macd_hist')} (prev: {i15.get('macd_hist_prev')})
  → LONG trigger: RSI crossed above 30 from below OR RSI 30-45 with MACD line > signal; AND MACD histogram increasing
  → SHORT trigger: RSI crossed below 70 from above OR RSI 55-70 with MACD line < signal; AND MACD histogram decreasing

ALL THREE STEPS MUST ALIGN. A 15m trigger against the 4H bias and 1H trend is a SKIP.
15m ATR: {i15.get('atr')} (for stop calculation)
BB width: {i15.get('bb_width')}
Vol ratio: {i15.get('vol_ratio')}

Return ONLY valid JSON with keys: direction, confidence, reasoning, entry_price, invalidation.
"""

    def check_entry(self, indicators: dict) -> SignalResult:
        """Deterministic RSI + MACD momentum check (no LLM)."""
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})

        rsi = i15.get("rsi")
        rsi_prev = i15.get("rsi_prev")
        macd_hist = i15.get("macd_hist")
        macd_hist_prev = i15.get("macd_hist_prev")
        macd_line = i15.get("macd_line")
        macd_signal = i15.get("macd_signal")
        price = i15.get("price")

        # --- Long conditions ---
        if i1h.get("price_vs_ema50") == "above":
            rsi_ok = False
            if rsi is not None and rsi_prev is not None:
                if rsi_prev < 30 < rsi:
                    rsi_ok = True
                elif 30 <= rsi <= 45 and macd_line is not None and macd_signal is not None and macd_line > macd_signal:
                    rsi_ok = True
            macd_ok = macd_hist is not None and macd_hist_prev is not None and macd_hist > macd_hist_prev
            if rsi_ok and macd_ok:
                return SignalResult(
                    direction="long", confidence=0.7,
                    reasoning="RSI+MACD long: EMA50 uptrend, RSI cross/bullish, MACD rising",
                    entry_price=price,
                )

        # --- Short conditions ---
        if i1h.get("price_vs_ema50") == "below":
            rsi_ok = False
            if rsi is not None and rsi_prev is not None:
                if rsi_prev > 70 > rsi:
                    rsi_ok = True
                elif 55 <= rsi <= 70 and macd_line is not None and macd_signal is not None and macd_line < macd_signal:
                    rsi_ok = True
            macd_ok = macd_hist is not None and macd_hist_prev is not None and macd_hist < macd_hist_prev
            if rsi_ok and macd_ok:
                return SignalResult(
                    direction="short", confidence=0.7,
                    reasoning="RSI+MACD short: EMA50 downtrend, RSI cross/bearish, MACD falling",
                    entry_price=price,
                )

        return SignalResult(direction="none", confidence=0.0, entry_price=price)

    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        i15 = indicators.get("15m", {})
        i1h = indicators.get("1h", {})

        # Attempt LLM-chosen values first, fallback to programmatic check
        direction = response.get("direction", "none")
        confidence = coerce_confidence(response.get("confidence"))

        # ---- Hard gate: long requires ALL conditions ----
        if direction == "long":
            rsi = i15.get("rsi")
            rsi_prev = i15.get("rsi_prev")
            macd_hist = i15.get("macd_hist")
            macd_hist_prev = i15.get("macd_hist_prev")
            macd_line = i15.get("macd_line")
            macd_signal = i15.get("macd_signal")
            ema_filter = i1h.get("price_vs_ema50") == "above"

            # RSI condition: crossed above 30 or 30-45 with bullish MACD
            rsi_ok = False
            if rsi is not None and rsi_prev is not None:
                if rsi_prev < 30 < rsi:
                    rsi_ok = True
                elif 30 <= rsi <= 45 and macd_line is not None and macd_signal is not None:
                    # bullish MACD cross (simple: macd > signal)
                    if macd_line > macd_signal:
                        rsi_ok = True

            # MACD increasing
            macd_ok = False
            if macd_hist is not None and macd_hist_prev is not None:
                if macd_hist > macd_hist_prev:
                    macd_ok = True

            if not all([rsi_ok, macd_ok, ema_filter]):
                confidence = min(confidence, 0.5)
                direction = "none"

        # ---- Hard gate: short requires ALL conditions ----
        elif direction == "short":
            rsi = i15.get("rsi")
            rsi_prev = i15.get("rsi_prev")
            macd_hist = i15.get("macd_hist")
            macd_hist_prev = i15.get("macd_hist_prev")
            macd_line = i15.get("macd_line")
            macd_signal = i15.get("macd_signal")
            ema_filter = i1h.get("price_vs_ema50") == "below"

            rsi_ok = False
            if rsi is not None and rsi_prev is not None:
                if rsi_prev > 70 > rsi:
                    rsi_ok = True
                elif 55 <= rsi <= 70 and macd_line is not None and macd_signal is not None:
                    if macd_line < macd_signal:
                        rsi_ok = True

            macd_ok = False
            if macd_hist is not None and macd_hist_prev is not None:
                if macd_hist < macd_hist_prev:
                    macd_ok = True

            if not all([rsi_ok, macd_ok, ema_filter]):
                confidence = min(confidence, 0.5)
                direction = "none"

        return SignalResult(
            direction=direction,
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            entry_price=i15.get("price"),
            invalidation=response.get("invalidation", ""),
        )
