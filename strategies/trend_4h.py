"""
Strategy 5: 4h Donchian Trend (long-only, chandelier exit)

Evidence (2026-09-24 research, Coinbase ETH/BTC 2020-2026, 0.19%/side
costs, rules fixed on <2023 data and checked on 2023+): a 20-bar 4h
breakout above the 4h EMA200, exited only by a 3.5x-ATR trailing stop, was
positive in ~94% of the neighbouring parameter grid out-of-sample on both
ETH and BTC — about +0.4R per trade on ETH long-only, with a 36-40% win
rate, ~2 trades a month and a median hold of ~4 days. It stayed positive in
ETH's down years (2022, 2025). The short side averaged ~0R, so it is off.

Every 15m strategy in this repo is negative after the 0.14%/fill fee. The
difference is scale: a ~6% stop makes the round-trip fee ~5% of the risk
instead of ~19%, and the trend rule makes its money on the rare 10-20R runner
that a fixed target or a 12h time exit would cut off. Hence its own
TradePolicy — no take-profit, no breakeven move, no time exit.

The cost is size: one ETH contract at a ~6% stop risks ~$16, which needs
max_risk_per_trade >= ~0.09 on a $200 account.
"""

from strategies.base import BaseStrategy, SignalResult, TradePolicy, coerce_confidence

MIN_4H_BARS = 300  # EMA200 is meaningless without history behind it
TRAIL_ATR_MULT = 3.5
ALLOW_SHORT = False


class Trend4hStrategy(BaseStrategy):
    name = "trend_4h"
    description = "4h Donchian-20 breakout above EMA200, long-only, 3.5x ATR chandelier exit"
    compatible_regimes = None  # backtested without a regime gate
    needs_4h_history = True
    policy = TradePolicy(
        stop_atr_mult=TRAIL_ATR_MULT,
        atr_timeframe="4h",
        # No fixed target — the trail is the exit. 50R is just "never": the
        # largest backtested winner was ~21R.
        take_profit_rr=50.0,
        breakeven_at_r=None,
        trail_activate_r=0.0,
        trail_atr_mult=TRAIL_ATR_MULT,
        trail_capped_at_risk=False,
        trail_update="4h",
        max_hold_h=None,
        legacy_trend_filters=False,
        losing_symbol_lock=False,
    )
    walk_forward_grid = {
        "atr_multiplier": [3.0, 3.5, 5.0],
        "trailing_atr_mult": [3.0, 3.5, 5.0],
    }

    def check_entry(self, indicators: dict) -> SignalResult:
        t = indicators.get("4h_trend") or {}
        close = t.get("close")
        upper = t.get("dc_upper_prev")
        lower = t.get("dc_lower_prev")
        ema200 = t.get("ema200")
        bar_time = t.get("bar_time")
        if None in (close, upper, lower, ema200) or t.get("n_bars", 0) < MIN_4H_BARS:
            return SignalResult(direction="none", confidence=0.0, entry_price=close)

        if close > upper and close > ema200:
            return SignalResult(
                direction="long", confidence=0.7,
                reasoning=f"4h close {close:.2f} > 20-bar high {upper:.2f}, above EMA200 {ema200:.2f}",
                entry_price=close, bar_time=bar_time,
            )
        if ALLOW_SHORT and close < lower and close < ema200:
            return SignalResult(
                direction="short", confidence=0.7,
                reasoning=f"4h close {close:.2f} < 20-bar low {lower:.2f}, below EMA200 {ema200:.2f}",
                entry_price=close, bar_time=bar_time,
            )
        return SignalResult(direction="none", confidence=0.0, entry_price=close, bar_time=bar_time)

    def build_prompt(self, indicators: dict, symbol: str,
                     regime: dict | None = None) -> str:
        t = indicators.get("4h_trend") or {}
        rule = self.check_entry(indicators)
        regime_line = f"\nMARKET REGIME: {regime.get('regime', 'unknown')}" if regime else ""
        return f"""
STRATEGY: 4h Donchian Trend Breakout (rules have ALREADY fired — you are a veto)
Asset: {symbol}{regime_line}

The mechanical rule triggered a {rule.direction.upper()} on the last closed 4h bar:
  4h close: {t.get('close')}
  Prior 20-bar 4h high: {t.get('dc_upper_prev')} | low: {t.get('dc_lower_prev')}
  4h EMA200: {t.get('ema200')}
  4h ATR(14): {t.get('atr')} (stop = 3.5x ATR, trailed; no fixed target)

This is a backtested trend-following entry with a ~40% win rate: most
trades lose 1R and a few run 5-20R. Do NOT veto because the move "looks
extended" or RSI is high — that is what breakouts look like, and vetoing
them removes the winners. Veto ONLY for a concrete, specific reason in the
context below (e.g. an exchange/hack/delisting headline, extreme funding
that makes holding for days costly).

Return ONLY valid JSON with keys: direction ("{rule.direction}" to allow,
"none" to veto), confidence (0-1), reasoning, entry_price, invalidation.
"""

    def parse_response(self, response: dict, indicators: dict) -> SignalResult:
        # The LLM can only agree or veto — it cannot pick a direction the
        # rule didn't.
        rule = self.check_entry(indicators)
        direction = response.get("direction", "none")
        if direction != rule.direction:
            direction = "none"
        return SignalResult(
            direction=direction,
            confidence=coerce_confidence(response.get("confidence")),
            reasoning=response.get("reasoning", "") or rule.reasoning,
            entry_price=rule.entry_price,
            invalidation=response.get("invalidation", ""),
            bar_time=rule.bar_time,
        )
