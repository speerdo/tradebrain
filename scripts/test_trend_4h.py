"""
Offline checks for the trend_4h strategy and per-strategy TradePolicy.

No network, no DB: exercises the pure pieces — 4h bar construction, the
entry rule, the signal engine's rules-first / once-per-bar gating, and the
position monitor's chandelier trail.

    venv/bin/python scripts/test_trend_4h.py
"""

import asyncio
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

logger.remove()

from agent import position_monitor as pm  # noqa: E402
from agent.executor import PaperPosition  # noqa: E402
from agent.indicator_engine import build_4h_bars, compute_trend_4h  # noqa: E402
from strategies import STRATEGIES  # noqa: E402
from strategies.trend_4h import MIN_4H_BARS  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _candles(n_bars: int, step_h: int, start="2026-01-01 00:00", trend=0.0) -> pd.DataFrame:
    t = pd.date_range(start, periods=n_bars, freq=f"{step_h}h")
    rng = np.random.default_rng(7)
    close = 2000 + np.cumsum(rng.normal(trend, 5, n_bars))
    return pd.DataFrame({"time": t, "open": close - 1, "high": close + 4,
                         "low": close - 4, "close": close, "volume": 1.0})


def test_bars():
    # 2h candles starting at 02:00 — consecutive-pair grouping would build
    # 02-06 bars; UTC alignment must give 00/04/08.
    df = _candles(10, 2, start="2026-01-01 02:00")
    bars = build_4h_bars(df)
    check("4h bars are UTC-aligned", all(t.hour % 4 == 0 for t in bars["time"]),
          str(list(bars["time"])))
    # A bar still in progress must be dropped.
    now = pd.Timestamp("2026-01-01 17:00").timestamp()   # 16:00 bar open until 20:00
    live = build_4h_bars(_candles(12, 2, start="2026-01-01 00:00"), now_ts=now)
    check("in-progress 4h bar dropped", live["time"].iloc[-1] == pd.Timestamp("2026-01-01 12:00"),
          str(live["time"].iloc[-1]))


def test_entry_rule():
    s = STRATEGIES["trend_4h"]
    base = {"close": 2100.0, "dc_upper_prev": 2050.0, "dc_lower_prev": 1900.0,
            "ema200": 2000.0, "atr": 30.0, "bar_time": 1.0, "n_bars": MIN_4H_BARS}
    check("breakout above EMA200 -> long", s.check_entry({"4h_trend": base}).direction == "long")
    check("no breakout -> none",
          s.check_entry({"4h_trend": {**base, "close": 2040.0}}).direction == "none")
    check("breakout below EMA200 -> none",
          s.check_entry({"4h_trend": {**base, "ema200": 2200.0}}).direction == "none")
    check("too little history -> none",
          s.check_entry({"4h_trend": {**base, "n_bars": 50}}).direction == "none")
    check("breakdown -> none (long-only)",
          s.check_entry({"4h_trend": {**base, "close": 1850.0, "ema200": 2200.0}}).direction == "none")
    check("LLM cannot flip direction",
          s.parse_response({"direction": "short", "confidence": 0.9}, {"4h_trend": base}).direction == "none")
    # End to end: a trending series actually produces a trend dict + signal.
    d = compute_trend_4h(_candles(900, 2, trend=0.8))
    check("compute_trend_4h has history", d.get("n_bars", 0) >= MIN_4H_BARS, str(d))


def test_policies():
    check("15m strategies keep the shared exit rules",
          all(pm.policy_for(n) is pm.SHARED_POLICY
              for n in ("rsi_macd", "donchian_breakout", "ema_pullback", "bollinger")))
    p = pm.policy_for("trend_4h")
    check("trend_4h: no breakeven, no time exit, uncapped 4h trail",
          p.breakeven_at_r is None and p.max_hold_h is None
          and not p.trail_capped_at_risk and p.trail_update == "4h")


class _FakeExecutor:
    class cfg:
        paper_trading = True
        fee_budget_pct_of_risk = 0.2

    def fee_for_leg(self, notional):
        return notional * 0.0014


def test_chandelier_trail():
    mon = pm.PositionMonitor(_FakeExecutor(), cb=None, risk=None)
    entry, atr = 2000.0, 40.0
    stop = entry - 3.5 * atr
    pos = PaperPosition(product_id="ETP", display_name="ETH PERP", direction="long",
                        entry_price=entry, stop_loss=stop, take_profit=entry + 50 * 3.5 * atr,
                        size_usdc=200.0, margin_usdc=50.0, leverage=5, risk_usdc=14.0,
                        strategy="trend_4h", original_stop=stop)
    mon.record_entry(pos, atr)
    loop = asyncio.new_event_loop()
    # Same 4h bucket: price runs +3R but the stop must not move until the bar closes.
    loop.run_until_complete(mon._manage_exits(pos, entry + 3 * 3.5 * atr))
    check("no ratchet inside the entry bar", pos.stop_loss == stop, f"{pos.stop_loss}")
    # Next bucket: price has pulled back, but the trail keys off the PEAK.
    mon._pos_meta["ETP"]["trail_bucket"] -= 1
    loop.run_until_complete(mon._manage_exits(pos, entry + 3.5 * atr))
    peak = entry + 3 * 3.5 * atr
    check("trail = peak - 3.5 ATR at the next 4h bar",
          abs(pos.stop_loss - (peak - 3.5 * atr)) < 1e-9, f"{pos.stop_loss} vs {peak - 3.5 * atr}")
    check("price back under the trailed stop -> stopped",
          mon._evaluate(pos, pos.stop_loss - 1).status == "stopped")
    above = pos.stop_loss + 10
    check("no TP / time exit while above the trail", mon._evaluate(pos, above).status == "open")
    pos.opened_at = time.time() - 30 * 86400
    check("30-day-old trend trade still open (no 12h time exit)",
          mon._evaluate(pos, above).status == "open")
    # A 15m-strategy position at +1.5R DOES get the shared breakeven move.
    leg = PaperPosition(product_id="L", display_name="L", direction="long", entry_price=entry,
                        stop_loss=entry - 40, take_profit=entry + 200, size_usdc=200.0,
                        margin_usdc=50.0, leverage=5, risk_usdc=4.0, strategy="rsi_macd",
                        original_stop=entry - 40, fees_usdc=0.28)
    mon.record_entry(leg, 16.0)
    loop.run_until_complete(mon._manage_exits(leg, entry + 60))
    check("15m strategy still gets breakeven at +1.5R", leg.stop_loss > entry, f"{leg.stop_loss}")
    loop.close()


def test_rules_first():
    """The LLM must not be called when the rule has no setup, and must be
    called at most once per 4h bar when it does."""
    from agent import signal_engine as se

    calls = []

    class _Engine(se.SignalEngine):
        def __init__(self):  # skip network client setup
            import config
            self.cfg = config.get_config()
            self._available = True
            self._memory_engine = None
            self._consumed_bars = {}

        async def _call_llm(self, *a, **k):
            calls.append(1)
            return {"direction": "long", "confidence": 0.8, "reasoning": "ok"}

        async def _log_signal(self, *a, **k):
            return 1

    eng = _Engine()
    s = STRATEGIES["trend_4h"]
    loop = asyncio.new_event_loop()
    none_ind = {"4h_trend": {"close": 1.0, "dc_upper_prev": 2.0, "dc_lower_prev": 0.5,
                             "ema200": 0.9, "atr": 0.1, "bar_time": 0.0, "n_bars": 400},
                "15m": {"price": 1.0}}
    loop.run_until_complete(eng.evaluate("ETP", s, none_ind))
    check("no rule setup -> no LLM call", not calls)
    bar = time.time() - 4 * 3600 - 60  # a bar that closed a minute ago
    ind = {"4h_trend": {"close": 2100.0, "dc_upper_prev": 2050.0, "dc_lower_prev": 1900.0,
                        "ema200": 2000.0, "atr": 30.0, "bar_time": bar, "n_bars": 400},
           "15m": {"price": 2100.0}}
    first = loop.run_until_complete(eng.evaluate("ETP", s, ind))
    second = loop.run_until_complete(eng.evaluate("ETP", s, ind))
    check("rule fires -> LLM asked once", len(calls) == 1 and first.direction == "long",
          f"calls={len(calls)} dir={first.direction}")
    check("same bar again -> no second signal", second.direction == "none" and len(calls) == 1)
    stale = {**ind, "4h_trend": {**ind["4h_trend"], "bar_time": bar - 3 * 4 * 3600}}
    eng._consumed_bars.clear()
    check("stale breakout bar refused",
          loop.run_until_complete(eng.evaluate("ETP", s, stale)).direction == "none")
    loop.close()


if __name__ == "__main__":
    test_bars()
    test_entry_rule()
    test_policies()
    test_chandelier_trail()
    test_rules_first()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)
