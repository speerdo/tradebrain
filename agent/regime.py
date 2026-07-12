"""
Market Regime Context Block

Computes once per signal loop iteration:
- BTC dominance proxy
- Market regime label (risk-on / risk-off / mixed / chop)
"""

from typing import Any

import pandas as pd
from loguru import logger

from agent.coinbase_client import CoinbaseClient, CbCandle


class RegimeEngine:
    """Builds the market-regime context block for Kimi prompts."""

    def __init__(self, cb: CoinbaseClient):
        self.cb = cb

    async def get_context(self) -> dict:
        """
        Returns context dict with:
        - btc_dominance (float)
        - regime (str)
        - context_tickers (list)
        """
        try:
            btc_candles = await self.cb.get_candles("BIP-20DEC30-CDE", "ONE_HOUR", limit=25)
            eth_candles = await self.cb.get_candles("ETP-20DEC30-CDE", "ONE_HOUR", limit=25)

            btc_regime = self._analyze_trend(btc_candles)
            eth_regime = self._analyze_trend(eth_candles)

            btc_change = self._pct_change(btc_candles)
            total_change = btc_change + self._pct_change(eth_candles)
            btc_dom = btc_change / total_change if total_change else 0.5

            regime = self._label_regime(btc_regime, eth_regime, btc_dom)

            return {
                "btc_dominance": round(btc_dom, 3),
                "regime": regime,
                "context_tickers": self._build_ticker_notes(btc_regime),
            }
        except Exception as exc:
            logger.warning(f"Regime context failed: {exc}")
            # "unknown" (not "mixed") so the regime gate fails open instead of
            # silently benching strategies whenever the fetch hiccups.
            return {"btc_dominance": 0.5, "regime": "unknown", "context_tickers": []}

    @staticmethod
    def _analyze_trend(candles: list[CbCandle]) -> dict:
        if len(candles) < 10:
            return {"trend": "unknown", "volatility": 0}
        closes = [c.close for c in candles]
        ema_short = pd.Series(closes).ewm(span=8).mean().iloc[-1]
        ema_long = pd.Series(closes).ewm(span=21).mean().iloc[-1]
        atr = max(closes) - min(closes)
        return {
            "trend": "up" if ema_short > ema_long else "down",
            "volatility": atr / closes[-1] * 100 if closes[-1] else 0,
        }

    @staticmethod
    def _pct_change(candles: list[CbCandle]) -> float:
        if len(candles) < 2:
            return 0.0
        first, last = candles[0].close, candles[-1].close
        return (last - first) / first if first else 0.0

    @staticmethod
    def _label_regime(btc: dict, eth: dict, btc_dom: float) -> str:
        trends = [btc.get("trend"), eth.get("trend")]
        if "unknown" in trends:
            return "mixed"
        if trends[0] == "up" and trends[1] == "up":
            return "risk-on" if btc_dom > 0.5 else "mixed"
        if trends[0] == "down" and trends[1] == "down":
            return "risk-off"
        return "chop"

    @staticmethod
    def _build_ticker_notes(regimes: dict) -> list[str]:
        return [f"BTC: {regimes.get('trend', 'unknown')} trend"]
