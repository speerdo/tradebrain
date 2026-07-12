"""
FCM Market Screener

Discovers perp-style products at runtime via Coinbase API,
and scores them for signal evaluation.
"""

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from agent.coinbase_client import CoinbaseClient
from agent.indicator_engine import compute_screener_indicators
from agent.database import get_db
from agent.derivatives import DerivativesContext
import config


@dataclass
class ScreenerScore:
    product_id: str
    display_name: str
    total_score: float
    volume_score: float
    volatility_score: float
    funding_score: float
    trend_score: float
    oi_score: float
    raw_metrics: dict | None = None


class Screener:

    VOLUME_WEIGHT = 0.30
    VOLATILITY_WEIGHT = 0.25
    FUNDING_WEIGHT = 0.20
    TREND_WEIGHT = 0.15
    OI_WEIGHT = 0.10

    MIN_VOLUME = 5_000_000
    MIN_LEVERAGE = 5
    MIN_PRICE = 0.0001

    def __init__(self, cb: CoinbaseClient):
        self.cb = cb
        self.cfg = config.get_config()
        self._derivatives = DerivativesContext()

    async def run(self, max_watchlist: int | None = None) -> list[str]:
        max_watchlist = max_watchlist or self.cfg.max_watchlist
        logger.info("Starting FCM screener...")

        # 1. Discover perp products
        products = await self.cb.list_future_products()
        if not products:
            logger.error("No perp products discovered")
            return []

        # 2. Hydrate per-product details
        products = await self.cb.hydrate_all(products)

        # 3. Apply thresholds
        candidates = []
        for p in products:
            if not p.trading_enabled:
                continue
            if (p.volume_24h or 0) < self.MIN_VOLUME:
                continue
            if (p.max_leverage or 0) < self.MIN_LEVERAGE:
                continue
            if (p.mark_price or 0) < self.MIN_PRICE:
                continue
            candidates.append(p)

        logger.info(f"Screener: {len(candidates)} candidates after filtering")
        if not candidates:
            return []

        # 4. Fetch 1H candles
        candle_map = await self.cb.get_candles_multi(
            [p.product_id for p in candidates], "ONE_HOUR"
        )

        # 4a. Record derivatives snapshots (C3) — funding + OI for each candidate
        for p in candidates:
            if p.funding_rate is not None or p.open_interest is not None:
                await self._derivatives.record_snapshot(
                    p.product_id, p.funding_rate, p.open_interest, p.mark_price
                )

        # 5. Score
        scores: list[ScreenerScore] = []
        for p in candidates:
            candles = candle_map.get(p.product_id, [])
            if len(candles) < 30:
                continue
            df = self._candles_to_df(candles)
            if df.empty:
                continue
            score = self._score(p, df)
            scores.append(score)

        scores = self._normalize(scores)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        top = scores[:max_watchlist]
        result = [s.product_id for s in top]

        await self._log(top)
        logger.info(f"Screener complete. Top {len(result)}: {[s.display_name for s in top]}")
        return result

    def _score(self, product, df: pd.DataFrame) -> ScreenerScore:
        inds = compute_screener_indicators(df)
        atr_pct = inds["atr_pct"]
        ema_sep = inds["ema_sep_pct"]

        volatility = self._volatility_score(atr_pct)
        funding = abs(product.funding_rate) if product.funding_rate else 0.5
        trend = min(ema_sep / 2.0, 1.0)

        raw = {
            "volume_24h": product.volume_24h,
            "mark_price": product.mark_price,
            "funding_rate": product.funding_rate,
            "open_interest": product.open_interest,
            "atr_pct": atr_pct,
            "ema_sep_pct": ema_sep,
        }

        return ScreenerScore(
            product_id=product.product_id,
            display_name=product.display_name,
            total_score=0.0,
            volume_score=product.volume_24h or 0,
            volatility_score=volatility,
            funding_score=funding,
            trend_score=trend,
            oi_score=product.open_interest or 0,
            raw_metrics=raw,
        )

    @staticmethod
    def _volatility_score(atr_pct: float) -> float:
        target = 2.5
        sigma = 2.0
        return float(max(0.0, math.exp(-((atr_pct - target) ** 2) / (2 * sigma ** 2))))

    @staticmethod
    def _normalize(scores: list[ScreenerScore]) -> list[ScreenerScore]:
        if not scores:
            return scores

        def norm(values: list[float]) -> list[float]:
            mn, mx = min(values), max(values)
            if mx == mn:
                return [0.5] * len(values)
            return [(v - mn) / (mx - mn) for v in values]

        vs = norm([s.volume_score for s in scores])
        os = norm([s.oi_score for s in scores])
        fs = norm([s.funding_score for s in scores])

        for s, v, o, f in zip(scores, vs, os, fs):
            s.volume_score = v
            s.oi_score = o
            s.funding_score = f
            s.total_score = (
                s.volume_score * Screener.VOLUME_WEIGHT +
                s.volatility_score * Screener.VOLATILITY_WEIGHT +
                s.funding_score * Screener.FUNDING_WEIGHT +
                s.trend_score * Screener.TREND_WEIGHT +
                s.oi_score * Screener.OI_WEIGHT
            )
        return scores

    async def _log(self, scores: list[ScreenerScore]) -> None:
        try:
            db = await get_db()
            await db.log_screener_run(
                selected_coins=[s.display_name for s in scores],
                scores={s.display_name: {"total": s.total_score, **s.raw_metrics} for s in scores},
            )
        except Exception as e:
            logger.warning(f"Failed to log screener: {e}")

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame([
            {"time": c.start, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").reset_index(drop=True)
