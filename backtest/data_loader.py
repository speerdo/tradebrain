"""
Historical candle fetcher for the backtester.

Fetches 15m + 1h candles from Coinbase Brokerage v3 with pagination,
caches them to a local parquet file keyed by (symbol, timeframe, day).
Subsequent runs read from cache — no API calls.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from agent.coinbase_client import CoinbaseClient

CACHE_DIR = Path(__file__).resolve().parent.parent / "backtest_cache"
GRAN_SECONDS = {
    "FIFTEEN_MINUTE": 15 * 60,
    "ONE_HOUR": 60 * 60,
}
GRAN_COL = {
    "FIFTEEN_MINUTE": "15m",
    "ONE_HOUR": "1h",
}


def _cache_path(symbol: str, granularity: str, days: int) -> Path:
    return CACHE_DIR / f"{symbol}_{granularity}_{days}d.parquet"


async def fetch_candles(
    cb: CoinbaseClient,
    symbol: str,
    granularity: str,
    days: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch `days` of historical candles for `symbol` at `granularity`.
    Returns DataFrame sorted ascending by time with columns:
    [time, open, high, low, close, volume] (time = datetime).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(symbol, granularity, days)
    if use_cache and cpath.exists():
        logger.info(f"Loading cached candles: {cpath}")
        df = pd.read_parquet(cpath)
        if not df.empty:
            return df

    gran_sec = GRAN_SECONDS[granularity]
    now = int(time.time())
    start = now - days * 86400
    # Coinbase max 300 candles per request
    batch_secs = gran_sec * 300

    all_candles = []
    cursor = start
    while cursor < now:
        batch_end = min(cursor + batch_secs, now)
        try:
            candles = await cb.get_candles(
                symbol, granularity=granularity,
                start=cursor, end=batch_end,
            )
        except Exception as exc:
            logger.warning(f"Candle fetch failed for {symbol} [{cursor}-{batch_end}]: {exc}")
            break
        all_candles.extend(candles)
        cursor = batch_end + 1
        # Be polite to the API
        await asyncio.sleep(0.15)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame([
        {"time": c.start, "open": c.open, "high": c.high,
         "low": c.low, "close": c.close, "volume": c.volume}
        for c in all_candles
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    df.to_parquet(cpath)
    logger.info(f"Cached {len(df)} {granularity} candles for {symbol} -> {cpath}")
    return df


async def load_pair(
    cb: CoinbaseClient,
    symbol: str,
    days: int = 180,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load (15m, 1h) candles for a symbol."""
    df_15m = await fetch_candles(cb, symbol, "FIFTEEN_MINUTE", days, use_cache)
    df_1h = await fetch_candles(cb, symbol, "ONE_HOUR", days, use_cache)
    return df_15m, df_1h