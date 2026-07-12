"""
Indicator Engine — Manual Implementation

All indicators computed locally using pandas/numpy.
Replaces pandas-ta (incompatible with Python 3.14).
"""

import pandas as pd
import numpy as np
from loguru import logger


def _validate_ohlcv(df: pd.DataFrame) -> None:
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


# =====================================================================
# Single indicators
# =====================================================================

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Welles Wilder smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def ema(series: pd.Series, length: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range (Welles Wilder smoothing)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD — returns DataFrame with ['macd', 'signal', 'hist']."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": hist,
    })


def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — returns DataFrame with ['lower', 'middle', 'upper', 'width']."""
    middle = series.rolling(window=length, min_periods=length).mean()
    sigma = series.rolling(window=length, min_periods=length).std()
    upper = middle + std * sigma
    lower = middle - std * sigma
    width = (upper - lower) / middle
    return pd.DataFrame({
        "lower": lower,
        "middle": middle,
        "upper": upper,
        "width": width,
    })


# =====================================================================
# Main compute function
# =====================================================================

def _compute_indicator_df(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all indicator columns on the full 15m + 1h frames.

    Returns the enriched (df_15m, df_1h) DataFrames. Indicators are computed
    on the full series — callers that need zero-look-ahead must slice AFTER
    computation and only inspect rows up to their "current" bar (see
    `compute_indicators_at`).
    """
    _validate_ohlcv(df_15m)
    _validate_ohlcv(df_1h)

    # --- 15m indicators ---
    df_15m = df_15m.copy()
    df_15m["rsi"] = rsi(df_15m["close"], length=14)

    macd_df = macd(df_15m["close"], fast=12, slow=26, signal=9)
    df_15m["macd"] = macd_df["macd"]
    df_15m["macd_signal"] = macd_df["signal"]
    df_15m["macd_hist"] = macd_df["hist"]

    bb = bbands(df_15m["close"], length=20, std=2.0)
    df_15m["bb_lower"] = bb["lower"]
    df_15m["bb_middle"] = bb["middle"]
    df_15m["bb_upper"] = bb["upper"]
    df_15m["bb_width"] = bb["width"]

    df_15m["atr"] = atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)

    vol_sma = df_15m["volume"].rolling(window=20, min_periods=20).mean()
    df_15m["vol_ratio"] = df_15m["volume"] / vol_sma

    # Donchian channels (20-bar) for the breakout strategy.
    # `_prev` versions exclude the current bar — that's what you compare against
    # to detect a *new* breakout (close > prior 20-bar high).
    df_15m["dc_upper"] = df_15m["high"].rolling(window=20, min_periods=20).max()
    df_15m["dc_lower"] = df_15m["low"].rolling(window=20, min_periods=20).min()
    df_15m["dc_middle"] = (df_15m["dc_upper"] + df_15m["dc_lower"]) / 2.0
    df_15m["dc_upper_prev"] = df_15m["dc_upper"].shift(1)
    df_15m["dc_lower_prev"] = df_15m["dc_lower"].shift(1)

    # Rolling-mean BB width — lets the breakout strategy detect volatility
    # *expansion* (current width > recent average) vs. contraction.
    df_15m["bb_width_avg20"] = df_15m["bb_width"].rolling(window=20, min_periods=20).mean()

    # --- 1h indicators ---
    df_1h = df_1h.copy()
    df_1h["ema20"] = ema(df_1h["close"], length=20)
    df_1h["ema50"] = ema(df_1h["close"], length=50)
    df_1h["rsi"] = rsi(df_1h["close"], length=14)

    return df_15m, df_1h


def aggregate_candles(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Aggregate a lower-timeframe candle DataFrame into a higher timeframe
    by grouping `factor` consecutive bars.

    E.g. aggregate_candles(df_2h, 2) → 4h candles.
    Input df must have columns: time, open, high, low, close, volume.
    """
    if df.empty or factor <= 1:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    df["grp"] = np.arange(len(df)) // factor
    agg = df.groupby("grp").agg({
        "time": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index(drop=True)
    return agg


def compute_4h_indicators(df_4h: pd.DataFrame) -> dict:
    """
    Compute 4h trend indicators for the cascading multi-timeframe prompt (C2).
    Returns the last-bar dict with price, ema20, ema50, rsi, price_vs_ema50.
    """
    if df_4h is None or df_4h.empty:
        return {}
    _validate_ohlcv(df_4h)
    df = df_4h.copy()
    df["ema20"] = ema(df["close"], length=20)
    df["ema50"] = ema(df["close"], length=50)
    df["rsi"] = rsi(df["close"], length=14)
    df["atr"] = atr(df["high"], df["low"], df["close"], length=14)
    last = df.iloc[-1]
    def _safe(val):
        return None if pd.isna(val) else float(val)
    return {
        "price": _safe(last["close"]),
        "ema20": _safe(last["ema20"]),
        "ema50": _safe(last["ema50"]),
        "price_vs_ema50": "above" if last["close"] > last["ema50"] else "below",
        "rsi": _safe(last["rsi"]),
        "atr": _safe(last["atr"]),
    }


def compute_indicators(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> dict:
    """
    Compute all indicators for signal evaluation.
    Returns structured dict matching BLUEPRINT.md Section 5.2.

    Convenience wrapper: computes indicators on the full frames and returns
    the dict at the latest bar. For backtesting, use `compute_indicators_at`.
    """
    df_15m_enriched, df_1h_enriched = _compute_indicator_df(df_15m, df_1h)
    return compute_indicators_at(
        df_15m_enriched, df_1h_enriched,
        len(df_15m_enriched) - 1, len(df_1h_enriched) - 1,
    )


def compute_indicators_at(df_15m_full: pd.DataFrame, df_1h_full: pd.DataFrame,
                          i_15m: int, i_1h: int) -> dict:
    """
    Zero-look-ahead indicator extraction.

    Given the FULL pre-computed indicator DataFrames and a "current" index,
    return the indicator dict as of the close of that bar — only using rows
    up to and including that index. This is the backtester's contract:
    no future data leaks into a decision.

    `i_15m` is the index into df_15m_full of the "current" 15m bar.
    `i_1h` is the index of the most recent CLOSED 1h bar as of that 15m bar.
    """
    df_15m = df_15m_full.iloc[: i_15m + 1]
    df_1h = df_1h_full.iloc[: i_1h + 1]
    if df_15m.empty or df_1h.empty:
        return {}

    last = df_15m.iloc[-1]
    prev = df_15m.iloc[-2] if len(df_15m) > 1 else last
    last_1h = df_1h.iloc[-1]

    def _safe(val):
        return None if pd.isna(val) else float(val)

    return {
        "15m": {
            "price": _safe(last["close"]),
            "high": _safe(last["high"]),
            "low": _safe(last["low"]),
            "rsi": _safe(last["rsi"]),
            "rsi_prev": _safe(prev["rsi"]),
            "macd_line": _safe(last["macd"]),
            "macd_signal": _safe(last["macd_signal"]),
            "macd_hist": _safe(last["macd_hist"]),
            "macd_hist_prev": _safe(prev["macd_hist"]),
            "bb_upper": _safe(last["bb_upper"]),
            "bb_middle": _safe(last["bb_middle"]),
            "bb_lower": _safe(last["bb_lower"]),
            "bb_width": _safe(last["bb_width"]),
            "bb_width_avg20": _safe(last["bb_width_avg20"]),
            "atr": _safe(last["atr"]),
            "vol_ratio": _safe(last["vol_ratio"]),
            "dc_upper": _safe(last["dc_upper"]),
            "dc_lower": _safe(last["dc_lower"]),
            "dc_middle": _safe(last["dc_middle"]),
            "dc_upper_prev": _safe(last["dc_upper_prev"]),
            "dc_lower_prev": _safe(last["dc_lower_prev"]),
        },
        "1h": {
            "price": _safe(last_1h["close"]),
            "ema20": _safe(last_1h["ema20"]),
            "ema50": _safe(last_1h["ema50"]),
            "price_vs_ema50": "above" if last_1h["close"] > last_1h["ema50"] else "below",
            "rsi": _safe(last_1h["rsi"]),
        }
    }


# =====================================================================
# Screener indicator helpers
# =====================================================================

def compute_screener_indicators(df_1h: pd.DataFrame) -> dict:
    """
    Compute screener-specific indicators from 1H candles.
    Returns normalized metrics needed for scoring.
    """
    _validate_ohlcv(df_1h)
    df = df_1h.copy()

    df["atr"] = atr(df["high"], df["low"], df["close"], length=14)
    df["ema20"] = ema(df["close"], length=20)
    df["ema50"] = ema(df["close"], length=50)

    last = df.iloc[-1]
    price = float(last["close"])
    atr_val = float(last["atr"])
    atr_pct = (atr_val / price * 100) if price else 0.0

    # Trend clarity: how separated are the EMAs?
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema_sep = abs(ema20 - ema50) / price * 100 if price else 0.0

    return {
        "atr_pct": atr_pct,
        "ema_sep_pct": ema_sep,
        "price_above_ema50": price > ema50,
        "ema20_above_ema50": ema20 > ema50,
    }
