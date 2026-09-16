"""
Walk-forward validation.

A single-window backtest can only tell you a set of parameters worked over
one historical stretch — it can't tell you whether those parameters were
genuinely good or just fit to that stretch's specific regime. Walk-forward
splits history into sequential (train, test) folds: for each fold, a small
parameter grid is searched on the TRAIN window only, the best combo is
picked, and THAT SAME combo is then run once on the TEST window it never
saw. Only the test-window results are trusted as an estimate of live
performance; a big gap between in-sample and out-of-sample scores is the
signature of overfitting.

Folds are anchored (train window grows each fold, test window slides
forward), so later folds validate on data no earlier fold's search touched:

    fold 0: train=[chunk 0]              test=[chunk 1]
    fold 1: train=[chunk 0..1]           test=[chunk 2]
    fold 2: train=[chunk 0..2]           test=[chunk 3]
    ...

Usage:
    python -m backtest.walk_forward --strategy donchian_breakout --symbol BTC-PERP --days 360 --folds 6
"""

import argparse
import asyncio
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd
from loguru import logger

import config
from agent.coinbase_client import CoinbaseClient
from backtest.data_loader import load_pair
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.run import resolve_symbol
from strategies import STRATEGIES
from strategies.base import BaseStrategy

# Kept small deliberately — each combo runs a full bar-by-bar backtest per
# fold, so grid size multiplies fold count directly.
DEFAULT_PARAM_GRID: dict[str, list[float]] = {
    "atr_multiplier": [1.0, 1.5, 2.0],
    "take_profit_rr": [1.5, 2.0, 3.0],
}


@dataclass
class Fold:
    index: int
    train_range: tuple[pd.Timestamp, pd.Timestamp]
    test_range: tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class FoldResult:
    fold: Fold
    best_params: dict
    train_summary: dict
    test_summary: dict


@dataclass
class WalkForwardResult:
    metric: str
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.folds:
            return {"n_folds": 0}
        train_scores = [_score(f.train_summary, self.metric) for f in self.folds]
        test_scores = [_score(f.test_summary, self.metric) for f in self.folds]
        oos_trades = sum(f.test_summary.get("n_trades", 0) for f in self.folds)
        oos_wins = sum(f.test_summary.get("wins", 0) for f in self.folds)
        oos_pnl = sum(f.test_summary.get("net_pnl", 0.0) for f in self.folds)
        train_avg = float(np.mean(train_scores)) if train_scores else 0.0
        test_avg = float(np.mean(test_scores)) if test_scores else 0.0
        return {
            "n_folds": len(self.folds),
            "oos_total_trades": oos_trades,
            "oos_win_rate_pct": oos_wins / oos_trades * 100 if oos_trades else 0.0,
            "oos_net_pnl_sum": oos_pnl,
            f"in_sample_avg_{self.metric}": train_avg,
            f"out_of_sample_avg_{self.metric}": test_avg,
            "overfit_gap": train_avg - test_avg,
        }


def _score(summary: dict, metric: str) -> float:
    """Folds with zero trades are worthless evidence either way, not zeros."""
    if summary.get("n_trades", 0) == 0:
        return float("-inf")
    val = summary.get(metric, float("-inf"))
    return val if np.isfinite(val) else 1e9  # infinite profit_factor (no losses) ranks top, not NaN


def _slice_by_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask].reset_index(drop=True)


def make_folds(df_15m: pd.DataFrame, n_folds: int, min_bars: int = 300) -> list[Fold]:
    """Split into n_folds+1 equal-sized chunks; fold k trains on chunks[0..k]
    and tests on chunk k+1. Drops folds whose train window is too short for
    indicators to warm up on."""
    times = df_15m["time"]
    total = len(times)
    n_chunks = n_folds + 1
    chunk_size = total // n_chunks
    if chunk_size < min_bars:
        raise ValueError(
            f"Not enough data for {n_folds} folds: {total} bars / {n_chunks} chunks "
            f"= {chunk_size} bars/chunk, need >= {min_bars}. Use fewer folds or --days."
        )

    folds = []
    for k in range(n_folds):
        train_end_idx = (k + 1) * chunk_size
        test_start_idx = train_end_idx
        test_end_idx = total if k == n_folds - 1 else (k + 2) * chunk_size
        if train_end_idx < min_bars:
            continue
        folds.append(Fold(
            index=k,
            train_range=(times.iloc[0], times.iloc[train_end_idx - 1]),
            test_range=(times.iloc[test_start_idx], times.iloc[test_end_idx - 1]),
        ))
    return folds


def _run_one(
    strategy: BaseStrategy, base_cfg: BacktestConfig, overrides: dict,
    df_15m: pd.DataFrame, df_1h: pd.DataFrame, symbol: str,
) -> dict:
    cfg = replace(base_cfg, **overrides)
    if len(df_15m) < 100:
        return {"n_trades": 0, "final_equity": cfg.balance_usdc}
    engine = BacktestEngine(strategy, cfg)
    result = engine.run(df_15m, df_1h, symbol=symbol)
    return result.summary()


def run_walk_forward(
    strategy: BaseStrategy,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    symbol: str,
    base_cfg: BacktestConfig,
    param_grid: dict[str, list[float]] = None,
    n_folds: int = 6,
    metric: str = "profit_factor",
) -> WalkForwardResult:
    param_grid = param_grid or DEFAULT_PARAM_GRID
    folds = make_folds(df_15m, n_folds)
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    wf_result = WalkForwardResult(metric=metric)
    for fold in folds:
        train_15m = _slice_by_time(df_15m, *fold.train_range)
        train_1h = _slice_by_time(df_1h, fold.train_range[0], fold.train_range[1])
        test_15m = _slice_by_time(df_15m, *fold.test_range)
        test_1h = _slice_by_time(df_1h, fold.test_range[0], fold.test_range[1])

        best_params, best_summary, best_score = None, None, float("-inf")
        for combo in combos:
            summary = _run_one(strategy, base_cfg, combo, train_15m, train_1h, symbol)
            s = _score(summary, metric)
            if s > best_score:
                best_score, best_params, best_summary = s, combo, summary

        if best_params is None:
            continue
        test_summary = _run_one(strategy, base_cfg, best_params, test_15m, test_1h, symbol)

        logger.info(
            f"Fold {fold.index}: train {fold.train_range[0].date()}..{fold.train_range[1].date()} "
            f"best={best_params} ({metric}={best_score:.2f}, n={best_summary.get('n_trades', 0)}) | "
            f"test {fold.test_range[0].date()}..{fold.test_range[1].date()} "
            f"oos_{metric}={_score(test_summary, metric):.2f} n={test_summary.get('n_trades', 0)} "
            f"net_pnl={test_summary.get('net_pnl', 0.0):.2f}"
        )
        wf_result.folds.append(FoldResult(
            fold=fold, best_params=best_params,
            train_summary=best_summary, test_summary=test_summary,
        ))

    return wf_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TradeBrain walk-forward validator")
    p.add_argument("--strategy", default="rsi_macd", choices=list(STRATEGIES.keys()))
    p.add_argument("--symbol", default="BTC-PERP", help="Coinbase FCM product_id or alias")
    p.add_argument("--days", type=int, default=360)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--balance", type=float, default=100_000.0)
    p.add_argument("--leverage", type=int, default=3)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--metric", default="profit_factor",
                   choices=["profit_factor", "net_pnl", "sharpe_annualized", "win_rate"])
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    cfg = config.get_config()
    if not cfg.coinbase_api_key:
        logger.error("COINBASE_API_KEY missing — cannot fetch candles")
        return 1

    strategy = STRATEGIES.get(args.strategy)
    if not strategy:
        logger.error(f"Unknown strategy: {args.strategy}")
        return 1

    cb = CoinbaseClient()
    try:
        symbol = await resolve_symbol(cb, args.symbol)
        logger.info(f"Resolved symbol: {args.symbol} -> {symbol}")
        df_15m, df_1h = await load_pair(cb, symbol, days=args.days, use_cache=not args.no_cache)
        if df_15m.empty or len(df_15m) < 500:
            logger.error(f"Not enough candle data: {len(df_15m)} 15m bars")
            return 1
        logger.info(f"Loaded {len(df_15m)} 15m bars, {len(df_1h)} 1h bars")

        base_cfg = BacktestConfig(
            balance_usdc=args.balance,
            leverage=args.leverage,
            risk_per_trade_pct=args.risk_per_trade,
            min_confidence=args.min_confidence,
        )

        wf = run_walk_forward(
            strategy, df_15m, df_1h, symbol, base_cfg,
            n_folds=args.folds, metric=args.metric,
        )

        print("\n" + "=" * 70)
        print(f"WALK-FORWARD — {args.strategy} on {args.symbol} ({args.folds} folds, metric={args.metric})")
        print("=" * 70)
        for fr in wf.folds:
            print(
                f"  fold {fr.fold.index}: best={fr.best_params}  "
                f"train_n={fr.train_summary.get('n_trades', 0):>3}  "
                f"test_n={fr.test_summary.get('n_trades', 0):>3}  "
                f"test_net_pnl={fr.test_summary.get('net_pnl', 0.0):>10.2f}  "
                f"test_win_rate={fr.test_summary.get('win_rate', 0.0):>6.1f}%"
            )
        print("-" * 70)
        for k, v in wf.summary().items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:>15.3f}")
            else:
                print(f"  {k:<30} {v:>15}")
        print("=" * 70)
        gap = wf.summary().get("overfit_gap", 0.0)
        if gap > 0.5:
            print(f"⚠️  In-sample score beats out-of-sample by {gap:.2f} — likely overfit to history, not a strategy edge.")
        return 0
    finally:
        await cb.close()


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
