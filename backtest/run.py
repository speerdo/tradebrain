"""
Backtest CLI — runs a strategy over historical candles.

Usage:
    python -m backtest.run --strategy rsi_macd --symbol BTC-PERP --days 180
    python -m backtest.run --strategy donchian_breakout --symbol ETH-PERP --days 90 --no-cache

Notes:
- `--symbol` is the Coinbase FCM product_id (e.g. "BIP-20DEC30-CDE" for BTC perp).
  For convenience, the CLI accepts common aliases: BTC-PERP, ETH-PERP, etc.,
  and tries to resolve them via the screener product discovery.
- Requires Coinbase API credentials in .env (for candle fetching).
- Does NOT require DATABASE_URL or OPENROUTER_API_KEY.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

import config
from agent.coinbase_client import CoinbaseClient
from backtest.data_loader import load_pair
from backtest.engine import BacktestEngine, BacktestConfig
from strategies import STRATEGIES

# Common aliases -> Coinbase FCM product_ids. The BTC/ETH perps have stable
# product_ids; others change with expiry so we resolve dynamically if needed.
SYMBOL_ALIASES = {
    "BTC-PERP": "BIP-20DEC30-CDE",
    "ETH-PERP": "ETP-20DEC30-CDE",
}


async def resolve_symbol(cb: CoinbaseClient, requested: str) -> str:
    if requested in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[requested]
    # If it looks like a FCM product id already, pass through
    if "-CDE" in requested or "-DEC" in requested:
        return requested
    # Try to discover via product list
    products = await cb.list_future_products()
    for p in products:
        if requested.upper() in p.display_name.upper():
            return p.product_id
    logger.warning(f"Could not resolve symbol '{requested}' — using as-is")
    return requested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TradeBrain backtester")
    p.add_argument("--strategy", default="rsi_macd", choices=list(STRATEGIES.keys()))
    p.add_argument("--symbol", default="BTC-PERP", help="Coinbase FCM product_id or alias")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--balance", type=float, default=100_000.0)
    p.add_argument("--leverage", type=int, default=3)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--atr-multiplier", type=float, default=1.5)
    p.add_argument("--take-profit-rr", type=float, default=2.0)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--no-cache", action="store_true", help="Skip parquet cache")
    p.add_argument("--no-trailing", action="store_true")
    p.add_argument("--no-breakeven", action="store_true")
    p.add_argument("--no-time-exit", action="store_true")
    p.add_argument("--no-partial-tp", action="store_true")
    p.add_argument("--trailing-mult", type=float, default=2.0)
    p.add_argument("--save-trades", type=str, default="", help="Path to save trades as JSON")
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
        if df_15m.empty or len(df_15m) < 100:
            logger.error(f"Not enough candle data: {len(df_15m)} 15m bars")
            return 1
        logger.info(f"Loaded {len(df_15m)} 15m bars, {len(df_1h)} 1h bars")

        bt_cfg = BacktestConfig(
            balance_usdc=args.balance,
            leverage=args.leverage,
            risk_per_trade_pct=args.risk_per_trade,
            atr_multiplier=args.atr_multiplier,
            take_profit_rr=args.take_profit_rr,
            min_confidence=args.min_confidence,
            max_concurrent_positions=args.max_concurrent,
            enable_trailing=not args.no_trailing,
            enable_breakeven=not args.no_breakeven,
            enable_time_exit=not args.no_time_exit,
            enable_partial_tp=not args.no_partial_tp,
            trailing_atr_mult=args.trailing_mult,
        )

        engine = BacktestEngine(strategy, bt_cfg)
        result = engine.run(df_15m, df_1h, symbol=symbol)
        summary = result.summary()

        print("\n" + "=" * 60)
        print(f"BACKTEST RESULT — {args.strategy} on {args.symbol}")
        print(f"Period: {args.days} days, {len(df_15m)} 15m bars")
        print("=" * 60)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:>15.2f}")
            else:
                print(f"  {k:<25} {v:>15}")
        print("=" * 60)

        if args.save_trades:
            trades_json = [
                {
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl_usdc": t.pnl_usdc,
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                    "fees_usdc": t.fees_usdc,
                    "funding_usdc": t.funding_usdc,
                }
                for t in result.trades
            ]
            Path(args.save_trades).write_text(json.dumps(trades_json, indent=2))
            logger.info(f"Saved {len(trades_json)} trades to {args.save_trades}")

        return 0
    finally:
        await cb.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))