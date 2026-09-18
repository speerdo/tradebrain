"""
Live preflight — run this BEFORE flipping PAPER_TRADING=false.

Places NO orders. It answers, against the real account and the real
exchange, the questions that decide whether live trading can work at all:

  1. Auth + futures provisioning + actual CFM balance.
  2. For every liquid perp: 1-contract notional, overnight margin, and the
     $-at-risk at the configured ATR stop — i.e. which products this account
     can actually trade in whole contracts, and what one loss costs.
  3. /orders/preview of a 1-contract market order + protective stop on the
     best affordable product: validates the payload shape the executor uses
     and reports the exchange's own commission and margin numbers, so the fee
     model in config.py can be checked instead of assumed.

Usage:
    venv/bin/python scripts/live_preflight.py [--balance 200]
"""

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from loguru import logger

import config
from agent.coinbase_client import CoinbaseClient
from agent.indicator_engine import atr as atr_series
from agent.risk_manager import ContractSpec, compute_contract_position
from agent.screener import Screener


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, default=None,
                    help="Account size to evaluate (default: live CFM balance, or paper_balance if 0)")
    args = ap.parse_args()
    cfg = config.get_config()
    # The running agent's knobs live in agent_config (UI/Burt edits), not
    # .env — pull them so the sizing below matches what the bot would do.
    if cfg.database_url:
        try:
            from agent.database import get_db
            db = await get_db()
            await db.sync_config()
            await db.close()
        except Exception as exc:
            logger.warning(f"Could not sync agent_config from DB ({exc}); using .env/defaults")
    cb = CoinbaseClient()
    problems: list[str] = []
    try:
        print("=== 1. AUTH / FUNDING ===")
        if not await cb.verify_auth():
            problems.append("Coinbase auth failed")
            return _finish(problems)
        summary = (await cb.get_futures_balance_summary()).get("balance_summary", {})
        cfm = float((summary.get("cfm_usd_balance") or {}).get("value") or 0)
        bp = float((summary.get("futures_buying_power") or {}).get("value") or 0)
        print(f"  CFM balance: ${cfm:,.2f}   futures buying power: ${bp:,.2f}")
        balance = args.balance or cfm or float(cfg.paper_balance)
        if cfm <= 0:
            problems.append("CFM balance is $0 — sweep funds from spot to futures before going live")
        print(f"  Evaluating sizing for a ${balance:,.2f} account "
              f"(risk/trade {cfg.risk_per_trade:.1%}, hard max {cfg.max_risk_per_trade:.0%}, "
              f"margin cap {cfg.max_margin_pct:.0%}, ATR stop x{cfg.atr_multiplier})")

        print("\n=== 2. WHAT THIS ACCOUNT CAN TRADE (whole contracts, overnight margin) ===")
        products = await cb.hydrate_all(await cb.list_future_products())
        liquid = [p for p in products if p.trading_enabled and (p.volume_24h or 0) >= Screener.MIN_VOLUME
                  and (p.max_leverage or 0) >= Screener.MIN_LEVERAGE]
        candles = await cb.get_candles_multi([p.product_id for p in liquid], "FIFTEEN_MINUTE")
        rows = []
        for p in sorted(liquid, key=lambda x: -(x.volume_24h or 0)):
            px = p.mark_price or p.price or 0
            cs = p.contract_size or 0
            if not px or not cs or not p.margin_rate_long or not p.margin_rate_short:
                rows.append((p.display_name, "no contract economics returned", None))
                continue
            import pandas as pd
            c = candles.get(p.product_id, [])
            stop_pct = None
            if len(c) >= 20:
                df = pd.DataFrame([{"high": x.high, "low": x.low, "close": x.close} for x in c])
                a = float(atr_series(df["high"], df["low"], df["close"]).iloc[-1])
                stop_pct = a * cfg.atr_multiplier / px
            spec = ContractSpec(cs, p.margin_rate_long, p.margin_rate_short)
            sp = stop_pct or 0.02
            sizing = compute_contract_position(
                px, px * (1 - sp), "long", balance, cfg.risk_per_trade, spec,
                cfg.max_risk_per_trade, cfg.max_margin_pct,
                cfg.taker_fee_pct, cfg.min_fee_usdc, cfg.entry_fee_budget_pct_of_risk,
            )
            notional = px * cs
            line = (f"1 ctr = ${notional:>8,.0f} | margin L ${notional * p.margin_rate_long:>7,.0f} "
                    f"S ${notional * p.margin_rate_short:>7,.0f} | ATR stop {sp:.2%} -> risk/ctr "
                    f"${notional * sp:>6.2f} ({notional * sp / balance:.1%} of acct) | "
                    + (f"OK: {sizing.contracts} contract(s)" if sizing.contracts else f"SKIP: {sizing.reason}"))
            rows.append((p.display_name, line, p if sizing.contracts else None))
        tradeable = []
        for name, line, p in rows:
            print(f"  {name:<14} {line}")
            if p is not None:
                tradeable.append(p)
        if not tradeable:
            problems.append(
                f"No liquid product is tradeable in whole contracts on a ${balance:,.0f} account "
                f"at these risk settings — the bot will sit idle live"
            )

        print("\n=== 3. ORDER PREVIEW (nothing is placed) ===")
        target = tradeable[0] if tradeable else next((p for p in liquid if p.display_name.startswith("ETH")), None)
        if target is None:
            print("  no product to preview")
        else:
            px = target.mark_price or target.price
            prev = await cb.preview_futures_market_order(target.product_id, "BUY", 1)
            comm = float(prev.get("commission_total") or 0)
            marg = float(prev.get("order_margin_total") or 0)
            notional = px * target.contract_size
            errs = prev.get("errs") or []
            print(f"  {target.display_name} BUY 1 contract (~${notional:,.2f}): commission ${comm:.4f} "
                  f"({comm / notional:.3%} of notional), margin ${marg:.2f} ({marg / notional:.1%}), errs={errs}")
            modeled = max(notional * cfg.taker_fee_pct, cfg.min_fee_usdc)
            if comm and abs(comm - modeled) / comm > 0.15:
                problems.append(
                    f"Fee model is off: config models ${modeled:.4f}/fill on ${notional:,.0f} notional, "
                    f"exchange quotes ${comm:.4f} — update TAKER_FEE_PCT"
                )
            bad = [e for e in errs if "INSUFFICIENT_FUNDS" not in e]
            if bad:
                problems.append(f"Order preview rejected the payload: {bad}")
            elif errs:
                print("  (insufficient-funds is expected until the CFM account is funded; payload shape accepted)")
        return _finish(problems)
    finally:
        await cb.close()


def _finish(problems: list[str]) -> int:
    print("\n=== RESULT ===")
    if not problems:
        print("  ✅ preflight passed")
        return 0
    for p in problems:
        print(f"  ❌ {p}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
