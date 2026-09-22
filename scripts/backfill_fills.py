"""
Backfill the fill audit trail for live trades placed before order-id logging.

From 2026-09-22 the executor records its own order ids and pulls each order's
fills as it goes (agent/executor.py). Trades placed before that have an entry
`order_id` at best and no exit order id at all, so their rows carry the bot's
MODELED entry/exit prices with nothing to check them against.

This script closes that gap after the fact:

  1. Pulls fills from Coinbase for every product traded live in the window and
     upserts them into `fills` (idempotent on fill_id — safe to re-run).
  2. Links each fill to its trade. Entry fills match on `trades.order_id`.
     Exit fills on legacy rows have no id to match, so they are matched by
     product + opposite side + a close timestamp within --exit-tolerance of
     the trade's `closed_at`; the matched order id is then written back to
     `trades.exit_order_id` so the match is permanent and auditable.
  3. Rolls the real numbers onto the trade row: filled_entry_price,
     filled_exit_price and exchange_fees_usdc (actual commission).

Read-only unless --apply is given.

Usage:
    venv/bin/python scripts/backfill_fills.py                 # dry run
    venv/bin/python scripts/backfill_fills.py --apply
    venv/bin/python scripts/backfill_fills.py --apply --prune-unrelated
"""

import argparse
import asyncio
import sys
from datetime import timedelta

sys.path.insert(0, ".")

from loguru import logger

from agent.coinbase_client import CoinbaseClient
from agent.database import get_db
from agent.executor import Executor


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="how far back to pull fills (default 30)")
    ap.add_argument("--exit-tolerance", type=float, default=300.0,
                    help="seconds between a fill and a trade's closed_at for the "
                         "fill to count as that trade's exit leg (default 300)")
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--prune-unrelated", action="store_true",
                    help="delete fills for products this bot never traded live "
                         "(e.g. old spot history pulled in by an unscoped sync)")
    args = ap.parse_args()
    wet = args.apply

    db = await get_db()
    await db.sync_config()
    cb = CoinbaseClient()
    ex = Executor(cb)
    try:
        products = await ex._audited_product_ids()
        print(f"Products traded live in the window: {products or '(none)'}")
        if not products:
            return 0

        # 1. Pull + upsert fills.
        raw = await cb.get_fills(product_ids=products, limit=250)
        print(f"Exchange returned {len(raw)} fill(s) for those products")
        if wet and raw:
            written = await ex._record_fills(raw)
            print(f"  upserted {written} fill row(s)")
        elif raw:
            print("  (dry run — not written)")

        # 2. Link legacy exit legs by time proximity.
        trades = await db.fetch(
            """
            SELECT id, product_id, direction, order_id, exit_order_id,
                   entry_price, exit_price, closed_at
            FROM trades
            WHERE is_paper = FALSE AND status <> 'open'
              AND created_at >= NOW() - ($1 || ' days')::INTERVAL
            ORDER BY id
            """,
            str(args.days),
        )
        for t in trades:
            fills = await db.fetch(
                "SELECT * FROM fills WHERE product_id = $1 ORDER BY filled_at",
                t["product_id"],
            )
            exit_side = "SELL" if t["direction"] == "long" else "BUY"

            entry = next((f for f in fills if f["order_id"] == (t["order_id"] or "")), None)
            exit_fill = next(
                (f for f in fills if f["order_id"] == (t["exit_order_id"] or "")), None
            )
            if exit_fill is None and t["closed_at"]:
                window = timedelta(seconds=args.exit_tolerance)
                candidates = [
                    f for f in fills
                    if f["side"] == exit_side and f["filled_at"] is not None
                    and abs(f["filled_at"] - t["closed_at"]) <= window
                    and (f["trade_id"] is None or f["trade_id"] == t["id"])
                ]
                if len(candidates) == 1:
                    exit_fill = candidates[0]
                elif len(candidates) > 1:
                    logger.warning(
                        f"trade {t['id']}: {len(candidates)} candidate exit fills "
                        f"within {args.exit_tolerance:.0f}s — skipping rather than guess"
                    )

            patch: dict = {}
            fee = 0.0
            if entry is not None:
                patch["filled_entry_price"] = entry["price"]
                patch["filled_contracts"] = entry["size"]
                fee += float(entry["commission_usdc"] or 0)
            if exit_fill is not None:
                patch["filled_exit_price"] = exit_fill["price"]
                patch["exit_order_id"] = exit_fill["order_id"]
                fee += float(exit_fill["commission_usdc"] or 0)
            if fee:
                patch["exchange_fees_usdc"] = round(fee, 6)
            if not patch:
                print(f"  trade {t['id']}: no matching fills")
                continue

            slip_in = (
                f"{t['entry_price']:.2f}→{patch['filled_entry_price']:.2f}"
                if "filled_entry_price" in patch else "entry unmatched"
            )
            slip_out = (
                f"{(t['exit_price'] or 0):.2f}→{patch['filled_exit_price']:.2f}"
                if "filled_exit_price" in patch else "exit unmatched"
            )
            print(f"  trade {t['id']} {t['product_id']} {t['direction']}: "
                  f"entry {slip_in} | exit {slip_out} | commission ${fee:.4f}")
            if wet:
                await db.update_trade_orders(t["id"], **patch)
                for f, leg in ((entry, "entry"), (exit_fill, "exit")):
                    if f is not None:
                        await db.execute(
                            "UPDATE fills SET trade_id = $1, leg = $2 WHERE id = $3",
                            t["id"], leg, f["id"],
                        )

        # 3. Optional prune of unrelated history.
        unrelated = await db.fetch(
            """
            SELECT product_id, count(*) n FROM fills
            WHERE product_id NOT IN (
                SELECT DISTINCT product_id FROM trades
                WHERE is_paper = FALSE AND product_id <> '')
            GROUP BY 1 ORDER BY 2 DESC
            """
        )
        if unrelated:
            total = sum(r["n"] for r in unrelated)
            print(f"\nFills for products never traded live: {total} row(s) — "
                  + ", ".join(f"{r['product_id']}×{r['n']}" for r in unrelated))
            if args.prune_unrelated and wet:
                await db.execute(
                    """
                    DELETE FROM fills WHERE product_id NOT IN (
                        SELECT DISTINCT product_id FROM trades
                        WHERE is_paper = FALSE AND product_id <> '')
                    """
                )
                print(f"  deleted {total} row(s)")
            elif args.prune_unrelated:
                print("  (dry run — not deleted)")
            else:
                print("  left in place (pass --prune-unrelated to remove)")

        if not wet:
            print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0
    finally:
        await cb.close()
        await db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
