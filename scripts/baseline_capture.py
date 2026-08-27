"""
P5 — Capture the pre-change baseline (IMPLEMENTATION_PLAN_V2 Pre-flight P5).

M7's gate is a before/after comparison, so record the "before" while it
still exists: LLM calls/day, symbols evaluated/day, signals logged/day,
parse-failure rate, and estimated LLM spend/day — all from the `signals`
table. Writes docs/baseline_P5_<date>.json for the M7 gate to diff against.

Usage:
    venv/bin/python scripts/baseline_capture.py [--days 14] [--cost-per-call 0.0014]
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from agent.database import get_db  # noqa: E402


async def capture(days: int, cost_per_call: float) -> dict:
    db = await get_db()
    rows = await db.fetch(
        """
        SELECT date_trunc('day', created_at)::date AS day,
               COUNT(*)                       AS signals,
               COUNT(DISTINCT symbol)         AS symbols,
               COUNT(*) FILTER (WHERE parse_failed)          AS parse_failures,
               COUNT(*) FILTER (WHERE acted_on)              AS acted,
               COUNT(*) FILTER (WHERE model IS NOT NULL)     AS with_model
        FROM signals
        WHERE created_at >= NOW() - ($1 || ' days')::interval
        GROUP BY day
        ORDER BY day
        """,
        str(days),
    )
    days_out = []
    for r in rows:
        calls = int(r["signals"])
        days_out.append({
            "date": str(r["day"]),
            "llm_calls": calls,
            "symbols_evaluated": int(r["symbols"]),
            "signals_logged": calls,
            "parse_failures": int(r["parse_failures"]),
            "acted_on": int(r["acted"]),
            "model_populated_pct": round(int(r["with_model"]) / calls * 100, 1) if calls else 0.0,
            "est_spend_usd": round(calls * cost_per_call, 2),
        })

    total_calls = sum(d["llm_calls"] for d in days_out)
    avg = {
        "llm_calls_per_day": round(total_calls / max(1, len(days_out)), 1),
        "symbols_evaluated_per_day": round(
            sum(d["symbols_evaluated"] for d in days_out) / max(1, len(days_out)), 1),
        "parse_failure_rate_pct": round(
            sum(d["parse_failures"] for d in days_out) / max(1, total_calls) * 100, 2),
        "est_spend_per_day_usd": round(total_calls * cost_per_call / max(1, len(days_out)), 2),
    }
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "cost_per_call_usd_assumed": cost_per_call,
        "signal_model": config.get_config().signal_model,
        "caveats": [
            "parse_failed column was added 2026-08-27 (P1); rows before that date "
            "have parse_failed=FALSE by default, NOT measured — parse-failure "
            "baselines only become meaningful once the agent runs post-P1.",
            "est_spend_usd uses an assumed blended cost-per-call, not provider data.",
        ],
        "averages": avg,
        "daily": days_out,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--cost-per-call", type=float, default=0.0014,
                    help="blended USD per signal call (K2.6 ≈ $60/mo at 1,440/day)")
    args = ap.parse_args()

    # DB only — no agent startup, no LLM calls
    os.environ.setdefault("PAPER_TRADING", "true")
    baseline = await capture(args.days, args.cost_per_call)

    print("══════════════ P5 BASELINE (pre-M7) ══════════════")
    print(f"captured: {baseline['captured_at']}")
    print(f"model:    {baseline['signal_model']}")
    print(f"averages over {args.days}d:")
    for k, v in baseline["averages"].items():
        print(f"  {k}: {v}")
    print("\ndaily:")
    for d in baseline["daily"]:
        print(f"  {d['date']}: {d['llm_calls']} calls, {d['symbols_evaluated']} symbols, "
              f"{d['parse_failures']} parse-fail, ~${d['est_spend_usd']}")

    out = f"docs/baseline_P5_{datetime.now(timezone.utc).date()}.json"
    os.makedirs("docs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nSaved: {out} — M7's gate diffs against this file.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
