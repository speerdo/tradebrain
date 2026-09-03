"""
Performance Analytics — equity curve, drawdown, risk metrics.

Reads closed trades from the trades table and computes:
- Equity curve (mark-to-mark by closed trade)
- Max drawdown
- Sharpe / Sortino (per-trade, annualized to 96 15m-bars/day * 365)
- Profit factor
- Win rate, expectancy
- Slices by strategy / symbol / hour-of-day / confidence bucket

Also produces a Burt-readable weekly self-review string:
"EMA pullback is 3W/9L in ranging regime — suggest disabling it there."

All queries are read-only and use the existing Database layer.
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np

from agent.database import get_db


CONFIDENCE_BUCKETS = [(0.0, 0.5, "0-50%"), (0.5, 0.65, "50-65%"),
                      (0.65, 0.75, "65-75%"), (0.75, 0.85, "75-85%"),
                      (0.85, 1.01, "85-100%")]


def _bucket(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return label
    return "unknown"


def _sharpe(returns: np.ndarray, periods_per_year: float = 96 * 365) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: float = 96 * 365) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * math.sqrt(periods_per_year))


def _max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


async def compute_analytics(starting_balance: float | None = None) -> dict:
    """
    Compute full analytics from the trades table.

    Returns a dict with:
      - equity_curve: list of {time, equity}
      - summary: overall metrics
      - by_strategy, by_symbol, by_hour, by_confidence: sliced stats

    `starting_balance` defaults to the persistent paper balance for paper
    mode (100k placeholder for live) so the equity curve reflects the real,
    PnL-updated account rather than a hardcoded number.
    """
    db = await get_db()
    if starting_balance is None:
        starting_balance = 100_000.0
        key = await db.get_config_value("paper_balance")
        if key is not None:
            try:
                starting_balance = float(key)
            except ValueError:
                pass
    rows = await db.fetch(
        """
        SELECT symbol, direction, strategy, confidence, entry_price, exit_price,
               pnl_usdc, created_at, closed_at, status
        FROM trades
        WHERE status != 'open' AND pnl_usdc IS NOT NULL
        ORDER BY closed_at ASC
        """
    )

    if not rows:
        return {"equity_curve": [], "summary": {"n_trades": 0},
                "by_strategy": {}, "by_symbol": {}, "by_hour": {}, "by_confidence": {}}

    pnls = [float(r["pnl_usdc"]) for r in rows]
    equity = starting_balance
    curve = []
    for r, p in zip(rows, pnls):
        equity += p
        curve.append({"time": r["closed_at"].isoformat() if r["closed_at"] else r["created_at"].isoformat(),
                       "equity": round(equity, 2)})

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    returns = np.array(pnls) / starting_balance
    expectancy = float(np.mean(pnls)) if pnls else 0.0

    summary = {
        "n_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0.0,
        "net_pnl": round(sum(pnls), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "max_drawdown_pct": round(_max_drawdown([starting_balance] + [e["equity"] for e in curve]) * 100, 2),
        "sharpe": round(_sharpe(returns), 2),
        "sortino": round(_sortino(returns), 2),
        "expectancy_usdc": round(expectancy, 2),
        "final_equity": round(equity, 2),
        "avg_win": round(np.mean(wins), 2) if wins else 0.0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0.0,
    }

    def _slice(key_fn):
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(float(r["pnl_usdc"]))
        out = {}
        for k, ps in groups.items():
            w = [p for p in ps if p > 0]
            l = [p for p in ps if p < 0]
            out[k] = {
                "n": len(ps),
                "wins": len(w),
                "losses": len(l),
                "win_rate_pct": round(len(w) / len(ps) * 100, 1) if ps else 0.0,
                "net_pnl": round(sum(ps), 2),
                "expectancy": round(np.mean(ps), 2) if ps else 0.0,
            }
        return out

    by_strategy = _slice(lambda r: r["strategy"] or "unknown")
    by_symbol = _slice(lambda r: r["symbol"] or "unknown")
    by_hour = _slice(lambda r: str(r["created_at"].hour) if r["created_at"] else "unknown")
    by_confidence = _slice(lambda r: _bucket(float(r["confidence"]) if r["confidence"] is not None else None))

    return {
        "equity_curve": curve,
        "summary": summary,
        "by_strategy": by_strategy,
        "by_symbol": by_symbol,
        "by_hour": by_hour,
        "by_confidence": by_confidence,
        "by_model": await analytics_by_model(),
    }


async def analytics_by_model() -> dict:
    """
    Performance sliced by the `model` column (MODELS.md §5/§7) — the A/B
    dimension for model swaps. Trades join to signals via symbol+time is
    lossy, so this reports per-model signal volume and parse-failure rate;
    PnL attribution per model lands when trades carry a model column.
    """
    db = await get_db()
    rows = await db.fetch(
        """
        SELECT model,
               COUNT(*) AS signals,
               COUNT(*) FILTER (WHERE acted_on) AS acted,
               COUNT(*) FILTER (WHERE parse_failed) AS parse_failures
        FROM signals
        WHERE model IS NOT NULL
          AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY model
        ORDER BY signals DESC
        """
    )
    out = {}
    for r in rows:
        signals = int(r["signals"]) or 1
        out[r["model"]] = {
            "signals": int(r["signals"]),
            "acted_on": int(r["acted"]),
            "parse_failures": int(r["parse_failures"]),
            "parse_failure_rate_pct": round(int(r["parse_failures"]) / signals * 100, 2),
        }
    return out


async def parse_failure_rate(days: int = 7) -> dict:
    """
    Parse-failure rate per model over the lookback window (P1).

    Distinguishes malformed LLM output from genuine no-signals — this is the
    metric that must be measured (>=200 calls) before/after any model swap
    (MODELS.md §6.3).
    """
    db = await get_db()
    rows = await db.fetch(
        """
        SELECT model,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE parse_failed) AS failed
        FROM signals
        WHERE created_at >= NOW() - ($1 || ' days')::interval
          AND model IS NOT NULL
        GROUP BY model
        ORDER BY total DESC
        """,
        str(days),
    )
    out = {}
    for r in rows:
        total = int(r["total"])
        failed = int(r["failed"])
        out[r["model"]] = {
            "calls": total,
            "parse_failures": failed,
            "failure_rate_pct": round(failed / total * 100, 2) if total else 0.0,
        }
    return out


async def weekly_self_review() -> str:
    """
    Produce a Burt-readable weekly review string.

    Format: "EMA pullback is 3W/9L in ranging regime — suggest disabling it there."
    (For now we slice by strategy only — regime tagging of past trades requires
    C1 wiring first. We still surface the worst-performing strategies.)
    """
    db = await get_db()
    rows = await db.fetch(
        """
        SELECT strategy, direction, pnl_usdc
        FROM trades
        WHERE status != 'open' AND pnl_usdc IS NOT NULL
          AND closed_at >= NOW() - INTERVAL '7 days'
        ORDER BY closed_at ASC
        """
    )
    if not rows:
        return "No closed trades in the last 7 days — nothing to review."

    by_strat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_strat[r["strategy"] or "unknown"].append(float(r["pnl_usdc"]))

    notes = []
    for strat, pnls in by_strat.items():
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        net = sum(pnls)
        if losses > wins and net < 0:
            notes.append(f"{strat} is {wins}W/{losses}L over 7d (net ${net:+.2f}) — consider disabling or raising its confidence bar.")
        elif wins > losses and net > 0:
            notes.append(f"{strat} is {wins}W/{losses}L over 7d (net ${net:+.2f}) — performing well, keep as is.")

    if not notes:
        return "All strategies are roughly break-even over the last 7 days."
    return "Weekly review: " + " | ".join(notes)