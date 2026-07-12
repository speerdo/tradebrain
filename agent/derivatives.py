"""
Derivatives Context — funding rate + open interest delta tracking (C3).

The screener already fetches funding/OI snapshots via `hydrate_product_details`.
This module records those snapshots to the DB on each screener run and computes
deltas (trend over recent snapshots) for prompt injection + a server-side
funding-cost rule.

Server-side rule: don't open positions that pay > max_funding_bps_per_day
against us (longs pay positive funding, shorts pay negative).
"""

import time
from typing import Any

from loguru import logger

from agent.database import get_db


MAX_FUNDING_BPS_PER_DAY = 5.0  # 5 bps/day = ~18% annual — reject entries paying more


class DerivativesContext:
    """Records funding/OI snapshots and computes deltas for prompts."""

    def __init__(self):
        pass

    async def record_snapshot(self, symbol: str, funding_rate: float | None,
                               open_interest: float | None,
                               mark_price: float | None) -> None:
        """Record a funding + OI snapshot to the DB."""
        try:
            db = await get_db()
            if funding_rate is not None:
                funding_annual = funding_rate * 3 * 365 * 100  # 8h funding * 3/day * 365
                await db.execute(
                    """INSERT INTO funding_snapshots (symbol, funding_rate, funding_annual, mark_price)
                       VALUES ($1, $2, $3, $4)""",
                    symbol, funding_rate, funding_annual, mark_price,
                )
            if open_interest is not None:
                await db.execute(
                    """INSERT INTO oi_snapshots (symbol, open_interest, mark_price)
                       VALUES ($1, $2, $3)""",
                    symbol, open_interest, mark_price,
                )
        except Exception as exc:
            logger.warning(f"Failed to record derivatives snapshot for {symbol}: {exc}")

    async def get_context(self, symbol: str) -> dict:
        """
        Returns derivatives context for a symbol:
        - funding_rate (latest)
        - funding_trend (8h trend: rising/falling/stable)
        - funding_bps_per_day (cost of holding in bps/day)
        - oi_trend (rising/falling/stable)
        - oi_price_divergence (OI rising while price falling = shorts in control)
        """
        try:
            db = await get_db()
            # Last 10 funding snapshots (~80h of history if recorded every 8h)
            funding_rows = await db.fetch(
                """SELECT funding_rate, mark_price, created_at
                   FROM funding_snapshots
                   WHERE symbol = $1
                   ORDER BY created_at DESC LIMIT 10""",
                symbol,
            )
            oi_rows = await db.fetch(
                """SELECT open_interest, mark_price, created_at
                   FROM oi_snapshots
                   WHERE symbol = $1
                   ORDER BY created_at DESC LIMIT 10""",
                symbol,
            )

            ctx = {
                "funding_rate": None,
                "funding_bps_per_day": None,
                "funding_trend": "unknown",
                "oi_trend": "unknown",
                "oi_price_divergence": "unknown",
            }

            if funding_rows:
                latest = funding_rows[0]
                ctx["funding_rate"] = float(latest["funding_rate"])
                # bps/day = funding_rate * 3 (3 funding periods/day) * 10000
                ctx["funding_bps_per_day"] = ctx["funding_rate"] * 3 * 10000
                if len(funding_rows) >= 3:
                    recent_avg = sum(float(r["funding_rate"]) for r in funding_rows[:3]) / 3
                    older_avg = sum(float(r["funding_rate"]) for r in funding_rows[3:6]) / min(3, len(funding_rows[3:6]))
                    if recent_avg > older_avg * 1.2:
                        ctx["funding_trend"] = "rising"
                    elif recent_avg < older_avg * 0.8:
                        ctx["funding_trend"] = "falling"
                    else:
                        ctx["funding_trend"] = "stable"

            if oi_rows:
                if len(oi_rows) >= 3:
                    recent_oi = sum(float(r["open_interest"]) for r in oi_rows[:3]) / 3
                    older_oi = sum(float(r["open_interest"]) for r in oi_rows[3:6]) / min(3, len(oi_rows[3:6]))
                    if recent_oi > older_oi * 1.05:
                        ctx["oi_trend"] = "rising"
                    elif recent_oi < older_oi * 0.95:
                        ctx["oi_trend"] = "falling"
                    else:
                        ctx["oi_trend"] = "stable"
                    # OI vs price divergence
                    recent_price = sum(float(r["mark_price"]) for r in oi_rows[:3] if r["mark_price"]) / 3
                    older_price = sum(float(r["mark_price"]) for r in oi_rows[3:6] if r["mark_price"]) / min(3, len([r for r in oi_rows[3:6] if r["mark_price"]]))
                    if older_price and recent_oi > older_oi and recent_price < older_price:
                        ctx["oi_price_divergence"] = "shorts_in_control"
                    elif older_price and recent_oi > older_oi and recent_price > older_price:
                        ctx["oi_price_divergence"] = "longs_in_control"
                    elif older_price and recent_oi < older_oi and recent_price > older_price:
                        ctx["oi_price_divergence"] = "longs_unwinding"
                    elif older_price and recent_oi < older_oi and recent_price < older_price:
                        ctx["oi_price_divergence"] = "shorts_unwinding"

            return ctx
        except Exception as exc:
            logger.warning(f"Failed to get derivatives context for {symbol}: {exc}")
            return {
                "funding_rate": None, "funding_bps_per_day": None,
                "funding_trend": "unknown", "oi_trend": "unknown",
                "oi_price_divergence": "unknown",
            }

    def check_funding_rule(self, direction: str, ctx: dict) -> str:
        """
        Server-side rule: reject entries that pay > MAX_FUNDING_BPS_PER_DAY
        against us. Longs pay positive funding, shorts pay negative.

        Returns "" if OK, or a skip reason.
        """
        bps = ctx.get("funding_bps_per_day")
        if bps is None:
            return ""
        # Longs pay when funding is positive; shorts pay when negative
        if direction == "long" and bps > MAX_FUNDING_BPS_PER_DAY:
            return f"Funding cost {bps:.1f} bps/day against long (limit {MAX_FUNDING_BPS_PER_DAY})"
        if direction == "short" and bps < -MAX_FUNDING_BPS_PER_DAY:
            return f"Funding cost {-bps:.1f} bps/day against short (limit {MAX_FUNDING_BPS_PER_DAY})"
        return ""

    @staticmethod
    def format_prompt_block(ctx: dict) -> str:
        """Format derivatives context as a prompt block."""
        if not ctx or ctx.get("funding_rate") is None and ctx.get("oi_trend") == "unknown":
            return ""
        lines = ["\nDERIVATIVES CONTEXT:"]
        if ctx.get("funding_rate") is not None:
            lines.append(
                f"  Funding: {ctx['funding_rate']:.6f} ({ctx.get('funding_bps_per_day', 0):.1f} bps/day, "
                f"trend: {ctx.get('funding_trend', 'unknown')})"
            )
        if ctx.get("oi_trend") != "unknown":
            lines.append(
                f"  OI trend: {ctx.get('oi_trend')} | Divergence: {ctx.get('oi_price_divergence', 'unknown')}"
            )
        return "\n".join(lines) + "\n"