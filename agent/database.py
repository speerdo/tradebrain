"""
TradeBrain Database Layer — Neon Postgres

Handles asyncpg pool, schema creation, and CRUD for all tables.
pgvector is used for semantic memory (memories.embedding).
"""

import json
import re
from typing import Any
import asyncpg
from loguru import logger

import config

# Defense-in-depth keyword blocklist for the read-only query tool. Even though
# we run inside a READ ONLY transaction, this rejects writes early with a clear
# error and blocks side-effecting commands the read-only flag doesn't cover.
_DANGEROUS_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|REVOKE|TRUNCATE|"
    r"COPY|VACUUM|REINDEX|CLUSTER|LOCK|SET|BEGIN|COMMIT|ROLLBACK|END|"
    r"SAVEPOINT|RELEASE|EXECUTE|CALL|DO|LISTEN|NOTIFY|UNLISTEN|PREPARE|"
    r"DEALLOCATE|REFRESH|RESET|DISCARD|LOAD|SECURITY)\b",
    re.IGNORECASE,
)


class Database:
    """Singleton-style Neon database manager."""

    def __init__(self, dsn: str | None = None):
        self.cfg = config.get_config()
        self.dsn = dsn or self.cfg.database_url
        self.pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create connection pool and register pgvector type."""
        if self.pool is not None:
            return
        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Connected to Neon DB")
        # pgvector type registration on first connection
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector';")
            except Exception:
                logger.warning("pgvector extension may not be enabled")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Closed Neon DB pool")

    async def execute(self, sql: str, *args) -> Any:
        """Low-level execute. Ensure connected first."""
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.execute(sql, *args)

    async def fetch(self, sql: str, *args) -> list[asyncpg.Record]:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args) -> asyncpg.Record | None:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args) -> Any:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.fetchval(sql, *args)

    async def read_only_query(self, sql: str, max_rows: int = 200) -> list[dict]:
        """
        Run a SELECT-only SQL statement with safety guards.

        Used by Burt's `query_database` tool to give the LLM arbitrary
        historical access without risking writes.
        """
        stripped = (sql or "").strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("Empty SQL")
        if len(stripped) > 4000:
            raise ValueError("Query too long (>4000 chars)")
        if ";" in stripped:
            raise ValueError("Multiple statements not allowed")
        first_word = stripped.split(None, 1)[0].upper()
        if first_word not in ("SELECT", "WITH"):
            raise ValueError(f"Only SELECT/WITH queries allowed; got '{first_word}'")
        if _DANGEROUS_SQL.search(stripped):
            raise ValueError("Query contains a forbidden keyword")

        cap = max(1, min(int(max_rows or 200), 1000))
        if not re.search(r"\bLIMIT\s+\d+\b", stripped, re.IGNORECASE):
            stripped = f"{stripped} LIMIT {cap}"

        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute("SET LOCAL statement_timeout = '5s'")
                rows = await conn.fetch(stripped)
        return [dict(r) for r in rows[:cap]]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def log_signal(self, signal: dict) -> int:
        sql = """
            INSERT INTO signals (
                symbol, direction, strategy, confidence, reasoning,
                acted_on, skip_reason, rsi_15m, macd_hist_15m, atr_15m, price, model,
                parse_failed, raw_response_snippet
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
        """
        vals = (
            signal["symbol"],
            signal["direction"],
            signal.get("strategy", ""),
            signal.get("confidence", 0.0),
            signal.get("reasoning", ""),
            signal.get("acted_on", False),
            signal.get("skip_reason", ""),
            signal.get("rsi_15m"),
            signal.get("macd_hist_15m"),
            signal.get("atr_15m"),
            signal.get("price"),
            signal.get("model"),
            signal.get("parse_failed", False),
            signal.get("raw_response_snippet"),
        )
        sid = await self.fetchval(sql, *vals)
        logger.debug(f"Logged signal {sid} for {signal['symbol']}")
        return sid

    async def get_recent_signals(self, limit: int = 100) -> list[asyncpg.Record]:
        return await self.fetch(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT $1", limit
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def log_trade(self, trade: dict) -> int:
        sql = """
            INSERT INTO trades (
                symbol, direction, strategy, confidence, entry_price,
                stop_loss, take_profit, size_usdc, margin_usdc, leverage,
                risk_usdc, is_paper, status, reasoning, order_id, signal_id,
                product_id, display_name, tax_treatment, product_type, fees_usdc,
                contracts, client_order_id, stop_order_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
            RETURNING id
        """
        vals = (
            trade["symbol"],
            trade["direction"],
            trade.get("strategy", ""),
            trade.get("confidence", 0.0),
            trade["entry_price"],
            trade["stop_loss"],
            trade["take_profit"],
            trade["size_usdc"],
            trade["margin_usdc"],
            trade["leverage"],
            trade["risk_usdc"],
            trade.get("is_paper", True),
            trade.get("status", "open"),
            trade.get("reasoning", ""),
            trade.get("order_id"),
            trade.get("signal_id"),
            trade.get("product_id", ""),
            trade.get("display_name", ""),
            trade.get("tax_treatment", "1256"),
            trade.get("product_type", "perp"),
            trade.get("fees_usdc", 0.0),
            int(trade.get("contracts", 0) or 0),
            trade.get("client_order_id"),
            trade.get("stop_order_id"),
        )
        tid = await self.fetchval(sql, *vals)
        logger.debug(f"Logged trade {tid} for {trade['symbol']}")
        return tid

    # ------------------------------------------------------------------
    # Live order audit — real exchange identifiers on the trade row
    # ------------------------------------------------------------------

    # Only these may be written by the order-audit path. A whitelist rather
    # than free-form kwargs: this builds SQL from the caller's keys, and the
    # caller is fed by exchange responses.
    _TRADE_ORDER_FIELDS = (
        "order_id", "client_order_id", "stop_order_id", "exit_order_id",
        "exit_client_order_id", "filled_entry_price", "filled_exit_price",
        "filled_contracts", "exchange_fees_usdc", "fills_synced_at",
    )

    async def update_trade_orders(self, trade_id: int, **fields) -> None:
        """
        Patch exchange identities / actual fill data onto a trade row.

        Kept separate from close_trade so an entry's real order id, average
        fill price and exchange-charged commission land on the row while the
        position is still OPEN — that is what makes the row auditable against
        Coinbase's order history in real time instead of at close.
        """
        updates = {k: v for k, v in fields.items()
                   if k in self._TRADE_ORDER_FIELDS and v is not None}
        if not trade_id or not updates:
            return
        cols = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
        await self.execute(
            f"UPDATE trades SET {cols} WHERE id = $1", trade_id, *updates.values()
        )

    async def record_fills(self, fills: list[dict]) -> int:
        """
        Upsert exchange fills. Returns the number of rows written.

        `fill_id` is unique, so re-syncing the same order (the fills endpoint
        lags, so we poll it more than once) updates in place instead of
        duplicating. That also makes a periodic full re-sync of recent fills
        safe to run on every monitor tick.
        """
        if not fills:
            return 0
        sql = """
            INSERT INTO fills (
                trade_id, fill_id, order_id, client_order_id, product_id, side,
                leg, price, size, commission_usdc, liquidity_indicator,
                filled_at, is_paper, raw
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (fill_id) DO UPDATE SET
                trade_id = COALESCE(EXCLUDED.trade_id, fills.trade_id),
                leg = COALESCE(EXCLUDED.leg, fills.leg),
                price = EXCLUDED.price,
                size = EXCLUDED.size,
                commission_usdc = EXCLUDED.commission_usdc,
                liquidity_indicator = EXCLUDED.liquidity_indicator,
                filled_at = EXCLUDED.filled_at,
                raw = EXCLUDED.raw
        """
        rows = [
            (
                f.get("trade_id") or None,
                f["fill_id"],
                f.get("order_id", ""),
                f.get("client_order_id"),
                f.get("product_id", ""),
                f.get("side"),
                f.get("leg"),
                f.get("price"),
                f.get("size"),
                f.get("commission_usdc"),
                f.get("liquidity_indicator"),
                f.get("filled_at"),
                bool(f.get("is_paper", False)),
                json.dumps(f["raw"]) if f.get("raw") is not None else None,
            )
            for f in fills
        ]
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, rows)
        return len(rows)

    async def get_fills_for_trade(self, trade_id: int) -> list[asyncpg.Record]:
        return await self.fetch(
            "SELECT * FROM fills WHERE trade_id = $1 ORDER BY filled_at ASC", trade_id
        )

    async def get_known_fill_ids(self, product_ids: list[str] | None = None,
                                 limit: int = 500) -> set[str]:
        """Fill ids already stored — lets the periodic re-sync skip work."""
        if product_ids:
            rows = await self.fetch(
                "SELECT fill_id FROM fills WHERE product_id = ANY($1) "
                "ORDER BY filled_at DESC LIMIT $2",
                product_ids, limit,
            )
        else:
            rows = await self.fetch(
                "SELECT fill_id FROM fills ORDER BY filled_at DESC LIMIT $1", limit
            )
        return {r["fill_id"] for r in rows}

    async def find_trade_by_order(self, order_id: str) -> asyncpg.Record | None:
        """Locate the trade an exchange order belongs to (entry, stop or exit)."""
        if not order_id:
            return None
        return await self.fetchrow(
            """
            SELECT * FROM trades
            WHERE order_id = $1 OR exit_order_id = $1 OR stop_order_id = $1
            ORDER BY created_at DESC LIMIT 1
            """,
            order_id,
        )

    # ------------------------------------------------------------------
    # Account equity snapshots (agent/account.py)
    # ------------------------------------------------------------------

    async def log_account_snapshot(self, snap: dict) -> None:
        await self.execute(
            """
            INSERT INTO account_snapshots (
                mode, equity_usdc, buying_power_usdc, cash_usdc,
                unrealized_pnl_usdc, daily_realized_pnl_usdc,
                initial_margin_usdc, maintenance_margin_usdc, source, raw
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            snap.get("mode", ""),
            snap.get("equity_usdc", 0.0),
            snap.get("buying_power_usdc", 0.0),
            snap.get("cash_usdc", 0.0),
            snap.get("unrealized_pnl_usdc", 0.0),
            snap.get("daily_realized_pnl_usdc", 0.0),
            snap.get("initial_margin_usdc", 0.0),
            snap.get("maintenance_margin_usdc", 0.0),
            snap.get("source", ""),
            snap.get("raw"),
        )

    async def get_latest_account_snapshot(self, mode: str | None = None) -> asyncpg.Record | None:
        if mode:
            return await self.fetchrow(
                "SELECT * FROM account_snapshots WHERE mode = $1 "
                "ORDER BY created_at DESC LIMIT 1", mode,
            )
        return await self.fetchrow(
            "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
        )

    async def get_account_equity_curve(self, mode: str, days: int = 30,
                                       source: str | None = None) -> list[asyncpg.Record]:
        """
        Equity snapshots for one mode.

        `source` filters to snapshots taken from the same balance_summary
        field. Different fields measure different things — cfm_usd_balance is
        the futures slice, total_usd_balance is the whole account — so
        splicing them into one series shows a step change in equity where
        nothing actually moved. Filtering means changing
        `live_equity_source` starts a new series rather than corrupting the
        old one.
        """
        sql = """
            SELECT created_at, equity_usdc, unrealized_pnl_usdc,
                   daily_realized_pnl_usdc, source
            FROM account_snapshots
            WHERE mode = $1 AND created_at >= NOW() - ($2 || ' days')::INTERVAL
        """
        args: list = [mode, str(int(days))]
        if source:
            sql += " AND source = $3"
            args.append(source)
        return await self.fetch(sql + " ORDER BY created_at ASC", *args)

    async def close_trade(self, trade_id: int, exit_price: float, pnl_usdc: float, status: str,
                          realized_partial: float | None = None, fees_usdc: float | None = None,
                          exit_order_id: str | None = None,
                          exit_client_order_id: str | None = None) -> None:
        await self.execute(
            """
            UPDATE trades
            SET exit_price = $1, pnl_usdc = $2, status = $3, closed_at = NOW(),
                realized_partial = COALESCE($5, realized_partial),
                fees_usdc = COALESCE($6, fees_usdc),
                exit_order_id = COALESCE($7, exit_order_id),
                exit_client_order_id = COALESCE($8, exit_client_order_id)
            WHERE id = $4
            """,
            exit_price, pnl_usdc, status, trade_id, realized_partial, fees_usdc,
            exit_order_id, exit_client_order_id,
        )
        logger.info(
            f"Closed trade {trade_id}: status={status} pnl=${pnl_usdc:.2f} "
            f"fees=${fees_usdc or 0:.2f}"
            + (f" exit_order={exit_order_id}" if exit_order_id else "")
        )

    # Every trades read below takes `is_paper`. Paper and live rows sit in one
    # table and used to be queried together, so a live session's "today's P&L",
    # open-position list and stats all silently included simulated fills — and
    # the circuit breaker, dashboards and Burt reported the blend as the live
    # account. `None` keeps the old unscoped behaviour for the rare caller
    # that genuinely wants both (e.g. a full audit export).

    @staticmethod
    def _mode_clause(is_paper: bool | None, param: int, prefix: str = "AND") -> str:
        return "" if is_paper is None else f" {prefix} is_paper = ${param}"

    async def get_open_trades(self, is_paper: bool | None = None) -> list[asyncpg.Record]:
        sql = ("SELECT * FROM trades WHERE status = 'open'"
               + self._mode_clause(is_paper, 1) + " ORDER BY created_at DESC")
        return await self.fetch(sql, *([is_paper] if is_paper is not None else []))

    async def get_recent_trades(self, limit: int = 50,
                                is_paper: bool | None = None) -> list[asyncpg.Record]:
        if is_paper is None:
            return await self.fetch(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT $1", limit
            )
        return await self.fetch(
            "SELECT * FROM trades WHERE is_paper = $1 ORDER BY created_at DESC LIMIT $2",
            is_paper, limit,
        )

    async def get_today_stats(self, is_paper: bool | None = None) -> dict:
        sql = """
            SELECT
                COUNT(*) FILTER (WHERE status != 'open') AS closed_count,
                COUNT(*) FILTER (WHERE pnl_usdc > 0) AS wins,
                COUNT(*) FILTER (WHERE pnl_usdc < 0) AS losses,
                COALESCE(SUM(pnl_usdc), 0) AS pnl_today,
                COALESCE(SUM(fees_usdc), 0) AS fees_total
            FROM trades
            WHERE created_at >= CURRENT_DATE
        """ + self._mode_clause(is_paper, 1)
        row = await self.fetchrow(sql, *([is_paper] if is_paper is not None else []))
        if row is None:
            return {"closed_count": 0, "wins": 0, "losses": 0, "pnl_today": 0.0,
                    "fees_total": 0.0, "win_rate": 0.0,
                    "is_paper": is_paper}
        total_closed = row["closed_count"] or 0
        return {
            "closed_count": total_closed,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "pnl_today": float(row["pnl_today"] or 0),
            "fees_total": float(row["fees_total"] or 0),
            "win_rate": (row["wins"] / total_closed * 100) if total_closed else 0.0,
            "is_paper": is_paper,
        }

    async def get_realized_pnl_today(self, is_paper: bool) -> float:
        """Realized P&L booked to closed trades since midnight, one mode only.

        The circuit breaker's own counter is in-memory and resets with the
        process; this is the durable number it re-seeds from on restart so a
        mid-day restart cannot hand the day a fresh loss budget."""
        val = await self.fetchval(
            """
            SELECT COALESCE(SUM(pnl_usdc), 0) FROM trades
            WHERE status != 'open' AND pnl_usdc IS NOT NULL
              AND is_paper = $1 AND closed_at >= CURRENT_DATE
            """,
            is_paper,
        )
        return float(val or 0.0)

    # ------------------------------------------------------------------
    # Config hot-reload
    # ------------------------------------------------------------------

    async def sync_config(self) -> None:
        """Read all keys from agent_config and update in-memory cfg."""
        rows = await self.fetch("SELECT key, value FROM agent_config")
        for row in rows:
            config.set_config_key(row["key"], row["value"])

    async def set_config(self, key: str, value: str) -> None:
        await self.execute(
            """
            INSERT INTO agent_config (key, value, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            key, value,
        )

    async def get_config_value(self, key: str) -> str | None:
        row = await self.fetchrow("SELECT value FROM agent_config WHERE key = $1", key)
        return row["value"] if row else None

    # ------------------------------------------------------------------
    # Screener
    # ------------------------------------------------------------------

    async def log_screener_run(self, selected_coins: list[str], scores: dict) -> int:
        sid = await self.fetchval(
            "INSERT INTO screener_runs (selected_coins, scores) VALUES ($1, $2) RETURNING id",
            selected_coins,
            json.dumps(scores),
        )
        return sid

    async def get_last_screener_run(self) -> asyncpg.Record | None:
        return await self.fetchrow("SELECT * FROM screener_runs ORDER BY created_at DESC LIMIT 1")

    # ------------------------------------------------------------------
    # Memory (Burt)
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_literal(embedding: list[float] | None) -> str | None:
        """Convert an embedding list to a pgvector literal string.

        asyncpg has no native codec for the pgvector `vector` type — binding a
        Python list fails with "expected str, got list". pgvector accepts its
        text representation ('[0.1,0.2,...]') with an explicit ::vector cast.
        """
        if embedding is None:
            return None
        return "[" + ",".join(f"{x!r}" for x in embedding) + "]"

    async def store_memory(self, memory_type: str, content: str, source: str = "",
                           symbol: str = "", strategy: str = "",
                           embedding: list[float] | None = None, importance: float = 0.5) -> int:
        sid = await self.fetchval(
            """
            INSERT INTO memories (memory_type, content, source, symbol, strategy, embedding, importance)
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7)
            RETURNING id
            """,
            memory_type, content, source, symbol, strategy,
            self._vector_literal(embedding),
            importance,
        )
        return sid

    async def search_memories(self, embedding: list[float], limit: int = 5, importance_threshold: float = 0.3) -> list[asyncpg.Record]:
        return await self.fetch(
            """
            SELECT content, importance, memory_type, symbol, strategy
            FROM memories
            WHERE importance > $1
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            importance_threshold, self._vector_literal(embedding), limit,
        )

    async def update_memory_importance(self, memory_id: int, importance: float) -> None:
        await self.execute(
            "UPDATE memories SET importance = $1, updated_at = NOW() WHERE id = $2",
            importance, memory_id,
        )

    # ------------------------------------------------------------------
    # Discord messages
    # ------------------------------------------------------------------

    async def add_discord_message(self, role: str, content: str,
                                   discord_user: str = "", message_id: str = "") -> None:
        await self.execute(
            "INSERT INTO discord_messages (role, content, discord_user, message_id) VALUES ($1, $2, $3, $4)",
            role, content, discord_user, message_id,
        )

    async def get_recent_discord_history(self, limit: int = 20) -> list[asyncpg.Record]:
        return await self.fetch(
            "SELECT role, content FROM discord_messages ORDER BY created_at DESC LIMIT $1", limit
        )

    # ------------------------------------------------------------------
    # Daily consolidations
    # ------------------------------------------------------------------

    async def add_daily_consolidation(self, date, summary: str,
                                       lessons: list[str], stats: dict) -> None:
        await self.execute(
            """
            INSERT INTO daily_consolidations (date, summary, lessons, stats)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (date) DO UPDATE SET
                summary = EXCLUDED.summary,
                lessons = EXCLUDED.lessons,
                stats = EXCLUDED.stats,
                created_at = NOW()
            """,
            date, summary, lessons, json.dumps(stats),
        )

    async def get_last_consolidation(self):
        return await self.fetchrow(
            "SELECT * FROM daily_consolidations ORDER BY date DESC LIMIT 1"
        )


# Singleton access
db_instance: Database | None = None


async def get_db() -> Database:
    global db_instance
    if db_instance is None:
        db_instance = Database()
        await db_instance.connect()
    return db_instance
