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
                product_id, display_name, tax_treatment, product_type, fees_usdc
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
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
        )
        tid = await self.fetchval(sql, *vals)
        logger.debug(f"Logged trade {tid} for {trade['symbol']}")
        return tid

    async def close_trade(self, trade_id: int, exit_price: float, pnl_usdc: float, status: str,
                          realized_partial: float | None = None, fees_usdc: float | None = None) -> None:
        await self.execute(
            """
            UPDATE trades
            SET exit_price = $1, pnl_usdc = $2, status = $3, closed_at = NOW(),
                realized_partial = COALESCE($5, realized_partial),
                fees_usdc = COALESCE($6, fees_usdc)
            WHERE id = $4
            """,
            exit_price, pnl_usdc, status, trade_id, realized_partial, fees_usdc,
        )
        logger.info(f"Closed trade {trade_id}: status={status} pnl=${pnl_usdc:.2f} fees=${fees_usdc or 0:.2f}")

    async def get_open_trades(self) -> list[asyncpg.Record]:
        return await self.fetch(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY created_at DESC"
        )

    async def get_recent_trades(self, limit: int = 50) -> list[asyncpg.Record]:
        return await self.fetch(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT $1", limit
        )

    async def get_today_stats(self) -> dict:
        row = await self.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status != 'open') AS closed_count,
                COUNT(*) FILTER (WHERE pnl_usdc > 0) AS wins,
                COUNT(*) FILTER (WHERE pnl_usdc < 0) AS losses,
                COALESCE(SUM(pnl_usdc) FILTER (WHERE created_at >= CURRENT_DATE), 0) AS pnl_today,
                COALESCE(SUM(pnl_usdc), 0) AS pnl_total,
                COALESCE(SUM(fees_usdc), 0) AS fees_total
            FROM trades
            WHERE created_at >= CURRENT_DATE
        """)
        if row is None:
            return {"closed_count": 0, "wins": 0, "losses": 0, "pnl_today": 0.0, "pnl_total": 0.0,
                    "fees_total": 0.0, "win_rate": 0.0}
        total_closed = row["closed_count"] or 0
        return {
            "closed_count": total_closed,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "pnl_today": float(row["pnl_today"] or 0),
            "pnl_total": float(row["pnl_total"] or 0),
            "fees_total": float(row["fees_total"] or 0),
            "win_rate": (row["wins"] / total_closed * 100) if total_closed else 0.0,
        }

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
