"""
Database schema setup script for TradeBrain.
Run once to create all tables, extensions, and indexes on Neon Postgres.
"""

import asyncio
import asyncpg
from loguru import logger

SCHEMA_SQL = """
-- Enable pgvector for semantic memory
CREATE EXTENSION IF NOT EXISTS vector;

-- =========================================================================
-- CORE TABLES (BLUEPRINT.md Section 10)
-- =========================================================================

CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    reasoning       TEXT,
    acted_on        BOOLEAN DEFAULT FALSE,
    skip_reason     TEXT,
    rsi_15m         FLOAT,
    macd_hist_15m   FLOAT,
    atr_15m         FLOAT,
    price           FLOAT,
    model           TEXT,
    parse_failed    BOOLEAN DEFAULT FALSE,
    raw_response_snippet TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    confidence      FLOAT,
    entry_price     FLOAT NOT NULL,
    stop_loss       FLOAT NOT NULL,
    take_profit     FLOAT NOT NULL,
    size_usdc       FLOAT NOT NULL,
    margin_usdc     FLOAT NOT NULL,
    leverage        INT NOT NULL,
    risk_usdc       FLOAT NOT NULL,
    is_paper        BOOLEAN DEFAULT TRUE,
    status          TEXT DEFAULT 'open',
    exit_price      FLOAT,
    pnl_usdc        FLOAT,
    closed_at       TIMESTAMPTZ,
    reasoning       TEXT,
    order_id        TEXT,
    signal_id       INT REFERENCES signals(id),
    -- Coinbase FCM product identity
    product_id      TEXT,
    display_name    TEXT,
    tax_treatment   TEXT DEFAULT '1256',
    product_type    TEXT DEFAULT 'perp',
    fees_usdc       FLOAT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_config (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS screener_runs (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    selected_coins  TEXT[],
    scores          JSONB
);

-- =========================================================================
-- BURT TABLES (BURT.md Section 4.2)
-- =========================================================================

CREATE TABLE IF NOT EXISTS memories (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    memory_type     TEXT NOT NULL,
    content         TEXT NOT NULL,
    source          TEXT,
    symbol          TEXT,
    strategy        TEXT,
    embedding       vector(1536),
    importance      FLOAT DEFAULT 0.5,
    times_retrieved INT DEFAULT 0,
    last_retrieved  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS discord_messages (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    discord_user    TEXT,
    message_id      TEXT
);

CREATE TABLE IF NOT EXISTS daily_consolidations (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    date            DATE NOT NULL UNIQUE,
    summary         TEXT NOT NULL,
    lessons         TEXT[],
    stats           JSONB
);

-- =========================================================================
-- INDEXES
-- =========================================================================

CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_product_id ON trades(product_id);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_acted ON signals(acted_on);
CREATE INDEX IF NOT EXISTS idx_screener_created ON screener_runs(created_at DESC);

-- Burt indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_symbol ON memories(symbol);
CREATE INDEX IF NOT EXISTS idx_discord_messages_created ON discord_messages(created_at DESC);

-- pgvector index for similarity search
-- (IVFFlat requires ~1k rows to build first; skip if empty)
-- We'll create this in a separate step after data exists.
COMMENT ON TABLE memories IS 'Burt semantic memory store. Run manual index creation after 1000+ rows.';

-- Insert default config values if not present
INSERT INTO agent_config (key, value) VALUES
    ('paper_trading', 'true'),
    ('leverage', '3'),
    ('risk_per_trade', '0.01'),
    ('daily_loss_limit', '0.05'),
    ('strategy', 'rsi_macd'),
    ('signal_interval', '300'),
    ('max_watchlist', '5'),
    ('min_confidence', '0.65'),
    ('atr_multiplier', '1.5'),
    ('take_profit_rr', '2.0'),
    ('stop_loss_method', 'atr'),
    ('signal_model', 'moonshotai/kimi-k2.6'),
    ('critic_model', ''),
    ('burt_model', ''),
    ('embedding_model', 'openai/text-embedding-3-small'),
    ('consolidation_model', ''),
    ('signal_provider', 'openrouter'),
    ('critic_provider', 'openrouter'),
    ('burt_provider', 'openrouter'),
    ('embedding_provider', 'openrouter'),
    ('consolidation_provider', 'openrouter')
ON CONFLICT (key) DO NOTHING;

-- =========================================================================
-- MIGRATIONS (additive only — never drop columns)
-- Run idempotently on every setup so existing DBs pick up new columns.
-- =========================================================================

ALTER TABLE signals ADD COLUMN IF NOT EXISTS model TEXT;

-- =========================================================================
-- DERIVATIVES CONTEXT (C3) — funding rate + open interest snapshots
-- =========================================================================

CREATE TABLE IF NOT EXISTS funding_snapshots (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          TEXT NOT NULL,
    funding_rate    FLOAT NOT NULL,
    funding_annual  FLOAT,
    mark_price      FLOAT
);
CREATE INDEX IF NOT EXISTS idx_funding_snap_created ON funding_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_funding_snap_symbol ON funding_snapshots(symbol, created_at DESC);

CREATE TABLE IF NOT EXISTS oi_snapshots (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          TEXT NOT NULL,
    open_interest   FLOAT NOT NULL,
    mark_price      FLOAT
);
CREATE INDEX IF NOT EXISTS idx_oi_snap_created ON oi_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_oi_snap_symbol ON oi_snapshots(symbol, created_at DESC);

-- =========================================================================
-- SENTIMENT & NEWS (C4) — Fear & Greed + CryptoPanic cache
-- =========================================================================

CREATE TABLE IF NOT EXISTS sentiment_cache (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    source          TEXT NOT NULL,           -- 'fear_greed' | 'cryptopanic'
    symbol          TEXT,                    -- NULL for global (Fear & Greed)
    value           FLOAT,                   -- F&G value 0-100
    classification  TEXT,                    -- F&G: 'Extreme Fear' etc.
    raw             JSONB                    -- raw API response
);
CREATE INDEX IF NOT EXISTS idx_sentiment_created ON sentiment_cache(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_cache(source, created_at DESC);

CREATE TABLE IF NOT EXISTS news_cache (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    symbol          TEXT,
    title           TEXT NOT NULL,
    url             TEXT,
    sentiment       TEXT,                    -- 'positive' | 'negative' | 'neutral'
    votes_positive  INT DEFAULT 0,
    votes_negative  INT DEFAULT 0,
    published_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_news_created ON news_cache(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_cache(symbol, created_at DESC);

-- =========================================================================
-- MIGRATIONS (idempotent — safe to re-run against existing databases)
-- =========================================================================

-- P1: signal parse-failure instrumentation
ALTER TABLE signals ADD COLUMN IF NOT EXISTS parse_failed BOOLEAN DEFAULT FALSE;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS raw_response_snippet TEXT;

-- Paper restart-safety: PnL already banked by partial take-profits on a paper
-- position must survive restarts so the final close books the correct total.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS realized_partial FLOAT DEFAULT 0.0;

-- Fee accounting: pnl_usdc is now net of modeled taker fees (entry + exit +
-- any partial-close legs) instead of gross. This column tracks the total
-- fee drag per trade for reporting and post-mortems.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fees_usdc FLOAT DEFAULT 0.0;
"""

PGVECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


async def create_schema(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        logger.info("Creating schema...")
        await conn.execute(SCHEMA_SQL)
        logger.info("Schema created successfully")

        # pgvector ivfflat index needs rows to exist; attempt it but don't fail
        try:
            await conn.execute(PGVECTOR_INDEX_SQL)
            logger.info("pgvector IVF index created")
        except Exception:
            logger.info("pgvector IVF index skipped (needs data first)")

        # Verify
        rows = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [r["table_name"] for r in rows]
        logger.info(f"Tables in DB: {', '.join(tables)}")

        if len(tables) < 7:
            raise RuntimeError(f"Expected 7+ tables, got {len(tables)}: {tables}")

        logger.success("✓ Database setup complete")
    finally:
        await conn.close()


async def main():
    import sys
    import os

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    await create_schema(dsn)


if __name__ == "__main__":
    asyncio.run(main())
