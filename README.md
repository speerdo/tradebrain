# TradeBrain (Burt)

> An AI-powered crypto trading agent with personality, memory, and risk management.
> **Now trading on Coinbase Advanced Perps (FCM)** — CFTC-regulated, USA-compliant.

## What is TradeBrain?

TradeBrain is a personal, locally-run AI trading agent for **Coinbase Financial Markets (FCM)** perp-style futures. It continuously screens ~20 FCM perp markets, evaluates setups using Kimi K2.6 via OpenRouter, and executes leveraged long/short positions — all with mandatory risk controls and Section 1256 tax treatment.

**Paper trading is ON by default.** One toggle switches to live trading.

### Meet Burt

Burt is the agent's personality layer. He's conversational, dry, self-aware, and remembers your past trades. He talks to you like a slightly nerdy friend who knows a lot about trading. He runs in Discord, sends proactive updates, learns from outcomes, and forms long-term semantic memories (powered by pgvector embeddings on Neon).

## Key Capabilities

| Feature | Status |
|---------|--------|
| FCM Market Screener | ✅ Discovers ~22 perp products, scores top 5 |
| AI Signal Evaluation (Kimi K2.6) | ✅ Working with OpenRouter |
| Risk Manager (stops, sizing, circuit breaker) | ✅ Full implementation |
| Paper Trading | ✅ Ready for testing |
| Live Trading (Coinbase FCM) | ✅ Live — real equity, real order ids, real fills |
| Position Monitor | ✅ Full implementation |
| Discord Bot (Burt personality) | ✅ Skeleton ready — needs `DISCORD_BOT_TOKEN` |
| Semantic Memory (pgvector) | ✅ Full implementation |
| SvelteKit Dashboard | ✅ Built with Svelte 5 runes |
| Neon Postgres Logging | ✅ All tables created |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        TradeBrain Agent                      │
├─────────────────────────────────────────────────────────────┤
│  Screener → Indicators → Signal Engine → Risk → Executor    │
│     ↑         ↑              ↑                               │
│  Coinbase   pandas       OpenRouter (Kimi K2.6)            │
│  FCM API                                               │
├─────────────────────────────────────────────────────────────┤
│  Position Monitor (30s) | Burt Bot (Discord) | FastAPI    │
│       ↓                        ↓                  ↓         │
│   Neon DB                  Neon DB            SvelteKit    │
│   (trades, signals,        (memories,          Dashboard   │
│    screener, config)        discord_msgs)       localhost  │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Signal Brain | Kimi K2.6 via OpenRouter |
| Exchange | Coinbase Financial Markets (FCM) — CFTC-regulated US DCM |
| Market Data | Native JWT-signed Coinbase Brokerage v3 client |
| Indicators | Manual pandas implementation (RSI, MACD, BB, ATR, EMA) |
| Backend API | FastAPI + uvicorn |
| Frontend | SvelteKit 2 + Svelte 5 runes |
| Database | Neon Postgres + pgvector |
| Notifications | Discord webhook + discord.py bot |
| Language | Python 3.11+ / TypeScript |

## Strategies

Three built-in strategies provide signal prompts to the AI:

1. **RSI + MACD Momentum** (default) — Trend reversals on 15m with 1h EMA filter
2. **Bollinger Band Mean Reversion** — Fade overextension back to the mean
3. **EMA Trend + Pullback** — Enter pullbacks to 20 EMA in strong trends

## Paper vs Live (they are separate accounts)

`PAPER_TRADING` in `.env` and `agent_config.paper_trading` in the DB select the
mode, and **they must agree** — the agent refuses to start otherwise, and
refuses to flip mode at runtime (a flip would leave open positions, the equity
source and the day's loss budget belonging to the mode it booted in).

| | PAPER | LIVE |
|---|---|---|
| Equity | `paper_balance` ledger in `agent_config`, updated by realized paper P&L | **Coinbase `/cfm/balance_summary`**, cached 30s (`agent/account.py`) |
| Position sizing & daily stop | % of that ledger | % of real account equity |
| Circuit breaker | modeled P&L of paper closes | worse of our modeled losses **and the exchange's own `daily_realized_pnl`** |
| Trades | `trades` rows with `is_paper = TRUE` | `is_paper = FALSE`, plus real `order_id` / `client_order_id` / `stop_order_id` / `exit_order_id` and rows in `fills` |
| Restart | open paper positions restored from `trades` | restored from `trades`, then reconciled against the exchange (exchange wins) |

Every read is scoped to one book: `/api/trades`, `/api/stats`, `/api/analytics`
and `/api/analytics/review` default to the running mode and take
`?mode=paper|live|all`. Burt's context, the nightly consolidation and the
weekly review are scoped the same way — blending simulated fills into live
numbers describes an account that does not exist.

A position stamped for one mode can never be **opened or reduced** while the
agent runs in the other (`agent/trading_mode.py`). Closing is deliberately
exempt: flattening must always work, so a mismatch is logged loudly and the
close proceeds rather than stranding a real position.

## Live Order Audit

Every live order records its identity before and after the fact, so each
`trades` row can be reconciled line-by-line against Coinbase's order history:

- `client_order_id` is minted **before** the request — the only handle that
  finds an order the exchange accepted but whose response never came back.
- `order_id`, `stop_order_id`, `exit_order_id` are the exchange's own ids.
- Fills land in the `fills` table (unique on Coinbase's `fill_id`, so re-syncs
  are idempotent), and their volume-weighted price and real commission are
  written back as `filled_entry_price` / `filled_exit_price` /
  `exchange_fees_usdc` — kept **separate** from the bot's modeled
  `entry_price` / `fees_usdc` so slippage and fee-model drift are visible.
- A sweep every 10 monitor ticks catches fills the bot never saw: a protective
  stop that triggered on its own, a position closed from the Coinbase app.

```bash
# reconcile trades placed before order-id logging existed (dry run by default)
venv/bin/python scripts/backfill_fills.py
venv/bin/python scripts/backfill_fills.py --apply
```

```sql
-- modeled vs actual, per live trade
SELECT id, entry_price, filled_entry_price, exit_price, filled_exit_price,
       fees_usdc AS modeled_fees, exchange_fees_usdc AS actual_fees
FROM trades WHERE is_paper = FALSE ORDER BY id DESC;
```

## Risk Management (Non-Negotiable)

- **Position sizing**: in **whole contracts** at the exchange's overnight margin rate (CFM fills whole contracts; 1 ETH PERP ≈ $260, 1 NEAR PERP ≈ $1,900). Target `risk_per_trade`, hard-capped by `max_risk_per_trade` (default 4%) and `max_margin_pct` (default 50% of balance for one position). Products whose single contract breaks either cap are dropped by the screener. Paper mode uses the same lot sizes so paper results mean something for live.
- **Stop loss**: ATR-based (default 2.5× 15m ATR) or fixed % — placed simultaneously with entry, plus an exchange-native stop in live mode
- **Take profit**: Configurable R:R (default 4.0); trailing stop takes over at +1.5R
- **4h bias gate**: long only above the 4h EMA50, short only below (`require_4h_bias`)
- **Circuit breaker**: Halts ALL trading after the daily loss limit (default 5% **of real account equity** in live mode). In live mode it also trips on the exchange's own realized P&L for the session, and the day's loss is re-seeded from closed trades on restart — a restart no longer hands the day a fresh loss budget.
- **Stale balance**: live entries are refused when the last successful balance read is older than `MAX_BALANCE_STALENESS_SEC` — sizing against a balance the exchange stopped confirming is how a 1% risk becomes an unbounded one.
- **Fees**: modeled at the measured 0.14%/fill taker rate; entries whose round-trip fee exceeds 20% of $-at-risk are rejected
- **Min confidence**: default 0.65 to act on a signal — *tunable live from the UI or Burt*

Before going live, run `venv/bin/python scripts/live_preflight.py` — it shows which products your balance can actually trade in whole contracts, what one loss costs, and previews (does not place) a 1-contract order to confirm fees and payload shape against the exchange.

## Quick Start

TradeBrain runs as **two processes**:

| Process | Command | What it runs |
|---|---|---|
| Agent | `python -m agent.main` | Trading loop + FastAPI on :8000 + position monitor + **Burt** (Discord) |
| Dashboard | `cd ui && npm run dev` | SvelteKit UI on :5173 |

```bash
# 1. Clone & setup
cd tradebrain
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (see "Required Environment Variables" below)

# 3. Initialize database (one-time)
python scripts/setup_db.py

# 4. (Optional) sanity-check the screener
python -c "
import asyncio
from agent.coinbase_client import CoinbaseClient
from agent.screener import Screener
async def test():
    cb = CoinbaseClient()
    s = Screener(cb)
    print(await s.run())
    await cb.close()
asyncio.run(test())
"

# 5. Terminal A — start the agent (this also starts Burt + the API)
venv/bin/python -m agent.main

# 6. Terminal B — start the dashboard
cd ui && npm install && npm run dev
# → open http://localhost:5173
```

## Running Burt (Discord)

Burt is **not** a separate process — he runs as a background asyncio task inside `python -m agent.main`. He only attaches if `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, and `DISCORD_USER_ID` are all set in `.env`. Otherwise the agent runs headless and Burt is silently skipped.

### One-time Discord setup

1. Create an application + bot at <https://discord.com/developers/applications>.
2. Bot tab → enable **Message Content Intent** (Burt won't see your messages without it).
3. OAuth2 → URL Generator → scopes: `bot`; permissions: `Send Messages`, `Read Message History`. Open the generated URL to invite Burt to your server.
4. In Discord (with Developer Mode on), right-click your channel → **Copy Channel ID** → set as `DISCORD_CHANNEL_ID`. Right-click your username → **Copy User ID** → set as `DISCORD_USER_ID`.
5. Bot tab → **Reset Token** → set as `DISCORD_BOT_TOKEN`.

Burt **only** responds in the configured channel, **only** to the configured user. Other messages are ignored.

### Verify he's online

Look for this line in the agent logs after startup:

```
Burt connected as Burt#1234
```

### What you can say to Burt

| Intent | Example |
|---|---|
| Ask about state | "what positions are open?" / "how did I do this week?" |
| Spot-check indicators | "what does BTC's RSI look like right now?" |
| Audit signals | "show me the last 20 signals for ETH" |
| Tune knobs | "lower min confidence to 0.5" / "drop me to 2x leverage" / "switch strategy to bollinger" |
| Pause / resume | "pause" / "stop trading" / "resume" |
| Close a position | "close BTC PERP" → Burt asks for confirmation → reply `confirm` |

Burt has read-only SQL access to the entire database (`query_database` tool), so he can answer arbitrary historical questions about trades, signals, screener picks, and his own memories.

## Running the UI

```bash
cd ui
npm install      # first time only
npm run dev
# → http://localhost:5173
```

The UI talks to FastAPI on `http://127.0.0.1:8000`, so the agent must be running first. CORS is preconfigured for `localhost:5173`.

### Live tunable knobs

Every slider in the sidebar writes to the `agent_config` table and is picked up on the **next loop iteration** (within `signal_interval` seconds — no restart needed):

| Knob | Range | Effect |
|---|---|---|
| Strategy | rsi_macd / bollinger / ema_pullback | Which prompt template Burt uses |
| Leverage | 1–20× | Position notional / margin |
| Risk/Trade | 0.5–5% of balance | $-at-risk per trade |
| Daily Loss Limit | 1–20% | Trips the circuit breaker |
| **Min Confidence** | **30–95%** | **The trade-frequency lever — lower = more trades** |
| ATR Multiplier | 0.5–5× | Stop distance when method = ATR |
| Take Profit RR | 0.5–10 | TP distance vs stop distance |
| Stop Method | atr / fixed | Use ATR or fixed-% stops |
| Signal Interval | 60–3600s | How often the agent re-evaluates the watchlist |
| Max Watchlist | 1–20 | How many symbols the screener keeps |

If you're getting no trades in a flat market, drop **Min Confidence** to ~0.5 — that's the gate most setups fail.

## Required Environment Variables

| Variable | Status | Where to Get |
|---|---|---|
| `DATABASE_URL` | ✅ Ready | [neon.tech](https://neon.tech) |
| `OPENROUTER_API_KEY` | ✅ Ready | [openrouter.ai](https://openrouter.ai) |
| `COINBASE_API_KEY` | ✅ Ready | [portal.cdp.coinbase.com](https://portal.cdp.coinbase.com) — CDP key name |
| `COINBASE_API_SECRET` | ✅ Ready | Same as above — EC private key PEM |
| `DISCORD_BOT_TOKEN` | ⏳ Optional | [discord.com/developers](https://discord.com/developers) |
| `DISCORD_CHANNEL_ID` | ⏳ Optional | Right-click channel → Copy ID |
| `DISCORD_USER_ID` | ⏳ Optional | Right-click your name → Copy ID |

Live-account knobs (all optional, defaults in `config.py`): `LIVE_EQUITY_SOURCE`,
`BALANCE_REFRESH_SEC`, `MAX_BALANCE_STALENESS_SEC`, `ACCOUNT_SNAPSHOT_INTERVAL_SEC`,
`USE_EXCHANGE_REALIZED_PNL`, `SYNC_FILLS`. See `.env.example`.

## Project Structure

```
tradebrain/
├── agent/
│   ├── main.py              # Entry point, startup, main loop
│   ├── api.py               # FastAPI backend
│   ├── coinbase_client.py   # Native JWT Coinbase Brokerage v3 client
│   ├── account.py           # Real account equity from /cfm/balance_summary (cached)
│   ├── trading_mode.py      # PAPER/LIVE source of truth + mismatch guards
│   ├── screener.py          # FCM perp universe discovery + scoring
│   ├── indicator_engine.py  # Manual pandas TA indicators
│   ├── signal_engine.py     # OpenRouter / Kimi K2.6 client
│   ├── risk_manager.py      # Position sizing + circuit breaker
│   ├── executor.py          # Order placement + paper trading
│   ├── position_monitor.py  # Track open positions
│   ├── maintenance.py       # Friday/quarterly maintenance window check
│   ├── regime.py            # BTC dominance + market regime context
│   ├── database.py          # Neon asyncpg client
│   ├── notifier.py          # Discord notifications
│   ├── burt.py              # Personality + Discord bot
│   └── memory_engine.py     # Semantic memory + RAG
├── strategies/
│   ├── base.py              # BaseStrategy ABC
│   ├── rsi_macd.py
│   ├── bollinger.py
│   └── ema_pullback.py
├── ui/                      # SvelteKit 2 dashboard
├── scripts/
│   └── setup_db.py          # One-time DB schema creation
├── config.py                # Central config loader
├── requirements.txt
└── .env                     # Your secrets (gitignored)
```

## Safety

- Paper trading is the default. Switching to live requires manual confirmation, and `.env` + `agent_config` must agree — the agent refuses to start on a mismatch and will not change mode while running.
- Paper and live are fully separated: a simulated position can never be opened or reduced in live mode (or vice versa), and simulated losses can never be charged against the live daily loss budget.
- No entry order is placed without a simultaneous stop loss order.
- Circuit breaker halts all trading at the daily loss limit.
- The agent runs entirely on your local machine. No cloud compute. Your keys stay local.

## Phase Roadmap

- ✅ Phase 1-8: Project scaffolding, database, indicators, screener, strategies, signal engine, risk manager
- ✅ Phase 9: Executor (paper trading ready, live skeleton for Coinbase FCM)
- ✅ Phase 10: Position monitor
- ✅ Phase 11: Memory engine
- ✅ Phase 12: Burt (Discord bot skeleton)
- ✅ Phase 13: Notifier
- ✅ Phase 14: FastAPI backend
- ✅ Phase 15: Main agent loop
- ✅ Phase 16: SvelteKit dashboard
- 🔮 Phase 17+: Integration testing, backtesting, go-live

**Do not enable live trading until paper mode runs cleanly for 2 weeks.**

## License

Private / personal use. Trade at your own risk.
