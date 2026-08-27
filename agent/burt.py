"""
Burt — Personality, Memory & Discord Bot

Runs as a background asyncio task alongside the main trading agent.
Requires DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, DISCORD_USER_ID.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
from loguru import logger

import config
from agent.database import get_db
from agent.executor import Executor
from agent import llm_router
from agent.indicator_engine import compute_indicators
from agent.risk_manager import RiskManager
from agent.screener import Screener

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tools whose execution mutates trading state and require user confirmation.
DESTRUCTIVE_TOOLS = {"close_position"}

# Whitelist of agent_config keys Burt is allowed to tune via set_config, with
# safety bounds. Anything else (DB credentials, API keys, etc.) is rejected.
TUNABLE_CONFIG: dict[str, dict] = {
    "leverage":          {"type": "int",    "min": 1,    "max": 20},
    "risk_per_trade":    {"type": "float",  "min": 0.001, "max": 0.05},
    "daily_loss_limit":  {"type": "float",  "min": 0.01, "max": 0.20},
    "min_confidence":    {"type": "float",  "min": 0.30, "max": 0.95},
    "atr_multiplier":    {"type": "float",  "min": 0.5,  "max": 5.0},
    "take_profit_rr":    {"type": "float",  "min": 0.5,  "max": 10.0},
    "fixed_stop_pct":    {"type": "float",  "min": 0.005, "max": 0.10},
    "stop_loss_method":  {"type": "enum",   "choices": ["atr", "fixed"]},
    "signal_model":      {"type": "str",    "min": None, "max": None},
    "strategy":          {"type": "enum",   "choices": ["rsi_macd", "bollinger", "ema_pullback", "donchian_breakout"]},
    "signal_interval":   {"type": "int",    "min": 60,   "max": 3600},
    "max_watchlist":     {"type": "int",    "min": 1,    "max": 20},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_open_positions",
            "description": "List currently open paper positions with entry, direction, stop, take-profit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicator_snapshot",
            "description": (
                "Fetch live 15m + 1h candles for a perp and compute current RSI, MACD, "
                "EMA20/50, Bollinger Bands, ATR, volume ratio. Use this when the user asks "
                "what an indicator looks like, why a signal fired/didn't, or to spot-check "
                "the math. Argument can be a product_id like 'BIP-20DEC30-CDE' or a display "
                "name like 'BTC PERP' or just 'BTC'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Product id or display name."}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_config",
            "description": (
                "Tune a hot-reloadable trading parameter. Writes to agent_config; "
                "the next loop iteration (within `signal_interval` seconds) picks it up. "
                "Allowed keys: leverage (1-20), risk_per_trade (0.001-0.05 = 0.1%-5%), "
                "daily_loss_limit (0.01-0.20), min_confidence (0.30-0.95 — LOWER means "
                "Burt takes more trades), atr_multiplier (0.5-5.0), take_profit_rr "
                "(0.5-10.0), fixed_stop_pct (0.005-0.10), stop_loss_method "
                "('atr'|'fixed'), strategy ('rsi_macd'|'bollinger'|'ema_pullback'), "
                "signal_interval (60-3600 sec), max_watchlist (1-20). Always tell the "
                "user what you changed and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Config key (see allowed list)."},
                    "value": {"type": "string", "description": "New value as a string."},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a read-only SQL SELECT against the TradeBrain Postgres database. "
                "Use this for ANY historical question — past trades, signals, screener "
                "picks, lessons, P&L over arbitrary date ranges, etc. The 'Today' stats "
                "in the live state are just a snapshot; this tool is how you reach "
                "everything else. SELECT/WITH only, single statement, auto-capped at "
                "200 rows (or `max_rows`, max 1000), 5s timeout. Schema is documented "
                "in the system prompt. Postgres dialect — use NOW(), INTERVAL, "
                "CURRENT_DATE, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT or WITH...SELECT statement."},
                    "max_rows": {"type": "integer", "description": "Row cap (default 200, max 1000)."},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_signals",
            "description": (
                "Pull the last N signal-evaluation rows from the database (every screened "
                "symbol logs a row each loop iteration even if it didn't fire). Use this to "
                "audit why trades were or weren't taken."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many rows (default 10, max 50)."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_trading",
            "description": "Set the manual_pause flag — the loop will skip new entries until resumed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_trading",
            "description": "Clear manual_pause so the loop can take new entries again.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_circuit_breaker",
            "description": "Manually clear a tripped daily-loss circuit breaker.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "force_screener_run",
            "description": (
                "Run the screener immediately and return the top picks. Note: does not "
                "override the main loop's live watchlist (which refreshes on its own "
                "cadence) — this is for inspection only."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_position",
            "description": (
                "DESTRUCTIVE: close an open paper position at market. The system will "
                "REFUSE to execute this unless the user's most recent message contains the "
                "literal word 'confirm'. Workflow: (1) describe what you're about to close "
                "and ask the user to reply 'confirm'; (2) on the next turn, after they "
                "confirm, call this tool. Argument is symbol or display name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Product id or display name to close."}
                },
                "required": ["symbol"],
            },
        },
    },
]

# discord.py is optional — graceful fallback if not installed
try:
    import discord
    _has_discord = True
except ImportError:
    _has_discord = False
    discord = None  # type: ignore

try:
    import pytz
    _has_pytz = True
except ImportError:
    _has_pytz = False
    pytz = None  # type: ignore


class Burt:
    """
    Burt's personality, memory, and Discord interface.
    Runs as a background asyncio task alongside the main trading agent.
    """

    def __init__(self, db: Any, executor: Executor, risk_manager: RiskManager,
                 screener: Screener | None = None):
        self.cfg = config.get_config()
        self.db = db
        self.executor = executor
        self.risk = risk_manager
        self.screener = screener
        self._client: Any = None
        self._task: asyncio.Task | None = None
        self._last_proactive_time: datetime | None = None
        self._products_cache: list[Any] = []
        self._products_cache_at: float = 0.0
        self._last_user_msg: str = ""

        if not _has_discord:
            logger.warning("discord.py not installed — Burt disabled")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._setup_events()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Discord bot."""
        if self._client is None:
            logger.warning("Burt client not available")
            return
        if not self.cfg.discord_bot_token:
            logger.warning("DISCORD_BOT_TOKEN missing — Burt not starting")
            return
        try:
            await self._client.start(self.cfg.discord_bot_token)
        except Exception as exc:
            logger.error(f"Burt failed to start: {exc}")

    def stop(self) -> None:
        if self._client:
            asyncio.create_task(self._client.close())

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    def _setup_events(self) -> None:
        @self._client.event
        async def on_ready():
            logger.info(f"Burt connected as {self._client.user}")

        @self._client.event
        async def on_message(message):
            await self._on_message(message)

    async def _on_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        logger.debug(
            f"Burt rx: channel={message.channel.id} author={message.author.id} "
            f"is_self={message.author == self._client.user} "
            f"cfg_channel={self.cfg.discord_channel_id} cfg_user={self.cfg.discord_user_id} "
            f"content={message.content!r}"
        )
        # Ignore bot's own messages
        if message.author == self._client.user:
            return

        # Only respond in designated channel
        if str(message.channel.id) != self.cfg.discord_channel_id:
            logger.info(f"Burt skip: channel {message.channel.id} != {self.cfg.discord_channel_id}")
            return

        # Only respond to designated user
        if str(message.author.id) != self.cfg.discord_user_id:
            logger.info(f"Burt skip: author {message.author.id} != {self.cfg.discord_user_id}")
            return

        # Store message
        try:
            await self.db.add_discord_message(
                role="user",
                content=message.content,
                discord_user=str(message.author),
                message_id=str(message.id),
            )
        except Exception as exc:
            logger.warning(f"Failed to store Discord message: {exc}")

        # Generate response
        try:
            response = await self._generate_response(message.content)
            await message.channel.send(response)
            await self.db.add_discord_message(
                role="assistant",
                content=response,
                discord_user="Burt",
                message_id="",
            )
        except Exception as exc:
            logger.error(f"Burt response failed: {exc}")

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def _generate_response(self, user_message: str) -> str:
        """Route to direct command or LLM."""
        msg_lower = user_message.lower().strip()

        # Side-effect commands bypass the LLM so it can't accidentally toggle trading
        if "stop trading" in msg_lower or msg_lower == "pause":
            self.risk.state.manual_pause = True
            return "Trading paused. Say 'resume' when you're ready."

        if msg_lower in ("resume", "start trading", "unpause"):
            self.risk.state.manual_pause = False
            return "Back in action."

        if msg_lower == "go live":
            return "You sure? We're in paper mode. Say 'yes go live' to confirm."

        return await self._llm_response(user_message)

    async def _get_products_summary(self) -> str:
        """Return a cached one-line summary of the perp universe."""
        import time
        now = time.time()
        if not self._products_cache or (now - self._products_cache_at) > 300:
            try:
                self._products_cache = await self.executor.cb.list_future_products()
                self._products_cache_at = now
            except Exception as exc:
                logger.warning(f"Burt: failed to refresh perp universe: {exc}")
                if not self._products_cache:
                    return "Universe: unavailable right now"

        names = [p.display_name for p in self._products_cache]
        tradable = sum(
            1 for p in self._products_cache
            if p.trading_enabled and p.status == "online"
        )
        return (
            f"Universe ({len(names)} perps on Coinbase FCM, {tradable} tradable): "
            + ", ".join(names)
        )

    async def _llm_response(self, user_message: str) -> str:
        """Query Kimi K2.6 with live trading context + tool-calling loop."""
        if not self.cfg.openrouter_api_key:
            return "(No OPENROUTER_API_KEY configured — can't reach my brain right now.)"

        self._last_user_msg = user_message
        system_prompt = await self._build_system_prompt()

        history_msgs: list[dict] = []
        try:
            recent = await self.db.get_recent_discord_history(limit=11)
            older = list(recent)[1:]  # drop the just-stored current user message
            history_msgs = [
                {"role": r["role"], "content": r["content"]}
                for r in reversed(older)
            ]
        except Exception as exc:
            logger.warning(f"Burt: could not load chat history: {exc}")

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": user_message})

        route = llm_router.route(self.cfg, "burt")

        # Tool-call loop: up to 4 rounds. Each round, the model can either return
        # final text (we stop) or call one or more tools (we execute, append
        # results, loop again).
        async with httpx.AsyncClient(timeout=route.timeout) as client:
            for _ in range(4):
                payload = {
                    "model": route.model,
                    "messages": messages,
                    "tools": TOOLS,
                    "temperature": 0.6,
                    "max_tokens": 800,
                }
                if route.provider == "openrouter":
                    payload["reasoning"] = {"exclude": True}
                try:
                    resp = await client.post(
                        f"{route.base_url}/chat/completions",
                        headers=route.headers, json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    logger.error(f"Burt LLM HTTP {exc.response.status_code}: {exc.response.text[:300]}")
                    return "(Brain hit an API error — try again in a sec.)"
                except Exception as exc:
                    logger.error(f"Burt LLM call failed: {exc}")
                    return "(Brain offline for a sec — try again.)"

                msg = data["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    content = (msg.get("content") or "").strip()
                    if not content:
                        finish = data["choices"][0].get("finish_reason")
                        logger.warning(f"Burt: empty LLM content; finish_reason={finish}")
                        return "(Brain returned empty — try again?)"
                    return content

                # Echo the assistant turn (with its tool_calls) back into the conversation.
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })

                # Execute each tool call and append its result.
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    result = await self._dispatch_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            # Loop budget exhausted
            return "(Brain went in circles — try rephrasing.)"

    async def _build_system_prompt(self) -> str:
        positions = self.executor.get_open_positions()
        if positions:
            pos_lines = "\n".join(
                f"  {p.symbol} {p.direction.upper()} entry={p.entry_price}"
                for p in positions
            )
        else:
            pos_lines = "  (none)"

        try:
            stats = await self.db.get_today_stats()
            stats_line = (
                f"Today: {stats.get('wins', 0)}W/{stats.get('losses', 0)}L  "
                f"P&L=${stats.get('pnl_today', 0):+.2f}  "
                f"closed={stats.get('closed_count', 0)}"
            )
        except Exception:
            stats_line = "Today: no data yet"

        pause_state = "paused" if self.risk.state.manual_pause else "active"
        cb_state = "TRIPPED" if self.risk.state.circuit_breaker_active else "ok"
        mode = "paper" if self.cfg.paper_trading else "LIVE"
        universe_line = await self._get_products_summary()

        return (
            "You are Burt, the personality and chat interface for a crypto-perps "
            "trading agent (TradeBrain on Coinbase Advanced). You speak as a terse, "
            "experienced, slightly dry trader. Plain text, short sentences, no "
            "markdown headers or bullet lists, no emojis unless the user uses them first. "
            "Be conversational — this is a Discord chat, not a status report.\n\n"
            "LIVE STATE (refreshed every turn):\n"
            f"Mode: {mode}  |  Risk: {pause_state}  |  Circuit breaker: {cb_state}\n"
            f"{stats_line}\n"
            f"{universe_line}\n"
            "Open positions:\n"
            f"{pos_lines}\n\n"
            "TOOL USE:\n"
            "- You have function tools: read-only (get_*, query_database), config "
            "(set_config), and control (pause/resume/reset_circuit_breaker/"
            "force_screener_run/close_position).\n"
            "- set_config tunes risk/strategy knobs live. The single biggest lever for "
            "trade frequency is min_confidence — drop it (e.g. 0.50) to take more "
            "trades, raise it to be picky. Always tell the user what you changed.\n"
            "- Strategy selector — match to regime: rsi_macd (trending momentum reversals), "
            "bollinger (range-bound mean reversion — DANGEROUS in strong trends), "
            "ema_pullback (continuation in established trends), donchian_breakout "
            "(volatility-expansion / range breakouts — the cleanest fit for leverage).\n"
            "- Use them when the user asks something the live state above can't answer "
            "(e.g. 'show me BTC's RSI right now' → get_indicator_snapshot).\n"
            "- close_position is destructive. NEVER call it without first describing the "
            "action and asking the user to reply with the literal word 'confirm'. The "
            "system will refuse the call otherwise.\n"
            "- Don't call tools just to summarize what the user can see in the live state.\n\n"
            "DATABASE ACCESS (query_database):\n"
            "The 'Today' line above is just a snapshot. For ANY question about the past "
            "— last week's P&L, win rate this month, trades in a specific symbol, "
            "screener history, your own past lessons — use query_database with a "
            "Postgres SELECT. The full schema is yours; only writes are blocked.\n"
            "Tables (Postgres, all timestamps are TIMESTAMPTZ):\n"
            "  trades(id, created_at, closed_at, symbol, direction, strategy, "
            "confidence, entry_price, stop_loss, take_profit, size_usdc, margin_usdc, "
            "leverage, risk_usdc, is_paper, status, exit_price, pnl_usdc, reasoning, "
            "order_id, signal_id, product_id, display_name, tax_treatment, product_type)\n"
            "    -- status is 'open' or a closed variant; pnl_usdc only set when closed\n"
            "  signals(id, created_at, symbol, direction, strategy, confidence, "
            "reasoning, acted_on, skip_reason, rsi_15m, macd_hist_15m, atr_15m, price)\n"
            "    -- one row per screened symbol per loop iteration; acted_on=true means "
            "it became a trade\n"
            "  screener_runs(id, created_at, selected_coins TEXT[], scores JSONB)\n"
            "  memories(id, created_at, updated_at, memory_type, content, source, "
            "symbol, strategy, importance, times_retrieved, last_retrieved)\n"
            "    -- your own long-term memory store; embedding column exists but skip it\n"
            "  discord_messages(id, created_at, role, content, discord_user, message_id)\n"
            "  daily_consolidations(id, created_at, date, summary, lessons TEXT[], "
            "stats JSONB)\n"
            "  agent_config(key, value, updated_at)\n"
            "Use NOW(), CURRENT_DATE, INTERVAL '7 days', date_trunc('day', ...), "
            "FILTER (WHERE ...), etc. Always include a WHERE on created_at when scanning "
            "trades/signals so you don't pull the whole table. Examples:\n"
            "  -- last 7 days P&L\n"
            "  SELECT COUNT(*) FILTER (WHERE pnl_usdc > 0) AS wins, "
            "COUNT(*) FILTER (WHERE pnl_usdc < 0) AS losses, "
            "ROUND(SUM(pnl_usdc)::numeric, 2) AS pnl FROM trades "
            "WHERE closed_at >= NOW() - INTERVAL '7 days' AND status != 'open';\n"
            "  -- best/worst trades this month\n"
            "  SELECT symbol, direction, pnl_usdc, closed_at FROM trades "
            "WHERE closed_at >= date_trunc('month', NOW()) AND status != 'open' "
            "ORDER BY pnl_usdc DESC LIMIT 10;\n\n"
            "REPLY STYLE:\n"
            "- Never invent numbers. If a tool result doesn't have what you need, say so plainly.\n"
            "- Keep replies under ~3 sentences unless the user explicitly asks for detail."
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return a JSON string result for the LLM."""
        # Server-side gate: destructive tools require the user's most recent message
        # to literally contain 'confirm'. Belt-and-suspenders alongside the prompt.
        if name in DESTRUCTIVE_TOOLS and "confirm" not in self._last_user_msg.lower():
            return json.dumps({
                "error": "REFUSED_NOT_CONFIRMED",
                "message": (
                    "Destructive action blocked. Tell the user what you intend to do "
                    "and ask them to reply with the word 'confirm' before retrying."
                ),
            })

        try:
            if name == "get_open_positions":
                positions = self.executor.get_open_positions()
                return json.dumps([
                    {
                        "symbol": p.symbol,
                        "direction": p.direction,
                        "entry_price": p.entry_price,
                        "stop_loss": getattr(p, "stop_loss", None),
                        "take_profit": getattr(p, "take_profit", None),
                        "size_usdc": getattr(p, "size_usdc", None),
                    }
                    for p in positions
                ])

            if name == "get_indicator_snapshot":
                symbol = args.get("symbol", "")
                product_id = await self._resolve_product_id(symbol)
                if not product_id:
                    return json.dumps({"error": "UNKNOWN_SYMBOL", "input": symbol})
                snap = await self._indicator_snapshot(product_id)
                return json.dumps(snap)

            if name == "set_config":
                key = (args.get("key") or "").strip()
                raw_value = args.get("value")
                if key not in TUNABLE_CONFIG:
                    return json.dumps({
                        "error": "FORBIDDEN_KEY",
                        "key": key,
                        "allowed": list(TUNABLE_CONFIG.keys()),
                    })
                spec = TUNABLE_CONFIG[key]
                try:
                    if spec["type"] == "int":
                        coerced = int(float(raw_value))
                        if not (spec["min"] <= coerced <= spec["max"]):
                            raise ValueError(f"out of range [{spec['min']}, {spec['max']}]")
                    elif spec["type"] == "float":
                        coerced = float(raw_value)
                        if not (spec["min"] <= coerced <= spec["max"]):
                            raise ValueError(f"out of range [{spec['min']}, {spec['max']}]")
                    elif spec["type"] == "enum":
                        coerced = str(raw_value)
                        if coerced not in spec["choices"]:
                            raise ValueError(f"must be one of {spec['choices']}")
                    elif spec["type"] == "str":
                        coerced = str(raw_value).strip()
                        if not coerced:
                            raise ValueError("must be a non-empty string")
                    else:
                        raise ValueError("unknown spec type")
                except (ValueError, TypeError) as exc:
                    return json.dumps({"error": "INVALID_VALUE", "key": key, "detail": str(exc)})
                str_val = str(coerced)
                await self.db.set_config(key, str_val)
                config.set_config_key(key, str_val)
                return json.dumps({"ok": True, "key": key, "value": str_val,
                                   "note": "Effective on next loop iteration."})

            if name == "query_database":
                sql = args.get("sql", "")
                max_rows = args.get("max_rows", 200) or 200
                try:
                    rows = await self.db.read_only_query(sql, max_rows=max_rows)
                except ValueError as exc:
                    return json.dumps({"error": "INVALID_SQL", "detail": str(exc)})
                return json.dumps({"row_count": len(rows), "rows": rows}, default=str)

            if name == "get_recent_signals":
                limit = min(int(args.get("limit", 10) or 10), 50)
                rows = await self.db.get_recent_signals(limit=limit)
                return json.dumps([dict(r) for r in rows], default=str)

            if name == "pause_trading":
                self.risk.state.manual_pause = True
                return json.dumps({"ok": True, "manual_pause": True})

            if name == "resume_trading":
                self.risk.state.manual_pause = False
                return json.dumps({"ok": True, "manual_pause": False})

            if name == "reset_circuit_breaker":
                self.risk.reset_circuit_breaker()
                return json.dumps({"ok": True, "circuit_breaker_active": False})

            if name == "force_screener_run":
                if not self.screener:
                    return json.dumps({"error": "NO_SCREENER"})
                top = await self.screener.run()
                return json.dumps({"ok": True, "top_picks": top})

            if name == "close_position":
                symbol = args.get("symbol", "")
                product_id = self._match_open_position(symbol)
                if not product_id:
                    return json.dumps({
                        "error": "NO_MATCHING_POSITION",
                        "input": symbol,
                        "open": [p.symbol for p in self.executor.get_open_positions()],
                    })
                result = await self.executor.close_position(product_id)
                return json.dumps({
                    "ok": getattr(result, "success", False),
                    "product_id": product_id,
                    "error": getattr(result, "error", None),
                })

            return json.dumps({"error": "UNKNOWN_TOOL", "name": name})
        except Exception as exc:
            logger.exception(f"Burt tool {name} failed")
            return json.dumps({"error": "TOOL_EXCEPTION", "name": name, "detail": str(exc)})

    # ------------------------------------------------------------------
    # Symbol resolution + indicator snapshot
    # ------------------------------------------------------------------

    async def _resolve_product_id(self, symbol: str) -> str | None:
        """Map a user-friendly symbol like 'BTC' or 'BTC PERP' to a product_id."""
        if not symbol:
            return None
        await self._get_products_summary()  # populates cache
        s = symbol.strip().upper()
        for p in self._products_cache:
            if p.product_id.upper() == s:
                return p.product_id
            if p.display_name.upper() == s:
                return p.product_id
        # prefix match on display_name (e.g. 'BTC' matches 'BTC PERP')
        for p in self._products_cache:
            if p.display_name.upper().startswith(s):
                return p.product_id
        return None

    def _match_open_position(self, symbol: str) -> str | None:
        s = (symbol or "").strip().upper()
        for p in self.executor.get_open_positions():
            if p.symbol.upper() == s:
                return p.symbol
            display = getattr(p, "display_name", "") or ""
            if display.upper() == s or display.upper().startswith(s):
                return p.symbol
        return None

    async def _indicator_snapshot(self, product_id: str) -> dict:
        """Fetch live candles + compute the same indicators used for trade signals."""
        cb = self.executor.cb
        candles_15m = await cb.get_candles(product_id, "FIFTEEN_MINUTE")
        candles_1h = await cb.get_candles(product_id, "ONE_HOUR")
        if len(candles_15m) < 30 or len(candles_1h) < 20:
            return {"error": "INSUFFICIENT_CANDLES",
                    "have_15m": len(candles_15m), "have_1h": len(candles_1h)}
        df_15m = self._candles_to_df(candles_15m)
        df_1h = self._candles_to_df(candles_1h)
        indicators = compute_indicators(df_15m, df_1h)
        indicators["product_id"] = product_id
        indicators["candles_used"] = {"15m": len(candles_15m), "1h": len(candles_1h)}
        return indicators

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame([
            {"time": c.start, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Proactive messaging
    # ------------------------------------------------------------------

    async def proactive_loop(self) -> None:
        """Background task for proactive messages."""
        # Skeleton — full implementation would schedule morning briefs, trade notifications, etc.
        pass

    # ------------------------------------------------------------------
    # Notifications (called by main agent)
    # ------------------------------------------------------------------

    async def notify_trade_opened(self, symbol: str, order: dict, signal: Any) -> None:
        """Send trade notification in Burt's voice."""
        if self._client is None:
            return
        channel = self._client.get_channel(int(self.cfg.discord_channel_id))
        if channel:
            await channel.send(f"Opened {signal.direction} {symbol}. Let's see how this goes.")

    async def notify_trade_closed(self, trade: Any) -> None:
        """Send close notification in Burt's voice."""
        if self._client is None:
            return
        channel = self._client.get_channel(int(self.cfg.discord_channel_id))
        if channel:
            pnl_text = f"P&L: ${trade.pnl_usdc:+.2f}" if hasattr(trade, 'pnl_usdc') else ""
            await channel.send(f"Closed {trade.symbol}. {pnl_text}")

    async def notify_circuit_breaker(self) -> None:
        """Send circuit breaker alert in Burt's voice."""
        if self._client is None:
            return
        channel = self._client.get_channel(int(self.cfg.discord_channel_id))
        if channel:
            await channel.send("Circuit breaker triggered. I'm done for the day.")

    async def morning_brief(self) -> None:
        """Send morning brief."""
        if self._client is None:
            return
        channel = self._client.get_channel(int(self.cfg.discord_channel_id))
        if channel:
            await channel.send("Morning. Screener's running. Will report back if anything looks good.")

    # ------------------------------------------------------------------
    # Active hours check
    # ------------------------------------------------------------------

    @staticmethod
    def is_active_hours() -> bool:
        if not _has_pytz:
            return True  # fallback: always active
        et = pytz.timezone("America/New_York")
        now = datetime.now(et)
        cfg = config.get_config()
        return cfg.burt_active_hours_start <= now.hour < cfg.burt_active_hours_end
