"""
FastAPI Backend — REST API for the SvelteKit dashboard.

Runs on localhost:8000, CORS allowed for localhost:5173.
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

import config
from agent import trading_mode
from agent.account import AccountService
from agent.database import get_db
from agent.executor import Executor
from agent.risk_manager import RiskManager
from agent.screener import Screener

app = FastAPI(title="TradeBrain API", version="1.0.0")

# CORS: allow SvelteKit dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    key: str
    value: str


# ------------------------------------------------------------------
# State (injected at startup)
# ------------------------------------------------------------------

_executor: Executor | None = None
_risk_manager: RiskManager | None = None
_screener: Screener | None = None
_account: AccountService | None = None


def set_agent_state(executor: Executor, risk_manager: RiskManager, screener: Screener,
                    account: AccountService | None = None) -> None:
    global _executor, _risk_manager, _screener, _account
    _executor = executor
    _risk_manager = risk_manager
    _screener = screener
    _account = account


def _mode_filter(mode: str | None) -> bool | None:
    """Resolve a ?mode= query param to an is_paper filter.

    Default is the mode the agent is RUNNING in: paper and live rows share
    the trades table, and a live dashboard showing simulated fills mixed in
    with real ones is worse than no dashboard. 'all' opts out explicitly.
    """
    if mode is None or mode == "":
        return bool(config.get_config().paper_trading)
    normalized = mode.strip().lower()
    if normalized == "all":
        return None
    if normalized in ("paper", "true"):
        return True
    if normalized in ("live", "false"):
        return False
    raise HTTPException(status_code=400, detail="mode must be paper, live or all")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> dict:
    cfg = config.get_config()
    rm_state = _risk_manager.state if _risk_manager else None
    # Tunable knobs come from `cfg` because `set_config_key` updates it
    # synchronously on every PATCH /api/config — so the UI sees the new value
    # immediately. RiskManager.state is a *lagging* cache that only refreshes
    # once per loop iteration (every signal_interval seconds), so reading knobs
    # from there caused the slider to snap back to the old value before the
    # agent had a chance to sync. RiskManager.state is still authoritative for
    # genuinely dynamic state (circuit breaker, daily loss, manual pause).
    snap = _account.last if _account else None
    return {
        "paper_trading": cfg.paper_trading,
        "mode": trading_mode.current_mode(cfg),
        "boot_mode": trading_mode.boot_mode(),
        "mode_drift": trading_mode.check_boot_drift(cfg),
        "strategy": cfg.strategy,
        "leverage": cfg.leverage,
        "risk_per_trade": cfg.risk_per_trade,
        "daily_loss_limit": cfg.daily_loss_limit,
        "min_confidence": cfg.min_confidence,
        "atr_multiplier": cfg.atr_multiplier,
        "take_profit_rr": cfg.take_profit_rr,
        "stop_loss_method": cfg.stop_loss_method,
        "signal_model": cfg.signal_model,
        "signal_interval": cfg.signal_interval,
        "max_watchlist": cfg.max_watchlist,
        "max_risk_per_trade": cfg.max_risk_per_trade,
        "max_margin_pct": cfg.max_margin_pct,
        "require_4h_bias": cfg.require_4h_bias,
        # Equity the agent is actually sizing off — the exchange's number in
        # live mode, the paper ledger in paper mode.
        "balance_usdc": rm_state.balance_usdc if rm_state else 0.0,
        "balance_source": rm_state.balance_source if rm_state else "",
        "balance_stale": rm_state.balance_stale if rm_state else False,
        "balance_age_sec": round(rm_state.balance_age_sec, 1) if rm_state else 0.0,
        "buying_power_usdc": snap.buying_power_usdc if snap else 0.0,
        "unrealized_pnl_usdc": snap.unrealized_pnl_usdc if snap else 0.0,
        "exchange_realized_pnl_usdc": (
            rm_state.exchange_realized_pnl_usdc if rm_state else 0.0
        ),
        "circuit_breaker_active": rm_state.circuit_breaker_active if rm_state else False,
        "daily_loss_usdc": rm_state.daily_loss_usdc if rm_state else 0.0,
        "daily_loss_limit_usdc": (
            rm_state.balance_usdc * rm_state.daily_loss_limit_pct if rm_state else 0.0
        ),
        "manual_pause": rm_state.manual_pause if rm_state else False,
        "open_positions_count": len(_executor.get_open_positions()) if _executor else 0,
    }


@app.get("/api/account")
async def get_account(refresh: bool = False) -> dict:
    """Live account equity straight from Coinbase (or the paper ledger)."""
    if _account is None:
        raise HTTPException(status_code=503, detail="Account service not initialized")
    snap = await _account.get(force=refresh)
    return {
        "mode": snap.mode,
        "equity_usdc": snap.equity_usdc,
        "buying_power_usdc": snap.buying_power_usdc,
        "cash_usdc": snap.cash_usdc,
        "unrealized_pnl_usdc": snap.unrealized_pnl_usdc,
        "daily_realized_pnl_usdc": snap.daily_realized_pnl_usdc,
        "initial_margin_usdc": snap.initial_margin_usdc,
        "maintenance_margin_usdc": snap.maintenance_margin_usdc,
        "source": snap.source,
        "stale": snap.stale,
        "age_sec": round(snap.age_sec, 1),
    }


@app.get("/api/account/history")
async def get_account_history(days: int = 30, mode: str | None = None,
                              all_sources: bool = False) -> list[dict]:
    """
    Persisted equity snapshots — the real account's equity curve.

    Scoped to the configured `live_equity_source` by default: snapshots taken
    from a different balance field measure a different slice of the account
    and would show a step change where nothing moved. `all_sources=true`
    returns the raw history.
    """
    db = await get_db()
    cfg = config.get_config()
    resolved = (mode or trading_mode.current_mode()).upper()
    source = None if (all_sources or resolved == "PAPER") else cfg.live_equity_source
    rows = await db.get_account_equity_curve(resolved, days, source=source)
    return [_record_to_dict(r) for r in rows]


@app.get("/api/fills")
async def get_fills(limit: int = 100, trade_id: int | None = None) -> list[dict]:
    """Exchange fills — the audit trail against Coinbase's own history."""
    db = await get_db()
    if trade_id:
        rows = await db.get_fills_for_trade(trade_id)
    else:
        rows = await db.fetch(
            "SELECT * FROM fills ORDER BY filled_at DESC NULLS LAST LIMIT $1", limit
        )
    return [_record_to_dict(r) for r in rows]


@app.patch("/api/config")
async def update_config(update: ConfigUpdate) -> dict:
    # Mode is not a runtime knob. Flipping paper_trading under a running
    # agent leaves in-memory positions, the paper ledger and the day's loss
    # budget belonging to the old mode — the exact mismatch the mode guards
    # exist to prevent. Refuse here rather than let it land and pause the loop.
    if update.key == "paper_trading":
        requested = trading_mode.parse_flag(update.value)
        if requested is not None and requested is not bool(config.get_config().paper_trading):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Trading mode is {trading_mode.current_mode()} and cannot be "
                    "changed while the agent is running. Stop it, set PAPER_TRADING "
                    "in .env and agent_config.paper_trading to match, then restart."
                ),
            )
    try:
        config.set_config_key(update.key, update.value)
        db = await get_db()
        await db.set_config(update.key, update.value)
        return {"success": True, "key": update.key, "value": update.value}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/trades")
async def get_trades(limit: int = 50, mode: str | None = None) -> list[dict]:
    db = await get_db()
    rows = await db.get_recent_trades(limit, is_paper=_mode_filter(mode))
    return [_record_to_dict(r) for r in rows]


@app.get("/api/signals")
async def get_signals(limit: int = 100) -> list[dict]:
    db = await get_db()
    rows = await db.get_recent_signals(limit)
    return [_record_to_dict(r) for r in rows]


@app.get("/api/stats")
async def get_stats(mode: str | None = None) -> dict:
    db = await get_db()
    return await db.get_today_stats(is_paper=_mode_filter(mode))


@app.get("/api/analytics")
async def get_analytics(mode: str | None = None) -> dict:
    from agent.analytics import compute_analytics
    return await compute_analytics(is_paper=_mode_filter(mode))


@app.get("/api/analytics/review")
async def get_weekly_review(mode: str | None = None) -> dict:
    from agent.analytics import weekly_self_review
    return {"review": await weekly_self_review(is_paper=_mode_filter(mode))}


@app.get("/api/watchlist")
async def get_watchlist() -> dict:
    db = await get_db()
    run = await db.get_last_screener_run()
    if run is None:
        return {"coins": [], "scores": {}}
    return {
        "coins": run.get("selected_coins", []),
        "scores": run.get("scores", {}),
        "run_at": run.get("created_at"),
    }


@app.post("/api/screener/run")
async def run_screener() -> dict:
    if _screener is None:
        raise HTTPException(status_code=503, detail="Screener not initialized")
    try:
        coins = await _screener.run()
        return {"success": True, "coins": coins}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/positions")
async def get_positions() -> list[dict]:
    if _executor is None:
        return []
    positions = _executor.get_open_positions()
    return [
        {
            "symbol": p.symbol,                          # product_id (used to close)
            "display_name": p.display_name or p.symbol,  # user-friendly name for the UI
            "direction": p.direction,
            "entry_price": p.entry_price,
            "stop_loss": p.stop_loss,
            "take_profit": p.take_profit,
            "size_usdc": p.size_usdc,
            "margin_usdc": p.margin_usdc,
            "leverage": p.leverage,
            "is_paper": p.is_paper,
            "mode": p.mode,
            "opened_at": p.opened_at,
            "strategy": p.strategy,
            "confidence": p.confidence,
            "trade_id": p.trade_id,
            "order_id": p.entry_order_id,
        }
        for p in positions
    ]


@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str) -> dict:
    if _executor is None:
        raise HTTPException(status_code=503, detail="Executor not initialized")
    result = await _executor.close_position(symbol)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "symbol": symbol}


@app.post("/api/circuit-breaker/reset")
async def reset_circuit_breaker() -> dict:
    if _risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    _risk_manager.reset_circuit_breaker()
    return {"success": True}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _record_to_dict(record: Any) -> dict:
    """Convert asyncpg Record to plain dict."""
    return dict(record)
