"""
Central configuration for TradeBrain.
Loads from .env, provides defaults, hot-reloadable.
"""

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Load .env from project root
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")


class Config(BaseModel):
    """TradeBrain configuration — all values loaded from env or defaults."""

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    openrouter_api_key: str = Field(default="")
    coinbase_api_key: str = Field(default="")
    coinbase_api_secret: str = Field(default="")
    database_url: str = Field(default="")

    # Burt / Discord
    discord_bot_token: str = Field(default="")
    discord_channel_id: str = Field(default="")
    discord_user_id: str = Field(default="")

    # ------------------------------------------------------------------
    # Optional / notifications
    # ------------------------------------------------------------------
    discord_webhook_url: str = Field(default="")
    moondev_api_key: str = Field(default="")

    # ------------------------------------------------------------------
    # Run-plane LLM providers (MODELS.md §2, §7) — pay-per-token keys only
    # ------------------------------------------------------------------
    zai_api_key: str = Field(default="")
    moonshot_api_key: str = Field(default="")
    ollama_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="https://ollama.com/v1")  # local: http://localhost:11434/v1

    # ------------------------------------------------------------------
    # Trading defaults (all overridable in UI)
    # ------------------------------------------------------------------
    paper_trading: bool = Field(default=True)
    # Screener only selects products with max_leverage >= 5 (screener.py
    # MIN_LEVERAGE), so 5x is always available on anything we'd trade. At 3x,
    # the 20%-of-balance margin cap binds on nearly every entry (ATR stops
    # here are routinely <1% away), crushing notional down to a size where
    # the $0.15/fill fee minimum dominates the trade's actual $-at-risk —
    # replaying 9 live paper trades, fee/risk ratios ran 13%-179%. 5x roughly
    # doubles achievable notional at the same margin, without increasing
    # $-at-risk (the stop still defines that) or approaching liquidation —
    # these stops are 10-100x closer than a 5x liquidation buffer (~20%).
    default_leverage: int = Field(default=5)
    default_risk_per_trade: float = Field(default=0.01)       # 1%
    default_daily_loss_limit: float = Field(default=0.05)     # 5%
    default_strategy: str = Field(default="rsi_macd")
    default_signal_interval: int = Field(default=300)         # 5 min
    default_max_watchlist: int = Field(default=5)
    burt_active_hours_start: int = Field(default=6)
    burt_active_hours_end: int = Field(default=22)

    # Config keys that can be hot-reloaded from DB
    leverage: int = Field(default=5)
    risk_per_trade: float = Field(default=0.01)
    daily_loss_limit: float = Field(default=0.05)
    strategy: str = Field(default="rsi_macd")
    signal_interval: int = Field(default=300)
    max_watchlist: int = Field(default=5)
    # How often the screener re-ranks the whole perp universe. Costs ~4s of
    # Coinbase calls and ZERO LLM tokens, so it is cheap to run often — unlike
    # signal_interval, where every tick spends one LLM call per symbol.
    screener_interval_h: float = Field(default=1.0)
    min_confidence: float = Field(default=0.65)
    # 2026-09-18: 1.5 -> 3.0. Sweep of 768 backtests (BTC/ETH/NEAR x
    # donchian/rsi_macd, real 0.14% fees): expectancy improves MONOTONICALLY
    # with stop width for both strategies (rsi_macd avgR -0.85 at 1.5x ->
    # -0.35 at 3.0x; donchian -0.72 -> -0.45). A 1.5x 15m-ATR stop is
    # routinely <1% away — inside normal noise, so ~80% of entries stopped
    # out before the thesis could play. Wider stops also make the fixed
    # 0.28% round-trip fee a smaller share of $-at-risk.
    atr_multiplier: float = Field(default=3.0)
    # 3.0 -> 5.0. The fixed target is mostly a cap — the trailing stop
    # decides the real exit on a genuine trend — and the sweep found TP
    # nearly irrelevant to expectancy (slightly better the higher it is).
    # 5R leaves room for the rare trade that runs straight through.
    take_profit_rr: float = Field(default=5.0)
    # Floor on stop distance as a fraction of entry. With the 0.28% round-trip
    # fee, fee/risk = 0.28% / stop%, so a stop under ~1.4% can't clear the
    # 20% entry fee budget no matter how the position is sized (ETH's 3x ATR
    # stop was 1.21% on 2026-09-18 and got rejected). 1.5% -> fee is 19% of
    # risk; it also means "deeper stops" holds even when ATR is compressed.
    min_stop_pct: float = Field(default=0.015)
    fixed_stop_pct: float = Field(default=0.02)
    stop_loss_method: str = Field(default="atr")
    # Paper-mode account size. RiskManager sizes every position and sets the
    # circuit-breaker threshold off this, so it must reflect the budget you
    # are actually simulating — the old hardcoded 100k made a "1% risk" trade
    # $1,000 and put the daily loss limit at $5,000. LIVE mode ignores this
    # entirely: equity comes from /cfm/balance_summary (agent/account.py).
    paper_balance: float = Field(default=200.0)

    # ------------------------------------------------------------------
    # Live account data (agent/account.py) — LIVE mode only
    # ------------------------------------------------------------------
    # Which balance_summary field is "the account" for sizing and the daily
    # stop, falling back through the others if the chosen one is 0.
    #
    # total_usd_balance is the default because it is the account: measured
    # 2026-09-22, cfm_usd_balance was $68.13 while cbi_usd_balance held
    # another $131.79 of USD that the exchange sweeps into futures on demand
    # (futures_buying_power $195.41 confirms it is usable). Sizing off the
    # CFM slice alone would have valued a $200 account at $68 — at 2%
    # risk/trade that is $1.36, less than one ETH contract's $-at-risk, so
    # the bot would have gone quiet; it would also have put the 5% daily
    # stop at $3.41, below a single round-trip fee-plus-stop loss, tripping
    # the breaker on the first losing trade of any day.
    live_equity_source: str = Field(default="total_usd_balance")
    # Cache TTL for the balance summary. The signal loop, the 30s position
    # monitor, the dashboard and Burt all read equity; without a shared cache
    # that is four /cfm/balance_summary calls where one will do.
    balance_refresh_sec: float = Field(default=30.0)
    # How often an equity snapshot is written to account_snapshots (audit
    # trail + live equity curve). 0 disables.
    account_snapshot_interval_sec: float = Field(default=300.0)
    # Refuse NEW live entries when the last good balance is older than this.
    # Sizing a position against a balance the exchange stopped confirming is
    # how a 1% risk becomes an unbounded one. 0 disables the check.
    max_balance_staleness_sec: float = Field(default=300.0)
    # Live circuit breaker: trip on the EXCHANGE's own realized P&L for the
    # session (balance_summary.daily_realized_pnl) as well as our modeled
    # P&L, whichever is worse. The exchange sees fills we never do — a stop
    # that filled while the bot was down, a manual close from the app.
    use_exchange_realized_pnl: bool = Field(default=True)
    # Pull fills from /orders/historical/fills after every live order and
    # reconcile them into the trades + fills tables (audit against Coinbase).
    sync_fills: bool = Field(default=True)
    # Attempts (1s apart) to find an order's fills before giving up — the
    # fills endpoint lags the order response by a beat.
    fill_sync_attempts: int = Field(default=4)
    signal_model: str = Field(default="moonshotai/kimi-k2.6")

    # ------------------------------------------------------------------
    # Trading fees — CFM perpetual-style futures are taker-only here (every
    # entry/exit is a market order). MEASURED 2026-09-18 via
    # /orders/preview on this account: commission_total is 0.14% of notional
    # on every product tried (1 ETH contract $262 -> $0.367; 1 BTC contract
    # $811 -> $1.136; 1 NEAR contract $1,866 -> $2.614), with no per-fill
    # minimum in play. The previous 0.02% + $0.15-minimum model under-
    # charged every paper trade by ~2.5x at $200 notional. Two fills
    # (entry+exit) per trade = 0.28% of notional round trip.
    # ------------------------------------------------------------------
    taker_fee_pct: float = Field(default=0.0014)     # 0.14% per fill (measured)
    min_fee_usdc: float = Field(default=0.0)          # no minimum observed
    # ------------------------------------------------------------------
    # Whole-contract sizing. CFM products trade in whole contracts and the
    # exchange charges margin at its own (overnight) rate — 24.5% on ETH,
    # 38% on NEAR, 44% on DOT — regardless of the `leverage` knob. Sizing
    # therefore works in contracts: how many whole contracts keep $-at-risk
    # near risk_per_trade, subject to these two hard ceilings.
    # ------------------------------------------------------------------
    # A single contract is usually MORE than risk_per_trade wants at this
    # account size. Allow rounding UP to 1 contract as long as the actual
    # $-at-risk stays under this fraction of balance; otherwise skip.
    # On $200, 1 ETH contract at a 2.5% stop risks $6.55 (3.3%) — 4% is the
    # floor that leaves ETH tradeable at all; NEAR/SOL/XRP/BTC are out
    # regardless ($12-$28 per contract at that stop).
    max_risk_per_trade: float = Field(default=0.04)
    # Exchange margin for one position may not exceed this fraction of
    # balance. 1 ETH contract needs $64 (long) / $88 (short) on a $200
    # account, so the old 20% cap made every product untradeable.
    max_margin_pct: float = Field(default=0.50)
    # Mechanical 4h-bias gate (long only above 4h EMA50, short only below).
    require_4h_bias: bool = Field(default=True)
    # Drought guard: if nothing has traded in this many hours, also try every
    # other regime-compatible strategy each tick. 0 disables it. Off by
    # default since 2026-09-18 — forced trades cost expectancy, and every
    # strategy it would fall back to backtests worse than the primary.
    drought_guard_hours: float = Field(default=0.0)
    # Partial take-profit adds a third fee leg. Skip it when that leg's fee
    # would eat more than this fraction of the trade's $-at-risk (risk_usdc) —
    # otherwise the "diversification" of banking early costs more than it's
    # worth at small size. Scales up automatically as risk_usdc grows.
    fee_budget_pct_of_risk: float = Field(default=0.15)
    # Reject an entry outright when its round-trip (entry+exit) fee alone —
    # unavoidable on every trade, unlike the partial's optional 3rd leg —
    # would exceed this fraction of $-at-risk. Catches stops so tight that
    # even a full-margin-cap position can't earn back its own transaction
    # cost (e.g. a 0.14% stop, notional-capped at $120: fees are 179% of
    # risk). Tightened 0.30→0.15 after the 2026-09-01..10 logs showed tiny-
    # risk trades slipping through ($0.17-$0.25 at-risk) whose full-size
    # wins capped at pocket change while losses still paid full fees.
    # 2026-09-18: with the measured 0.28% round-trip fee this ratio is purely
    # a function of stop distance (0.28% / stop%): a 1.5% stop is 19%, 2%
    # is 14%, 2.5% is 11%. 0.20 lets the wider ATR stops through and rejects
    # anything tighter than ~1.4% — where the fee alone needs a +0.2R edge.
    entry_fee_budget_pct_of_risk: float = Field(default=0.20)

    # ------------------------------------------------------------------
    # Run-plane model roles (MODELS.md §6, §7) — hot-reloadable
    # ------------------------------------------------------------------
    critic_model: str = Field(default="")                  # "" → signal_model
    burt_model: str = Field(default="")                    # "" → signal_model
    embedding_model: str = Field(default="openai/text-embedding-3-small")
    consolidation_model: str = Field(default="")           # "" → signal_model
    signal_provider: str = Field(default="openrouter")     # openrouter|zai|moonshot|ollama
    critic_provider: str = Field(default="openrouter")
    burt_provider: str = Field(default="openrouter")
    embedding_provider: str = Field(default="openrouter")
    consolidation_provider: str = Field(default="openrouter")
    # Reasoning effort per role ("" = don't send the field at all).
    # On GLM-5.3-Flash this is the single biggest cost+latency lever: measured
    # 2026-08-27, "low" cut a real signal call from 17-20s/1038 output tokens
    # to 4-5s/148 — ~7x cheaper and ~4x faster, still valid JSON. Reasoning
    # cannot be disabled entirely on that endpoint (HTTP 400).
    signal_reasoning_effort: str = Field(default="low")
    critic_reasoning_effort: str = Field(default="high")    # critic should think
    burt_reasoning_effort: str = Field(default="low")
    embedding_reasoning_effort: str = Field(default="")
    consolidation_reasoning_effort: str = Field(default="")
    signal_timeout: float = Field(default=30.0)            # seconds, per-role
    critic_timeout: float = Field(default=90.0)            # critic is relaxed (M11)
    burt_timeout: float = Field(default=45.0)
    embedding_timeout: float = Field(default=15.0)
    consolidation_timeout: float = Field(default=90.0)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def check_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            logger.warning("DATABASE_URL does not look like a Postgres connection string")
        return v

    @field_validator("default_leverage")
    @classmethod
    def check_leverage(cls, v: int) -> int:
        return max(1, min(50, v))

    @field_validator("default_risk_per_trade", "default_daily_loss_limit")
    @classmethod
    def check_pct(cls, v: float) -> float:
        return max(0.001, min(0.20, float(v)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """'PAPER' or 'LIVE' — see agent/trading_mode.py for the guards that
        keep this in step with the DB flag and each position's is_paper."""
        return "PAPER" if self.paper_trading else "LIVE"

    def is_required_key_present(self, key: str) -> bool:
        """Check if a specific env var is present."""
        val = getattr(self, key, "")
        return bool(val and val.strip())

    def missing_keys(self) -> list[str]:
        """Return list of required keys that are missing."""
        return [k for k in self._required_keys if not self.is_required_key_present(k)]

    @property
    def _required_keys(self) -> list[str]:
        return [
            "openrouter_api_key",
            "coinbase_api_key",
            "coinbase_api_secret",
            "database_url",
        ]


# ------------------------------------------------------------------
# Singleton + hot-reload helpers
# ------------------------------------------------------------------
_config_instance: Config | None = None


def get_config() -> Config:
    """Return the cached config singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = _build_config()
    return _config_instance


def reload_config() -> Config:
    """Force reload from env (useful after UI changes)."""
    global _config_instance
    _config_instance = _build_config()
    logger.info("Config reloaded from environment")
    return _config_instance


def set_config_key(key: str, value: Any) -> None:
    """Update a single config key in-memory (called from FastAPI / DB sync).

    Coerces the incoming value to the field's annotated type. Without this, ints
    and floats stored as TEXT in agent_config get re-injected as strings and
    silently break arithmetic (e.g. `cfg.signal_interval` becomes "300", and
    `interval // 300` raises TypeError).
    """
    cfg = get_config()
    if key not in cfg.model_fields:
        logger.warning(f"Attempted to set unknown config key: {key}")
        return
    annotation = cfg.model_fields[key].annotation
    try:
        if annotation is bool:
            coerced: Any = str(value).strip().lower() in ("true", "1", "yes", "on")
        elif annotation is int:
            coerced = int(float(value))
        elif annotation is float:
            coerced = float(value)
        else:
            coerced = value
    except (ValueError, TypeError) as exc:
        logger.warning(f"Could not coerce config '{key}'={value!r} to {annotation}: {exc}")
        return
    setattr(cfg, key, coerced)
    logger.info(f"Config updated: {key} = {coerced!r}")


def _build_config() -> Config:
    """Construct Config from current environment."""
    def _env(key: str, default="") -> str:
        return os.getenv(key, os.getenv(key.upper(), default))

    def _bool(key: str, default: bool = False) -> bool:
        return _env(key, str(default)).lower() in ("true", "1", "yes", "on")

    def _int(key: str, default: int = 0) -> int:
        return int(_env(key, str(default)))

    def _float(key: str, default: float = 0.0) -> float:
        return float(_env(key, str(default)))

    return Config(
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        coinbase_api_key=_env("COINBASE_API_KEY"),
        coinbase_api_secret=_env("COINBASE_API_SECRET"),
        database_url=_env("DATABASE_URL"),
        discord_bot_token=_env("DISCORD_BOT_TOKEN"),
        discord_channel_id=_env("DISCORD_CHANNEL_ID"),
        discord_user_id=_env("DISCORD_USER_ID"),
        discord_webhook_url=_env("DISCORD_WEBHOOK_URL"),
        moondev_api_key=_env("MOONDEV_API_KEY"),
        zai_api_key=_env("ZAI_API_KEY"),
        moonshot_api_key=_env("MOONSHOT_API_KEY"),
        ollama_api_key=_env("OLLAMA_API_KEY"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "https://ollama.com/v1"),
        paper_trading=_bool("PAPER_TRADING", True),
        default_leverage=_int("DEFAULT_LEVERAGE", 5),
        default_risk_per_trade=_float("DEFAULT_RISK_PER_TRADE", 0.01),
        default_daily_loss_limit=_float("DEFAULT_DAILY_LOSS_LIMIT", 0.05),
        default_strategy=_env("DEFAULT_STRATEGY", "rsi_macd"),
        default_signal_interval=_int("DEFAULT_SIGNAL_INTERVAL", 300),
        default_max_watchlist=_int("DEFAULT_MAX_WATCHLIST", 5),
        screener_interval_h=_float("SCREENER_INTERVAL_H", 1.0),
        burt_active_hours_start=_int("BURT_ACTIVE_HOURS_START", 6),
        burt_active_hours_end=_int("BURT_ACTIVE_HOURS_END", 22),
        # Run-plane roles (MODELS.md §6, §7). These MUST be listed here — Config
        # is built from explicit kwargs, so a field that is not passed silently
        # keeps its class default and every *_MODEL / *_PROVIDER in .env is
        # ignored no matter what .env.example documents.
        paper_balance=_float("PAPER_BALANCE", 200.0),
        live_equity_source=_env("LIVE_EQUITY_SOURCE", "total_usd_balance"),
        balance_refresh_sec=_float("BALANCE_REFRESH_SEC", 30.0),
        account_snapshot_interval_sec=_float("ACCOUNT_SNAPSHOT_INTERVAL_SEC", 300.0),
        max_balance_staleness_sec=_float("MAX_BALANCE_STALENESS_SEC", 300.0),
        use_exchange_realized_pnl=_bool("USE_EXCHANGE_REALIZED_PNL", True),
        sync_fills=_bool("SYNC_FILLS", True),
        fill_sync_attempts=_int("FILL_SYNC_ATTEMPTS", 4),
        taker_fee_pct=_float("TAKER_FEE_PCT", 0.0014),
        min_fee_usdc=_float("MIN_FEE_USDC", 0.0),
        max_risk_per_trade=_float("MAX_RISK_PER_TRADE", 0.04),
        max_margin_pct=_float("MAX_MARGIN_PCT", 0.50),
        require_4h_bias=_bool("REQUIRE_4H_BIAS", True),
        drought_guard_hours=_float("DROUGHT_GUARD_HOURS", 0.0),
        min_stop_pct=_float("MIN_STOP_PCT", 0.015),
        fee_budget_pct_of_risk=_float("FEE_BUDGET_PCT_OF_RISK", 0.15),
        entry_fee_budget_pct_of_risk=_float("ENTRY_FEE_BUDGET_PCT_OF_RISK", 0.20),
        signal_model=_env("SIGNAL_MODEL", "moonshotai/kimi-k2.6"),
        critic_model=_env("CRITIC_MODEL"),
        burt_model=_env("BURT_MODEL"),
        embedding_model=_env("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        consolidation_model=_env("CONSOLIDATION_MODEL"),
        signal_provider=_env("SIGNAL_PROVIDER", "openrouter"),
        critic_provider=_env("CRITIC_PROVIDER", "openrouter"),
        burt_provider=_env("BURT_PROVIDER", "openrouter"),
        embedding_provider=_env("EMBEDDING_PROVIDER", "openrouter"),
        consolidation_provider=_env("CONSOLIDATION_PROVIDER", "openrouter"),
        signal_reasoning_effort=_env("SIGNAL_REASONING_EFFORT", "low"),
        critic_reasoning_effort=_env("CRITIC_REASONING_EFFORT", "high"),
        burt_reasoning_effort=_env("BURT_REASONING_EFFORT", "low"),
        embedding_reasoning_effort=_env("EMBEDDING_REASONING_EFFORT"),
        consolidation_reasoning_effort=_env("CONSOLIDATION_REASONING_EFFORT"),
        signal_timeout=_float("SIGNAL_TIMEOUT", 30.0),
        critic_timeout=_float("CRITIC_TIMEOUT", 90.0),
        burt_timeout=_float("BURT_TIMEOUT", 45.0),
        embedding_timeout=_float("EMBEDDING_TIMEOUT", 15.0),
        consolidation_timeout=_float("CONSOLIDATION_TIMEOUT", 90.0),
    )
