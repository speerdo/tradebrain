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
    default_leverage: int = Field(default=3)
    default_risk_per_trade: float = Field(default=0.01)       # 1%
    default_daily_loss_limit: float = Field(default=0.05)     # 5%
    default_strategy: str = Field(default="rsi_macd")
    default_signal_interval: int = Field(default=300)         # 5 min
    default_max_watchlist: int = Field(default=5)
    burt_active_hours_start: int = Field(default=6)
    burt_active_hours_end: int = Field(default=22)

    # Config keys that can be hot-reloaded from DB
    leverage: int = Field(default=3)
    risk_per_trade: float = Field(default=0.01)
    daily_loss_limit: float = Field(default=0.05)
    strategy: str = Field(default="rsi_macd")
    signal_interval: int = Field(default=300)
    max_watchlist: int = Field(default=5)
    min_confidence: float = Field(default=0.65)
    atr_multiplier: float = Field(default=1.5)
    take_profit_rr: float = Field(default=2.0)
    fixed_stop_pct: float = Field(default=0.02)
    stop_loss_method: str = Field(default="atr")
    signal_model: str = Field(default="moonshotai/kimi-k2.6")

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
        default_leverage=_int("DEFAULT_LEVERAGE", 3),
        default_risk_per_trade=_float("DEFAULT_RISK_PER_TRADE", 0.01),
        default_daily_loss_limit=_float("DEFAULT_DAILY_LOSS_LIMIT", 0.05),
        default_strategy=_env("DEFAULT_STRATEGY", "rsi_macd"),
        default_signal_interval=_int("DEFAULT_SIGNAL_INTERVAL", 300),
        default_max_watchlist=_int("DEFAULT_MAX_WATCHLIST", 5),
        burt_active_hours_start=_int("BURT_ACTIVE_HOURS_START", 6),
        burt_active_hours_end=_int("BURT_ACTIVE_HOURS_END", 22),
    )
