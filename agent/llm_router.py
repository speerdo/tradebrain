"""
LLM Router — per-role provider routing for the run plane (MODELS.md §2, §7).

Roles (config keys):
    signal        — Tier 1 evaluation, highest volume (role: signal_model/signal_provider)
    critic        — risk critic (M11, few calls/day)
    burt          — Burt chat + tool loop (Discord latency matters)
    embedding     — semantic memory embeddings
    consolidation — nightly memory consolidation

Providers (all OpenAI-compatible — base-URL + key swap, not a client rewrite):
    openrouter | zai | moonshot | ollama
"""

import os
from dataclasses import dataclass

from loguru import logger

import config

PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "zai": "https://api.z.ai/api/paas/v4",
    "moonshot": "https://api.moonshot.ai/v1",
    "ollama": None,  # resolved from cfg.ollama_base_url (cloud or local)
}

PROVIDER_KEY_FIELDS = {
    "openrouter": "openrouter_api_key",
    "zai": "zai_api_key",
    "moonshot": "moonshot_api_key",
    "ollama": "ollama_api_key",
}

# Seat-subscription key prefixes that must never serve the run plane (MODELS.md §2).
SEAT_KEY_PREFIXES = ("sk-ant-",)

# Roles whose call volume makes a seat subscription impossible (§2 math).
HIGH_VOLUME_ROLES = ("signal",)


@dataclass(frozen=True)
class Route:
    role: str
    provider: str
    base_url: str
    api_key: str
    headers: dict
    model: str
    timeout: float


def resolve_model(cfg, role: str) -> str:
    """Model for a role; empty role config falls back to signal_model."""
    model = getattr(cfg, f"{role}_model", "") or ""
    return model or cfg.signal_model


def resolve_provider(cfg, role: str) -> str:
    return (getattr(cfg, f"{role}_provider", "") or "openrouter").lower()


def resolve_timeout(cfg, role: str) -> float:
    return float(getattr(cfg, f"{role}_timeout", 0) or 30.0)


def route(cfg, role: str) -> Route:
    """Resolve base URL, key, headers, model and timeout for a run-plane role."""
    provider = resolve_provider(cfg, role)
    if provider not in PROVIDER_BASE_URLS:
        raise ValueError(
            f"Unknown provider '{provider}' for role '{role}' "
            f"(valid: {sorted(PROVIDER_BASE_URLS)})"
        )
    base_url = PROVIDER_BASE_URLS[provider] or cfg.ollama_base_url
    if not base_url:
        raise ValueError(f"Role '{role}' routes to ollama but OLLAMA_BASE_URL is not set")
    api_key = getattr(cfg, PROVIDER_KEY_FIELDS[provider], "") or ""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/tradebrain"
        headers["X-Title"] = "TradeBrain"

    return Route(
        role=role,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        headers=headers,
        model=resolve_model(cfg, role),
        timeout=resolve_timeout(cfg, role),
    )


def validate_run_plane_config(cfg) -> list[str]:
    """
    Enforce MODELS.md §2 in code: the run plane pays per token.

    Fails loudly (returns errors) when a run-plane credential looks like a
    seat-subscription key, and when the highest-volume role routes to a seat
    metering that cannot sustain the loop. Returns a list of fatal errors;
    an empty list means the routing table is sane.
    """
    errors: list[str] = []
    for role in ("signal", "critic", "burt", "embedding", "consolidation"):
        try:
            r = route(cfg, role)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        for prefix in SEAT_KEY_PREFIXES:
            if r.api_key.startswith(prefix):
                errors.append(
                    f"Role '{role}' uses a seat-subscription key ({prefix}…). "
                    "The run plane must use pay-per-token keys (MODELS.md §2)."
                )

        allow_cloud = os.getenv("RUN_PLANE_ALLOW_OLLAMA_CLOUD", "").strip().lower() \
            in ("1", "true", "yes", "on")
        if role in HIGH_VOLUME_ROLES and r.provider == "ollama" and r.api_key \
                and "ollama.com" in r.base_url and not allow_cloud:
            errors.append(
                f"Role '{role}' routes to Ollama Cloud with an account key — a seat "
                "subscription cannot sustain the loop volume (MODELS.md §2). Use a "
                "pay-per-token provider, local Ollama (no key), or set "
                "RUN_PLANE_ALLOW_OLLAMA_CLOUD=1 to acknowledge."
            )

        # The local daemon proxies `:cloud` tags to ollama.com using the
        # signed-in account, so a localhost base_url with no API key still
        # spends the Ollama subscription — the error above cannot see that.
        # Warn rather than block: at post-G1 volume (~50 calls/day) this is
        # comfortably sustainable, and it is the right setup for a test run.
        if role in HIGH_VOLUME_ROLES and r.provider == "ollama" \
                and r.model.endswith(":cloud"):
            logger.warning(
                f"Role '{role}' uses the Ollama cloud model '{r.model}'. Even via "
                "localhost this bills the Ollama subscription, not a metered key. "
                "Fine for a test run and for post-G1 volume; revisit before "
                "running 300s x N symbols continuously (MODELS.md §2)."
            )
    return errors


def log_routing_table(cfg) -> None:
    for role in ("signal", "critic", "burt", "embedding", "consolidation"):
        try:
            r = route(cfg, role)
            logger.info(
                f"LLM route [{role}]: {r.provider}/{r.model} "
                f"via {r.base_url} (timeout {r.timeout:.0f}s)"
            )
        except ValueError as exc:
            logger.warning(f"LLM route [{role}]: {exc}")


def import_config():
    """Re-export for callers that want the live config singleton."""
    return config.get_config()
