"""
Trading mode — the single source of truth for PAPER vs LIVE.

Three things used to answer "are we live?" independently:

  1. PAPER_TRADING in .env (read once at boot into config)
  2. agent_config.paper_trading in the DB (hot-reloaded into that SAME config
     object on every loop tick, so it can flip mid-run)
  3. the per-position `is_paper` flag stamped on each trade at entry

Nothing cross-checked them. The failure that matters: the DB says
paper_trading=true while .env says false (or a UI/Burt edit flips it at
runtime), so `Executor.enter_position` routes to the paper branch and stamps
is_paper=True — while every log line, the dashboard, and the circuit breaker
are reporting a LIVE account. The trade is fiction, the account is real, and
nothing says so.

This module makes that combination fatal rather than silent:

  - `lock_boot_mode()` freezes the mode the process started in.
  - `check_db_agreement()` compares the DB's stored flag with it (startup
    refuses to run on a mismatch; each tick re-checks for a runtime flip).
  - `guard_trade()` / `assert_trade_allowed()` refuse to act on a position
    whose `is_paper` does not match the running mode.

Flattening is deliberately exempt: closing a position must work in any mode,
including one stamped by the other mode, or a mismatch would strand real
money in an unmanageable position.
"""

from typing import Any

from loguru import logger

import config

PAPER = "PAPER"
LIVE = "LIVE"


class ModeMismatch(RuntimeError):
    """Raised when an action's mode does not match the running trading mode."""


def mode_of(is_paper: bool) -> str:
    """The mode a boolean `is_paper` flag belongs to."""
    return PAPER if is_paper else LIVE


def current_mode(cfg: Any = None) -> str:
    """The mode the agent is running in RIGHT NOW (respects hot-reload)."""
    cfg = cfg or config.get_config()
    return mode_of(bool(cfg.paper_trading))


def is_live(cfg: Any = None) -> bool:
    return current_mode(cfg) == LIVE


# The mode the process booted in. Positions, balances, and the circuit
# breaker are all scoped to it; a runtime flip invalidates all three, so it
# is recorded once and compared against, never updated.
_boot_mode: str | None = None


def lock_boot_mode(cfg: Any = None) -> str:
    """Freeze the boot mode. Called once, from startup."""
    global _boot_mode
    _boot_mode = current_mode(cfg)
    logger.info(f"Trading mode locked for this process: {_boot_mode}")
    return _boot_mode


def boot_mode() -> str | None:
    return _boot_mode


def parse_flag(value: Any) -> bool | None:
    """Parse a stored agent_config value into a bool ('' / None → None)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return None


def check_db_agreement(db_value: Any, cfg: Any = None) -> str:
    """
    Compare the DB's stored paper_trading flag with the running mode.

    Returns "" when they agree (or the DB has no opinion), otherwise a
    human-readable description of the disagreement. The caller decides
    whether that is fatal (startup) or an alert (runtime).
    """
    cfg = cfg or config.get_config()
    db_paper = parse_flag(db_value)
    if db_paper is None:
        return ""
    if db_paper is bool(cfg.paper_trading):
        return ""
    return (
        f"agent_config.paper_trading={str(db_paper).lower()} "
        f"({mode_of(db_paper)}) disagrees with the running config "
        f"paper_trading={str(bool(cfg.paper_trading)).lower()} "
        f"({current_mode(cfg)})"
    )


def check_boot_drift(cfg: Any = None) -> str:
    """
    Returns "" unless the mode has changed since the process booted.

    A mid-run flip is never safe to act on: in-memory positions, the paper
    balance ledger, and the circuit breaker's daily loss all belong to the
    boot mode.
    """
    if _boot_mode is None:
        return ""
    now = current_mode(cfg)
    if now == _boot_mode:
        return ""
    return (
        f"trading mode changed at runtime: booted {_boot_mode}, config now says "
        f"{now} — restart the agent to switch modes"
    )


def guard_trade(is_paper: bool, action: str, cfg: Any = None) -> str:
    """
    Returns "" when `action` may proceed, or the refusal reason.

    This is the hard check: a paper-stamped position may never be acted on
    while the agent is LIVE, and a live-stamped one may never be acted on
    while it is in PAPER.
    """
    cfg = cfg or config.get_config()
    drift = check_boot_drift(cfg)
    if drift:
        return f"Refusing to {action} — {drift}"
    running = current_mode(cfg)
    stamped = mode_of(is_paper)
    if stamped != running:
        return (
            f"Refusing to {action} — MODE MISMATCH: position is stamped "
            f"is_paper={str(bool(is_paper)).lower()} ({stamped}) but the agent is "
            f"running in {running}"
        )
    return ""


def assert_trade_allowed(is_paper: bool, action: str, cfg: Any = None) -> None:
    """`guard_trade`, as an exception. Use where a return value would be ignored."""
    reason = guard_trade(is_paper, action, cfg)
    if reason:
        raise ModeMismatch(reason)
