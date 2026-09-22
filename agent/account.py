"""
Account state — real equity from Coinbase, one normalized shape for both modes.

Everything that scales with account size (position sizing, the portfolio
risk cap, the drawdown scale, and above all the daily-loss circuit breaker)
is a percentage of "the balance". In live mode that number has to come from
the exchange, not from the paper ledger in agent_config — a $200 CFM account
sized against a stale paper balance is not a rounding error, it is the
difference between a 5% daily stop and an unbounded one.

This module is the single reader of /cfm/balance_summary. It:

  - normalizes the API's {value, currency} envelopes into plain floats
  - caches for `balance_refresh_sec` so the 30s position monitor, the signal
    loop, the API and Burt share ONE request instead of four
  - marks the snapshot `stale` when a fetch fails, so callers can refuse to
    size new risk against an unknown balance rather than silently reusing
    the last good number forever
  - persists periodic snapshots to `account_snapshots` — the live equity
    curve and the audit trail against Coinbase's own statements
  - serves paper mode from the persistent paper ledger through the same
    interface, so no caller needs its own `if paper_trading:` branch

`daily_realized_pnl` is the exchange's own realized P&L for the session. It
is what the live circuit breaker trips on: the exchange knows about fills
the bot never saw (a stop that filled while we were down, a manual close),
and our modeled P&L does not.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import config
from agent import trading_mode
from agent.database import get_db

# Preference order when `live_equity_source` yields nothing usable.
# total_usd_balance is the whole account (CFM cash + the spot USD the
# exchange sweeps in on demand); buying power is what it will actually let us
# margin against; cfm_usd_balance is only the futures slice and reads far
# lower than the account is worth — see config.live_equity_source.
_EQUITY_FALLBACKS = ("total_usd_balance", "futures_buying_power", "cfm_usd_balance")


def _num(raw: Any) -> float:
    """Coinbase returns numerics as {"value": "12.34", "currency": "USD"},
    as bare strings, or as "" for empty. All three land here."""
    if raw is None:
        return 0.0
    if isinstance(raw, dict):
        raw = raw.get("value")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    """One normalized view of the trading account, in either mode."""
    mode: str                          # trading_mode.PAPER | trading_mode.LIVE
    equity_usdc: float                 # what risk sizing and the daily stop scale off
    buying_power_usdc: float = 0.0
    cash_usdc: float = 0.0
    unrealized_pnl_usdc: float = 0.0
    daily_realized_pnl_usdc: float = 0.0
    initial_margin_usdc: float = 0.0
    maintenance_margin_usdc: float = 0.0
    liquidation_buffer_usdc: float = 0.0
    fetched_at: float = field(default_factory=time.time)
    source: str = ""                   # which field equity came from
    stale: bool = False                # last refresh failed — equity is a carry-over
    raw: dict = field(default_factory=dict)

    @property
    def age_sec(self) -> float:
        return max(0.0, time.time() - self.fetched_at)

    @property
    def is_live(self) -> bool:
        return self.mode == trading_mode.LIVE

    def summary_line(self) -> str:
        if not self.is_live:
            return f"${self.equity_usdc:,.2f} (PAPER ledger)"
        return (
            f"${self.equity_usdc:,.2f} equity [{self.source}] | "
            f"buying power ${self.buying_power_usdc:,.2f} | "
            f"uP&L ${self.unrealized_pnl_usdc:+,.2f} | "
            f"realized today ${self.daily_realized_pnl_usdc:+,.2f}"
            + (f" | STALE {self.age_sec / 60:.0f}m" if self.stale else "")
        )


class AccountService:
    """Cached, mode-aware access to account equity. One instance per agent."""

    def __init__(self, cb: Any):
        self.cfg = config.get_config()
        self.cb = cb
        self._snapshot: AccountSnapshot | None = None
        self._last_persist: float = 0.0
        self._last_stale_warning: float = 0.0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def last(self) -> AccountSnapshot | None:
        """The most recent snapshot without touching the network — for sync
        callers (API handlers, Burt's prompt) that must not block."""
        return self._snapshot

    async def get(self, force: bool = False) -> AccountSnapshot:
        """Current account state, refreshed at most once per `balance_refresh_sec`."""
        ttl = float(getattr(self.cfg, "balance_refresh_sec", 30.0) or 0.0)
        snap = self._snapshot
        if not force and snap is not None and not snap.stale and snap.age_sec < ttl:
            return snap
        if self.cfg.paper_trading:
            snap = await self._paper_snapshot()
        else:
            snap = await self._live_snapshot()
        self._snapshot = snap
        await self._maybe_persist(snap)
        return snap

    async def _paper_snapshot(self) -> AccountSnapshot:
        """Paper equity is the persistent ledger in agent_config, which
        `Executor._apply_paper_pnl` banks realized P&L into."""
        balance = float(self.cfg.paper_balance or 0.0)
        try:
            db = await get_db()
            stored = await db.get_config_value("paper_balance")
            if stored is not None:
                balance = float(stored)
        except Exception as exc:
            logger.warning(f"Paper balance read failed — using config ${balance:,.2f}: {exc}")
        return AccountSnapshot(
            mode=trading_mode.PAPER,
            equity_usdc=balance,
            buying_power_usdc=balance,
            cash_usdc=balance,
            source="agent_config.paper_balance",
        )

    async def _live_snapshot(self) -> AccountSnapshot:
        try:
            raw = await self.cb.get_futures_balance_summary()
        except Exception as exc:
            return self._carry_forward(f"balance fetch failed: {exc}")
        # The API nests everything under balance_summary; reading the top
        # level reported $0.00 on a funded account.
        bs = (raw or {}).get("balance_summary", raw) or {}
        if not bs:
            return self._carry_forward("balance_summary came back empty")

        preferred = str(getattr(self.cfg, "live_equity_source", "") or "").strip()
        order = [preferred] + [k for k in _EQUITY_FALLBACKS if k != preferred]
        equity, source = 0.0, ""
        for key in order:
            if not key:
                continue
            val = _num(bs.get(key))
            if val > 0:
                equity, source = val, key
                break
        if equity <= 0:
            return self._carry_forward(
                f"no positive balance field in {list(bs.keys())} — account unfunded?"
            )

        snap = AccountSnapshot(
            mode=trading_mode.LIVE,
            equity_usdc=equity,
            buying_power_usdc=_num(bs.get("futures_buying_power")),
            cash_usdc=_num(bs.get("cfm_usd_balance")),
            unrealized_pnl_usdc=_num(bs.get("unrealized_pnl")),
            daily_realized_pnl_usdc=_num(bs.get("daily_realized_pnl")),
            initial_margin_usdc=_num(bs.get("initial_margin")),
            maintenance_margin_usdc=_num(bs.get("liquidation_threshold")),
            liquidation_buffer_usdc=_num(bs.get("liquidation_buffer_amount")),
            source=source,
            raw=bs,
        )
        prev = self._snapshot
        if prev is None or prev.stale or abs(prev.equity_usdc - equity) > 0.01:
            logger.info(f"Live account: {snap.summary_line()}")
        return snap

    def _carry_forward(self, why: str) -> AccountSnapshot:
        """
        A failed refresh keeps the last known equity but flags it stale.

        Zeroing would make every percentage-of-balance limit collapse to $0
        (and the circuit breaker trip on the next cent lost); silently
        reusing the old number would let the bot size new risk against a
        balance it can no longer see. Stale does both: callers keep reading,
        `RiskManager` refuses NEW entries once it ages past
        `max_balance_staleness_sec`.
        """
        prev = self._snapshot
        now = time.time()
        if now - self._last_stale_warning > 60:
            self._last_stale_warning = now
            logger.warning(
                f"Live balance unavailable ({why}) — carrying "
                f"${(prev.equity_usdc if prev else 0.0):,.2f} forward as STALE"
            )
        if prev is None:
            return AccountSnapshot(
                mode=trading_mode.LIVE, equity_usdc=0.0, source="unavailable",
                stale=True, fetched_at=now,
            )
        # Keep the original fetched_at: `age_sec` must measure how old the
        # DATA is, not how recently we failed to replace it.
        return AccountSnapshot(
            mode=prev.mode, equity_usdc=prev.equity_usdc,
            buying_power_usdc=prev.buying_power_usdc, cash_usdc=prev.cash_usdc,
            unrealized_pnl_usdc=prev.unrealized_pnl_usdc,
            daily_realized_pnl_usdc=prev.daily_realized_pnl_usdc,
            initial_margin_usdc=prev.initial_margin_usdc,
            maintenance_margin_usdc=prev.maintenance_margin_usdc,
            liquidation_buffer_usdc=prev.liquidation_buffer_usdc,
            fetched_at=prev.fetched_at, source=prev.source, stale=True, raw=prev.raw,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _maybe_persist(self, snap: AccountSnapshot) -> None:
        """Write an equity snapshot every `account_snapshot_interval_sec`.

        Throttled deliberately: this is an audit/equity-curve trail, not a
        tick log — the 30s monitor would otherwise write 2,880 rows a day."""
        interval = float(getattr(self.cfg, "account_snapshot_interval_sec", 300.0) or 0.0)
        if interval <= 0 or snap.stale or snap.equity_usdc <= 0:
            return
        now = time.time()
        if now - self._last_persist < interval:
            return
        self._last_persist = now
        try:
            db = await get_db()
            await db.log_account_snapshot({
                "mode": snap.mode,
                "equity_usdc": snap.equity_usdc,
                "buying_power_usdc": snap.buying_power_usdc,
                "cash_usdc": snap.cash_usdc,
                "unrealized_pnl_usdc": snap.unrealized_pnl_usdc,
                "daily_realized_pnl_usdc": snap.daily_realized_pnl_usdc,
                "initial_margin_usdc": snap.initial_margin_usdc,
                "maintenance_margin_usdc": snap.maintenance_margin_usdc,
                "source": snap.source,
                "raw": json.dumps(snap.raw) if snap.raw else None,
            })
        except Exception as exc:
            logger.warning(f"Account snapshot not persisted: {exc}")
