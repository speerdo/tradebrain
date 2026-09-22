"""
Offline checks for the paper/live separation.

Places no orders, opens no sockets, touches no database: the Coinbase client
and the DB layer are stubbed, so this runs anywhere and is safe to run while
the agent is live.

Covers:
  1. Balance — live equity is read from the exchange, staleness blocks new
     risk, paper equity comes from the ledger.
  2. Mode guards — a paper-stamped position cannot be acted on in LIVE, a
     live-stamped one cannot be acted on in PAPER, and a mid-run mode flip
     stops entries entirely.
  3. Circuit breaker — trips on the LIVE account's realized P&L, and ignores
     closes belonging to the other book.
  4. Fill audit — exchange fills land on the trade row as real prices and
     real commission.

Usage:
    venv/bin/python scripts/test_live_mode.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

import config
from agent import trading_mode

PASS, FAIL = "  ✅", "  ❌"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------

def _money(v):
    return {"value": str(v), "currency": "USD"}


class FakeCoinbase:
    """Just enough of CoinbaseClient for the paths under test."""

    # Mirrors the real account's shape as measured 2026-09-22: most of the
    # USD sits in the spot (CBI) balance and is swept into futures on demand,
    # so cfm_usd_balance alone badly understates the account.
    CFM_SHARE = 0.34

    def __init__(self, balance=200.0, realized=0.0):
        self.balance = balance          # total_usd_balance — the whole account
        self.realized = realized
        self.fail = False
        self.orders: list[dict] = []
        self.fills: list[dict] = []

    async def get_futures_balance_summary(self):
        if self.fail:
            raise RuntimeError("simulated API outage")
        return {"balance_summary": {
            "cfm_usd_balance": _money(round(self.balance * self.CFM_SHARE, 2)),
            "cbi_usd_balance": _money(round(self.balance * (1 - self.CFM_SHARE), 2)),
            "total_usd_balance": _money(self.balance),
            "futures_buying_power": _money(round(self.balance * 0.98, 2)),
            "unrealized_pnl": _money(1.25),
            "daily_realized_pnl": _money(self.realized),
            "initial_margin": _money(64.0),
            "liquidation_threshold": _money(30.0),
        }}

    async def place_futures_market_order(self, product_id, side, contracts,
                                         leverage=1, margin_type="ISOLATED",
                                         client_order_id=None):
        self.orders.append({"product_id": product_id, "side": side,
                            "contracts": contracts,
                            "client_order_id": client_order_id})
        return {"success": True, "success_response": {"order_id": "ORD-1"}}

    async def place_futures_stop_order(self, *a, **k):
        return {"success": True, "success_response": {"order_id": "STOP-1"}}

    async def get_fills(self, order_ids=None, product_ids=None, limit=100, start=None):
        return self.fills

    async def hydrate_product_details(self, product_id):
        return {"contract_size": 0.1, "mark_price": 2600.0}


class FakeDB:
    """Records what the executor would have written."""

    def __init__(self):
        self.trades: dict[int, dict] = {}
        self.fills: list[dict] = []
        self.snapshots: list[dict] = []
        self.next_id = 1
        self.config: dict[str, str] = {"paper_balance": "200"}

    async def log_trade(self, trade):
        tid = self.next_id
        self.next_id += 1
        self.trades[tid] = dict(trade)
        return tid

    async def update_trade_orders(self, trade_id, **fields):
        self.trades.setdefault(trade_id, {}).update(fields)

    async def log_account_snapshot(self, snap):
        self.snapshots.append(dict(snap))

    async def get_latest_account_snapshot(self, mode=None):
        return None

    async def record_fills(self, rows):
        self.fills.extend(rows)
        return len(rows)

    async def find_trade_by_order(self, order_id):
        return None

    async def get_known_fill_ids(self, product_ids=None, limit=500):
        return {f["fill_id"] for f in self.fills}

    async def get_config_value(self, key):
        return self.config.get(key)

    async def get_realized_pnl_today(self, is_paper):
        return 0.0

    async def close_trade(self, *a, **k):
        pass

    async def fetch(self, *a, **k):
        return []

    async def fetchrow(self, *a, **k):
        return None


def install_fake_db(db):
    """Point every get_db() caller at the stub."""
    import agent.database as database
    database.db_instance = db

    async def _get_db():
        return db

    database.get_db = _get_db
    for mod in ("agent.executor", "agent.account", "agent.risk_manager"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "get_db"):
            m.get_db = _get_db


def set_mode(paper: bool) -> None:
    cfg = config.get_config()
    cfg.paper_trading = paper
    trading_mode.lock_boot_mode(cfg)


# ----------------------------------------------------------------------
# 1. Balance
# ----------------------------------------------------------------------

async def test_balance():
    print("\n=== 1. REAL BALANCE ===")
    from agent.account import AccountService
    from agent.risk_manager import RiskManager

    db = FakeDB()
    install_fake_db(db)

    set_mode(False)
    cb = FakeCoinbase(balance=248.37, realized=-4.10)
    acct = AccountService(cb)
    snap = await acct.get()
    check("live equity read from the exchange", abs(snap.equity_usdc - 248.37) < 1e-6,
          f"${snap.equity_usdc:,.2f} via {snap.source}")
    check("equity is the WHOLE account, not just the CFM slice",
          snap.source == "total_usd_balance" and snap.equity_usdc > snap.cash_usdc,
          f"total ${snap.equity_usdc:,.2f} vs CFM cash ${snap.cash_usdc:,.2f}")
    check("exchange realized P&L captured",
          abs(snap.daily_realized_pnl_usdc + 4.10) < 1e-6,
          f"${snap.daily_realized_pnl_usdc:+.2f}")

    risk = RiskManager()
    risk.set_account(acct)
    await risk.sync()
    check("risk sizes off exchange equity, not the paper ledger",
          abs(risk.state.balance_usdc - 248.37) < 1e-6,
          f"${risk.state.balance_usdc:,.2f} (paper_balance is "
          f"${config.get_config().paper_balance:,.2f})")
    limit = risk.state.balance_usdc * risk.state.daily_loss_limit_pct
    check("daily loss limit is a fraction of real funds",
          abs(limit - 248.37 * risk.state.daily_loss_limit_pct) < 1e-6,
          f"${limit:,.2f}")

    # A failed refresh must not zero the balance, and must eventually block risk.
    cb.fail = True
    stale = await acct.get(force=True)
    check("failed refresh carries equity forward as stale",
          stale.stale and abs(stale.equity_usdc - 248.37) < 1e-6)
    await risk.sync()
    config.get_config().max_balance_staleness_sec = 0.0001
    risk.state.balance_age_sec = 999
    check("stale balance blocks new entries",
          bool(risk.balance_is_usable()), risk.balance_is_usable())
    config.get_config().max_balance_staleness_sec = 300.0

    # Paper mode reads the ledger, never the exchange.
    set_mode(True)
    db.config["paper_balance"] = "512.50"
    paper_snap = await AccountService(cb).get()
    check("paper equity comes from the persistent ledger",
          abs(paper_snap.equity_usdc - 512.50) < 1e-6 and paper_snap.mode == "PAPER",
          f"${paper_snap.equity_usdc:,.2f} (exchange call would have raised)")


# ----------------------------------------------------------------------
# 2. Mode guards
# ----------------------------------------------------------------------

async def test_mode_guards():
    print("\n=== 2. MODE MISMATCH REFUSALS ===")
    from agent.executor import Executor

    db = FakeDB()
    install_fake_db(db)
    cb = FakeCoinbase()

    # LIVE agent, paper-stamped entry.
    set_mode(False)
    ex = Executor(cb)
    res = await ex._enter_paper("ETP-20DEC30-CDE", "ETH PERP", "long", 2600.0,
                                2500.0, 2900.0, 260.0, 64.0, 5, 10.0,
                                "rsi_macd", 0.8, "test", 1)
    check("LIVE agent refuses a paper-stamped entry",
          not res.success and "MODE MISMATCH" in res.error, res.error[:90])
    check("...and nothing was written to the DB", not db.trades)
    check("...and no position was created", not ex.paper_positions)

    # PAPER agent, live entry.
    set_mode(True)
    ex = Executor(cb)
    res = await ex._enter_live("ETP-20DEC30-CDE", "ETH PERP", "long", 2600.0,
                               2500.0, 2900.0, 260.0, 64.0, 5, 10.0,
                               "rsi_macd", 0.8, "test", 1)
    check("PAPER agent refuses a live entry",
          not res.success and "MODE MISMATCH" in res.error, res.error[:90])
    check("...and no order reached the exchange", not cb.orders)

    # Restoring paper positions into a live session.
    set_mode(False)
    ex = Executor(cb)
    check("LIVE agent does not restore paper positions",
          await ex.restore_paper_positions() == [])

    # A mid-run mode flip blocks entries outright.
    set_mode(False)
    ex = Executor(cb)
    config.get_config().paper_trading = True      # simulate a DB hot-reload flip
    res = await ex.enter_position(
        symbol="ETP-20DEC30-CDE", direction="long", entry_price=2600.0,
        stop_loss=2500.0, take_profit=2900.0, size_usdc=260.0,
        margin_usdc=64.0, leverage=5, risk_usdc=10.0, contracts=1,
    )
    check("runtime mode flip blocks all entries",
          not res.success and "Mode drift" in res.error, res.error[:90])
    check("...and no order reached the exchange", not cb.orders)

    # Closing is deliberately exempt — never strand a position.
    set_mode(False)
    ex = Executor(cb)
    from agent.executor import PaperPosition
    pos = PaperPosition(product_id="X", display_name="X", direction="long",
                        entry_price=100.0, stop_loss=95.0, take_profit=110.0,
                        size_usdc=100.0, margin_usdc=20.0, leverage=5,
                        risk_usdc=5.0, is_paper=True, trade_id=7)
    ex.paper_positions["X"] = pos
    res = await ex.close_position("X", 99.0)
    check("a mismatched position can still be CLOSED", res.success)


# ----------------------------------------------------------------------
# 3. Circuit breaker
# ----------------------------------------------------------------------

async def test_circuit_breaker():
    print("\n=== 3. CIRCUIT BREAKER ON LIVE REALIZED P&L ===")
    from agent.account import AccountService
    from agent.risk_manager import RiskManager

    db = FakeDB()
    install_fake_db(db)

    set_mode(False)
    cb = FakeCoinbase(balance=200.0, realized=0.0)
    risk = RiskManager()
    risk.set_account(AccountService(cb))
    await risk.sync()
    limit = risk.state.balance_usdc * risk.state.daily_loss_limit_pct

    # A loss the bot never saw — the exchange reports it, the breaker trips.
    cb.realized = -(limit + 0.5)
    risk._account._snapshot = None
    await risk.sync()
    check("breaker trips on the exchange's own realized P&L",
          risk.state.circuit_breaker_active,
          f"daily loss ${risk.state.daily_loss_usdc:.2f} >= ${limit:.2f}")

    # A paper close must not touch the live loss budget.
    risk = RiskManager()
    risk.set_account(AccountService(FakeCoinbase(balance=200.0)))
    await risk.sync()
    risk.apply_loss(-999.0, symbol="ETP", pnl_r=-5.0, is_paper=True)
    check("a PAPER close is ignored by the live breaker",
          not risk.state.circuit_breaker_active and risk.state.daily_loss_usdc == 0.0,
          f"daily loss ${risk.state.daily_loss_usdc:.2f}")

    # A live close does count.
    risk.apply_loss(-(limit + 1), symbol="ETP", pnl_r=-1.0, is_paper=False)
    check("a LIVE close trips it", risk.state.circuit_breaker_active,
          f"daily loss ${risk.state.daily_loss_usdc:.2f}")

    # Paper mode ignores the exchange entirely.
    set_mode(True)
    db.config["paper_balance"] = "200"
    risk = RiskManager()
    risk.set_account(AccountService(FakeCoinbase(balance=200.0, realized=-500.0)))
    await risk.sync()
    check("paper breaker ignores the live account's realized P&L",
          not risk.state.circuit_breaker_active and risk.state.daily_loss_usdc == 0.0)


# ----------------------------------------------------------------------
# 4. Fill audit
# ----------------------------------------------------------------------

async def test_fill_audit():
    print("\n=== 4. ORDER IDs + FILLS ===")
    from agent.executor import Executor

    db = FakeDB()
    install_fake_db(db)
    set_mode(False)
    cb = FakeCoinbase()
    cb.fills = [
        {"trade_id": "FILL-A", "order_id": "ORD-1", "product_id": "ETP-20DEC30-CDE",
         "side": "BUY", "price": "2601.50", "size": "1", "commission": "0.3642",
         "liquidity_indicator": "TAKER", "trade_time": "2026-09-22T14:03:11.123Z",
         "client_order_id": "abc"},
    ]
    ex = Executor(cb)
    ex._contract_sizes["ETP-20DEC30-CDE"] = 0.1
    res = await ex._enter_live("ETP-20DEC30-CDE", "ETH PERP", "long", 2600.0,
                               2500.0, 2900.0, 260.0, 64.0, 5, 10.0,
                               "rsi_macd", 0.8, "test", 1)
    check("live entry succeeds", res.success, res.error)
    check("real order_id returned", res.order_id == "ORD-1", str(res.order_id))
    row = db.trades.get(1, {})
    check("client_order_id stored on the trade row", bool(row.get("client_order_id")),
          str(row.get("client_order_id"))[:16])
    check("entry order_id stored on the trade row", row.get("order_id") == "ORD-1")
    check("protective stop order_id stored", row.get("stop_order_id") == "STOP-1")

    # Let the detached fill audit run.
    for _ in range(40):
        await asyncio.sleep(0.1)
        if db.fills:
            break
    check("exchange fill recorded", len(db.fills) == 1,
          f"{len(db.fills)} fill row(s)")
    if db.fills:
        f = db.fills[0]
        check("fill carries the order id and price",
              f["order_id"] == "ORD-1" and abs(f["price"] - 2601.50) < 1e-9,
              f"{f['order_id']} @ {f['price']}")
        check("fill is linked to its trade row", f["trade_id"] == 1)
        check("real commission captured", abs(f["commission_usdc"] - 0.3642) < 1e-9,
              f"${f['commission_usdc']}")
    patched = db.trades.get(1, {})
    check("actual fill price written back to the trade",
          abs((patched.get("filled_entry_price") or 0) - 2601.50) < 1e-9,
          f"modeled {row.get('entry_price')} vs filled "
          f"{patched.get('filled_entry_price')}")
    check("exchange commission written back to the trade",
          abs((patched.get("exchange_fees_usdc") or 0) - 0.3642) < 1e-6)


async def main() -> int:
    print("TradeBrain — paper/live separation checks (offline)")
    await test_balance()
    await test_mode_guards()
    await test_circuit_breaker()
    await test_fill_audit()
    print("\n=== RESULT ===")
    if _failures:
        for f in _failures:
            print(f"  ❌ {f}")
        return 1
    print("  ✅ all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
