"""
P0 Gate — live-mode position tracking (IMPLEMENTATION_PLAN_V2 Pre-flight P0)

Run with PAPER_TRADING=false against a funded-but-idle account. READ-ONLY:
reconciles positions from the exchange, then injects synthetic positions
locally (never sent to the exchange) and asserts the portfolio caps reject
a synthetic 4th entry. Zero orders are placed.

Usage:
    PAPER_TRADING=false venv/bin/python scripts/gate_p0.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from agent.coinbase_client import CoinbaseClient  # noqa: E402
from agent.executor import Executor, PaperPosition  # noqa: E402
from agent.risk_manager import RiskManager  # noqa: E402


PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(f"{name}" + (f" — {detail}" if detail else ""))
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}: {name}" + (f" — {detail}" if detail else ""))


def synth(product_id: str, direction: str, entry: float, notional: float) -> PaperPosition:
    return PaperPosition(
        product_id=product_id, display_name=product_id, direction=direction,
        entry_price=entry,
        stop_loss=entry * (0.98 if direction == "long" else 1.02),
        take_profit=entry, size_usdc=notional, margin_usdc=notional / 3,
        leverage=3, risk_usdc=notional * 0.02, strategy="gate-synth",
        is_paper=False,
    )


async def main() -> int:
    cfg = config.get_config()
    print("══════════════════════════════════════════════")
    print(" P0 GATE — live position tracking")
    print("══════════════════════════════════════════════")

    # --- 0. Environment guard ---
    check("PAPER_TRADING=false", not cfg.paper_trading,
          "gate must run in live mode; export PAPER_TRADING=false")
    if cfg.paper_trading:
        return 1

    cb = CoinbaseClient()
    executor = Executor(cb)
    executor.set_notifier(None)  # silent gate run
    risk = RiskManager()  # defaults: max_concurrent=3, max_total_risk=3%, max_correlated=2

    # --- 1. Auth + futures provisioning (read-only) ---
    ok = await cb.verify_auth()
    check("Coinbase auth", ok)
    if not ok:
        return 1
    await cb.verify_futures_provisioned()

    # --- 2. Reconciliation against the real account ---
    live = await executor.reconcile_live_positions()
    print(f"\n  Exchange reports {len(live)} open position(s):")
    for pos in live:
        meta = executor.live_meta.get(pos.product_id, {})
        print(f"    - {pos.product_id} {pos.direction.upper()} "
              f"{meta.get('contracts')} contracts @ {pos.entry_price:.2f} "
              f"(uP&L ${meta.get('unrealized_pnl', 0):+.2f})")
    check("reconcile_live_positions() returns", isinstance(live, list))
    check("accessors see live positions",
          all(executor.has_position(p.product_id) and
              executor.get_position(p.product_id) is not None for p in live))

    # --- 3. Synthetic 4th-position test (local only, NO orders) ---
    before = len(executor.get_open_positions())
    n_synth = risk.state.max_concurrent_positions - before
    if n_synth < 1:
        n_synth = 0
        print(f"\n  Account already holds {before} positions — injecting none; "
              f"testing rejection directly against existing exposure")
    for i in range(n_synth):
        executor.live_positions[f"GATE-SYNTH-{i}"] = synth(
            f"GATE-SYNTH-{i}", "long", 80_000.0 + i, 1_000.0,
        )
    total_open = executor.get_open_positions()
    print(f"\n  Injected {len(total_open) - before} synthetic position(s) "
          f"locally → {len(total_open)} open in accessor view")

    check("has_position() sees synthetic live positions",
          executor.has_position("GATE-SYNTH-0") or before > 0)

    candidate = type("Sig", (), {
        "direction": "long", "confidence": 0.9, "entry_price": 80_000.0,
    })()
    reject = risk.check_portfolio_risk(candidate, "GATE-CANDIDATE", total_open)
    print(f"  check_portfolio_risk → {reject!r}")
    check("4th position rejected by portfolio caps", bool(reject),
          "expected a non-empty skip reason")

    # If a slot was genuinely free the count cap won't fire — verify the
    # count cap specifically only when we actually filled it.
    if n_synth > 0 or before >= risk.state.max_concurrent_positions:
        check("rejection is the position-count cap",
              "Max concurrent positions" in (reject or ""),
              f"got {reject!r}")

    # --- 4. Cleanup (local only) ---
    for i in range(n_synth):
        executor.live_positions.pop(f"GATE-SYNTH-{i}", None)
    await cb.close()
    check("synthetic state cleaned up",
          not executor.has_position("GATE-SYNTH-0"))

    # --- Verdict ---
    print("\n══════════════════════════════════════════════")
    if FAIL:
        print(f" P0 GATE: ❌ FAILED ({len(FAIL)} check(s))")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print(f" P0 GATE: ✅ PASSED ({len(PASS)} checks)")
    print(" Portfolio caps and re-entry suppression are live-aware.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
