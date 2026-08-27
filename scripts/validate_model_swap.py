"""
Model-swap validation harness (P4 gate — MODELS.md §6.3, §8).

Runs N real signal evaluations against a CANDIDATE model and reports:
  - parse-failure rate   (the primary gate metric)
  - p95 TTFT             (time-to-first-token proxy: full round-trip)
  - confidence histogram (calibration vs the incumbent)

Usage (candidate first, then optional incumbent re-run for a same-day baseline):
    venv/bin/python scripts/validate_model_swap.py z-ai/glm-5.3 --n 200
    venv/bin/python scripts/validate_model_swap.py moonshotai/kimi-k2.6 --n 200 --baseline

Reads real market data + strategy prompts (same path as live), logs NOTHING
to the signals table — evaluation only. Costs real tokens: --n 200 at GLM-5.3
blended pricing is roughly $0.20.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from loguru import logger  # noqa: E402

import config  # noqa: E402
from agent import llm_router  # noqa: E402
from agent.coinbase_client import CoinbaseClient  # noqa: E402
from agent.indicator_engine import compute_indicators  # noqa: E402
from agent.signal_engine import SignalEngine  # noqa: E402
from strategies import STRATEGIES  # noqa: E402


async def fetch_candidate_setups(cb: CoinbaseClient, symbols: list[str], strategy_name: str) -> list[dict]:
    """Build (symbol, indicators) setups the same way the live loop does."""
    candles_map = await cb.get_candles_multi(symbols, "FIFTEEN_MINUTE")
    hourly_map = await cb.get_candles_multi(symbols, "ONE_HOUR")
    strategy = STRATEGIES[strategy_name]
    setups = []
    for sym in symbols:
        c15, c1h = candles_map.get(sym, []), hourly_map.get(sym, [])
        if len(c15) < 60 or len(c1h) < 60:
            continue
        try:
            indicators = compute_indicators(c15, c1h)
            setups.append({"symbol": sym, "strategy": strategy, "indicators": indicators})
        except Exception as exc:
            logger.warning(f"{sym}: indicator failure — skipped ({exc})")
    return setups


async def run_calls(model: str, setups: list[dict], provider: str) -> dict:
    cfg = config.get_config()
    cfg = cfg.model_copy(update={"signal_model": model, "signal_provider": provider})
    engine = SignalEngine.__new__(SignalEngine)
    engine.cfg = cfg
    engine._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    engine._memory_engine = None
    engine._available = True

    latencies, parse_failures, confidences, errors = [], 0, [], 0
    for setup in setups:
        for _ in range(max(1, 200 // max(1, len(setups)))):
            t0 = time.perf_counter()
            try:
                resp = await engine._call_llm(
                    setup["symbol"], setup["strategy"], setup["indicators"],
                    regime=None, extra_context="",
                )
                latencies.append(time.perf_counter() - t0)
                if resp.get("parse_failed"):
                    parse_failures += 1
                else:
                    try:
                        confidences.append(float(resp.get("confidence", 0.0)))
                    except (TypeError, ValueError):
                        parse_failures += 1
            except Exception as exc:
                latencies.append(time.perf_counter() - t0)
                errors += 1
                logger.warning(f"{setup['symbol']}: call failed — {str(exc)[:120]}")

    await engine._client.aclose()
    n = len(latencies)
    return {
        "model": model,
        "calls": n,
        "errors": errors,
        "parse_failures": parse_failures,
        "parse_failure_rate_pct": round(parse_failures / n * 100, 2) if n else None,
        "error_rate_pct": round(errors / n * 100, 2) if n else None,
        "ttft_p50_s": round(statistics.median(latencies), 2) if n else None,
        "ttft_p95_s": round(sorted(latencies)[int(n * 0.95)], 2) if n >= 20 else None,
        "confidence_mean": round(statistics.mean(confidences), 3) if confidences else None,
        "confidence_std": round(statistics.stdev(confidences), 3) if len(confidences) > 1 else None,
        "confidence_histogram": _histogram(confidences),
    }


def _histogram(values: list[float]) -> dict:
    buckets = {"0-30": 0, "30-50": 0, "50-65": 0, "65-75": 0, "75-85": 0, "85-100": 0}
    for v in values:
        if v < 0.3: buckets["0-30"] += 1
        elif v < 0.5: buckets["30-50"] += 1
        elif v < 0.65: buckets["50-65"] += 1
        elif v < 0.75: buckets["65-75"] += 1
        elif v < 0.85: buckets["75-85"] += 1
        else: buckets["85-100"] += 1
    return buckets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="candidate model ID (e.g. z-ai/glm-5.3)")
    ap.add_argument("--n", type=int, default=200, help="number of calls (default 200)")
    ap.add_argument("--provider", default="openrouter", help="provider for the candidate")
    ap.add_argument("--strategy", default="rsi_macd")
    ap.add_argument("--symbols", default="BIP-20DEC30-CDE,ETP-20DEC30-CDE,SOP-20DEC30-CDE",
                    help="comma-separated CFM product_ids")
    ap.add_argument("--baseline", action="store_true",
                    help="label the run as the incumbent baseline")
    args = ap.parse_args()

    cfg = config.get_config()
    if not cfg.openrouter_api_key and args.provider == "openrouter":
        print("OPENROUTER_API_KEY missing", file=sys.stderr)
        return 1

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cb = CoinbaseClient()
    try:
        setups = await fetch_candidate_setups(cb, symbols, args.strategy)
    finally:
        await cb.close()
    if not setups:
        print("No usable setups (candle fetch failed?)", file=sys.stderr)
        return 1
    print(f"Validating {args.model} via {args.provider}: "
          f"{args.n} calls across {len(setups)} {args.strategy} setup(s)")

    result = await run_calls(args.model, setups, args.provider)
    result["role"] = "baseline (incumbent)" if args.baseline else "candidate"

    print("\n════════════════ MODEL VALIDATION ══════════════")
    print(json.dumps(result, indent=2))
    print("════════════════════════════════════════════════")

    # P4 gate thresholds (MODELS.md §6.3/§8) — advisory here, decisive when
    # both baseline and candidate exist in one report.
    if not args.baseline and result["parse_failure_rate_pct"] is not None:
        if result["parse_failure_rate_pct"] > 5:
            print("⚠️  Parse-failure rate >5% — GLM has no schema enforcement; "
                  "investigate before promoting (P4 gate).")
        if result["ttft_p95_s"] and result["ttft_p95_s"] > 25:
            print("⚠️  p95 TTFT >25s — must sit comfortably under signal_interval "
                  "(300s default) with headroom (P4 gate).")
        if result["ttft_p95_s"] and result["ttft_p95_s"] > 40:
            print("🛑 p95 TTFT >40s — GLM-5.3-Flash territory (~42s). Plan says NO.")

    out = f"docs/model_validation_{args.model.replace('/', '_')}.json"
    os.makedirs("docs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
