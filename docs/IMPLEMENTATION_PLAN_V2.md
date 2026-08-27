# IMPLEMENTATION_PLAN_V2.md — Step-by-Step Plan for ROADMAP_V2

> Companion to `docs/ROADMAP_V2.md`. It converts Milestones M7–M12 into ordered,
> executable steps against the codebase as it exists today. Each step names the file(s)
> it touches and the gate that must pass before moving on.
>
> Model assignments follow `docs/MODELS.md` — note that **Kimi K3 is always invoked
> via Ollama Cloud** (build-plane escalation, run-plane critic, consolidation).
>
> Written 2026-08-27. **Revised 2026-08-27** after a code review against `HEAD`:
> added the P0 live-position blocker, swapped the scanning and Governor milestones,
> and corrected three assumptions that don't hold against the current code. Phase IDs
> (F1…F7, G1…G5, H1…H6) are the stable anchors — milestone *numbers* were reordered.

---

## Pre-flight (do once, before M7)

These unblock everything downstream. Items P0–P2 are correctness work, not plumbing.

> **Status (2026-08-27):** P0–P3 and P5 **implemented**, pending gates/review:
>
> - **P0** — done in code: mode-aware accessors (`get_open_positions`/`has_position`/
>   `get_position`/`close_position` cover both modes), reconciliation on startup + every
>   monitor tick (adopts unknown exchange positions, alerts on divergence, drops local
>   state for externally-closed positions), `_enter_live` rebuilt (market entry, whole
>   contracts rounded DOWN, ISOLATED margin, exchange-native stop-limit crash stop,
>   best-effort with loud alert on rejection), live close/partial via
>   `/orders/close_position`. **Gate not yet run** — `PAPER_TRADING=false
>   venv/bin/python scripts/gate_p0.py` (read-only; injects synthetic positions locally).
> - **P1** — done: `signals.parse_failed` + `raw_response_snippet` columns (migration
>   applied to Neon 2026-08-27), `_extract_json` marks every fallback path, LLM exception
>   path marked, `analytics.parse_failure_rate(days)` surfaces per-model rates.
> - **P2** — done: `check_entry` is `@abstractmethod`; all four strategies verified.
> - **P3** — done: `agent/llm_router.py` routes per role (signal/critic/burt/embedding/
>   consolidation × openrouter/zai/moonshot/ollama), per-role timeouts, model fallback to
>   `signal_model`, `agent_config` seeded, `BURT_MODEL` + both `signal_engine.py`
>   hardcodes de-hardcoded, test scripts read `cfg.signal_model`, startup assertion
>   rejects seat-subscription keys (`sk-ant-…`) and high-volume roles on Ollama Cloud
>   (override: `RUN_PLANE_ALLOW_OLLAMA_CLOUD=1`), routing table logged at startup.
>   Analytics gained a per-model slice (`by_model`).
> - **P4** — **blocked on data**: harness ready (`scripts/validate_model_swap.py`); run
>   ≥200 calls on candidate AND incumbent before flipping the pin. Pin unchanged.
> - **P5** — baseline captured from the May 8–11 run (same 300s×5 config): ~1,215
>   calls/day steady-state, 4–5 symbols/day → `docs/baseline_P5_2026-08-27.json`.
>   Parse-failure rate in that window is the column default, not measured (pre-P1).
>   Re-capture once the agent has run post-P1 for a cleaner before.

> **Review pass (2026-08-27), 7 defects found and fixed** — all in the P0 live path
> except the first, which predates it:
>
> 1. **`hydrate_product_details` never returned `mark_price`**, but
>    `PositionMonitor._check` reads exactly that key — so every position marked at
>    its own entry price and **no stop, take-profit, trailing ratchet, or time exit
>    could ever fire, in paper or live.** Pre-existing (the key was never there), but
>    it silently voided the whole B2 exit stack and would have made the P0 gate
>    meaningless. Fixed: return the product's `price` as `mark_price`.
> 2. **Adopted positions were force-closed on the next monitor tick.**
>    `_adopt_live_position` sets `stop_loss=0/take_profit=0` as an "unknown levels"
>    marker, but `_evaluate` compares price against those zeros: an adopted **short**
>    reports `stopped` and a **long** reports `taken_profit`, immediately, always.
>    `_handle_exit` then market-closes at `exit_price=0.0` and books ±full notional —
>    a fake profit, or a fake loss that instantly trips the circuit breaker, on a
>    position the bot did not open. Fixed: the monitor reports unmanaged positions
>    and never acts on them.
> 3. **Orphaned exchange stops.** `stop_order_id` was stored and never used;
>    `cancel_orders` had zero callers. The stop is a plain stop-limit, **not
>    reduce-only** — left open after a close it would, on trigger, **open a new
>    position in the opposite direction**. Fixed: `_cancel_protective_stop` on full
>    close and on the reconciliation-drop path.
> 4. **Partial closes left an oversized stop.** After reducing contracts the stop
>    still covered the original count — on trigger it closes the remainder and
>    reverses with the excess. Fixed: `sync_live_stop` re-places at the new size.
> 5. **Trailing/breakeven never reached the exchange.** The bot ratcheted
>    `pos.stop_loss` locally while the exchange-native crash stop stayed at the
>    original wide level — so the protection that exists *because* the bot might die
>    was always stale. Fixed: monitor calls `sync_live_stop` when the stop moves.
> 6. **`_close_live` discarded the caller's exit price**, using the last reconciled
>    mark or falling back to `entry_price` — which yields PnL≈0 and hides a real loss
>    from the circuit breaker. Fixed: honor the passed price first.
> 7. **Externally-closed positions skipped risk accounting entirely.** The
>    reconciliation drop path logged a close with no PnL and never called
>    `apply_loss`, so a stop filling between monitor ticks was invisible to the daily
>    loss limit and to the protections' R-tracking. Fixed: estimate from the last
>    mark and book it via `Executor.set_risk_manager` (wired in `main.py`).
>
> Also: `_enter_live` records the pre-trade *estimate* as `entry_price`; reconciliation
> now overwrites it with the exchange's `avg_entry_price` (logging the slippage) so
> stop distance and R-multiples aren't anchored to a price that never traded.
>
> **Resolved after review:**
>
> 8. **Partial take-profit now runs in both modes.** It was gated on `pos.is_paper`,
>    which made paper and live take structurally different exits — so every paper
>    result was invalid as evidence about live behaviour, the same divergence class
>    as the P0 bug itself. `ENABLE_PARTIAL_TP` is now the single switch for both
>    modes; if partials turn out to be a bad idea, turn them off for both, never one.
>    Verified end-to-end: partial fires once at +1R, the exchange stop is resized to
>    the remaining contracts, the old stop is cancelled, and a live position too
>    small to split (1 contract) is marked done rather than retried every 30s.
>
> **Accepted as-is:** an adopted position counts full notional as risk, so it will
> exceed `max_total_risk_pct` and halt all new entries until closed. This is the
> intended conservative behaviour — confirmed 2026-08-27 — but it is a trading halt
> triggered by a position the bot did not open, so the alert on adoption must stay
> loud and must say that new entries are blocked.

### P0 — 🔴 Live-mode position tracking (blocker for the Governor)

`agent/executor.py:63` declares `self.live_positions: dict[str, Any] = {}` and **nothing
in the codebase reads or writes it.** Every accessor reads `paper_positions` only:
`get_open_positions()`, `has_position()`, `get_position()`, `close_position()`.

Consequences with `PAPER_TRADING=false` **today**:

- `check_portfolio_risk()` receives an empty list, so `max_concurrent_positions`,
  `max_total_risk_pct`, and `max_correlated_directions` **all pass unconditionally**.
- `has_position()` never suppresses a re-entry, so `_evaluate` can re-enter a symbol
  that already has a live position open.
- `close_position()` cannot close a live position at all.

In other words the portfolio caps are **paper-only**. This is also why the Governor cannot
be built before it is fixed: M8's heat budget and correlation math both read
`get_open_positions()`, so the Governor would validate perfectly in paper and be a **silent
no-op in live** — the worst available failure shape.

> **Corrected (2026-08-27, during P0):** the plan understates the damage. `_enter_live` was
> **deleted** in commit `a008817` — `enter_position` in live mode calls a literal `...`
> stub of a method that no longer exists, so live mode cannot open positions at all
> (AttributeError on first live signal), and `get_futures_positions()` had zero callers.
> P0 therefore includes rebuilding live entry (market order + exchange-native protective
> stop), live close/partial-close via `/orders/close_position`, and exchange-first
> reconciliation that adopts unknown positions and alerts on divergence. CFM products use
> **whole contracts** (`future_product_details.contract_size`, e.g. 0.01 BTC) — sizes are
> rounded DOWN to contracts so live notional never exceeds the risk-sized notional.

Steps:

1. Populate `live_positions` by reconciling from `cb.get_futures_positions()`
   (already implemented, `agent/coinbase_client.py:307`) on startup and on each
   position-monitor tick.
2. Make `get_open_positions()`, `has_position()`, `get_position()`, and `close_position()`
   mode-aware — one accessor, both modes, normalized to a shared shape so
   `check_portfolio_risk()` and the Governor never care which mode they're in.
3. Reconciliation is authoritative: the exchange is the source of truth. Log and alert on
   any divergence between local state and `get_futures_positions()` (a position we don't
   know about, or one we think is open that isn't).
4. **Gate:** with `PAPER_TRADING=false` against a funded-but-idle account, assert that the
   portfolio caps actually reject a synthetic 4th position. Do not start M8 until this
   passes.

### P1 — Signal parse-failure instrumentation

`SignalEngine._extract_json` collapses malformed LLM output into
`{"direction": "none", "confidence": 0.0}` — **indistinguishable from a genuine no-signal.**
Until that is separable, MODELS.md §6.3's "measure parse-failure rate over ≥200 calls"
cannot be executed, and a model swap is unmeasurable.

1. Add a `parse_failed` boolean (and optionally `raw_response_snippet`) to the `signals`
   table and to `_log_signal`.
2. Set it on every fallback path in `_extract_json` and on the `_call_llm` exception path.
3. Surface parse-failure rate per model in `agent/analytics.py`.

### P2 — Make `check_entry` mandatory

`strategies/base.py:56` gives `check_entry` a default returning `direction="none"`. Once G1
gates LLM calls on it (M7), any future strategy that forgets to override is **silently
gated to zero trades forever** — no error, no signal, no trade.

- Promote `check_entry` to `@abstractmethod`, or assert at import in `strategies/__init__.py`
  that every registered strategy overrides it.
- All four current strategies (`rsi_macd`, `bollinger`, `ema_pullback`, `donchian_breakout`)
  already implement it, so this is a guard against future work, not a migration.

### P3 — Run-plane model config plumbing

1. Add `critic_model`, `burt_model`, `embedding_model`, `consolidation_model` to
   `config.py` alongside the existing `signal_model`. (`MODELS.md` §7.)
2. Seed the keys in `agent_config` via `scripts/setup_db.py` so they are hot-reloadable
   from the UI and from Burt.
3. De-hardcode `BURT_MODEL` (`agent/burt.py:26`) and the two model constants in
   `agent/signal_engine.py` (lines 230, 252) — replace with `cfg.<key>` lookups.
4. Point the test scripts (`scripts/test_openrouter.py`, `test_strong_prompt.py`,
   `test_signal_response.py`, `test_quick.py`) at `config.get_config().signal_model`.
5. **Provider routing key** — add a per-role `provider` config (`openrouter` | `zai` |
   `moonshot` | `ollama`); all four are OpenAI-compatible so this is a base-URL + key swap
   in `agent/signal_engine.py`. Ollama Cloud is the provider for Kimi K3 roles.
6. **Per-role timeout** — replace the single global `httpx.Timeout(30.0)`
   (`signal_engine.py:41`) with a per-role map (Tier 1 fast, critic relaxed).
7. **Startup assertion** — fail loudly if a run-plane credential looks like a
   seat-subscription key; enforces `MODELS.md` §2 in code.
8. Verify `signals.model` is populated on every row, including fallback/error paths — it is
   the A/B join key and the audit trail (`MODELS.md` §5).

### P4 — Replace the deprecated model pin

`moonshotai/kimi-k2.6` is deprecated (`MODELS.md` §6). Treat this as maintenance.

1. Set `signal_model` → **GLM-5.3 non-reasoning**, `burt_model` → **GLM-5.2**.
2. **Requires P1.** Before promoting, run ≥200 signal calls and record parse-failure rate,
   p95 TTFT, and confidence calibration against the K2.6 baseline (`MODELS.md` §8).
   GLM supports JSON mode but **not** schema enforcement, so parse failure is a live risk.
3. Confirm the Tier 1 timeout from P3.6 comfortably exceeds observed p95 TTFT **and** stays
   well under `signal_interval`.
4. Do **not** adopt GLM-5.3-**Flash** here — ~42s TTFT (`MODELS.md` §6.3).

### P5 — Capture the pre-change baseline

M7's gate is a before/after comparison, so record the "before" while it still exists:
LLM calls/day, symbols evaluated/day, signals logged/day, and current LLM spend/day.
Both numbers come from the `signals` table.

---

## M7 — Attentive scanning, part 1 (G1 + G2 + G5)

> **Reordered:** this was M8. It now runs first. G1 has no dependency on the Governor —
> it gates the LLM call inside today's `_evaluate` and does not need the two-pass
> structure. Doing it first is strictly better: it is a far smaller change, it cuts
> ~96% of LLM spend *before* the multi-week paper validations begin (M8's ≥2-week run
> at 1,440 calls/day is ~$60 that this reduces to ~$2), and — most importantly — it makes
> the **live path match the backtested path**. The backtester evaluates `check_entry()`
> mechanical rules, not the LLM, so gating live entries on the same rules is what makes
> M8's Governor A/B representative of live behavior.

**Goal:** cut LLM spend ~96%, look at 3× more symbols, 5× more often.

1. **G1 trigger gate**: in `agent/main.py:_evaluate`, run `BaseStrategy.check_entry()`
   (pure Python, built for the backtester) before the signal LLM call. Only call the LLM
   when it fires. Log suppressed evaluations at debug level with the reason.
2. **G2 tiered cadence** in `agent/main.py`:
   - Hot tier: top 3 screener + all open positions, every 60s, full indicator + `check_entry`.
   - Warm tier: rest of watchlist (widen to ~15), every 5 min, same work.
   - Cold tier: full tradeable universe, screener re-rank every 4h (current behavior).
   Tier assignment lives in a small structure refreshed each cold pass.
3. **G5 idle-state surfacing**: when the regime gate disables all strategies, log it, send
   a Discord notification via `agent/notifier.py`, and expose an `idle` flag on the API
   (`agent/api.py`) for the UI.
   > Note: G5 *reports* idleness; **F6 (M9) is what fixes it.** Don't let this milestone's
   > gate pass while treating the idle-bot problem as solved.
4. **Measure** against the P5 baseline: LLM calls/day should drop to ~30–75; symbols
   evaluated/day should rise materially.
   **Gate: calls down, evaluations up — both printed side-by-side against P5 before M8.**

---

## M8 — Governor core (F1 + F2 + F3)

> **Reordered:** this was M7. **Requires P0** — the heat budget and correlation math read
> `get_open_positions()`, which is paper-only until P0 lands.

**Goal:** replace first-come-first-served entry with collect → rank → budget.

1. **Create `agent/governor.py`** with a `Candidate` dataclass:
   `symbol, direction, confidence, reasoning, entry, atr, strategy, regime_fit,
   expectancy_prior, liquidity_score`.
2. **Restructure `agent/main.py:_loop`** into the two-pass form:
   - Pass 1: evaluate every watchlist symbol into `list[Candidate]` — **no orders placed**.
     The M7 trigger gate stays in front of the LLM call, so pass 1 is cheap.
   - Pass 2: hand the list to the Governor; it returns 0..N order decisions.
   - Keep `has_position()` suppression in pass 1 for now, but structure it as a filter on
     the candidate list rather than an early `return` — M12 pyramiding needs open-position
     symbols to reach the Governor as *add* candidates.
3. **Implement candidate scoring** (F2):
   `score = confidence × regime_fit × expectancy_prior × liquidity_score ÷ correlation_penalty`.
   - `expectancy_prior` reads rolling R-expectancy per (strategy, regime) from
     `agent/analytics.py`; cold-start to 1.0.
   - `correlation_penalty = 1 + β` to already-open exposure.
4. **Portfolio heat budget** (F3) in `agent/risk_manager.py`:
   - Add `get_heat_budget(regime)` — base % of balance at risk, scaled: risk-on 1.5×,
     mixed/chop 1.0×, risk-off 0.5× (with a higher long-confidence bar), unknown 1.0×.
   - Add `correlation_adjusted_heat(positions)` — `sqrt(Σ wᵢwⱼρᵢⱼ)` over a rolling
     correlation matrix of 1h returns.
     > **Corrected:** ROADMAP_V2 F3 says 30 days and "candles already pulled." Neither
     > holds. `get_candles` is **max 300 per request with no pagination**
     > (`coinbase_client.py:240`) and the loop passes no `start`/`end`; 30d × 1h = 720 bars.
     > **Start with a 7-day window (168 bars, one request per symbol).** Crypto major
     > correlations are stable enough that 7d vs 30d won't materially move the allocation.
     > Widen later only if a paginated `get_candles_range()` is worth building.
   - Cache the matrix per cold-tier pass; do not recompute per cycle.
   - Keep an absolute position-count ceiling as backstop, regime-scaled (e.g. 3 chop /
     5 risk-on). Existing `max_concurrent_positions` and `max_correlated_directions`
     become backstops, not primary gates.
5. **Paper-trade A/B**: run the same historical window through the backtester twice —
   current first-come-first-served vs Governor allocation on identical signals.
   **Gate: Governor must win on expectancy and max drawdown before M9 starts.**

---

## M9 — Governor breadth (F6 + F4 + F7)

**Goal:** multi-strategy competition, anti-clustering, and full explainability.

1. **F6 multi-strategy concurrency**: run every strategy whose `compatible_regimes`
   matches the current regime; let the Governor arbitrate between their candidates.
   `cfg.strategy` demoted to a preference/tiebreaker. **This is the actual fix for the
   idle-bot failure mode that M7/G5 only reports.**
2. **F4 entry spacing**: enforce a minimum gap between new entries (2 bars or 10 min,
   whichever is longer) inside the Governor.
3. **F7 decision records**: per cycle, the Governor writes one record — every candidate,
   its score, budget state, taken/passed, and the reason. New table `governor_decisions`.
   > **Corrected:** ROADMAP_V2 F7 implies `search_memories`-style retrieval. **Don't embed
   > decision records** — it adds per-cycle embedding cost and couples this table to the
   > pgvector dimension-migration problem in `MODELS.md` §8 for no benefit. Plain SQL
   > filtered by time/symbol is enough for Burt to narrate a cycle in chat.
   **Gate: no idle-regime gaps over a 2-week paper run; every order has a complete
   decision record.**

---

## M10 — Attentive scanning, part 2 (G3 + G4)

**Goal:** WebSocket speed without losing REST resilience.

1. **G3 WS price feed**: add ticker/candle subscriptions for hot-tier symbols in
   `agent/coinbase_client.py`. REST stays as fallback — implement reconnect with backoff
   and an automatic downgrade to REST polling on WS failure.
2. **Position monitor on WS**: `agent/position_monitor.py` stops polling marks every 30s;
   trailing stops and breakeven moves trigger off the stream (much tighter).
3. **G4 event detectors**: cheap checks on the stream — volume spike vs 20-bar SMA, range
   expansion vs ATR, funding flip, OI surge — that promote a symbol to hot tier and force
   an immediate evaluation.
   **Gate: survive a forced WS disconnect without missed stops; backtest shows tighter
   exit timing vs M9.**

---

## M11 — Sharper decisions (H1 + H3 + H4 + H5)

**Goal:** veto layer, dynamic thresholds, equity-based drawdown, maker entries.

1. **H1 risk critic**: after the Governor selects candidates, a second LLM call using
   `critic_model` = **Kimi K3 via Ollama Cloud** (different family from `signal_model` =
   GLM — `MODELS.md` §6) may veto or size-down. Runs a handful of times per day; every
   critic verdict logged with the M9 decision record.
2. **H3 dynamic confidence threshold**: per-strategy threshold nudged by rolling
   expectancy, bounded ±0.1 around base, **hard floor 0.60** (MQL5 sub-0.55 evidence).
3. **H4 equity-based MaxDrawdown**: change `get_drawdown_scale()` in
   `agent/risk_manager.py` from closed-trade peak-to-trough to account-equity drawdown
   (balance + cumulative PnL + unrealized), mirroring Freqtrade 2026.2's change.
   Unrealized PnL requires open positions in both modes — **depends on P0.**
4. **H5 maker entries**: in `agent/executor.py`, enter post-only limit at trigger price
   with a short chase-then-cancel window (fall back to market only on expiry).
   > **Check first:** does `backtest/engine.py` model *unfilled* limit orders, or does it
   > assume every order fills? If it assumes fills, post-only entries cannot be honestly
   > gated on "beats M10 out-of-sample" — the backtest would credit fee savings while
   > ignoring missed trades. Either add non-fill modeling to the engine as step 0 of H5,
   > or gate H5 on live paper fill-rate data instead of the backtester.
   **Gate: each item independently beats the M10 configuration out-of-sample.**

---

## M12 — Pyramiding (F5) — ships last

**Goal:** add to winners without raising worst-case exposure.

1. Adds allowed only to positions ≥ +1R and still trending (strategy-supplied check).
2. Diminishing adds: 50%, then 25% of base size; total pyramid ≤ 150% of base.
3. Stack stop ratchets to average entry on each add; **never average down** enforced
   server-side in `executor.py`.
4. Pyramided exposure counts against the M8 heat budget like any new risk.
   **Gate: backtest shows expectancy improvement with no increase in tail drawdown;
   only ships after M8 budget math has ≥4 weeks of trusted live paper data.**

---

## Cross-cutting rules (apply at every milestone)

- **Deterministic rules are law**: any risk limit implied by a prompt is mirrored by a
  server-side check (`MODELS.md` §4). A persuasive rationale never overrides a hard cap.
- **Paper and live must share one code path.** P0 exists because they diverged silently.
  Any new state the Governor or monitor depends on gets a mode-aware accessor from day
  one — never a `paper_*` dict read directly.
- **Every change logs its model** — the `model` column is the A/B key and the audit trail
  (`MODELS.md` §5). FCA/ESMA scrutiny of retail AI products starts H2 2026.
- **Mandatory review**: diffs touching `risk_manager.py`, `governor.py`, `executor.py`,
  `position_monitor.py`, or risk defaults in `config.py` get Claude review regardless of
  which builder wrote them (`MODELS.md` §4).
- **Progression discipline**: each milestone must beat its predecessor out-of-sample in the
  backtester, or over ≥2 weeks of paper trading, before the next starts (`ROADMAP_V2.md` §5).
- **The bull-market caveat**: the Governor's job is not more bullishness — it is the *best*
  exposure, correctly sized for correlation, with capital preserved for when the regime
  turns (`ROADMAP_V2.md` §6).

## Deferred (unchanged)

- D4 ML win-probability layer (gated on data volume)
- E1/E2 grid + funding-harvest strategies
- H6 sub-account isolation — before live capital, not before paper
