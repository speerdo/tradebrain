# MODELS.md — Single Source of Truth for LLM Selection

> **This file supersedes every model reference in `BLUEPRINT.md`, `BURT.md`, and
> `ACTION_PLAN.md`.** Those docs were written against Kimi K2.6 in early 2026 and are
> now historical. When a model ID appears anywhere else in this repo, it is either
> a stale doc or a hardcode that should be migrated to a config key (see §7).
>
> Last reviewed: 2026-08-27

---

## 1. Why this file exists

Model IDs were scattered across `config.py`, `agent/burt.py`, `agent/signal_engine.py`,
`scripts/setup_db.py`, and four test scripts. Frontier models now turn over roughly
every 6–10 weeks. Chasing that from seven places is how a bot ends up silently running
a deprecated endpoint in production.

One file, one review cadence, one migration checklist.

---

## 2. Two planes — and the rule that separates them

TradeBrain uses LLMs for two completely different jobs, on completely different
commercial terms. Conflating them is the mistake to avoid.

| | **Build plane** | **Run plane** |
|---|---|---|
| **What** | Writing, planning, and reviewing TradeBrain's code | Burt's live brain: signal eval, risk critic, chat, embeddings |
| **Who invokes** | Adam, interactively, in an agent CLI | The bot, autonomously, 24/7 |
| **Paid via** | **Seat subscriptions** (Claude Pro, Ollama Pro, OpenCode Go) | **Pay-per-token API keys** (OpenRouter, or direct z.ai / Moonshot) |
| **Volume** | Bursty, human-paced, capped by 5h/weekly windows | Continuous, machine-paced |
| **Governed by** | §4 (the planner/builder/reviewer workflow) | §6 (role assignments) |

> ### 🔴 The hard rule
>
> **Never point the run plane at a seat subscription.**
>
> Two independent reasons, either sufficient on its own:
>
> 1. **The math doesn't work.** Current loop = 300s interval × 5 symbols = **1,440 LLM
>    calls/day**. Widening the watchlist to ~15 (per `ROADMAP_V2.md` G2) = **4,320/day**.
>    OpenCode Go allows roughly **110 Kimi K3 requests per 5 hours** (~528/day) under its
>    `$12 / 5h` cap ([Bitdoze](https://www.bitdoze.com/opencode-go-plan/)). Claude Pro is
>    ~45 prompts per 5-hour window ([Morph](https://www.morphllm.com/claude-code-usage-limits)).
>    An unattended bot drains either in a couple of hours, then silently degrades to
>    `direction: "none"` on every symbol — which looks exactly like a quiet market.
> 2. **Terms.** These are interactive-coding-assistant plans. Wiring a 24/7 autonomous
>    trading agent through one invites rate-limiting or suspension, and takes your dev
>    environment down with it.
>
> The run plane gets its own metered API key. Cost is controlled by **calling the LLM
> less** (see §6.1), not by borrowing a coding seat.

---

## 3. Subscription inventory (build plane)

Verified 2026-08-27. Re-check on renewal — these terms move constantly.

| Sub | Cost | Limits | Notable models |
|---|---|---|---|
| **Claude Pro** | $20/mo | ~45 prompts / 5h window, plus a weekly cap across all models. Weekly limits boosted 50% on paid plans through 2026-08-31. | Sonnet tier. **Verify Opus access on your account** — public write-ups say Opus is Max-only on Claude Code, which would conflict with using Opus 4.8 as your planner. If it's gated, plan with Sonnet and reserve Opus for the few reviews that genuinely need it. |
| **Ollama Pro** | $20/mo | 3 cloud models, ~50× free-tier cloud usage, private models. **Unlimited local models.** Metered by tokens, not a fixed call cap. | Open-weight catalog, cloud + local |
| **OpenCode Go** | $10/mo ($5 first month) | Dollar-based: **$12/5h, $30/week, $60/month**. Request count scales inversely with model price. | 18 models incl. **GLM-5.2/5.1, Kimi K3 / K2.7 Code / K2.6**, DeepSeek V4, Qwen3.8 Max, Grok 4.5, MiniMax M3 |

**Practical read:** OpenCode Go is the workhorse — it's where GLM and Kimi K3 live, and
$10/mo of dollar-metered budget goes a long way on cheap models. Claude Pro is the scarce,
high-value resource ("use sparingly" — correct instinct). Ollama Pro's real strategic value
here is **unlimited local inference**, which is the one thing that *can* legitimately touch
the run plane (§6.2).

---

## 4. Build plane — the planner / builder / reviewer workflow

This is the workflow already in use, written down so it survives context resets and so
the model tiers are matched to blast radius rather than to habit.

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │   PLANNER    │ ───▶ │   BUILDER    │ ───▶ │   REVIEWER   │
   │ Claude Opus  │      │  GLM-5.2 /   │      │ Claude Opus  │
   │   4.8 high   │      │ GLM-5.3-Flash│      │   4.8 high   │
   │ (Claude Pro) │      │ (OpenCode Go)│      │ (Claude Pro) │
   └──────────────┘      └──────────────┘      └──────────────┘
     scarce, deep          cheap, high-volume     scarce, deep
```

| Stage | Model | Owns |
|---|---|---|
| **Planner** | Claude Opus 4.8 high | Architecture decisions, spec-writing, breaking a milestone into steps. Anything where a wrong call costs a week. Output is a written plan committed to `docs/`. |
| **Builder** | GLM-5.2 (or GLM-5.3-Flash for bulk edits) | Implementing against an approved plan. Tests, boilerplate, refactors, UI, migrations. High volume, low judgment. |
| **Reviewer** | Claude Opus 4.8 high | Review of **risk-critical paths only** (see below). Not every diff — Claude Pro's 5h window won't take it. |
| **Escalation** | Kimi K3 (via **Ollama Cloud**) | Long-context work where 1M tokens matters — whole-repo reasoning, big backtest-output analysis — without spending Claude budget. Kimi K3 should be invoked through Ollama Cloud, not OpenCode Go. |

### Mandatory-review paths

Claude reviews every diff touching these, no exceptions. This is the code that can lose
real money, and it is a short enough list to stay affordable:

- `agent/risk_manager.py` — sizing, stops, circuit breaker, protections
- `agent/governor.py` — the allocator (once it exists; `ROADMAP_V2.md` Phase F)
- `agent/executor.py` — order placement, reduce-only logic
- `agent/position_monitor.py` — exit management
- `config.py` — anything that changes a live risk default

Everything else — strategies, screener, UI, analytics, docs, tests — GLM builds and
self-reviews, with Claude pulled in only when a backtest disagrees with expectations.

### Why this split, specifically

It mirrors the run plane's own architecture, and that symmetry is not a coincidence:
**cheap models do volume, expensive models do judgment, and deterministic rules are law.**
The 2026 production consensus is ML for classification, LLM for strategy-level decisions,
and hard-coded rules for risk limits, with the note that *"a persuasive rationale should
never override hard limits"* ([MQL5](https://www.mql5.com/en/blogs/post/769403)). A GLM
builder writing risk code that no one reviews is the same failure mode as an LLM overriding
a position cap.

---

## 5. The evidence, and what it means for the run plane

The temptation is to pin whatever tops the leaderboard. The data says don't.

| Finding | Source | Implication |
|---|---|---|
| 55 models, 8 seasons, 2,724 trades since Jan 2026 — **39% of model-seasons profitable**; rankings shift season to season and "no model is reliably best" | [TradeRank Arena](https://www.traderank.ai/llm-trading-benchmark) | Model choice is a **tunable parameter to be A/B'd on our own logged outcomes**, not a leaderboard lookup |
| Season 5 winner Gemini 3.5 Flash (+13.76%) was a *cheap* model, not a flagship | [TradeRank 2026 ranking](https://www.traderank.ai/blog/best-ai-models-for-crypto-trading-2026) | Don't assume the most expensive reasoning model wins. A cheap signal model is a legitimate first choice, not a compromise — but pick it on **latency and JSON reliability**, not on sticker price (§6.3). |
| Alpha Arena S1: 4 of 6 frontier models lost real money (GPT-5 −$6,267, Claude −$3,081) | [nof1](https://nof1.ai/) · [Protos](https://protos.com/llm-crypto-trading-contest-finds-llms-cant-trade-crypto/) | Raw LLM judgment is **not** the alpha source. It is a filter over mechanical rules. |
| Across 8,400 LLM decision calls (Oct 2025–Mar 2026), decisions below **0.55 confidence won 48.3%** — below breakeven at typical spreads | [MQL5](https://www.mql5.com/en/blogs/post/769403) | Our `min_confidence` floor of 0.65 is well-placed. Never lower it below 0.60. |
| 2026 production pattern: **ML for classification, LLM for strategy-level judgment, deterministic rules for hard risk limits** | [MQL5](https://www.mql5.com/en/blogs/post/769403) | Matches our architecture. Keep every prompt rule mirrored by a server-side check. |
| FCA/ESMA signalled H2 2026 scrutiny of "AI-driven" retail products — audit trail of inference calls, confidence scores, and rationales | [MQL5](https://www.mql5.com/en/blogs/post/769403) | We already log every signal with model, confidence, and reasoning. **Keep `model` populated on every row** — it is both our A/B key and our audit trail. |

**Standing rule:** a model change ships only when it beats the incumbent on *our* logged
signals (see `agent/analytics.py`), not because it launched with a good benchmark.

---

## 6. Run plane — role assignments

### 6.0 There is no "always-on scanner model" — and that's the design

The instinct is to pick one model and let it watch the market. Don't. The always-on layer
should contain **no LLM at all**. Three tiers, each with a different duty cycle:

| Tier | What runs | Cadence | Volume/day | Cost | Latency |
|---|---|---|---|---|---|
| **0 — Scanner** | `BaseStrategy.check_entry()`, pure Python. Later: WebSocket event detectors (`ROADMAP_V2.md` G4) | 60s hot / 5min warm | ~7,800 evaluations | **$0** | sub-ms |
| **1 — Confirmer** | `signal_model` — fires **only when Tier 0 triggers** | on trigger | ~30–75 calls | ~$5/mo | must be fast |
| **2 — Critic** | `critic_model` — fires only on Governor-selected candidates | on candidate | a handful | ~$1/mo | latency-tolerant |

This is the same principle as §4 and §5's architecture rule: **cheap/deterministic does
volume, expensive does judgment.** An LLM polling every symbol on a timer is paying frontier
prices to answer "nothing is happening" 95% of the time.

The consequence for model choice is important: **once G1 lands, cost stops being the
deciding factor.** At ~50 calls/day even Kimi K3 (run via Ollama Cloud) costs ~$11/mo. So Tier 1 gets picked on
**latency and JSON reliability**, not price — because a trigger is time-sensitive, and a
confirmer that takes 40s to answer has already missed the bar it was asked about.

| Role | Config key | Current pin | Candidate | Why |
|---|---|---|---|---|
| **Signal evaluation** (Tier 1) | `signal_model` | `moonshotai/kimi-k2.6` ⚠️ **deprecated** | **GLM-5.3 (non-reasoning)** | TTFT **1.60s**, 62.6 t/s, ~$0.90/1M blended. Fastest credible option, and cheap enough that post-G1 volume is ~$4.50/mo. **Not** GLM-5.3-Flash — see §6.3. |
| **Risk critic** (planned, `ROADMAP_V2.md` H1) | `critic_model` | *unset* | **Kimi K3 (via Ollama Cloud)** | A handful of calls/day, so $3/$15 and a 7.2s TTFT are both fine. Must be a **different family** than `signal_model`: a second opinion from the same model is not a second opinion. Signal = GLM ⇒ critic = Kimi (or Claude). Route Kimi K3 via Ollama Cloud. |
| **Burt chat** | `burt_model` | `moonshotai/kimi-k2.6` (hardcoded, `agent/burt.py:26`) | **GLM-5.2** | Tool-calling + personality. Latency matters (Discord). |
| **Embeddings** | `embedding_model` | `openai/text-embedding-3-small` (hardcoded, `signal_engine.py:230`) | Keep, or **local via Ollama** (§6.2) | 1536-dim. **Changing this invalidates the entire `memories` pgvector table** — see §8. |
| **Nightly consolidation** | `consolidation_model` | falls through to `burt_model` | **Kimi K3 (via Ollama Cloud)** | Batch, offline, latency-irrelevant, benefits from 1M context over a full day of trades. Cheapest place to trial a new model. Route Kimi K3 via Ollama Cloud. |

> ⚠️ **`kimi-k2.6` is deprecated** ([NxCode](https://www.nxcode.io/resources/news/kimi-k2-5-pricing-plans-api-costs-2026)).
> It is our current pin on both `signal_model` and `burt_model`. Migrating off it is now
> maintenance, not optimization — deprecated endpoints get withdrawn on the provider's
> schedule, not ours, and the failure mode is a dead signal engine.

### 6.1 Cost is controlled by call volume, not by model price

The single biggest lever isn't price per token — it's **`ROADMAP_V2.md` G1**, the
mechanical trigger gate. `BaseStrategy.check_entry()` already exists (built for the
backtester) and is pure Python. Run it on every poll; only spend an LLM call when it fires.

The difference that makes, on the same model:

| | Calls/day | GLM-5.3 | Kimi K3 (Ollama Cloud) |
|---|---|---|---|
| **Today** (300s × 5 symbols, LLM on every poll) | 1,440 | ~$128/mo | ~$330/mo |
| **Post-G1** (15 symbols, LLM on trigger only) | ~50 | **~$4.50/mo** | ~$11/mo |

Scanning 3× more symbols, 5× more often, for ~4% of the cost. **Do G1 before optimizing
model price** — afterwards the choice is nearly free either way.

### 6.2 Ollama's legitimate role in the run plane

Ollama Pro includes **unlimited local models**, which is the one subscription capability
that can serve the bot without a metered key or a ToS problem:

- **Embedding backfill / re-embedding.** High-volume, latency-tolerant, zero marginal cost — the ideal local job. Note the dimension-migration rule in §8 before switching.
- **Fallback brain.** If OpenRouter is down or rate-limited, a local model returning `direction: "none"` beats an exception loop. Degraded but honest.
- **Privacy.** Positions and PnL never leave the machine.

Local inference is slow on consumer hardware, so it belongs on the offline/fallback path —
never on the hot signal path where a 300s loop is waiting.

> **Ollama Cloud, as actually configured (2026-08-27).** `:cloud` tags are *not* local —
> the daemon proxies them to ollama.com using the signed-in account, so they bill the
> Ollama Pro subscription even though the base URL is `localhost` and no API key is set.
> `validate_run_plane_config` cannot detect that from the URL, so it emits a **warning**
> (not a block) for a `:cloud` model on the signal role. That is the right call: at
> post-G1 volume (~50 calls/day ≈ 19k tokens/day) this is comfortably sustainable, and a
> bounded test run is exactly what the exception is for. Revisit before running
> 300s × N symbols continuously.
>
> **Cloud tags expire.** The three tags on this machine had all been retired server-side —
> `glm-5` (2026-07-15), `kimi-k2.5` and `minimax-m2.5` (2026-07-31) — and returned
> `410 Gone`, not a clean error the agent would surface as anything but an LLM failure.
> Re-`ollama pull` and re-test after any gap in use.

### 6.3 ⚠️ Do not use GLM-5.3-**Flash** as the confirmer

The cheap-tokens argument points at GLM-5.3-Flash ($0.15/$0.50 per 1M, halved through
2026-09-09). Reject it on latency. Artificial Analysis measures:

| Model | TTFT | Output speed | Blended price |
|---|---|---|---|
| **GLM-5.3 (max)** | **1.60s** | 62.6 t/s (non-reasoning) | ~$0.90/1M |
| Kimi K3 (max, via Ollama Cloud) | 7.20s | 35.3 t/s | ~$2.31/1M |
| **GLM-5.3-Flash** | **~42s** ❌ | 49.4 t/s, *"notably slow and very verbose"* | ~$0.15/$0.50 |

The Flash variant is **26× slower to first token than the full model** despite the name —
it front-loads reasoning. Two consequences for us:

1. `agent/signal_engine.py:41` sets `httpx.Timeout(30.0)`. Flash would time out on most
   calls and fail closed to `direction: "none"` — silent, total signal loss that looks
   exactly like a quiet market.
2. Even with a raised timeout it's wrong for Tier 1. A trigger-confirmer answering 42s
   after a breakout fired is answering about a bar that has already closed.

Per §6.1, its token discount is worth ~$3/mo at post-G1 volume. That is not worth buying
a 42-second decision latency. **Use GLM-5.3 non-reasoning.**

Whichever model lands, still do this before promoting it:
1. Confirm the signal-engine timeout comfortably exceeds observed p95 TTFT **and** stays
   well under `signal_interval`.
2. Cap `max_tokens` (currently 3000) and tighten the prompt — we need a small JSON object,
   and on reasoning models those tokens are billed as output.
3. Measure JSON parse-failure rate across ≥200 calls. GLM supports JSON mode but **not
   schema enforcement**, so malformed output is a live risk that `_extract_json` would be
   quietly absorbing into `direction: "none"`.

### Known stale references (fix on next touch)

- `docs/BLUEPRINT.md:51` and `:369` say `openai/kimi-k2.6` — **wrong vendor prefix**. The working ID is `moonshotai/kimi-k2.6`.
- `agent/burt.py:26` — `BURT_MODEL` constant, not hot-reloadable.
- `agent/signal_engine.py:252` — `chat()` hardcodes the model, ignoring config.
- `agent/signal_engine.py:230` — embedding model hardcoded.
- `scripts/test_openrouter.py`, `test_strong_prompt.py`, `test_signal_response.py` — hardcoded.

---

## 7. Migration checklist

**Run-plane config (do this once):**

- [ ] Add `critic_model`, `burt_model`, `embedding_model`, `consolidation_model` to `config.py` alongside the existing `signal_model`
- [ ] Seed all five in `agent_config` via `scripts/setup_db.py` so they are hot-reloadable from the UI and from Burt
- [ ] Replace `BURT_MODEL` and the two hardcodes in `signal_engine.py` with `cfg.<key>` lookups
- [ ] Point test scripts at `config.get_config().signal_model`
- [ ] Confirm `signals.model` is populated on **every** row, including fallback/error paths — it is the A/B join key
- [ ] Add a `model` dimension to `analytics.py` slices (win rate, expectancy, confidence calibration, **and cost per signal**)

**Provider routing:**

- [ ] Add a `provider` config key per role (`openrouter` | `zai` | `moonshot` | `ollama`) so a role can bypass OpenRouter for a cheaper direct rate. All four are OpenAI-compatible, so this is a base-URL + key swap, not a client rewrite.
- [ ] Add `OLLAMA_BASE_URL` to `.env.example` for the local fallback path
- [ ] Make the timeout per-role, not a single global 30s (see §6.3)
- [ ] Add a startup assertion: **fail loudly if a run-plane key looks like a seat-subscription credential.** Enforces §2's hard rule in code rather than in prose.

---

## 8. Changing a model — procedure

**Signal / critic / Burt / consolidation models:**

1. Set the candidate on `critic_model` or `consolidation_model` first — the low-blast-radius roles.
2. Run ≥200 logged signals (or a backtest replay where the model is the only variable).
3. Compare against incumbent on: win rate, expectancy in R, **confidence calibration** (does 0.8 actually win ~80%?), JSON parse-failure rate, p95 latency, cost per signal.
4. Promote only on a win in expectancy *and* no regression in parse-failure rate.
5. Record the swap in §9 with the numbers that justified it.

**Embedding model — different rules.** Dimension or family changes silently corrupt
semantic memory: old vectors and new vectors are not comparable, so `search_memories`
starts returning nonsense and the C5 memory loop quietly feeds garbage into prompts.
To change it you must re-embed the full `memories` table in one migration, or version
the column and dual-write during cutover. Do not swap it casually. (This applies equally
to moving embeddings onto local Ollama — most local embedders are 768-dim, not 1536.)

**Build-plane models** need none of this ceremony — swap freely. The only standing
constraint is §4's mandatory-review paths: risk-critical code gets Claude eyes regardless
of which model wrote it.

---

## 9. Change log

| Date | Plane | Change | Justification |
|---|---|---|---|
| 2026-08-27 | — | File created. Run-plane roles at their as-built pins; nothing swapped yet. | — |
| 2026-08-27 | Build | Documented the Claude-plans / GLM-builds / Claude-reviews workflow and the mandatory-review path list. | Existing practice; written down so it survives context resets and so review effort is spent where money is at risk. |
| 2026-08-27 | Run | Added §2 build/run separation and the seat-subscription prohibition. | Volume arithmetic: bot needs 1,440–4,320 calls/day; OpenCode Go allows ~528 Kimi K3 calls/day, Claude Pro ~45 per 5h window. |
| 2026-08-27 | Run | **Reversed the GLM-5.3-Flash signal-model recommendation → GLM-5.3 non-reasoning.** | Flash TTFT is ~42s vs GLM-5.3's 1.60s. The earlier pick was made on token price before §6.1's volume math showed cost stops binding post-G1; latency is the real constraint for a trigger-confirmer. |
| 2026-08-27 | Run | Flagged `kimi-k2.6` (current `signal_model` + `burt_model` pin) as **deprecated**. | Provider-scheduled withdrawal risk; migration is now maintenance, not optimization. |
| 2026-08-27 | Run | **Test-run routing: signal/critic/burt/consolidation → `glm-5.2:cloud` via local Ollama daemon** (`http://localhost:11434/v1`, no API key — the daemon proxies with the signed-in account). Embeddings stay on OpenRouter. | Measured on this machine: **3.2–3.7s round trip, 3/3 valid JSON**, and a full `SignalEngine` call logged `model='glm-5.2:cloud'`, `parse_failed=False`. Comfortably inside the 30s signal timeout, and it retires the deprecated K2.6 pin for the signal path. |
| 2026-08-27 | Both | **Kimi K3 must be used with Ollama Cloud** everywhere it appears (build-plane escalation, run-plane critic, consolidation). | Provider routing standardization: Kimi K3 is served via Ollama Cloud; OpenCode Go reference removed. |

---

## Sources

**Subscriptions & pricing**
- [Ollama pricing](https://ollama.com/pricing) · [Ollama Cloud plans 2026](https://devtoolhub.com/ollama-cloud-free-vs-pro-limits-pricing-2026/)
- [OpenCode Go review (limits + model list)](https://www.bitdoze.com/opencode-go-plan/) · [OpenCode plans & pricing](https://www.codeagentswarm.com/en/guides/opencode-plans-and-pricing)
- [Claude Code usage limits 2026](https://www.morphllm.com/claude-code-usage-limits) · [Claude Code pricing 2026](https://www.cloudzero.com/blog/claude-code-pricing/)
- [Z.ai pricing](https://docs.z.ai/guides/overview/pricing) · [GLM-5.3 vs Kimi K3 latency](https://artificialanalysis.ai/models/comparisons/glm-5-3-vs-kimi-k3) · [Kimi K2.6 deprecation](https://www.nxcode.io/resources/news/kimi-k2-5-pricing-plans-api-costs-2026) · [GLM-5.3 pricing](https://glm5.app/blog/glm-5-3-pricing) · [GLM-5.3-Flash providers & speed](https://artificialanalysis.ai/models/glm-5-3-flash/providers)
- [Kimi K3 API pricing](https://benchlm.ai/moonshot/api-pricing) · [Kimi K3 on OpenRouter](https://openrouter.ai/moonshotai/kimi-k3)

**Trading-specific model evidence**
- [TradeRank Arena](https://www.traderank.ai/llm-trading-benchmark) · [Best AI models for crypto trading 2026](https://www.traderank.ai/blog/best-ai-models-for-crypto-trading-2026)
- [nof1 Alpha Arena](https://nof1.ai/) · [Season 1 results](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/) · [Protos](https://protos.com/llm-crypto-trading-contest-finds-llms-cant-trade-crypto/)
- [MQL5: LLM + trading architecture that works, 2026](https://www.mql5.com/en/blogs/post/769403)
