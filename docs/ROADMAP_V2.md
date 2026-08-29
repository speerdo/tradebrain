# TradeBrain V2 — The Governor & Attentive Scanning

> Successor to `IMPROVEMENT_PROPOSAL.md` (2026-07-12), whose Phases **A, B, and C shipped**
> in commit `2309590`. This document covers what's left, re-researched for a bullish
> 2026-H2 regime. Written 2026-08-27.
>
> Model selection has moved to its own living doc: **`docs/MODELS.md`**.

---

## 0. Where we actually stand

Shipped and working (don't rebuild these):

- Backtest engine + `analytics.py` (Phase A)
- Protections: StoplossGuard, per-symbol cooldown, LosingSymbolLock, churn limit (B1)
- Exit management: breakeven, trailing, partials, time exit (B2)
- Portfolio risk: max concurrent 3, max total risk 3%, correlation cap, drawdown-scaled sizing (B3)
- Regime engine wired + mechanically gating strategies (C1)
- Cascading 4h→1h→15m prompts (C2), funding/OI deltas (C3), Fear & Greed + CryptoPanic + news veto (C4), semantic memory retrieval at signal time (C5)

That is a genuinely strong base. The remaining gaps are **allocation** and **attention** —
not more indicators.

---

## 1. The real "15 longs" problem (and it isn't what it looks like)

There are hard caps in `risk_manager.py` — `max_concurrent_positions=3` and
`max_correlated_directions=2` — and **in paper mode (the default) they work.**

> ⚠️ **Corrected 2026-08-27:** those caps are **paper-only**. `agent/executor.py:63`
> declares `self.live_positions` and nothing reads or writes it; every accessor
> (`get_open_positions`, `has_position`, `get_position`, `close_position`) reads
> `paper_positions`. With `PAPER_TRADING=false`, `check_portfolio_risk()` receives an
> empty list and **every portfolio cap passes unconditionally** — so the runaway-exposure
> scenario *is* reachable in live mode today. Fixed in `IMPLEMENTATION_PLAN_V2.md` **P0**,
> which blocks the Governor.

The allocation defect below is separate, and applies in both modes. In
`agent/main.py:_loop`, the watchlist is walked
**sequentially**, and `_evaluate` fires an order the instant a symbol passes:

```
for symbol in self.watchlist:      # screener rank order, fixed
    await self._evaluate(symbol)   # → enters immediately on pass
```

So in a pump, where every symbol throws a long signal, the caps fill on a
**first-come-first-served** basis. Burt takes the first two symbols in screener order —
not the two *best* setups. He commits his entire correlated-long budget to whatever
happened to be at index 0 and 1, then spends the rest of the pump locked out, watching
better setups he is no longer allowed to take.

Three compounding failure modes on top of that:

1. **Single-strategy tunnel vision.** `_loop` runs exactly one strategy (`cfg.strategy`).
   If the regime gate disables it, `for symbol in watchlist` is skipped entirely and Burt
   **does nothing at all** — no fallback strategy, no logging of the idle state. A
   mean-reversion strategy selected during a bull run means an idle bot.
2. **Static caps across regimes.** 3 positions / 3% risk is correct for chop and
   over-conservative for a confirmed risk-on trend. The budget should breathe with the regime.
3. **Naive risk summation.** `check_portfolio_risk` adds per-position dollar risk linearly.
   Three correlated alt longs are not 3× a position's risk — they are closer to one
   position at ~2.6× size. The correlation cap is a blunt count, not a risk adjustment.

**The fix is a Governor: an allocator that sits between signal generation and execution
and turns "first N signals win" into "best N signals win, within a regime-scaled,
correlation-adjusted risk budget."**

---

## 2. Phase F — The Governor (`agent/governor.py`)

The single highest-value change on this list. Everything else is downstream of it.

### F1. Collect-then-allocate loop

Restructure `_loop` from *evaluate-and-fire* to a two-pass cycle:

```
PASS 1 (gather):   evaluate every watchlist symbol → list[Candidate]   (no orders placed)
PASS 2 (allocate): Governor ranks candidates, computes the budget,
                   allocates, and emits 0..N orders
```

`Candidate` carries `symbol, direction, confidence, reasoning, entry, atr, strategy,
regime_fit, expectancy_prior, liquidity_score`.

This is the whole ballgame: it lets Burt see all his options before committing capital
to any of them.

### F2. Candidate scoring

Rank by a composite, not raw LLM confidence alone:

```
score = confidence
      × regime_fit            (strategy × current regime, 0..1)
      × expectancy_prior      (this strategy's rolling R-expectancy in this regime,
                               from analytics.py — cold-start to 1.0)
      × liquidity_score       (screener composite; penalize thin books)
      ÷ correlation_penalty   (1 + β to already-open exposure)
```

The `expectancy_prior` term is the point where the analytics we already compute finally
feed back into decisions. Right now `analytics.py` produces numbers nobody acts on.

### F3. Risk budget, not position count

Replace the hard `max_concurrent_positions` gate with a **portfolio heat budget**:

- Budget expressed as % of balance at risk, regime-scaled:
  - `risk-on` → 1.5× base budget
  - `mixed` / `chop` → 1.0×
  - `risk-off` → 0.5×, and longs require a materially higher confidence bar
  - `unknown` → 1.0× (fail open, consistent with the existing regime gate)
- **Correlation-adjusted heat.** Sum risk as `sqrt(Σ wᵢwⱼρᵢⱼ)` rather than `Σ wᵢ`, using
  a rolling correlation matrix of 1h returns over the watchlist. Alts vs BTC will land
  ~0.8–0.9; two uncorrelated bets should cost less budget than two correlated ones. This
  is what the count-based `max_correlated_directions` is crudely approximating today.
  > ⚠️ **Corrected 2026-08-27:** this originally said 30 days and "we already pull 1h
  > candles." Neither holds — `get_candles` is max 300/request with no pagination
  > (`coinbase_client.py:240`) and the loop passes no `start`/`end`, while 30d × 1h = 720
  > bars. **Start at 7 days (168 bars, one request/symbol)**; correlations among crypto
  > majors are stable enough that the window length won't materially move allocation.
- Keep an absolute position-count ceiling as a backstop — but let it be regime-scaled
  (e.g. 3 in chop, 5 in confirmed risk-on).
- Research anchor: retail guidance converges on 1–2% per trade, ≤5–10% per strategy, and
  **20–30% total capital in active bots**, with the rest held as recovery reserve
  ([Altrady](https://www.altrady.com/blog/crypto-bots/ai-crypto-trading-bot-risk-management)).

### F4. Entry spacing / anti-clustering

Minimum spacing between new entries (e.g. 2 bars, or 10 minutes) regardless of how many
candidates pass. Prevents the entire budget landing inside one 15m candle at the top of a
squeeze — the mechanical version of "don't chase."

### F5. Pyramiding (adding to winners only)

Bull-market upside that we currently have no way to capture: our exits are good but our
entries are one-and-done.

- Add only to positions already **≥ +1R** and still trending.
- Diminishing size: adds at 50%, then 25% of base
  ([TradersPost](https://blog.traderspost.io/article/pyramiding-trading-strategies-guide)).
- Total pyramid capped at ~150% of a normal position.
- Stop for the whole stack ratchets to the average entry on each add.
- **Never average down.** Hard rule, server-side.

### F6. Multi-strategy concurrency

Run *every* strategy whose `compatible_regimes` matches the current regime, and let the
Governor arbitrate between their candidates. Fixes the idle-bot failure mode and gives
`expectancy_prior` cross-strategy competition to rank against. `cfg.strategy` becomes a
default/preference rather than an exclusive selector.

### F7. Explainability

Governor writes a per-cycle decision record: every candidate, its score, the budget state,
and why each one was taken or passed. This is a Burt superpower — *"I saw four longs, took
SOL and ETH; passed AVAX because it's 0.91 correlated to SOL and I was out of heat"* — and
it doubles as the regulator-friendly audit trail flagged in `MODELS.md` §5.

---

## 3. Phase G — Attentive scanning

Today the bot is **REST-polling only** (no WebSocket anywhere in `coinbase_client.py`) on a
fixed 300s loop, against a 5-symbol watchlist refreshed every 4 hours. A breakout at t+10s
waits up to five minutes for a look. That is the "not attentive enough" problem.

### G1. Mechanical trigger gate → then LLM

`BaseStrategy.check_entry()` already exists (built for the backtester) and is pure Python.
Use it live as a **cheap precondition**: poll fast, run `check_entry` on every tick, and
only spend an LLM call when a mechanical trigger fires.

This inverts the cost curve — it makes Burt *more* attentive and *cheaper* at the same
time, because the expensive step stops firing on symbols that had no setup.

### G2. Tiered scan cadence

| Tier | Contents | Cadence | Work done |
|---|---|---|---|
| **Hot** | Top 3 by screener score + all open positions | 60s | Full indicator + `check_entry` |
| **Warm** | Rest of watchlist (widen to ~15) | 5 min | Full indicator + `check_entry` |
| **Cold** | Full tradeable universe | 4h (current) | Screener re-rank |

Widening the watchlist from 5 to ~15 is only affordable *because* of G1 — LLM cost is
driven by triggers, not by watchlist size.

### G3. WebSocket price feed

Subscribe to Coinbase Advanced's ticker/candle channels for hot-tier symbols. WebSocket
delivers in ~100ms vs ~1s for REST polling, without burning rate limit
([CoinGecko](https://www.coingecko.com/learn/top-5-best-crypto-websocket-apis)).

Two wins beyond scan latency: `position_monitor` stops polling marks every 30s (trailing
stops and breakeven moves get *much* tighter), and volume/price spikes become **events**
that can promote a symbol into the hot tier mid-cycle.

Keep REST as the fallback path — WS reconnect logic is where these systems break.

### G4. Event-driven promotion

Cheap detectors on the WS stream that promote a symbol to hot tier and force an immediate
evaluation: volume spike vs 20-bar SMA, range expansion vs ATR, funding flip, OI surge.
This is what makes Burt feel like he's *watching* rather than *checking*.

### G5. Don't let the regime gate zero him out

When the gate disables all strategies, log it explicitly, notify Burt, and surface it in
the UI. Silent idleness is currently indistinguishable from a healthy quiet market.

---

## 4. Phase H — Sharper decisions (carry-over from D/E, re-prioritized)

- **H1. Risk critic (was D1).** Second LLM call, *different model family*
  (see `MODELS.md` §6), whose only job is to veto. Runs on Governor-selected candidates
  only — a handful per day. Veto or size-down on disagreement.
- **H2. Model A/B (was D2).** Unblocked by the `MODELS.md` migration checklist. Given that
  only 39% of model-seasons are profitable and no model is reliably best, this must be
  decided on our own logged outcomes.
- **H3. Dynamic confidence threshold (was D3).** Per-strategy, nudged by rolling
  expectancy, bounded ±0.1. **Hard floor at 0.60** — sub-0.55 LLM decisions won 48.3%
  across 8,400 calls, below breakeven ([MQL5](https://www.mql5.com/en/blogs/post/769403)).
- **H4. Account-equity MaxDrawdown protection.** Freqtrade 2026.2 added a drawdown mode
  measured on the **account equity curve** (starting balance + cumulative absolute profit)
  and deprecated the old trade-relative calculation
  ([Freqtrade](https://www.freqtrade.io/en/stable/plugins/)). Our `get_drawdown_scale()`
  measures peak-to-trough on closed-trade PnL only — it ignores open-position drawdown.
  Move it to account equity.
- **H5. Maker entries.** We are market-order-only, paying taker on every entry. Post-only
  limit at the trigger price with a short chase-then-cancel window. On a 15m strategy with
  2% stops, fee drag is a real fraction of expected edge.
- **H6. Sub-account isolation.** Common Freqtrade practice: cap what the bot can reach
  ([Gainium](https://gainium.io/review/freqtrade)). Relevant before going live.

Deferred, unchanged from the original proposal: **D4** (ML win-probability layer, gated on
data volume), **E1/E2** (grid + funding-harvest strategies).

---

## 5. Sequencing

> Step-by-step execution detail lives in **`docs/IMPLEMENTATION_PLAN_V2.md`**, which is
> authoritative on ordering. The table below is kept in sync with it.

| Milestone | Contents | Gate to advance |
|---|---|---|
| **P0–P5** | Pre-flight. **P0 is a blocker:** `executor.live_positions` is declared and never used, so the portfolio caps are paper-only and the Governor would be a silent no-op in live mode. Plus parse-failure instrumentation, mandatory `check_entry`, model-config plumbing, deprecated-pin swap, baseline capture. | Portfolio caps demonstrably reject a 4th position with `PAPER_TRADING=false`. |
| **M7** | G1 + G2 + G5 (trigger gate + tiered cadence + idle surfacing) | LLM calls/day **down**, opportunities seen **up**. Both measured against the P5 baseline. |
| **M8** | F1 + F2 + F3 (Governor core: collect → rank → budget) | Backtest: same signals, better allocation. Compare vs current first-come-first-served on the same historical window. |
| **M9** | F6 + F4 + F7 (multi-strategy, spacing, explainability) | No idle-regime gaps; Burt can narrate every pass. |
| **M10** | G3 + G4 (WebSocket + event promotion) | Reconnect survives a forced disconnect; tighter trailing stops show up in backtest exits. |
| **M11** | H1 + H3 + H4 + H5 | Each beats its predecessor out-of-sample. |
| **M12** | F5 pyramiding | **Last.** It increases exposure — it only ships once the Governor's budget math is trusted. |

The discipline from the original proposal still stands and still matters more than any
item on this list: **each milestone must beat the previous configuration out-of-sample in
the backtester, or over ≥2 weeks of paper trading, before the next one starts.**

---

## 6. The bull-market caveat

Worth stating plainly, because it's the trap this project is walking into by restarting now:
in a strong bull market, most bots underperform buy-and-hold, and the ones that blow up do
it by over-levering into a euphoric tape. The Governor's job is *not* to let Burt express
more bullishness. It is to make sure that when he is bullish, the exposure he takes is his
**best** exposure, correctly sized for how correlated it all is — and that he still has
capital when the regime turns.

---

## Sources

- [Freqtrade Plugins/Protections](https://www.freqtrade.io/en/stable/plugins/) · [Releases](https://github.com/freqtrade/freqtrade/releases) · [Gainium review](https://gainium.io/review/freqtrade)
- [MQL5: LLM + trading architecture that works, 2026](https://www.mql5.com/en/blogs/post/769403)
- [Altrady: AI crypto bot risk management 2026](https://www.altrady.com/blog/crypto-bots/ai-crypto-trading-bot-risk-management)
- [TradersPost: pyramiding strategies](https://blog.traderspost.io/article/pyramiding-trading-strategies-guide) · [QuantVPS: trading bot strategies 2026](https://www.quantvps.com/blog/trading-bot-strategies)
- [CoinGecko: best crypto WebSocket APIs 2026](https://www.coingecko.com/learn/top-5-best-crypto-websocket-apis) · [QVeris: WebSocket for AI agents](https://qveris.ai/guides/crypto-websocket-api-for-ai-agents/)
- [TradeRank Arena](https://www.traderank.ai/llm-trading-benchmark) · [nof1 Alpha Arena](https://nof1.ai/) · [Protos](https://protos.com/llm-crypto-trading-contest-finds-llms-cant-trade-crypto/)
