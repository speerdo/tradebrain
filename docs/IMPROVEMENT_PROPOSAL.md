# TradeBrain Improvement Proposal

> Research-driven roadmap for making TradeBrain/Burt more profitable, borrowing the
> best ideas from the leading open-source trading bots and LLM trading frameworks.
> Written 2026-07-12. Based on a survey of Freqtrade, Hummingbot, Jesse, Passivbot,
> OctoBot, TradingAgents, ai-hedge-fund, HedgeAgents, and the nof1 Alpha Arena
> LLM trading competitions.

---

## 1. Where TradeBrain stands today

Strong foundation, honest gaps:

**What we already have that most hobby bots don't:**
- Mandatory risk controls (ATR stops placed with entry, circuit breaker, sizing from $-at-risk)
- Live-tunable config (UI + Burt), full signal/trade audit trail in Postgres
- Semantic memory infrastructure (pgvector) and a personality layer with DB access
- A screener that adapts the watchlist instead of trading a fixed pair list

**What the successful bots have that we don't:**

| Gap | Consequence |
|---|---|
| **No backtesting at all** | We cannot measure whether any strategy/knob change is an improvement. Every tuning decision is a guess. |
| **No exit management** | Fixed SL/TP only. No trailing stop, no breakeven move, no partial profits, no time-based exit. Winners give back profit; stale positions tie up margin. |
| **No trade-outcome protections** | Circuit breaker only counts realized daily loss. Nothing stops the bot from taking the same losing setup 5x in a row. |
| **`regime.py` and `memory_engine.py` are not wired into the trade loop** | The LLM evaluates each symbol in a vacuum: no market regime, no past-outcome context. We built the organs but never connected them. |
| **Single LLM call = the whole brain** | One Kimi call on 15m/1h indicators decides everything. No second opinion, no sentiment/news/funding context, no self-consistency check. |
| **Market orders only** | We pay taker fees on every entry. On a 15m strategy with 2% stops, fee drag is a meaningful % of expected edge. |
| **No portfolio-level risk** | 5 concurrent alt positions are effectively one levered BTC position (alts are highly correlated). Per-trade risk understates true exposure. |
| **No performance analytics** | No equity curve, Sharpe, win-rate-by-strategy/regime. Burt can't learn what's working because nothing computes it. |

---

## 2. What the research says

### 2.1 Freqtrade (~25k stars) — the risk-plumbing gold standard
The most battle-tested retail bot. The features that map directly onto us:

- **Protections** ([docs](https://www.freqtrade.io/en/stable/includes/protections/)):
  - *StoplossGuard* — if N stop-outs occur within a lookback window, pause trading (globally or per pair) for a cooldown period.
  - *MaxDrawdown protection* — if trailing realized drawdown over a window exceeds X%, stop trading for a duration.
  - *CooldownPeriod* — don't immediately re-enter a pair after closing it.
  - *LowProfitPairs* — lock pairs that keep losing.
- **Trailing + custom stoploss** ([docs](https://www.freqtrade.io/en/stable/stoploss/)): hard stop as floor, trailing activates after profit threshold, stop price can only ratchet in the trade's favor.
- **FreqAI** ([docs](https://www.freqtrade.io/en/stable/freqai/)): continuously-retrained ML models (regressors/classifiers/RL) that adapt to shifting markets — the key insight is *scheduled re-fitting to recent data*, not any specific model.

### 2.2 Jesse — backtest honesty
Jesse's reputation rests on **zero look-ahead bias** backtesting. Lesson: an honest backtest engine is the single highest-leverage feature — a bot without one is flying blind, and a bot with a dishonest one is worse.

### 2.3 Passivbot & Hummingbot — a different edge in chop
Passivbot ([repo](https://github.com/enarjord/passivbot)) is a contrarian grid market-maker: it doesn't predict, it absorbs volatility with limit-order grids and takes small profits on reversals. Hummingbot specializes in market making/spread capture. Lesson: **directional trend strategies and mean-absorbing grid strategies are profitable in opposite regimes.** Our current strategies are all directional; in ranging markets we either sit idle or get chopped up. (Also a caution: public Passivbot results are mixed — grid strategies carry tail risk in trends, so this is a later, carefully-capped addition.)

### 2.4 TradingAgents / ai-hedge-fund / HedgeAgents — multi-agent LLM design
[TradingAgents](https://github.com/TauricResearch/TradingAgents) models a trading desk: analyst agents (technical, sentiment, news, fundamentals) → bull/bear researcher debate → trader → risk manager. Documented run: ~7% in 30 days vs S&P's 4.5%, but with 22% drawdowns and no repeatability guarantee. Lessons:
- Separating *analysis*, *debate*, and *risk veto* into distinct LLM roles improves decision quality over one mega-prompt.
- It's a research tool, not production — we should steal the **structure**, not the codebase.

### 2.5 nof1 Alpha Arena — the sobering LLM benchmark
Six frontier LLMs each traded $10k of real money on Hyperliquid perps ([nof1.ai](https://nof1.ai/), [results analysis](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/)). Season 1: **four of six lost money** (GPT-5 lost $6,267; Claude lost $3,081); only Qwen3 Max and DeepSeek profited. Hard-won lessons that apply directly to us:
- **Raw LLM judgment is not an edge.** The LLM should be a *filter and context-synthesizer* on top of mechanical rules, not the sole alpha source.
- **Every prompt rule needs a server-side check.** "LLMs are suggestible, not reliable — treat prompt instructions as guidelines and server validation as law." We already do this (risk manager gates the LLM) — keep extending it.
- **Lower trade frequency won.** Trade count correlated *negatively* with returns. A churn limit beats frantic activity.
- **Cascading multi-timeframe context** (daily → 4h → 1h → 15m as a decision tree) produced more consistent behavior than flat indicator dumps.

### 2.6 Free context data we're not using
- **Fear & Greed Index** — free JSON API, no key: `https://api.alternative.me/fng/` ([alternative.me](https://alternative.me/crypto/fear-and-greed-index/))
- **CryptoPanic** — free-tier news + sentiment API with per-currency filters ([developers](https://cryptopanic.com/developers/))
- **Funding rate & OI history** — we already fetch snapshots from Coinbase; we're not tracking *changes* (OI rising + price falling = shorts in control, etc.)

### 2.7 Validation methodology
[vectorbt](https://github.com/polakowo/vectorbt) can run thousands of parameter combos in seconds on pandas/NumPy. Walk-forward optimization (optimize on window N, test blind on window N+1, roll forward) is the standard defense against overfitting; an out-of-sample/in-sample profit ratio below ~0.5 means the system memorized history.

---

## 3. The proposal

Ordered by expected impact per unit of effort. The theme: **measure first, defend second, enrich third, get clever last.**

### Phase A — Measurement (do this before anything else)

**A1. Backtesting engine.**
The single biggest gap. Plan:
- Make each strategy **dual-mode**: extract the deterministic entry/exit rules that already exist implicitly in the prompts (RSI cross + MACD flip + EMA filter, etc.) into pure-Python `check_entry(indicators) -> SignalResult` methods on `BaseStrategy`. The LLM remains a *live-only confidence overlay*; the mechanical rules are what we backtest.
- Build `backtest/engine.py`: replay historical Coinbase candles bar-by-bar through `indicator_engine` → strategy rules → `risk_manager.compute_stops`/`compute_position_size` → simulated fills with **fees + slippage + funding** modeled. Reuse the exact production code paths so backtest == live logic (Jesse's zero-look-ahead discipline: indicators may only see bars ≤ current index).
- Add `vectorbt` for fast parameter sweeps (ATR multiplier, RR, confidence threshold) and use walk-forward splits for any tuning.
- CLI: `python -m backtest.run --strategy rsi_macd --symbol BTC-PERP --days 180`.
- **Acceptance test for every future change:** does it improve out-of-sample results? If we can't answer that, we don't ship it.

**A2. Performance analytics.**
- New `analytics.py` + DB views: equity curve, max drawdown, Sharpe/Sortino, profit factor, win rate and expectancy **sliced by strategy, symbol, regime, hour-of-day, and confidence bucket**.
- Surface in the dashboard (equity curve tab) and give Burt a weekly self-review: "EMA pullback is 3W/9L in ranging regime — suggest disabling it there."

### Phase B — Defense (Freqtrade-style protections; cheap, high value)

**B1. Protections in `risk_manager.py`:**
- *StoplossGuard*: ≥3 stop-outs in 4h → global 2h cooldown.
- *Per-symbol cooldown*: 1h lockout after closing a position in a symbol (prevents revenge-trading the same chart).
- *LosingSymbolLock*: symbol with ≤ -2R cumulative over 7 days gets benched until the weekly review.
- *Churn limit*: max N new positions per day (Alpha Arena lesson — fewer trades won).
- All tunable from UI/Burt like existing knobs; all logged with skip reasons.

**B2. Exit management in `position_monitor.py`:**
- **Breakeven move**: at +1R, move stop to entry (+fees).
- **Trailing stop**: after +1.5R, trail by ATR × multiplier; stop only ever ratchets toward profit (Freqtrade semantics).
- **Partial take-profit**: close 50% at +1R, let the rest run to the trailing stop — converts the current binary win/lose into a fatter right tail.
- **Time-based exit**: 15m-strategy positions that go nowhere for 12h get closed (thesis expired; stop paying funding).
- Paper mode simulates all of this; live mode amends the reduce-only orders.

**B3. Portfolio-level risk in `risk_manager.py`:**
- Max concurrent positions (default 3) and max total open risk (e.g. 3% of balance across all positions).
- **Correlation cap**: treat all alts as ~1 BTC-beta position; cap net same-direction exposure (e.g. max 2 correlated longs).
- **Drawdown-scaled sizing**: risk-per-trade scales down 50% when trailing 7-day equity drawdown exceeds half the daily-loss limit — survive losing streaks with capital intact.

### Phase C — Context (make the LLM's input worth its judgment)

**C1. Wire in `regime.py` (it already exists!).**
Compute market regime (BTC trend/chop via ADX + EMA structure, BTC dominance direction, fear/greed) once per loop and:
- Inject it into every strategy prompt as the top-level context block.
- Gate strategies mechanically: momentum strategies disabled in chop, mean-reversion disabled in strong trends. (Server-side check, not just a prompt hint.)

**C2. Cascading multi-timeframe prompt.**
Restructure prompts from flat indicator dumps to a decision tree: `4h bias → 1h structure → 15m trigger`, requiring alignment. Add the 4h candle fetch (one extra API call per symbol).

**C3. Derivatives context.**
Track funding-rate and open-interest **deltas** per symbol in the DB (screener already fetches snapshots). Feed "funding 8h trend" and "OI change vs price change" into prompts, and add a server-side rule: don't open positions that pay > X bps/day funding against us.

**C4. Sentiment & news feed.**
- Poll alternative.me Fear & Greed (1 req/hour, free) → regime block.
- Poll CryptoPanic free tier for watchlist currencies → one-line headline digest in the prompt + a "news veto" (skip entries within N minutes of high-panic news for that asset).
- Also lets Burt answer "why did ETH just dump?" — a personality win, not just an alpha win.

**C5. Close the memory loop.**
`memory_engine` already stores trade outcomes with embeddings. At signal time, retrieve the 3 most similar past setups (same symbol/strategy/regime) and append their outcomes to the prompt: *"Similar setup on 2026-06-30: stopped out, -1R."* This is our unique asset — none of the mainstream bots have it — and it's currently dark.

### Phase D — Intelligence (only after A–C prove out)

**D1. Two-stage signal evaluation (TradingAgents-lite).**
Full 7-agent debate is expensive and unproven. A cheap version:
- Stage 1 (current call) produces a candidate signal.
- If confidence ≥ threshold, Stage 2 runs a **risk-critic prompt**: a second LLM call whose only job is to find reasons to veto (regime conflict, funding cost, similar-setup losses, news risk). Veto or size-down on disagreement.
- ~2x LLM cost only on trade candidates (a few calls/day), not on every evaluation.

**D2. Model routing & A/B via OpenRouter.**
We're one config key away from testing models. Alpha Arena showed big per-model dispersion (Qwen/DeepSeek profited while GPT-5 and Claude lost). Make `signal_model` a hot-reloadable config knob, log the model on every signal, and let the Phase A analytics score models against each other on *our* setups. Data decides, not leaderboard vibes.

**D3. Dynamic confidence threshold.**
Replace the static `min_confidence` with a per-strategy threshold nudged by rolling expectancy (Kelly-lite): strategies on a hot streak in the current regime get a slightly lower bar, cold ones a higher bar. Bounded (e.g. ±0.1 around the user's setting) and fully logged.

**D4. FreqAI-inspired adaptive layer (stretch).**
A small gradient-boosted classifier retrained nightly on our own logged signals+outcomes, predicting P(win) from indicators+regime. Runs alongside the LLM; both must agree. Only worth doing once we have a few hundred logged outcomes — which Phase A analytics will tell us.

### Phase E — New strategy classes (diversification)

- **E1. Range/grid strategy** (Passivbot-inspired): in confirmed chop regimes only, tight capped grid on the top-1 screener symbol, hard kill-switch on regime flip to trend. Strictly capped margin (e.g. 10%).
- **E2. Funding-rate harvest**: when a symbol's funding is extreme and price structure agrees, take the funding-collecting side with a wide stop — a carry trade, not a prediction.

---

## 4. Suggested sequencing

| Milestone | Contents | Why first |
|---|---|---|
| **M1 (now)** | A1 backtester + A2 analytics | Everything else becomes measurable |
| **M2** | B1 protections + B2 exit management + B3 portfolio risk | Cuts left tail; provably safe changes |
| **M3** | C1 regime wiring + C2 cascading prompts + C3 derivatives deltas | Better inputs → better LLM output |
| **M4** | C4 sentiment/news + C5 memory loop | Unique differentiators |
| **M5** | D1 risk critic + D2 model A/B + D3 dynamic threshold | Intelligence, now measurable |
| **M6** | D4 ML layer, E1/E2 new strategies | Diversification, gated on data |

Each milestone must beat the previous configuration **out-of-sample in the backtester and/or over ≥2 weeks of paper trading** before the next one starts. That discipline — not any single feature — is what separates the profitable bots from the graveyard of GitHub trading projects.

## 5. A note on "most profitable in the world"

The honest research finding: most LLMs *lost money* trading real capital in Alpha Arena, and most public bots (grid included) underperform buy-and-hold in bull markets. What actually compounds:
1. **Not dying** — protections, portfolio caps, drawdown scaling (Phase B).
2. **Knowing your edge** — backtest + analytics (Phase A).
3. **Better information, cheaper execution** — context and fees (Phase C).
4. **Adaptation** — memory, regime gating, retraining (Phases C–D).

TradeBrain's genuine differentiators — a persistent outcome memory, a personality that can explain itself, and live-tunable everything — sit on top of that stack, not instead of it.

---

## Sources

- [Freqtrade Protections](https://www.freqtrade.io/en/stable/includes/protections/) · [Stoploss/Trailing](https://www.freqtrade.io/en/stable/stoploss/) · [FreqAI](https://www.freqtrade.io/en/stable/freqai/)
- [Gainium: Best Open Source Crypto Bots 2026](https://gainium.io/best/open-source) · [CoinCodeCap comparison](https://coincodecap.com/open-source-trading-bots-on-github)
- [TradingAgents (TauricResearch)](https://github.com/TauricResearch/TradingAgents) · [paper site](https://tradingagents-ai.github.io/) · [HedgeAgents](https://hedgeagents.github.io/) · [ai-hedge-fund overview](https://ultralab.tw/en/blog/ai-finance-github-projects-2026)
- [nof1 Alpha Arena](https://nof1.ai/) · [Season 1 results analysis](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/) · [Protos: LLMs can't trade crypto](https://protos.com/llm-crypto-trading-contest-finds-llms-cant-trade-crypto/) · [System-prompt lessons](https://medium.com/@kojott/teaching-an-ai-to-trade-system-prompt-engineering-and-performance-monitoring-9d55258cca06) · [TradeRank prompt engineering](https://www.traderank.ai/blog/ai-trading-prompts-engineering)
- [Passivbot](https://github.com/enarjord/passivbot) · [Passivbot review (mixed results)](https://gainium.io/review/passivbot)
- [vectorbt](https://github.com/polakowo/vectorbt) · [Walk-forward testing](https://www.alphanova.tech/blog/walk-forward-test)
- [alternative.me Fear & Greed API](https://alternative.me/crypto/fear-and-greed-index/) · [CryptoPanic API](https://cryptopanic.com/developers/)
