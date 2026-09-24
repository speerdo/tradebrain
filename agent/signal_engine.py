"""
Signal Engine — Kimi K2.6 via OpenRouter

Builds structured prompts from strategy + indicators, sends to LLM,
returns SignalResult. Everything is logged to DB regardless of outcome.
"""

import json
import time
from typing import Any

import httpx
from loguru import logger

import config
from agent import llm_router
from strategies.base import BaseStrategy, SignalResult
from agent.database import get_db

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

SYSTEM_PROMPT = """\
You are TradeBrain, an expert crypto futures trading signal evaluator.
You analyze live technical indicator data and determine whether a trading
signal meets the defined strategy criteria for a leveraged perpetual-futures
position on Coinbase Financial Markets (CFM).

Rules:
- Only signal "long" or "short" when ALL required conditions are clearly met
- When in doubt, return "none" — missing a trade is better than a bad trade
- Be concise in reasoning — max 2 sentences
- ALWAYS return valid JSON only, no markdown, no preamble
- Never recommend a trade without clear technical justification from the data provided
- Consider funding rate when provided — extreme positive funding favors shorts, extreme negative favors longs
"""


class SignalEngine:
    """Handles OpenRouter API calls for signal evaluation."""

    def __init__(self):
        self.cfg = config.get_config()
        # Client timeout is an upper bound; the per-role timeout from
        # llm_router overrides it on each request (MODELS.md §6.3).
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._available = self._signal_route_usable()
        if not self._available:
            logger.warning("SignalEngine: no usable run-plane credential for the signal role — evaluation disabled")
        self._memory_engine = None  # wired in by main.py for C5 memory loop
        # (symbol, strategy) -> bar_time of the last bar a directional rule
        # signal was handed on for. Bar-based strategies act once per bar.
        self._consumed_bars: dict[tuple[str, str], float] = {}

    def _signal_route_usable(self) -> bool:
        """Signal evaluation works when its route has a key, or routes to
        key-less local Ollama."""
        try:
            r = llm_router.route(self.cfg, "signal")
        except ValueError:
            return False
        return bool(r.api_key) or r.provider == "ollama"

    def set_memory_engine(self, memory_engine) -> None:
        """Wire in the MemoryEngine for similar-setup retrieval (C5)."""
        self._memory_engine = memory_engine

    async def _get_similar_setups(self, symbol: str, strategy: str,
                                  indicators: dict, regime: dict | None) -> str:
        """
        Retrieve the 3 most similar past setups from semantic memory (C5).
        Returns a prompt fragment string (empty if none found).
        """
        if self._memory_engine is None:
            return ""
        i15 = indicators.get("15m", {})
        regime_label = regime.get("regime", "unknown") if regime else "unknown"
        query = (
            f"{strategy} {symbol} regime={regime_label} "
            f"rsi={i15.get('rsi')} macd_hist={i15.get('macd_hist')} "
            f"price_vs_ema50={indicators.get('1h', {}).get('price_vs_ema50')}"
        )
        try:
            memories = await self._memory_engine.search_memories(query, limit=3, importance_threshold=0.5)
        except Exception as exc:
            logger.warning(f"Memory retrieval failed: {exc}")
            return ""
        if not memories:
            return ""
        lines = ["\nSIMILAR PAST SETUPS (from memory):"]
        for m in memories:
            content = m.get("content", "")[:150]
            lines.append(f"- {content}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    BAR_SECONDS = 4 * 3600      # bar_time is the start of a 4h bar
    MAX_SIGNAL_AGE_H = 1.0

    async def evaluate(self, symbol: str, strategy: BaseStrategy,
                       indicators: dict, regime: dict | None = None,
                       extra_context: str = "") -> SignalResult:
        """
        Rules first, LLM as a veto.

        `strategy.check_entry` — the same deterministic rule the backtester
        replays — decides whether there is a setup at all. Only when it fires
        is the LLM asked, and the LLM can only agree (possibly lowering the
        confidence) or veto; it can never originate a trade. Before
        2026-09-24 this ran the other way round: the LLM was called on every
        tick for every symbol (~400 calls/day on ETH alone) and picked the
        direction, with the rules only able to veto it — so no backtest was
        measuring the system that traded.

        An LLM error or unparseable reply fails OPEN (the rule's signal
        stands): the rule is the tested part, the LLM filter is not. Set
        config.llm_veto=false to skip the LLM entirely.

        `extra_context` is a pre-formatted string (derivatives + sentiment blocks)
        appended to the prompt after the strategy fragment.
        """
        rule = strategy.check_entry(indicators)
        if rule.entry_price is None:
            rule.entry_price = indicators.get("15m", {}).get("price")
        if rule.direction == "none":
            rule.skip_reason = "No rule setup"
            rule.signal_id = await self._log_signal(symbol, strategy.name, rule, indicators, model=None)
            return rule

        if rule.bar_time is not None:
            key = (symbol, strategy.name)
            # A breakout bar stays "true" for the 4 hours until the next one
            # closes; act on it once (the backtest enters at the bar close),
            # not on every tick — and not at all if we only noticed it late
            # (e.g. after a restart), when the fill would be hours off the
            # price that was tested.
            if self._consumed_bars.get(key) == rule.bar_time:
                return SignalResult(direction="none", entry_price=rule.entry_price,
                                    bar_time=rule.bar_time,
                                    skip_reason="No rule setup (bar already acted on)")
            self._consumed_bars[key] = rule.bar_time
            age_h = (time.time() - (rule.bar_time + self.BAR_SECONDS)) / 3600
            if age_h > self.MAX_SIGNAL_AGE_H:
                rule.direction = "none"
                rule.skip_reason = f"No rule setup (breakout bar closed {age_h:.1f}h ago — stale)"
                rule.signal_id = await self._log_signal(symbol, strategy.name, rule, indicators, model=None)
                return rule

        if not (self._available and self.cfg.llm_veto):
            rule.signal_id = await self._log_signal(symbol, strategy.name, rule, indicators, model=None)
            return rule

        try:
            llm_response = await self._call_llm(symbol, strategy, indicators, regime, extra_context)
        except Exception as exc:
            logger.error(f"LLM call failed for {symbol}: {exc}")
            llm_response = {
                "direction": "none", "confidence": 0.0,
                "parse_failed": True,
                "raw_response_snippet": f"LLM error: {exc}"[:200],
            }

        if llm_response.get("parse_failed"):
            logger.warning(
                f"{symbol} [{strategy.name}]: LLM veto unavailable (error/unparseable) — "
                f"rule signal {rule.direction} stands"
            )
            rule.parse_failed = True
            rule.raw_response_snippet = str(llm_response.get("raw_response_snippet") or "")
            rule.signal_id = await self._log_signal(symbol, strategy.name, rule, indicators)
            return rule

        # The strategy re-applies its hard gates to whatever the LLM said.
        llm_sig = strategy.parse_response(llm_response, indicators)
        if llm_sig.direction != rule.direction:
            signal = SignalResult(
                direction="none", confidence=llm_sig.confidence,
                reasoning=llm_sig.reasoning or rule.reasoning,
                entry_price=rule.entry_price, bar_time=rule.bar_time,
                skip_reason=f"LLM veto of rule {rule.direction}",
            )
            logger.info(f"{symbol} [{strategy.name}]: LLM vetoed rule {rule.direction} — {signal.reasoning[:150]}")
        else:
            signal = rule
            signal.confidence = llm_sig.confidence
            signal.reasoning = llm_sig.reasoning or rule.reasoning
            signal.invalidation = llm_sig.invalidation
        signal.signal_id = await self._log_signal(symbol, strategy.name, signal, indicators)
        return signal

    async def _call_llm(self, symbol: str, strategy: BaseStrategy,
                        indicators: dict, regime: dict | None = None,
                        extra_context: str = "") -> dict:
        """POST to OpenRouter and parse the JSON response."""
        prompt = strategy.build_prompt(indicators, symbol, regime=regime)
        # C5: append similar past setups from memory
        similar = await self._get_similar_setups(
            symbol, strategy.name, indicators, regime
        )
        if similar:
            prompt = prompt + similar
        # C3/C4: append derivatives + sentiment context
        if extra_context:
            prompt = prompt + extra_context

        route = llm_router.route(self.cfg, "signal")
        payload = {
            "model": route.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 3000,
            # Note: response_format json_object can cause content=None with some models
            # We parse JSON manually from content instead
        }

        if route.reasoning_effort:
            payload["reasoning"] = {"effort": route.reasoning_effort}

        resp = await self._client.post(
            f"{route.base_url}/chat/completions",
            headers=route.headers, json=payload, timeout=route.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data["choices"][0]["message"]
        content = message.get("content")

        # Fallback: if content is None but reasoning exists, extract JSON from reasoning
        if content is None:
            content = message.get("reasoning", "")

        return self._extract_json(content)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON object from text, handling markdown code blocks.

        Every fallback path returns `parse_failed: True` so the signals table
        can separate malformed LLM output from a genuine no-signal.
        """
        if not text:
            return {"direction": "none", "confidence": 0.0, "parse_failed": True}

        text = text.strip()

        # Try direct parse first. Valid JSON that isn't an object (a bare list
        # or number) is still unusable — fall through to the extractors rather
        # than returning something `.get()` will blow up on downstream.
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                result["parse_failed"] = False
                return result
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    result["parse_failed"] = False
                return result
            except json.JSONDecodeError:
                pass

        # Try finding first JSON object in text
        pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    result["parse_failed"] = False
                return result
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not extract JSON from LLM response: {text[:200]}")
        return {
            "direction": "none", "confidence": 0.0,
            "parse_failed": True,
            "raw_response_snippet": text[:200],
        }

    _MODEL_UNSET = object()

    async def _log_signal(self, symbol: str, strategy_name: str,
                          signal: SignalResult, indicators: dict,
                          model: Any = _MODEL_UNSET) -> int | None:
        """Write the signals row; returns its id so main.py can record the
        outcome. `model=None` marks a row the LLM never saw."""
        i15 = indicators.get("15m", {})
        if model is self._MODEL_UNSET:
            model = self.cfg.signal_model if self._available else None
        try:
            db = await get_db()
            return await db.log_signal({
                "symbol": symbol,
                "direction": signal.direction,
                "strategy": strategy_name,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "acted_on": False,
                "skip_reason": signal.skip_reason,
                "rsi_15m": i15.get("rsi"),
                "macd_hist_15m": i15.get("macd_hist"),
                "atr_15m": i15.get("atr"),
                "price": i15.get("price"),
                "model": model,
                "parse_failed": signal.parse_failed,
                "raw_response_snippet": signal.raw_response_snippet or None,
            })
        except Exception as exc:
            logger.warning(f"Failed to log signal: {exc}")
            return None

    # ------------------------------------------------------------------
    # Embeddings (for Burt semantic memory)
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embeddings for semantic memory via the `embedding` role route."""
        route = llm_router.route(self.cfg, "embedding")
        payload = {
            "model": route.model,
            "input": text,
        }
        resp = await self._client.post(
            f"{route.base_url}/embeddings",
            headers=route.headers, json=payload, timeout=route.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # Burt conversation
    # ------------------------------------------------------------------

    async def chat(self, messages: list[dict], role: str = "burt") -> str:
        """Generic chat completion routed by role (`burt` or `consolidation`)."""
        route = llm_router.route(self.cfg, role)
        payload = {
            "model": route.model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 512,
        }
        resp = await self._client.post(
            f"{route.base_url}/chat/completions",
            headers=route.headers, json=payload, timeout=route.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
