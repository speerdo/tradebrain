"""
Signal Engine — Kimi K2.6 via OpenRouter

Builds structured prompts from strategy + indicators, sends to LLM,
returns SignalResult. Everything is logged to DB regardless of outcome.
"""

import json
from typing import Any

import httpx
from loguru import logger

import config
from strategies.base import BaseStrategy, SignalResult
from agent.database import get_db

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

SYSTEM_PROMPT = """\
You are TradeBrain, an expert crypto futures trading signal evaluator.
You analyze live technical indicator data and determine whether a trading
signal meets the defined strategy criteria for a leveraged position on Hyperliquid.

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
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._available = bool(self.cfg.openrouter_api_key)
        if not self._available:
            logger.warning("SignalEngine: OPENROUTER_API_KEY missing — signal evaluation disabled")
        self._memory_engine = None  # wired in by main.py for C5 memory loop

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

    async def evaluate(self, symbol: str, strategy: BaseStrategy,
                       indicators: dict, regime: dict | None = None,
                       extra_context: str = "") -> SignalResult:
        """
        Run full LLM evaluation for one symbol + strategy.
        Falls back to "none" signal if key missing / API error / parse failure.

        `extra_context` is a pre-formatted string (derivatives + sentiment blocks)
        appended to the prompt after the strategy fragment.
        """
        llm_response = {"direction": "none", "confidence": 0.0}

        if self._available:
            try:
                llm_response = await self._call_llm(symbol, strategy, indicators, regime, extra_context)
            except Exception as exc:
                logger.error(f"LLM call failed for {symbol}: {exc}")
                llm_response = {"direction": "none", "confidence": 0.0}

        # Strategy may apply its own hard gates on top of LLM output
        signal = strategy.parse_response(llm_response, indicators)

        # Override with LLM-derived entry price if strategy didn't provide one
        if signal.entry_price is None:
            signal.entry_price = indicators.get("15m", {}).get("price")

        # Log to DB
        await self._log_signal(symbol, strategy.name, signal, indicators)
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

        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/tradebrain",
            "X-Title": "TradeBrain",
        }

        payload = {
            "model": self.cfg.signal_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 3000,
            # Note: response_format json_object can cause content=None with some models
            # We parse JSON manually from content instead
        }

        resp = await self._client.post(
            OPENROUTER_URL, headers=headers, json=payload
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
        """Extract JSON object from text, handling markdown code blocks."""
        if not text:
            return {"direction": "none", "confidence": 0.0}

        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first JSON object in text
        pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not extract JSON from LLM response: {text[:200]}")
        return {"direction": "none", "confidence": 0.0}

    async def _log_signal(self, symbol: str, strategy_name: str,
                          signal: SignalResult, indicators: dict) -> None:
        i15 = indicators.get("15m", {})
        try:
            db = await get_db()
            await db.log_signal({
                "symbol": symbol,
                "direction": signal.direction,
                "strategy": strategy_name,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "acted_on": False,
                "skip_reason": "",
                "rsi_15m": i15.get("rsi"),
                "macd_hist_15m": i15.get("macd_hist"),
                "atr_15m": i15.get("atr"),
                "price": i15.get("price"),
                "model": self.cfg.signal_model if self._available else None,
            })
        except Exception as exc:
            logger.warning(f"Failed to log signal: {exc}")

    # ------------------------------------------------------------------
    # Embeddings (for Burt semantic memory)
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float]:
        """Generate 1536-dim embedding via OpenRouter (text-embedding-3-small)."""
        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/tradebrain",
            "X-Title": "TradeBrain",
        }
        payload = {
            "model": "openai/text-embedding-3-small",
            "input": text,
        }
        resp = await self._client.post(
            OPENROUTER_EMBEDDING_URL, headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # Burt conversation
    # ------------------------------------------------------------------

    async def chat(self, messages: list[dict]) -> str:
        """Generic chat completion for Burt personality responses."""
        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/tradebrain",
            "X-Title": "TradeBrain",
        }
        payload = {
            "model": "moonshotai/kimi-k2.6",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 512,
        }
        resp = await self._client.post(
            OPENROUTER_URL, headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
