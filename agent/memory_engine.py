"""
Memory Engine — Semantic memory + RAG for Burt.

Handles embedding generation, memory CRUD, and nightly consolidation.
"""

from datetime import date
from typing import Any

from loguru import logger

import config
from agent.database import get_db
from agent.signal_engine import SignalEngine


class MemoryEngine:
    """Manages Burt's semantic memory using pgvector + OpenRouter embeddings."""

    def __init__(self, signal_engine: SignalEngine | None = None):
        self.cfg = config.get_config()
        self._signal_engine = signal_engine

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Generate embedding via OpenRouter."""
        if self._signal_engine is None:
            logger.warning("No SignalEngine available for embeddings")
            return None
        try:
            return await self._signal_engine.get_embedding(text)
        except Exception as exc:
            logger.error(f"Embedding failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    async def store_memory(
        self,
        content: str,
        memory_type: str,  # 'lesson' | 'preference' | 'observation' | 'feedback'
        source: str = "",  # 'trade_outcome' | 'user_message' | 'consolidation' | 'pattern'
        symbol: str = "",
        strategy: str = "",
        importance: float = 0.5,
    ) -> int | None:
        """Embed and store a memory."""
        embedding = await self._get_embedding(content)
        try:
            db = await get_db()
            mid = await db.store_memory(
                memory_type=memory_type,
                content=content,
                source=source,
                symbol=symbol,
                strategy=strategy,
                embedding=embedding,
                importance=importance,
            )
            logger.debug(f"Stored memory {mid}: {content[:60]}...")
            return mid
        except Exception as exc:
            logger.warning(f"Failed to store memory: {exc}")
            return None

    async def search_memories(
        self, query: str, limit: int = 5, importance_threshold: float = 0.3
    ) -> list[dict]:
        """Search memories by semantic similarity."""
        embedding = await self._get_embedding(query)
        if embedding is None:
            return []
        try:
            db = await get_db()
            rows = await db.search_memories(embedding, limit, importance_threshold)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"Memory search failed: {exc}")
            return []

    async def update_importance(self, memory_id: int, importance: float) -> None:
        try:
            db = await get_db()
            await db.update_memory_importance(memory_id, importance)
        except Exception as exc:
            logger.warning(f"Failed to update memory importance: {exc}")

    # ------------------------------------------------------------------
    # Automatic memory formation
    # ------------------------------------------------------------------

    async def form_trade_memory(
        self, symbol: str, direction: str, entry_price: float,
        exit_price: float, pnl: float, strategy: str, reasoning: str
    ) -> int | None:
        """Form a memory from a closed trade."""
        outcome = "win" if pnl >= 0 else "loss"
        content = (
            f"{date.today()} Went {direction} {symbol} at {entry_price:.2f}, "
            f"closed at {exit_price:.2f} ({pnl:+.2f} USDC). "
            f"Strategy: {strategy}. Reasoning: {reasoning}"
        )
        return await self.store_memory(
            content=content,
            memory_type="lesson",
            source="trade_outcome",
            symbol=symbol,
            strategy=strategy,
            importance=0.8 if abs(pnl) > 50 else 0.6,
        )

    async def form_user_feedback_memory(
        self, user_message: str, context: str
    ) -> int | None:
        """Form a memory from user feedback in Discord."""
        return await self.store_memory(
            content=f"User feedback: {user_message}. Context: {context}",
            memory_type="feedback",
            source="user_message",
            importance=0.7,
        )

    # ------------------------------------------------------------------
    # Nightly consolidation
    # ------------------------------------------------------------------

    async def nightly_consolidation(self) -> dict:
        """
        Run at 10:30 PM ET.
        Summarize the day, extract lessons, store as memories.
        """
        try:
            db = await get_db()
            trades = await db.get_recent_trades(limit=50)
            stats = await db.get_today_stats()

            # Build consolidation prompt
            trades_text = "\n".join(
                f"- {t['direction'].upper()} {t['symbol']} @ {t['entry_price']:.2f} "
                f"-> {t.get('exit_price', 'open')} P&L=${t.get('pnl_usdc', 0):+.2f}"
                for t in trades[:20]
            )

            prompt = f"""You are Burt, reviewing today's trading session.

Today's trades:
{trades_text}

Stats: {stats['wins']} wins, {stats['losses']} losses, P&L=${stats['pnl_today']:.2f}

Return JSON:
{{
  "narrative": "brief summary in Burt's voice",
  "lessons": ["lesson 1", "lesson 2"],
  "day_rating": "good" | "okay" | "bad",
  "rating_reason": "one sentence"
}}"""

            if self._signal_engine:
                response = await self._signal_engine.chat([
                    {"role": "system", "content": "You are Burt. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ], role="consolidation")
                # Parse JSON from response
                import json
                import re
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {}

                # Store lessons as memories
                for lesson in data.get("lessons", []):
                    await self.store_memory(
                        content=lesson,
                        memory_type="lesson",
                        source="consolidation",
                        importance=0.7,
                    )

                # Store daily consolidation
                await db.add_daily_consolidation(
                    date=date.today(),
                    summary=data.get("narrative", ""),
                    lessons=data.get("lessons", []),
                    stats=stats,
                )
                return data

            return {}

        except Exception as exc:
            logger.error(f"Nightly consolidation failed: {exc}")
            return {}
