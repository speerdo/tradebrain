"""
Sentiment & News Context (C4).

Polls free APIs for market sentiment:
- Fear & Greed Index: https://api.alternative.me/fng/ (free, no key, 1 req/hour)
- CryptoPanic: https://cryptopanic.com/api/v1/posts/ (free tier, needs API token)

Caches responses to the DB (sentiment_cache + news_cache tables) so we don't
re-fetch more than necessary. Injects a one-line digest into strategy prompts.
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from agent.database import get_db

FNG_URL = "https://api.alternative.me/fng/"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
CACHE_TTL_SEC = 3600  # 1 hour


class SentimentContext:
    """Fetches and caches Fear & Greed + CryptoPanic news."""

    def __init__(self, cryptopanic_token: str = ""):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        self._cryptopanic_token = cryptopanic_token
        self._fng_cache: dict | None = None
        self._fng_cache_ts: float = 0.0
        self._news_cache: dict[str, list] = {}
        self._news_cache_ts: dict[str, float] = {}

    async def get_fear_greed(self) -> dict:
        """
        Returns {'value': int 0-100, 'classification': str, 'cached': bool}.
        """
        now = time.time()
        if self._fng_cache and now - self._fng_cache_ts < CACHE_TTL_SEC:
            return self._fng_cache

        try:
            resp = await self._client.get(FNG_URL, params={"limit": 1})
            resp.raise_for_status()
            data = resp.json()
            entry = data.get("data", [{}])[0]
            result = {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
                "cached": False,
            }
            self._fng_cache = result
            self._fng_cache_ts = now
            # Cache to DB
            await self._cache_to_db("fear_greed", None, result["value"],
                                     result["classification"], data)
            return result
        except Exception as exc:
            logger.warning(f"Fear & Greed fetch failed: {exc}")
            return {"value": 50, "classification": "Neutral", "cached": False}

    async def get_news(self, currency: str) -> list[dict]:
        """
        Get recent news for a currency from CryptoPanic.
        Returns list of {'title', 'url', 'sentiment', 'votes_positive', 'votes_negative'}.
        """
        if not self._cryptopanic_token:
            return []

        now = time.time()
        if currency in self._news_cache and now - self._news_cache_ts.get(currency, 0) < CACHE_TTL_SEC:
            return self._news_cache[currency]

        try:
            resp = await self._client.get(
                CRYPTOPANIC_URL,
                params={
                    "auth_token": self._cryptopanic_token,
                    "currencies": currency,
                    "kind": "news",
                    "filter": "hot",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            news = [
                {
                    "title": r.get("title", "")[:200],
                    "url": r.get("url", ""),
                    "sentiment": self._classify_sentiment(r),
                    "votes_positive": r.get("votes", {}).get("positive", 0),
                    "votes_negative": r.get("votes", {}).get("negative", 0),
                    "published_at": r.get("published_at", ""),
                }
                for r in results[:5]  # top 5 headlines
            ]
            self._news_cache[currency] = news
            self._news_cache_ts[currency] = now
            # Cache to DB
            await self._cache_news_to_db(currency, news)
            return news
        except Exception as exc:
            logger.warning(f"CryptoPanic fetch failed for {currency}: {exc}")
            return []

    @staticmethod
    def _classify_sentiment(article: dict) -> str:
        votes = article.get("votes", {})
        pos = votes.get("positive", 0)
        neg = votes.get("negative", 0)
        if pos > neg + 2:
            return "positive"
        if neg > pos + 2:
            return "negative"
        return "neutral"

    async def _cache_to_db(self, source: str, symbol: str | None,
                            value: float, classification: str, raw: dict) -> None:
        try:
            db = await get_db()
            import json
            await db.execute(
                """INSERT INTO sentiment_cache (source, symbol, value, classification, raw)
                   VALUES ($1, $2, $3, $4, $5)""",
                source, symbol, value, classification, json.dumps(raw),
            )
        except Exception:
            pass  # non-critical

    async def _cache_news_to_db(self, symbol: str, news: list[dict]) -> None:
        try:
            db = await get_db()
            for n in news:
                # asyncpg needs a datetime (not an ISO string) for TIMESTAMPTZ
                published = None
                raw_ts = n.get("published_at")
                if raw_ts:
                    try:
                        published = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    except ValueError:
                        pass
                await db.execute(
                    """INSERT INTO news_cache (symbol, title, url, sentiment, votes_positive, votes_negative, published_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    symbol, n["title"], n["url"], n["sentiment"],
                    n["votes_positive"], n["votes_negative"],
                    published,
                )
        except Exception:
            pass

    def format_prompt_block(self, fng: dict, news: list[dict], symbol: str) -> str:
        """Format sentiment + news as a prompt block."""
        lines = ["\nSENTIMENT & NEWS:"]
        lines.append(f"  Fear & Greed: {fng.get('value', 50)}/100 ({fng.get('classification', 'Neutral')})")
        if news:
            negative = [n for n in news if n["sentiment"] == "negative"]
            if negative:
                lines.append(f"  ⚠️ {len(negative)} negative headline(s) for {symbol}:")
                for n in negative[:2]:
                    lines.append(f"    - {n['title']}")
            else:
                lines.append(f"  News sentiment: {news[0]['sentiment']} — {news[0]['title'][:80]}")
        return "\n".join(lines) + "\n"

    def has_high_panic_news(self, news: list[dict]) -> bool:
        """Check if there are high-panic negative headlines (news veto)."""
        if not news:
            return False
        negative = [n for n in news if n["sentiment"] == "negative"]
        return len(negative) >= 2  # 2+ negative headlines = high panic

    async def close(self) -> None:
        await self._client.aclose()