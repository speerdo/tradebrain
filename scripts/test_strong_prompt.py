"""Test stronger JSON-only prompt with Kimi K2.6."""
import sys
sys.path.insert(0, '.')
import asyncio, json
import httpx
from config import get_config

async def test():
    cfg = get_config()
    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/tradebrain",
        "X-Title": "TradeBrain",
    }

    # Much stronger system prompt
    system = """You are a JSON-only API. You evaluate crypto trading signals.

STRICT RULES:
1. Output NOTHING except valid JSON
2. No explanations, no reasoning, no markdown, no code blocks
3. The JSON must have exactly these keys: direction, confidence, reasoning, entry_price, invalidation
4. direction must be "long", "short", or "none"
5. confidence must be a float between 0.0 and 1.0
6. reasoning must be under 100 characters
7. entry_price is the current price
8. invalidation describes what would cancel the setup

EXAMPLE OUTPUT:
{"direction":"none","confidence":0.0,"reasoning":"No clear setup","entry_price":80000.0,"invalidation":"N/A"}

DO NOT output anything before or after the JSON."""

    prompt = """BTC signal evaluation. RSI=68.8, MACD hist increasing, price below 1h EMA50. Strategy: RSI+MACD momentum with EMA50 filter."""

    payload = {
        "model": get_config().signal_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as c:
        resp = await c.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        reasoning = msg.get("reasoning", "")
        print(f"Content: {content}")
        print(f"Reasoning: {reasoning[:200]}...")
        if content:
            print(f"Parsed JSON attempt: ", end="")
            try:
                print(json.loads(content))
            except Exception as e:
                print(f"FAIL: {e}")

asyncio.run(test())
