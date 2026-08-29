"""Verify OpenRouter model ID fix."""
import asyncio, httpx, json
from config import get_config

async def test():
    cfg = get_config()
    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/tradebrain",
        "X-Title": "TradeBrain",
    }
    payload = {
        "model": get_config().signal_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return JSON: {\"hello\": \"world\"}"},
        ],
        "temperature": 0.1,
        "max_tokens": 100,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        try:
            data = resp.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
        except Exception:
            print(f"Raw body:\n{resp.text}")

if __name__ == "__main__":
    asyncio.run(test())
