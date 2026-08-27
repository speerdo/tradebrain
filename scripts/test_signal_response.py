"""Debug full OpenRouter response for signal prompt."""
import sys
sys.path.insert(0, '.')
import asyncio, json
import httpx
from agent.data_client import HyperliquidDataClient
from agent.indicator_engine import compute_indicators
from strategies import STRATEGIES
from config import get_config

async def test():
    cfg = get_config()
    client = HyperliquidDataClient()
    try:
        df_15m = await client.get_candles('BTC', '15m', limit=100)
        df_1h = await client.get_candles('BTC', '1h', limit=100)
        indicators = compute_indicators(df_15m, df_1h)
        strategy = STRATEGIES['rsi_macd']
        prompt = strategy.build_prompt(indicators, 'BTC')

        headers = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/tradebrain",
            "X-Title": "TradeBrain",
        }
        payload = {
            "model": get_config().signal_model,
            "messages": [
                {"role": "system", "content": "You are TradeBrain, an expert crypto futures trading signal evaluator. Analyze live technical indicator data and determine whether a trading signal meets the defined strategy criteria. Rules: Only signal 'long' or 'short' when ALL required conditions are clearly met. When in doubt, return 'none'. Be concise in reasoning. ALWAYS return valid JSON only, no markdown, no preamble. Never recommend a trade without clear technical justification."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as c:
            resp = await c.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            print(f"Status: {resp.status_code}")
            data = resp.json()
            print(f"Full response:\n{json.dumps(data, indent=2)}")
    finally:
        await client.close()

asyncio.run(test())
