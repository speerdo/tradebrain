"""Integration test: run one full signal loop iteration, then stop."""
import asyncio, sys
sys.path.insert(0, '.')

from agent.main import TradeBrainAgent

async def test():
    agent = TradeBrainAgent()
    
    # Override signal interval to 0 for instant iteration
    agent.cfg.default_signal_interval = 0
    
    # Start
    await agent.startup()
    
    # Run exactly one iteration of the signal loop
    print("\n=== Running one signal loop iteration ===")
    await agent.db.sync_config()
    await agent.risk.sync()
    
    from strategies import STRATEGIES
    strategy = STRATEGIES.get(agent.cfg.strategy)
    for symbol in agent.watchlist[:2]:  # Test first 2 only
        print(f"\nEvaluating {symbol}...")
        await agent._evaluate(symbol, [strategy] if strategy else [])
    
    print("\n=== Iteration complete ===")
    
    # Check what was logged
    trades = await agent.db.get_open_trades()
    signals = await agent.db.get_recent_signals(10)
    print(f"\nOpen trades: {len(trades)}")
    print(f"Recent signals: {len(signals)}")
    
    await agent.shutdown()
    print("\n✅ Integration test passed!")

if __name__ == "__main__":
    asyncio.run(test())
