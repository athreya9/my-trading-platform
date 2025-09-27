import pytest
from backtesting.core import BacktestEngine
from backtesting.strategies import MovingAverageCrossover

def test_backtesting_engine():
    """Test our backtesting engine with real data"""
    print("\nTesting Backtesting Engine...")
    
    # Initialize engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Register strategy
    strategy = MovingAverageCrossover(short_window=10, long_window=30)
    engine.register_strategy("Test MA Crossover", strategy)
    
    # Run backtest
    engine.run_backtest('Test MA Crossover', 'RELIANCE.NS', '2023-01-01', '2024-01-01')
    
    # Get results
    results = engine.results.get('Test MA Crossover')
    assert results is not None, "Backtest did not produce results."
    assert not results['portfolio'].empty, "Backtest results are empty."
    print("✅ Backtest ran successfully and produced results.")

    # Calculate metrics
    metrics = engine.calculate_performance_metrics('Test MA Crossover')
    assert metrics is not None, "Could not calculate performance metrics."
    assert isinstance(metrics, dict), "Metrics should be a dictionary."
    print("✅ Performance metrics calculated successfully.")
    
    print("\nBacktest Results:")
    for metric, value in metrics.items():
        # The value is a float, so we format it for printing
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")

def test_nifty_strategy():
    """Test with Indian market data"""
    print("\nTesting NIFTY Strategy...")
    engine = BacktestEngine()
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    engine.register_strategy("NIFTY MA Crossover", strategy)
    engine.run_backtest('NIFTY MA Crossover', '^NSEI', '2023-01-01', '2024-01-01')
    
    results = engine.results.get('NIFTY MA Crossover')
    assert results is not None, "Backtest did not produce results for NIFTY."
    assert not results['portfolio'].empty, "NIFTY backtest results are empty."
    print("✅ NIFTY backtest ran successfully.")

    metrics = engine.calculate_performance_metrics('NIFTY MA Crossover')
    assert metrics is not None, "Could not calculate metrics for NIFTY."
    print("✅ NIFTY performance metrics calculated successfully.")

    print("\nNIFTY Backtest Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    test_backtesting_engine()
    test_nifty_strategy()
