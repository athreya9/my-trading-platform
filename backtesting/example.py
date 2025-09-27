from backtesting.core import BacktestEngine
from backtesting.strategies import MovingAverageCrossover

# --- 2. Run the Backtest ---
if __name__ == '__main__':
    # Instantiate the engine
    engine = BacktestEngine(initial_capital=100000)

    # Create and register the strategy
    ma_crossover = MovingAverageCrossover(short_window=40, long_window=100)
    engine.register_strategy("MA Crossover", ma_crossover)

    # Define backtest parameters
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Run the backtest
    engine.run_backtest("MA Crossover", symbol, start_date, end_date)

    # --- 3. Analyze the Results ---
    # Calculate and print performance metrics
    metrics = engine.calculate_performance_metrics("MA Crossover")
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    # Plot the results
    engine.plot_results("MA Crossover")
