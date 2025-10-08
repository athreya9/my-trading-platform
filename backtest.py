"""
Main entry point for running AI-driven backtesting and strategy optimization.

This script orchestrates the following steps:
1. Fetches historical market data for a given symbol.
2. Runs a backtest simulation using the AIAnalysisEngine to generate trading signals.
3. Logs the trades to a JSON file.
4. Analyzes the results to calculate performance metrics.
5. Optimizes the trading strategy by finding the best confidence threshold.
"""

import argparse
import logging
import json
import pandas as pd

from backtesting.historical_data_fetcher import fetch_historical_data
from backtesting.backtest_engine import run_backtest
from backtesting.strategy_optimizer import optimize_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_results(trades):
    """
    Analyzes the trades from a backtest and prints a performance summary.
    """
    if not trades:
        logger.info("No trades were made during the backtest.")
        return

    df = pd.DataFrame(trades)
    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]

    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = df['pnl'].sum()
    average_pnl = df['pnl'].mean()
    average_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    average_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0

    logger.info("--- Backtest Performance Summary ---")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total PnL: {total_pnl:.2f}")
    logger.info(f"Average PnL per Trade: {average_pnl:.2f}")
    logger.info(f"Average Winning Trade: {average_win:.2f}")
    logger.info(f"Average Losing Trade: {average_loss:.2f}")
    logger.info("------------------------------------")

def main(symbol, period, interval):
    """
    Orchestrates the full backtesting and optimization workflow.
    """
    logger.info(f"Starting full backtest and optimization for {symbol}...")

    # Step 1: Fetch historical data
    fetch_historical_data(symbol, period=period, interval=interval)

    # Step 2: Run the backtest using the AI engine
    # The run_backtest function will save the trades to results/backtest_log.json
    trades = run_backtest(symbol)

    if not trades:
        logger.error("Backtest finished with no trades. Aborting optimization.")
        return

    # Step 3: Analyze and print the results of the initial backtest
    analyze_results(trades)

    # Step 4: Optimize the strategy
    logger.info("--- Strategy Optimization ---")
    best_threshold, best_pnl = optimize_strategy(trades)

    if best_threshold is not None:
        logger.info(f"Optimal confidence threshold found: {best_threshold}%")
        logger.info(f"PnL at this threshold: {best_pnl:.2f}")

        # Analyze results if we only took trades above the optimal threshold
        optimized_trades = [t for t in trades if t.get('confidence', 0) >= best_threshold]
        logger.info("\n--- Optimized Performance Summary ---")
        analyze_results(optimized_trades)
    else:
        logger.warning("Could not determine an optimal strategy.")

    logger.info("Backtesting and optimization process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a full backtest and strategy optimization."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="RELIANCE.NS",
        help="The stock symbol to backtest (e.g., 'RELIANCE.NS')."
    )
    parser.add_argument(
        "--period",
        type=str,
        default="90d",
        help="The historical data period to fetch (e.g., '30d', '3mo')."
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="The data interval to fetch (e.g., '15m', '1h', '1d')."
    )
    args = parser.parse_args()

    main(args.symbol, args.period, args.interval)