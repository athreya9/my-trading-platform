import pandas as pd
import json
import os
import logging
from datetime import datetime
from api.ai_analysis_engine import AIAnalysisEngine # Assuming this contains generate_intelligent_signal
from api.news_sentiment import fetch_news_sentiment # For sentiment scoring in backtest
from api.live_data_provider import get_live_quote # For live price if needed, though backtest uses historical
from backtesting.historical_data_fetcher import fetch_historical_data # To ensure data is available

logger = logging.getLogger(__name__)

def run_backtest(symbol, historical_data_path=None):
    """
    Simulates trades using historical data and existing signal logic.
    """
    logger.info(f"Starting backtest for {symbol}...")

    # Ensure historical data is fetched
    if historical_data_path is None:
        historical_data_path = f"data/historical/{symbol}.csv"
        if not os.path.exists(historical_data_path):
            logger.info(f"Historical data for {symbol} not found. Attempting to fetch...")
            fetch_historical_data(symbol, period="90d", interval="1h") # Fetch if not present

    if not os.path.exists(historical_data_path):
        logger.error(f"Historical data file not found for {symbol} at {historical_data_path}. Skipping backtest.")
        return []

    df = pd.read_csv(historical_data_path, index_col=0, parse_dates=True)
    if df.empty:
        logger.warning(f"Historical data for {symbol} is empty. Skipping backtest.")
        return []

    trades = []
    ai_engine = AIAnalysisEngine()

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    backtest_log_path = os.path.join(results_dir, "backtest_log.json")

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Simulate fetching sentiment (using historical context or a simplified approach)
        # For simplicity in backtesting, we'll assume neutral sentiment for now,
        # or you could implement a historical sentiment analysis.
        sentiment_string = "neutral" # Placeholder for backtesting sentiment

        # Prepare data for AI analysis engine
        # The AIAnalysisEngine expects kite_data, market_context, stock_name
        # We need to adapt historical row data to fit kite_data structure
        kite_data_for_analysis = {
            'rsi': 50, # Placeholder, ideally calculated from historical data
            'sma_20': row['Close'], # Placeholder
            'sma_50': row['Close'], # Placeholder
            'volume': row['Volume'],
            'volume_avg': row['Volume'], # Placeholder
            'macd': 0, # Placeholder
            'macd_signal': 0, # Placeholder
            'atr': 0, # Placeholder
            'atr_avg': 0, # Placeholder
            'symbol': symbol,
            'Close': row['Close']
        }
        market_context = {} # Placeholder

        # Run AI analysis
        analysis = ai_engine.analyze_trading_opportunity(kite_data_for_analysis, market_context, symbol)
        signal = ai_engine.generate_intelligent_signal(analysis)

        if signal['action'] in ["BUY", "SELL"]:
            entry_price = row['Close']
            exit_price = next_row['Close']
            
            pnl = 0
            if signal['action'] == "BUY":
                pnl = exit_price - entry_price
            elif signal['action'] == "SELL": # Assuming short sell
                pnl = entry_price - exit_price

            trades.append({
                "timestamp": row.name.isoformat(),
                "signal": signal['action'],
                "confidence": signal['confidence'],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "reasoning": signal['reasoning']
            })
    
    with open(backtest_log_path, "w") as f:
        json.dump(trades, f, indent=2)
    
    logger.info(f"Backtest for {symbol} completed. {len(trades)} trades logged to {backtest_log_path}")
    return trades

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Example backtest run
    # Ensure you have historical data for these symbols in data/historical/
    run_backtest("^NSEI")
    run_backtest("RELIANCE.NS")