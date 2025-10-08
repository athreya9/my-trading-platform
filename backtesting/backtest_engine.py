import pandas as pd
import json
import os
import logging
import pandas_ta as ta
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

    column_names = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = pd.read_csv(historical_data_path, skiprows=3, header=None, names=column_names, index_col='Price', parse_dates=['Price'])
    # Ensure numerical columns are correctly typed
    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_cols, inplace=True) # Drop rows where numerical conversion failed

    if df.empty:
        logger.warning(f"Historical data for {symbol} is empty after cleaning. Skipping backtest.")
        return []

    # --- Indicator Calculation ---
    # Calculate all indicators on the dataframe at once for efficiency
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df['volume_avg_20'] = df['Volume'].rolling(window=20).mean()

    # Drop rows with NaN values that were created by the indicator calculations
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

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
            'rsi': row['RSI_14'],
            'sma_20': row['SMA_20'],
            'sma_50': row['SMA_50'],
            'volume': row['Volume'],
            'volume_avg': row['volume_avg_20'],
            'macd': row['MACD_12_26_9'],
            'macd_signal': row['MACDs_12_26_9'],
            'atr': row['ATRr_14'],
            'atr_avg': row['ATRr_14'], # Using ATR itself as average for this context
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
                "timestamp": row['Price'].isoformat(),
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