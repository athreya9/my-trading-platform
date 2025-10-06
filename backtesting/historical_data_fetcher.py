import yfinance as yf
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def fetch_historical_data(symbol, period="90d", interval="1h"):
    """
    Fetches historical OHLC data for a given symbol from Yahoo Finance
    and saves it to data/historical/{symbol}.csv.
    """
    try:
        # Ensure the historical data directory exists
        historical_data_dir = "data/historical"
        os.makedirs(historical_data_dir, exist_ok=True)

        file_path = os.path.join(historical_data_dir, f"{symbol}.csv")

        logger.info(f"Fetching historical data for {symbol} (period={period}, interval={interval})...")
        data = yf.download(tickers=symbol, period=period, interval=interval)

        if data.empty:
            logger.warning(f"No historical data found for {symbol}.")
            return None

        data.to_csv(file_path)
        logger.info(f"Successfully fetched and saved historical data for {symbol} to {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Fetching NIFTY historical data...")
    fetch_historical_data("^NSEI", period="30d", interval="1h")
    print("Fetching Reliance historical data...")
    fetch_historical_data("RELIANCE.NS", period="30d", interval="1h")