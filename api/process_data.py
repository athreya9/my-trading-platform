# process_data.py
# A single, combined script for GitHub Actions.
# It fetches data, generates signals, and updates Google Sheets.

import gspread
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import sys

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard" # The name of your Google Sheet
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"

# Data collection settings
SYMBOLS = ['RELIANCE.NS', '^NSEI']

# Signal generation settings
TARGET_INSTRUMENT = '^NSEI'
SIGNAL_HEADERS = ['timestamp', 'instrument', 'signal']

# --- Main Functions ---

def connect_to_google_sheets():
    """Connects to Google Sheets using credentials from an environment variable."""
    print("Attempting to authenticate with Google Sheets...")
    creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
    if not creds_json_str:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS environment variable not found. Ensure it's set in GitHub Actions secrets.")

    try:
        print("Authenticating with Google Sheets via environment variable.")
        creds_dict = json.loads(creds_json_str)
        client = gspread.service_account_from_dict(creds_dict)
        spreadsheet = client.open(SHEET_NAME)
        print(f"Successfully connected to Google Sheet: '{SHEET_NAME}'")
        return spreadsheet
    except Exception as e:
        raise Exception(f"Error connecting to Google Sheet: {e}")

def fetch_historical_data(ticker, period='5d', interval='15m'):
    """Fetches historical data from Yahoo Finance for a given ticker."""
    print(f"Fetching data for {ticker}...")
    try:
        stock_data = yf.download(
            tickers=ticker, period=period, interval=interval, auto_adjust=True
        )
        if stock_data.empty:
            print(f"No data downloaded for {ticker}.")
            return pd.DataFrame()

        stock_data.reset_index(inplace=True)
        
        clean_df = pd.DataFrame()
        clean_df['timestamp'] = stock_data['Datetime']
        clean_df['instrument'] = ticker
        clean_df['open'] = stock_data['Open']
        clean_df['high'] = stock_data['High']
        clean_df['low'] = stock_data['Low']
        clean_df['close'] = stock_data['Close']
        clean_df['volume'] = stock_data.get('Volume')
        
        return clean_df
    except Exception as e:
        print(f"Warning: Could not fetch data for {ticker}: {e}")
        return pd.DataFrame()

def run_data_collection():
    """Fetches data for all symbols and returns a combined DataFrame."""
    all_data = [fetch_historical_data(symbol) for symbol in SYMBOLS]
    all_data = [df for df in all_data if not df.empty]

    if not all_data:
        raise Exception("No data was fetched for any symbol. Halting process.")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully fetched a total of {len(combined_df)} rows of data.")
    return combined_df

def generate_signals(price_df):
    """Generates trading signals from a DataFrame of price data."""
    print(f"Generating signals for {TARGET_INSTRUMENT}...")
    
    instrument_df = price_df[price_df['instrument'] == TARGET_INSTRUMENT].copy()

    if instrument_df.empty:
        print(f"No data found for instrument '{TARGET_INSTRUMENT}'. Cannot generate signals.")
        return pd.DataFrame()

    instrument_df['timestamp'] = pd.to_datetime(instrument_df['timestamp'])
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        instrument_df[col] = pd.to_numeric(instrument_df[col], errors='coerce')

    instrument_df.sort_values('timestamp', inplace=True)
    
    # Use pandas' built-in, efficient rolling mean calculation instead of pandas_ta
    instrument_df['SMA_20'] = instrument_df['close'].rolling(window=20).mean()
    instrument_df['SMA_50'] = instrument_df['close'].rolling(window=50).mean()
    instrument_df.dropna(inplace=True)
    
    if instrument_df.empty:
        print("Not enough data to calculate moving averages. No signals generated.")
        return pd.DataFrame()
        
    instrument_df['position'] = np.where(instrument_df['SMA_20'] > instrument_df['SMA_50'], 1, -1)
    instrument_df['crossover'] = instrument_df['position'].diff()
    
    signals = instrument_df[instrument_df['crossover'] != 0].copy()
    
    if signals.empty:
        print("No new signals generated.")
        return pd.DataFrame()
        
    signals['signal'] = np.where(signals['crossover'] > 0, 'BUY', 'SELL')
    signals['instrument'] = TARGET_INSTRUMENT
    
    final_signals_df = signals[SIGNAL_HEADERS]
    print(f"Generated {len(final_signals_df)} signals.")
    return final_signals_df

def write_to_sheets(spreadsheet, price_df, signals_df):
    """Writes the price data and signal data to their respective sheets."""
    
    DATA_HEADERS = ['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume']
    # --- Write Price Data ---
    print(f"Writing {len(price_df)} rows to '{DATA_WORKSHEET_NAME}'...")
    price_df['timestamp'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    price_df.fillna('', inplace=True)
    
    price_data_to_write = [DATA_HEADERS] + price_df[DATA_HEADERS].values.tolist()
    
    data_worksheet = spreadsheet.worksheet(DATA_WORKSHEET_NAME)
    data_worksheet.clear()
    data_worksheet.update(price_data_to_write, value_input_option='USER_ENTERED')
    print("Price data written successfully.")

    # --- Write Signal Data ---
    if not signals_df.empty:
        print(f"Writing {len(signals_df)} rows to '{SIGNALS_WORKSHEET_NAME}'...")
        signals_df['timestamp'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        signal_data_to_write = [SIGNAL_HEADERS] + signals_df.values.tolist()
        
        signals_worksheet = spreadsheet.worksheet(SIGNALS_WORKSHEET_NAME)
        signals_worksheet.clear()
        signals_worksheet.update(signal_data_to_write, value_input_option='USER_ENTERED')
        print("Signal data written successfully.")
    else:
        # If no signals, still clear the sheet to remove old signals
        print("No signals to write. Clearing old signals from sheet.")
        spreadsheet.worksheet(SIGNALS_WORKSHEET_NAME).clear()


def main():
    """Main function that runs the entire process."""
    # Step 1: Connect to Google Sheets
    spreadsheet = connect_to_google_sheets()
    
    # Step 2: Collect new data
    price_df = run_data_collection()
    
    # Step 3: Generate signals from the new data
    signals_df = generate_signals(price_df.copy()) # Pass a copy to avoid pandas warnings
    
    # Step 4: Write both data and signals to the sheets
    write_to_sheets(spreadsheet, price_df, signals_df)

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting trading signal process...")
    try:
        main()
        print("\n✅ Process completed successfully.")
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        sys.exit(1) # Exit with a non-zero code to fail the GitHub Action