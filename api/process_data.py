# process_data.py

# A single, combined script for Vercel deployment.
# It fetches data, generates signals, and updates Google Sheets.

import gspread
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import json
from http.server import BaseHTTPRequestHandler

# --- Configuration ---
SERVICE_ACCOUNT_FILE = 'service_account.json'
SHEET_NAME = "Algo Trading Dashboard"
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"

# Data collection settings
SYMBOLS = ['RELIANCE.NS', '^NSEI']
DATA_HEADERS = ['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume']

# Signal generation settings
TARGET_INSTRUMENT = '^NSEI'
SIGNAL_HEADERS = ['timestamp', 'instrument', 'signal']

# --- Main Functions ---

def connect_to_google_sheets():
    """Connects to Google Sheets using credentials from env vars or a local file."""
    try:
        # Vercel: Get credentials from environment variable (as a JSON string)
        creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        if creds_json_str:
            print("Authenticating with Google Sheets via environment variable.")
            creds_dict = json.loads(creds_json_str)
            client = gspread.service_account_from_dict(creds_dict)
        else:
            # Local: Fallback to the service account file for local testing
            print("GOOGLE_SHEETS_CREDENTIALS env var not found. Falling back to local file.")
            client = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)

        spreadsheet = client.open(SHEET_NAME)
        print("Successfully connected to Google Sheet.")
        return spreadsheet
    except Exception as e:
        # Raise the exception to be caught by the Vercel handler
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
    
    instrument_df['SMA_20'] = ta.sma(instrument_df['close'], length=20)
    instrument_df['SMA_50'] = ta.sma(instrument_df['close'], length=50)
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

# --- Vercel Serverless Handler ---
# This is the entry point for the Vercel serverless function.
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            main()
            # If main() completes without error, send a success response
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Process completed successfully.')
        except Exception as e:
            # If any error occurs, log it and send a server error response
            error_message = f"An error occurred: {e}"
            print(error_message)
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(error_message.encode('utf-8'))

if __name__ == "__main__":
    # This block allows you to test the script locally
    # without needing to deploy it to Vercel.
    print("Running script locally...")
    main()
    print("\nLocal run finished.")