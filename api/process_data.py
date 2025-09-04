# process_data.py
# A single, combined script for GitHub Actions.
# It fetches data, generates signals, and updates Google Sheets.

import gspread
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import json
import sys

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard" # The name of your Google Sheet
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"

# Data collection settings
SYMBOLS = ['RELIANCE.NS', '^NSEI', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']

# Signal generation settings
SIGNAL_HEADERS = ['timestamp', 'instrument', 'signal', 'stop_loss', 'take_profit']

# Risk Management settings
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = 2.0  # e.g., 2 * ATR below entry price
TAKE_PROFIT_MULTIPLIER = 4.0 # e.g., 4 * ATR above entry price (for a 1:2 risk/reward ratio)

# Microstructure settings
REALIZED_VOL_WINDOW = 20 # e.g., 20 periods for volatility calculation

# --- Main Functions ---

def read_manual_controls(spreadsheet):
    """Reads manual override settings from the 'Manual Control' sheet."""
    print("Reading data from 'Manual Control' sheet...")
    try:
        worksheet = spreadsheet.worksheet("Manual Control")
        records = worksheet.get_all_records()
        if not records:
            print("No manual controls found or sheet is empty.")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # Ensure key columns exist, even if empty
        for col in ['instrument', 'limit_price', 'hold_status']:
            if col not in df.columns:
                df[col] = None
        
        df.set_index('instrument', inplace=True)
        print("Manual controls loaded successfully.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        print("Warning: 'Manual Control' worksheet not found. Skipping manual overrides.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not read manual controls: {e}")
        return pd.DataFrame()

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

def calculate_indicators(price_df):
    """
    Calculates all technical indicators (SMA, RSI, MACD, ATR) for all instruments
    using efficient, vectorized operations.
    """
    print("Calculating indicators for all instruments...")
    # --- Data Cleaning and Preparation ---
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    # Ensure all key columns are numeric, coercing errors to NaN
    for col in ['open', 'high', 'low', 'close', 'volume']:
        price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Drop any rows where essential data (like close price or volume) is missing
    price_df.dropna(subset=['close', 'volume'], inplace=True)
    price_df.sort_values(['instrument', 'timestamp'], inplace=True)

    # Define a function to apply all indicators to a group (a single instrument's data)
    def apply_indicators(group):
        group = group.copy()
        # Simple Moving Averages
        group['SMA_20'] = group['close'].rolling(window=20).mean()
        group['SMA_50'] = group['close'].rolling(window=50).mean()
        # pandas-ta indicators
        group.ta.rsi(length=14, append=True)
        group.ta.macd(fast=12, slow=26, signal=9, append=True)
        group.ta.atr(length=ATR_PERIOD, append=True)

        # --- New Microstructure Indicators ---
        # 1. Realized Volatility (rolling standard deviation of log returns)
        group['log_return'] = np.log(group['close'] / group['close'].shift(1))
        group['realized_vol'] = group['log_return'].rolling(window=REALIZED_VOL_WINDOW).std()

        # 2. VWAP (Volume-Weighted Average Price) - calculated per day
        # This requires a nested groupby to reset the calculation for each new day.
        def calculate_daily_vwap(daily_group):
            # Fill NaN volumes with 0 to prevent issues in cumulative sum
            daily_group['volume'] = daily_group['volume'].fillna(0)
            daily_group['vwap'] = (daily_group['close'] * daily_group['volume']).cumsum() / daily_group['volume'].cumsum()
            return daily_group
        group = group.groupby(group['timestamp'].dt.date, group_keys=False).apply(calculate_daily_vwap)
        return group

    # Use groupby().apply() to run the indicator calculations for each instrument
    price_df = price_df.groupby('instrument', group_keys=False).apply(apply_indicators)
    
    print("Indicators calculated successfully for all instruments.")
    return price_df

def generate_signals(price_df, manual_controls_df):
    """Generates trading signals for all instruments, applying manual overrides."""
    print("Generating signals for all instruments...")
    
    all_signals_list = []

    # Iterate over each instrument's data
    for instrument, group in price_df.groupby('instrument'):
        instrument_signals = []
        # Drop rows where indicators are not yet calculated
        group = group.copy().dropna(subset=['SMA_50', 'RSI_14', f'ATRr_{ATR_PERIOD}'])
        if group.empty:
            continue

        latest_row = group.iloc[-1]
        
        # --- Manual Override Logic ---
        manual_override_triggered = False
        if not manual_controls_df.empty and instrument in manual_controls_df.index:
            control = manual_controls_df.loc[instrument]
            
            # 1. Hold Status Override
            if str(control['hold_status']).upper() == 'HOLD':
                print(f"'{instrument}' is on HOLD. Skipping automated signal.")
                manual_override_triggered = True
            elif str(control['hold_status']).upper() == 'SELL':
                print(f"'{instrument}' has manual SELL override.")
                instrument_signals.append({'signal': 'SELL (MANUAL)', 'stop_loss': np.nan, 'take_profit': np.nan})
                manual_override_triggered = True

            # 2. Limit Price Alert
            limit_price = pd.to_numeric(control['limit_price'], errors='coerce')
            if pd.notna(limit_price) and latest_row['close'] > limit_price:
                print(f"'{instrument}' price ({latest_row['close']:.2f}) is above limit ({limit_price:.2f}).")
                instrument_signals.append({'signal': 'SELL (LIMIT HIT)', 'stop_loss': np.nan, 'take_profit': np.nan})
                manual_override_triggered = True

        # --- Automated Signal Logic ---
        # This new logic checks the CURRENT STATE of the instrument, not just a crossover event.
        if not manual_override_triggered:
            # Check for an active "BUY" state
            if latest_row['SMA_20'] > latest_row['SMA_50'] and latest_row['RSI_14'] < 70:
                signal_text = "BUY"
                atr_val = latest_row[f'ATRr_{ATR_PERIOD}']
                sl, tp = np.nan, np.nan
                if pd.notna(atr_val):
                    sl = latest_row['close'] - (atr_val * STOP_LOSS_MULTIPLIER)
                    tp = latest_row['close'] + (atr_val * TAKE_PROFIT_MULTIPLIER)
                instrument_signals.append({'signal': signal_text, 'stop_loss': sl, 'take_profit': tp})

            # Check for an active "SELL" state
            elif latest_row['SMA_20'] < latest_row['SMA_50']:
                signal_text = "SELL"
                instrument_signals.append({'signal': signal_text, 'stop_loss': np.nan, 'take_profit': np.nan})

        # Add common data to all signals found for this instrument
        for sig in instrument_signals:
            sig['timestamp'] = latest_row['timestamp']
            sig['instrument'] = instrument
            all_signals_list.append(sig)

    if not all_signals_list:
        print("No new signals generated for any instrument.")
        return pd.DataFrame()
        
    final_signals_df = pd.DataFrame(all_signals_list)
    print(f"Generated {len(final_signals_df)} total signals.")
    return final_signals_df[SIGNAL_HEADERS] # Ensure correct column order

def write_to_sheets(spreadsheet, price_df, signals_df):
    """Writes the price data and signal data to their respective sheets."""
    
    # --- Write Price Data ---
    # Headers are now dynamically generated from the DataFrame columns
    DATA_HEADERS = price_df.columns.tolist()
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
    
    # Step 2: Read manual controls from the sheet
    manual_controls_df = read_manual_controls(spreadsheet)
    
    # Step 3: Collect new data
    price_df = run_data_collection()
    
    # Step 4: Calculate all indicators for all instruments
    price_df = calculate_indicators(price_df)
    
    # Step 5: Generate signals using data and manual controls
    signals_df = generate_signals(price_df.copy(), manual_controls_df) # Pass a copy to avoid pandas warnings
    
    # Step 6: Write both data and signals to the sheets
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