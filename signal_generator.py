# signal_generator.py

import gspread
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard"
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"
SERVICE_ACCOUNT_FILE = 'service_account.json'

# The instrument we want to analyze from the 'Price Data' sheet
TARGET_INSTRUMENT = '^NSEI'

# Headers for the output sheet to ensure consistency
SIGNAL_HEADERS = ['timestamp', 'instrument', 'signal']


def connect_to_google_sheets():
    """Connects to Google Sheets and returns the spreadsheet object."""
    try:
        # Use the modern gspread authentication method
        client = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        spreadsheet = client.open(SHEET_NAME)
        print("Successfully connected to Google Sheet.")
        return spreadsheet
    except Exception as e:
        print(f"Error connecting to Google Sheet: {e}")
        exit()


def read_price_data(spreadsheet):
    """Reads price data from the sheet, returning a clean pandas DataFrame."""
    try:
        print(f"Reading data from '{DATA_WORKSHEET_NAME}' tab...")
        worksheet = spreadsheet.worksheet(DATA_WORKSHEET_NAME)
        
        # get_all_values() is faster and more reliable than get_all_records()
        records = worksheet.get_all_values()
        if len(records) < 2: # Must have header + at least one data row
            print("No data found in 'Price Data' tab. Run data_collector.py first.")
            return pd.DataFrame()
        
        # Create DataFrame from records, using the first row as the header
        df = pd.DataFrame(records[1:], columns=records[0])
        print(f"Successfully read {len(df)} rows.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{DATA_WORKSHEET_NAME}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading data: {e}")
        return pd.DataFrame()


def generate_signals(df):
    """Generates trading signals based on a moving average crossover strategy."""
    print(f"Generating signals for {TARGET_INSTRUMENT}...")
    
    instrument_df = df[df['instrument'] == TARGET_INSTRUMENT].copy()

    if instrument_df.empty:
        print(f"No data found for instrument '{TARGET_INSTRUMENT}'.")
        return pd.DataFrame()

    # --- Data Preparation ---
    # Convert columns to correct data types for calculations
    instrument_df['timestamp'] = pd.to_datetime(instrument_df['timestamp'])
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        instrument_df[col] = pd.to_numeric(instrument_df[col], errors='coerce')

    instrument_df.sort_values('timestamp', inplace=True)
    
    # --- Strategy Calculation (SMA Crossover) ---
    instrument_df['SMA_20'] = ta.sma(instrument_df['close'], length=20)
    instrument_df['SMA_50'] = ta.sma(instrument_df['close'], length=50)
    instrument_df.dropna(inplace=True) # Remove rows where SMAs aren't calculated
    
    # --- Signal Logic ---
    # 1 for bullish state (SMA20 > SMA50), -1 for bearish
    instrument_df['position'] = np.where(instrument_df['SMA_20'] > instrument_df['SMA_50'], 1, -1)
    # Find the crossover: a change from the previous row's position
    instrument_df['crossover'] = instrument_df['position'].diff()
    
    signals = instrument_df[instrument_df['crossover'] != 0].copy()
    
    if signals.empty:
        print("No new signals generated.")
        return pd.DataFrame()
        
    # --- Format Final Output ---
    signals['signal'] = np.where(signals['crossover'] > 0, 'BUY', 'SELL')
    signals['timestamp'] = signals['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Use the target instrument name for the output
    signals['instrument'] = TARGET_INSTRUMENT
    
    final_signals_df = signals[SIGNAL_HEADERS]
    print(f"Generated {len(final_signals_df)} signals.")
    return final_signals_df


def write_signals_to_sheet(spreadsheet, signals_df):
    """Clears the 'Signals' sheet and writes the new signals."""
    if signals_df.empty:
        return

    try:
        worksheet = spreadsheet.worksheet(SIGNALS_WORKSHEET_NAME)
        print(f"Writing {len(signals_df)} signals to '{SIGNALS_WORKSHEET_NAME}' tab...")
        
        data_to_write = [signals_df.columns.values.tolist()] + signals_df.values.tolist()
        
        worksheet.clear()
        worksheet.update(data_to_write, value_input_option='USER_ENTERED')
        
        print("\n✅ Signals written successfully!")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{SIGNALS_WORKSHEET_NAME}' not found. Please create it.")
    except Exception as e:
        print(f"An error occurred while writing signals: {e}")


def main():
    """Main function to run the entire signal generation process."""
    spreadsheet = connect_to_google_sheets()
    price_df = read_price_data(spreadsheet)
    
    if not price_df.empty:
        signals_df = generate_signals(price_df)
        write_signals_to_sheet(spreadsheet, signals_df)


if __name__ == "__main__":
    main()