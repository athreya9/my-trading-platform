# backtest.py
# A script to backtest a trading strategy using historical data from Google Sheets.

import gspread
import pandas as pd
import numpy as np
import os
import json
import sys

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard"
DATA_WORKSHEET_NAME = "Price Data"
TARGET_INSTRUMENT = '^NSEI'
INITIAL_CAPITAL = 100000.0  # Starting capital for the simulation
SERVICE_ACCOUNT_FILE = 'service_account.json' # For local execution

# --- Main Functions ---

def connect_to_google_sheets():
    """Connects to Google Sheets using credentials from a local file."""
    print("Attempting to authenticate with Google Sheets...")
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"'{SERVICE_ACCOUNT_FILE}' not found. "
            "Please ensure your service account key file is in the same directory."
        )
    try:
        client = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        spreadsheet = client.open(SHEET_NAME)
        print(f"Successfully connected to Google Sheet: '{SHEET_NAME}'")
        return spreadsheet
    except Exception as e:
        raise Exception(f"Error connecting to Google Sheet: {e}")

def read_price_data(spreadsheet):
    """Reads historical price data from the sheet, returning a clean DataFrame."""
    print(f"Reading historical data from '{DATA_WORKSHEET_NAME}' tab...")
    try:
        worksheet = spreadsheet.worksheet(DATA_WORKSHEET_NAME)
        records = worksheet.get_all_records() # Reads data into a list of dicts
        if not records:
            raise ValueError("No data found in 'Price Data' tab.")
        
        df = pd.DataFrame(records)
        print(f"Successfully read {len(df)} rows of historical data.")
        
        # --- Data Cleaning and Preparation ---
        # Filter for the target instrument
        df = df[df['instrument'] == TARGET_INSTRUMENT].copy()
        if df.empty:
            raise ValueError(f"No data found for target instrument '{TARGET_INSTRUMENT}'.")

        # Convert columns to correct data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Replace empty strings with NaN and convert to numeric
            df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
        
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Data cleaned and prepared for backtesting.")
        return df

    except Exception as e:
        raise Exception(f"An error occurred while reading or processing data: {e}")

def run_backtest(price_df, initial_capital):
    """Simulates the trading strategy and returns performance results."""
    print("Running backtest simulation...")

    # --- Strategy Calculation (SMA Crossover) ---
    price_df['SMA_20'] = price_df['close'].rolling(window=20).mean()
    price_df['SMA_50'] = price_df['close'].rolling(window=50).mean()
    
    # Determine position based on crossover: 1 for long (buy), -1 for short (sell state)
    price_df['position'] = np.where(price_df['SMA_20'] > price_df['SMA_50'], 1, -1)
    # Find the exact point of crossover to trigger a trade
    price_df['signal'] = price_df['position'].diff()

    # --- Portfolio Simulation ---
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position
    trades = []
    portfolio_history = []

    for i, row in price_df.iterrows():
        # --- Buy Signal ---
        if row['signal'] > 0 and position == 0: # Crossover from -1 to 1
            position = 1
            entry_price = row['close']
            entry_date = row['timestamp']

        # --- Sell Signal ---
        elif row['signal'] < 0 and position == 1: # Crossover from 1 to -1
            position = 0
            exit_price = row['close']
            exit_date = row['timestamp']
            
            # Log the completed trade
            profit = exit_price - entry_price
            trades.append({
                'entry_date': entry_date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'profit': profit
            })
            capital += profit # Simple profit calculation (ignores shares)

        portfolio_history.append({'timestamp': row['timestamp'], 'capital': capital})

    return pd.DataFrame(portfolio_history), pd.DataFrame(trades)

def calculate_and_print_performance(portfolio_df, trades_df, initial_capital):
    """Calculates key performance metrics and prints a summary report."""
    print("\n--- Backtest Performance Report ---")

    if portfolio_df.empty:
        print("No portfolio history to analyze.")
        return

    # 1. Total Return
    final_capital = portfolio_df['capital'].iloc[-1]
    total_return_pct = ((final_capital / initial_capital) - 1) * 100
    print(f"Ending Capital:         ${final_capital:,.2f}")
    print(f"Total Return:           {total_return_pct:.2f}%")

    if trades_df.empty:
        print("No trades were executed.")
        return

    # 2. Trade Statistics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\nTotal Trades:           {total_trades}")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Average Profit/Trade:   ${trades_df['profit'].mean():,.2f}")
    print(f"Average Win:            ${winning_trades['profit'].mean():,.2f}")
    print(f"Average Loss:           ${losing_trades['profit'].mean():,.2f}")

    # 3. Maximum Drawdown
    portfolio_df['peak'] = portfolio_df['capital'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['capital'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min() * 100
    
    print(f"\nMaximum Drawdown:       {max_drawdown:.2f}%")
    print("-------------------------------------\n")

def main():
    """Main function that runs the entire backtesting process."""
    try:
        spreadsheet = connect_to_google_sheets()
        price_df = read_price_data(spreadsheet)
        portfolio_history, trades = run_backtest(price_df, INITIAL_CAPITAL)
        calculate_and_print_performance(portfolio_history, trades, INITIAL_CAPITAL)
    except Exception as e:
        print(f"\n❌ An error occurred during the backtest: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()