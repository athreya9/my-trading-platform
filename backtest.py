# backtest.py
# A script to backtest a trading strategy using historical data from Google Sheets.

import gspread
import pandas_ta as ta
import pandas as pd
import numpy as np
import os
import json
import time
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard"
DATA_WORKSHEET_NAME = "Price Data"
TARGET_INSTRUMENT = '^NSEI'
INITIAL_CAPITAL = 100000.0  # Starting capital for the simulation
SERVICE_ACCOUNT_FILE = 'service_account.json' # For local execution

# --- Strategy & Risk Configuration (to match process_data.py) ---
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = 2.0
TAKE_PROFIT_MULTIPLIER = 4.0

# --- Helper Functions ---
def retry(tries=3, delay=5, backoff=2, logger=print):
    """
    A retry decorator with exponential backoff.
    Catches common network-related exceptions for gspread.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                # Catches API errors from gspread and general connection errors
                except (gspread.exceptions.APIError, ConnectionError) as e:
                    msg = f"'{f.__name__}' failed with {e}. Retrying in {mdelay} seconds..."
                    if logger:
                        logger(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs) # Last attempt, if it fails, it fails
        return f_retry
    return deco_retry

def get_atm_strike(price, instrument):
    """
    Calculates a theoretical at-the-money (ATM) strike price by rounding.
    This is a rule-based estimation as we don't have live options chain data.
    """
    # For Nifty 50 (^NSEI), strike prices are typically in multiples of 50.
    if instrument == '^NSEI':
        return round(price / 50) * 50
    # For individual stocks, a simple rounding is used as a fallback.
    else:
        return round(price)

# --- Main Functions ---

@retry()
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

@retry()
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

def run_backtest(price_df, initial_capital, sma_short, sma_long, rsi_period):
    """Simulates the trading strategy and returns performance results."""
    print(f"Running backtest simulation with SMA({sma_short}/{sma_long}) and RSI({rsi_period})...")

    # --- Strategy Calculation (SMA Crossover + RSI + ATR) ---
    price_df[f'SMA_{sma_short}'] = price_df['close'].rolling(window=sma_short).mean()
    price_df[f'SMA_{sma_long}'] = price_df['close'].rolling(window=sma_long).mean()
    price_df.ta.rsi(length=rsi_period, append=True)
    price_df.ta.atr(length=ATR_PERIOD, append=True)

    # Generate raw state signals (1 for bullish, -1 for bearish)
    price_df['state'] = np.where(price_df[f'SMA_{sma_short}'] > price_df[f'SMA_{sma_long}'], 1, -1)
    # Find the exact point of state change to trigger a trade
    price_df['signal'] = price_df['state'].diff()

    # --- Portfolio Simulation ---
    # NOTE: This backtest simulates P&L based on the UNDERLYING asset's price movement
    # as a proxy for the option's performance, since we don't have option premium data.
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long (Call), -1 = short (Put)
    stop_loss = 0
    take_profit = 0
    trades = []
    portfolio_history = []

    for i, row in price_df.iterrows():
        # --- Check for Exit Conditions First ---
        if position == 1: # Exit a LONG (Call) position
            # 1. Stop-Loss Hit
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'Stop-Loss'
            # 2. Take-Profit Hit
            elif row['high'] >= take_profit:
                exit_price = take_profit
                exit_reason = 'Take-Profit'
            # 3. Signal flips to bearish
            elif row['signal'] < 0:
                exit_price = row['close']
                exit_reason = 'Signal Flip'
            else:
                portfolio_history.append({'timestamp': row['timestamp'], 'capital': capital})
                continue

            # Execute Long Exit
            position = 0
            profit = exit_price - entry_price
            capital += profit
            trades.append({
                'entry_date': entry_date, 'exit_date': row['timestamp'], 'type': 'Call',
                'entry_price': entry_price, 'exit_price': exit_price, 'profit': profit, 
                'exit_reason': exit_reason
            })

        elif position == -1: # Exit a SHORT (Put) position
            # 1. Stop-Loss Hit (price goes too high)
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'Stop-Loss'
            # 2. Take-Profit Hit (price goes low enough)
            elif row['low'] <= take_profit:
                exit_price = take_profit
                exit_reason = 'Take-Profit'
            # 3. Signal flips to bullish
            elif row['signal'] > 0:
                exit_price = row['close']
                exit_reason = 'Signal Flip'
            else:
                portfolio_history.append({'timestamp': row['timestamp'], 'capital': capital})
                continue

            # Execute Short Exit
            position = 0
            profit = entry_price - exit_price # Profit is reversed for shorts
            capital += profit
            trades.append({
                'entry_date': entry_date, 'exit_date': row['timestamp'], 'type': 'Put',
                'entry_price': entry_price, 'exit_price': exit_price, 'profit': profit, 
                'exit_reason': exit_reason
            })

        # --- Check for Entry Signal (if not in a position) ---
        if position == 0:
            atr_value = row[f'ATRr_{ATR_PERIOD}']
            # CALL signal: Bullish state change AND RSI is not overbought
            if row['signal'] > 0 and row[f'RSI_{rsi_period}'] < 70:
                position = 1
                entry_price = row['close']
                entry_date = row['timestamp']
                if pd.notna(atr_value):
                    stop_loss = entry_price - (atr_value * STOP_LOSS_MULTIPLIER)
                    take_profit = entry_price + (atr_value * TAKE_PROFIT_MULTIPLIER)
                else: # Fallback if ATR is not available
                    stop_loss = entry_price * 0.95
                    take_profit = entry_price * 1.10

            # PUT signal: Bearish state change
            elif row['signal'] < 0:
                position = -1
                entry_price = row['close']
                entry_date = row['timestamp']
                if pd.notna(atr_value):
                    stop_loss = entry_price + (atr_value * STOP_LOSS_MULTIPLIER)
                    take_profit = entry_price - (atr_value * TAKE_PROFIT_MULTIPLIER)
                else: # Fallback if ATR is not available
                    stop_loss = entry_price * 1.05
                    take_profit = entry_price * 0.90

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
    
    # 3. Exit Reason Analysis
    if not trades_df.empty and 'exit_reason' in trades_df.columns:
        print("\nExit Reasons:")
        print(trades_df['exit_reason'].value_counts().to_string())

    # 4. Sharpe Ratio (assuming 0 risk-free rate)
    if not portfolio_df.empty:
        daily_returns = portfolio_df['capital'].pct_change().dropna()
        if daily_returns.std() > 0:
            # Assuming 252 trading days in a year for annualization
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            print(f"\nSharpe Ratio (Annualized): {sharpe_ratio:.2f}")
        else:
            print("\nSharpe Ratio:           N/A (no volatility in returns)")

    # 5. Maximum Drawdown
    portfolio_df['peak'] = portfolio_df['capital'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['capital'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min() * 100
    
    print(f"\nMaximum Drawdown:       {max_drawdown:.2f}%")
    print("-------------------------------------\n")

def plot_equity_curve(portfolio_df, price_df, initial_capital, sma_short, sma_long, rsi_period):
    """Plots the portfolio's equity curve over time and saves it to a file."""
    if portfolio_df.empty:
        print("Cannot plot equity curve: No portfolio history available.")
        return

    print("Generating equity curve plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the equity curve
    ax.plot(portfolio_df['timestamp'], portfolio_df['capital'], label='Equity Curve', color='royalblue', linewidth=2)

    # --- Add Benchmark (Buy and Hold) ---
    if not price_df.empty:
        benchmark_df = price_df.copy()
        first_price = benchmark_df['close'].iloc[0]
        benchmark_df['benchmark_value'] = initial_capital * (benchmark_df['close'] / first_price)
        ax.plot(benchmark_df['timestamp'], benchmark_df['benchmark_value'], label='Buy & Hold Benchmark', color='darkorange', linestyle='--')

    # Formatting
    title = f'Portfolio Equity Curve\nSMA({sma_short}/{sma_long}), RSI({rsi_period})'
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    
    # Format y-axis to show currency
    formatter = mticker.FormatStrFormatter('$%.0f')
    ax.yaxis.set_major_formatter(formatter)
    
    # Add a horizontal line for the initial capital
    ax.axhline(y=initial_capital, color='grey', linestyle='--', label=f'Initial Capital (${initial_capital:,.0f})')
    
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to a file
    plot_filename = 'equity_curve.png'
    plt.savefig(plot_filename)
    print(f"Equity curve plot saved as '{plot_filename}'")

@retry()
def write_trade_log_to_sheets(spreadsheet, trades_df):
    """Writes the completed trades to a 'Trade Log' sheet for persistence."""
    if trades_df.empty:
        print("No trades to log.")
        return
    print("Writing trades to 'Trade Log' sheet...")
    try:
        worksheet = spreadsheet.worksheet("Trade Log")
    except gspread.exceptions.WorksheetNotFound:
        print("Creating 'Trade Log' worksheet.")
        worksheet = spreadsheet.add_worksheet(title="Trade Log", rows="1000", cols="20")

    # Convert datetime objects to strings for JSON compatibility
    log_df = trades_df.copy()
    log_df['entry_date'] = log_df['entry_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    log_df['exit_date'] = log_df['exit_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    headers = log_df.columns.tolist()
    data_to_write = [headers] + log_df.values.tolist()
    
    worksheet.clear()
    worksheet.update('A1', data_to_write, value_input_option='USER_ENTERED')
    print("Trade log written successfully to Google Sheets.")

def main(sma_short, sma_long, rsi_period):
    """Main function that runs the entire backtesting process."""
    try:
        spreadsheet = connect_to_google_sheets()
        price_df = read_price_data(spreadsheet)
        portfolio_history, trades = run_backtest(price_df, INITIAL_CAPITAL, sma_short, sma_long, rsi_period)
        calculate_and_print_performance(portfolio_history, trades, INITIAL_CAPITAL)
        # Generate and save the equity curve plot
        plot_equity_curve(portfolio_history, price_df, INITIAL_CAPITAL, sma_short, sma_long, rsi_period)
        # Write the results to Google Sheets to be used by the main script
        write_trade_log_to_sheets(spreadsheet, trades)
    except Exception as e:
        print(f"\n❌ An error occurred during the backtest: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest.")
    parser.add_argument('--sma_short', type=int, default=SMA_SHORT_WINDOW, 
                        help=f'Short SMA window (default: {SMA_SHORT_WINDOW})')
    parser.add_argument('--sma_long', type=int, default=SMA_LONG_WINDOW, 
                        help=f'Long SMA window (default: {SMA_LONG_WINDOW})')
    parser.add_argument('--rsi', type=int, default=RSI_PERIOD, 
                        help=f'RSI period (default: {RSI_PERIOD})')
    args = parser.parse_args()

    main(sma_short=args.sma_short, sma_long=args.sma_long, rsi_period=args.rsi)