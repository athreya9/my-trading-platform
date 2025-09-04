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
from transformers import pipeline
import torch
import sys

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard" # The name of your Google Sheet
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"

# Data collection settings
SYMBOLS = ['RELIANCE.NS', '^NSEI', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']

# Signal generation settings
SIGNAL_HEADERS = ['timestamp', 'instrument', 'option_type', 'strike_price', 'underlying_price', 'stop_loss', 'take_profit', 'kelly_pct', 'sentiment_score']

# Risk Management settings
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = 2.0  # e.g., 2 * ATR below entry price
TAKE_PROFIT_MULTIPLIER = 4.0 # e.g., 4 * ATR above entry price (for a 1:2 risk/reward ratio)

# Microstructure settings
REALIZED_VOL_WINDOW = 20 # e.g., 20 periods for volatility calculation
KELLY_CRITERION_CAP = 0.20 # Maximum percentage of capital to risk, as per Kelly Criterion

# NLP Sentiment Analysis settings
NLP_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_THRESHOLD = 0.1 # Minimum positive/negative sentiment score to influence a signal

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

def read_trade_log(spreadsheet):
    """Reads the historical trade log from the 'Trade Log' sheet."""
    print("Reading data from 'Trade Log' sheet...")
    try:
        worksheet = spreadsheet.worksheet("Trade Log")
        records = worksheet.get_all_records()
        if not records:
            print("Trade log is empty. Cannot calculate Kelly Criterion.")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # Convert profit to numeric, coercing errors to NaN and then dropping them
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
        df.dropna(subset=['profit'], inplace=True)
        
        print(f"Successfully read {len(df)} trades from the log.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        print("Warning: 'Trade Log' worksheet not found. Cannot calculate Kelly Criterion.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not read trade log: {e}")
        return pd.DataFrame()

def calculate_kelly_criterion(trades_df):
    """Calculates the Kelly Criterion percentage from a DataFrame of historical trades."""
    # Require a minimum number of trades for a statistically significant calculation
    if trades_df.empty or len(trades_df) < 20:
        print("Not enough historical trades (< 20) to calculate Kelly Criterion.")
        return np.nan

    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]

    if len(losing_trades) == 0:
        return 0.20 # If no losses, return max capped value.
    if len(winning_trades) == 0:
        return 0.0 # If no wins, risk nothing.

    # 1. Calculate Win Rate (W)
    win_rate = len(winning_trades) / len(trades_df)

    # 2. Calculate Win/Loss Ratio (R)
    average_win = winning_trades['profit'].mean()
    average_loss = abs(losing_trades['profit'].mean())
    win_loss_ratio = average_win / average_loss

    # 3. Calculate Kelly Criterion: K% = W – [(1 – W) / R]
    kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Cap the Kelly percentage to a reasonable maximum to avoid over-leveraging
    kelly_pct = max(0, min(kelly_pct, KELLY_CRITERION_CAP)) 
    print(f"Win Rate: {win_rate:.2%}, Win/Loss Ratio: {win_loss_ratio:.2f}, Calculated Kelly Criterion: {kelly_pct:.2%}")
    return kelly_pct

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

def get_atm_strike(price, instrument):
    """
    Calculates a theoretical at-the-money (ATM) strike price by rounding.
    This is a rule-based estimation as we don't have live options chain data.
    """
    # For Nifty 50 (^NSEI), strike prices are typically in multiples of 50.
    if instrument == '^NSEI':
        return round(price / 50) * 50
    # For individual stocks, this varies. We'll use a simple rounding for demonstration.
    # A robust solution would have a mapping of tickers to their strike steps.
    else:
        return round(price)

def get_news_sentiment(instrument, analyzer):
    """Fetches news for an instrument and returns an aggregated sentiment score."""
    print(f"Fetching and analyzing sentiment for {instrument}...")
    try:
        # Fetch news using yfinance's built-in news feature
        ticker_news = yf.Ticker(instrument).news
        if not ticker_news:
            print(f"No news found for {instrument}.")
            return 0.0 # Return neutral sentiment if no news

        headlines = [news['title'] for news in ticker_news[:8]] # Analyze latest 8 headlines
        
        # Analyze sentiment using the pre-loaded FinBERT model
        sentiments = analyzer(headlines)
        
        # Convert sentiment labels and scores to a single numerical value
        score = 0.0
        for sentiment in sentiments:
            if sentiment['label'] == 'positive':
                score += sentiment['score']
            elif sentiment['label'] == 'negative':
                score -= sentiment['score']
        
        # Return the average score
        avg_score = score / len(sentiments) if sentiments else 0.0
        print(f"Average sentiment score for {instrument}: {avg_score:.3f}")
        return avg_score

    except Exception as e:
        print(f"Warning: Could not get sentiment for {instrument}: {e}")
        return 0.0 # Return neutral on error

def generate_signals(price_df, manual_controls_df, trade_log_df, sentiment_analyzer):
    """Generates trading signals for all instruments, applying manual overrides."""
    print("Generating signals for all instruments...")
    
    # Calculate Kelly Criterion once based on the historical trade log
    kelly_pct = calculate_kelly_criterion(trade_log_df)

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
                instrument_signals.append({'option_type': 'PUT (MANUAL)', 'strike_price': get_atm_strike(latest_row['close'], instrument), 'underlying_price': latest_row['close']})
                manual_override_triggered = True

            # 2. Limit Price Alert
            limit_price = pd.to_numeric(control['limit_price'], errors='coerce')
            if pd.notna(limit_price) and latest_row['close'] > limit_price:
                print(f"'{instrument}' price ({latest_row['close']:.2f}) is above limit ({limit_price:.2f}).")
                instrument_signals.append({'option_type': 'PUT (LIMIT HIT)', 'strike_price': get_atm_strike(latest_row['close'], instrument), 'underlying_price': latest_row['close']})
                manual_override_triggered = True

        # --- Automated Signal Logic ---
        # This new logic checks the CURRENT STATE of the instrument, not just a crossover event.
        if not manual_override_triggered:
            # Get sentiment score for the current instrument
            sentiment_score = get_news_sentiment(instrument, sentiment_analyzer)

            # Check for an active "BUY" state
            if latest_row['SMA_20'] > latest_row['SMA_50'] and latest_row['RSI_14'] < 70 and sentiment_score > SENTIMENT_THRESHOLD:
                option_type = "CALL"
                strike_price = get_atm_strike(latest_row['close'], instrument)
                atr_val = latest_row[f'ATRr_{ATR_PERIOD}']
                sl, tp = np.nan, np.nan
                if pd.notna(atr_val):
                    sl = latest_row['close'] - (atr_val * STOP_LOSS_MULTIPLIER)
                    tp = latest_row['close'] + (atr_val * TAKE_PROFIT_MULTIPLIER)
                instrument_signals.append({
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'underlying_price': latest_row['close'],
                    'stop_loss': sl,
                    'take_profit': tp,
                    'kelly_pct': kelly_pct,
                    'sentiment_score': sentiment_score
                })

            # Check for an active "SELL" state -> Generate PUT signal
            elif latest_row['SMA_20'] < latest_row['SMA_50'] and sentiment_score < -SENTIMENT_THRESHOLD:
                option_type = "PUT"
                strike_price = get_atm_strike(latest_row['close'], instrument)
                instrument_signals.append({
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'underlying_price': latest_row['close'],
                    'kelly_pct': kelly_pct,
                    'sentiment_score': sentiment_score
                })

        # Add common data to all signals found for this instrument
        for sig in instrument_signals:
            sig['timestamp'] = latest_row['timestamp']
            sig['instrument'] = instrument
            # Fill missing risk keys for manual signals
            sig.setdefault('stop_loss', np.nan)
            sig.setdefault('take_profit', np.nan)
            sig.setdefault('kelly_pct', kelly_pct)
            sig.setdefault('sentiment_score', np.nan)
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
        # Replace NaN values with empty strings to make them JSON compliant for gspread
        signals_df.fillna('', inplace=True)
        
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

    # Step 3: Read historical trade log to calculate performance stats
    trade_log_df = read_trade_log(spreadsheet)
    
    # Initialize the sentiment analysis pipeline once
    print("Initializing sentiment analysis model (this may take a moment on first run)...")
    # Use GPU if available, otherwise CPU.
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline("sentiment-analysis", model=NLP_MODEL_NAME, device=device)
    print("Sentiment model initialized.")

    # Step 4: Collect new data
    price_df = run_data_collection()
    
    # Step 5: Calculate all indicators for all instruments
    price_df = calculate_indicators(price_df)
    
    # Step 6: Generate signals using data, manual controls, historical performance, and sentiment
    signals_df = generate_signals(price_df.copy(), manual_controls_df, trade_log_df, sentiment_analyzer) # Pass a copy to avoid pandas warnings
    
    # Step 7: Write both data and signals to the sheets
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