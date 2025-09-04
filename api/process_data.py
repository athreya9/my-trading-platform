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
    """Calculates the Kelly Criterion percentage, win rate, and win/loss ratio."""
    if trades_df.empty or len(trades_df) < 20:
        print("Not enough historical trades (< 20) to calculate Kelly Criterion.")
        return np.nan, np.nan, np.nan

    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]

    win_rate = len(winning_trades) / len(trades_df)

    if len(losing_trades) == 0:
        win_loss_ratio = np.inf
    else:
        average_win = winning_trades['profit'].mean()
        average_loss = abs(losing_trades['profit'].mean())
        win_loss_ratio = average_win / average_loss

    # Kelly Criterion: K% = W – [(1 – W) / R]
    if win_loss_ratio > 0:
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    else:
        kelly_pct = 0

    # Use Half-Kelly for safety
    kelly_pct *= 0.5
    
    kelly_pct = max(0, min(kelly_pct, KELLY_CRITERION_CAP))
    
    print(f"Win Rate: {win_rate:.2%}, Win/Loss Ratio: {win_loss_ratio:.2f}, Calculated Kelly Criterion: {kelly_pct:.2%}")
    return kelly_pct, win_rate, win_loss_ratio

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

def fetch_historical_data(ticker, period, interval):
    """Fetches historical data from Yahoo Finance for a given ticker."""
    print(f"Fetching {interval} data for {ticker}...")
    try:
        stock_data = yf.download(
            tickers=ticker, period=period, interval=interval, auto_adjust=True
        )
        if stock_data.empty:
            print(f"No data downloaded for {ticker} at {interval} interval.")
            return pd.DataFrame()

        stock_data.reset_index(inplace=True)
        
        clean_df = pd.DataFrame()
        # The column name is 'Datetime' for intraday data, and 'Date' for daily data
        timestamp_col = 'Datetime' if 'Datetime' in stock_data.columns else 'Date'
        clean_df['timestamp'] = stock_data[timestamp_col]
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
    """Fetches data for all symbols and timeframes and returns a dictionary of DataFrames."""
    data_frames = {"15m": [], "1h": []}
    
    for symbol in SYMBOLS:
        # Fetch 15-minute data for the last 5 days
        df_15m = fetch_historical_data(symbol, period='5d', interval='15m')
        if not df_15m.empty:
            data_frames["15m"].append(df_15m)
            
        # Fetch 1-hour data for a longer period to establish a trend
        df_1h = fetch_historical_data(symbol, period='60d', interval='1h')
        if not df_1h.empty:
            data_frames["1h"].append(df_1h)

    if not data_frames["15m"]:
        raise Exception("No 15m data was fetched for any symbol. Halting process.")

    # Combine the lists of dataframes into single dataframes
    combined_df_15m = pd.concat(data_frames["15m"], ignore_index=True)
    combined_df_1h = pd.concat(data_frames["1h"], ignore_index=True)
    
    print(f"Successfully fetched {len(combined_df_15m)} rows of 15m data and {len(combined_df_1h)} rows of 1h data.")
    
    return {"15m": combined_df_15m, "1h": combined_df_1h}

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
    
    # Drop any rows where the close price is missing, as it's essential for all indicators
    price_df.dropna(subset=['close'], inplace=True)
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

        group['volume_avg_20'] = group['volume'].rolling(window=20).mean()

        # --- New Microstructure Indicators ---
        # 1. Realized Volatility (rolling standard deviation of log returns)
        group['log_return'] = np.log(group['close'] / group['close'].shift(1))
        group['realized_vol'] = group['log_return'].rolling(window=REALIZED_VOL_WINDOW).std()

        # 2. VWAP (Volume-Weighted Average Price) - calculated per day
        # This requires a nested groupby to reset the calculation for each new day.
        def calculate_daily_vwap(daily_group):
            # Fill NaN volumes with 0 to prevent issues in cumulative sum
            daily_group['volume'] = daily_group['volume'].fillna(0)
            cum_vol = daily_group['volume'].cumsum()
            # Avoid division by zero; use close price as VWAP if volume is zero.
            vwap_calc = (daily_group['close'] * daily_group['volume']).cumsum() / cum_vol.replace(0, np.nan)
            # If VWAP is still NaN (e.g., at the start), fill with the current close price
            daily_group['vwap'] = vwap_calc.fillna(daily_group['close'])
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

def generate_signals(price_data_dict, manual_controls_df, trade_log_df, sentiment_analyzer):
    """Generates trading signals for all instruments, applying a suite of validation and risk rules."""
    print("Generating signals with advanced rule validation...")
    
    # --- Setup ---
    kelly_pct, win_rate, win_loss_ratio = calculate_kelly_criterion(trade_log_df)
    all_signals_list = []
    price_df_15m = price_data_dict['15m']
    price_df_1h = price_data_dict['1h']

    # --- Iterate over each instrument's 15-minute data ---
    for instrument, group_15m in price_df_15m.groupby('instrument'):
        instrument_signals = []
        group_15m = group_15m.copy().dropna(subset=['SMA_50', 'RSI_14', f'ATRr_{ATR_PERIOD}', 'volume_avg_20'])
        if group_15m.empty:
            continue

        latest_15m = group_15m.iloc[-1]

        # --- Rule 1: Volatility Filter ---
        atr_percentage = (latest_15m[f'ATRr_{ATR_PERIOD}'] / latest_15m['close']) * 100
        if atr_percentage > 3.0:
            print(f"Skipping {instrument}: High volatility detected (ATR is {atr_percentage:.2f}% of price).")
            continue

        # --- Rule 2: Multi-Timeframe Confluence ---
        group_1h = price_df_1h[price_df_1h['instrument'] == instrument].copy()
        group_1h = group_1h.dropna(subset=['SMA_20', 'SMA_50'])
        if group_1h.empty:
            print(f"Skipping {instrument}: Not enough 1-hour data for trend analysis.")
            continue
        latest_1h = group_1h.iloc[-1]
        is_1h_bullish = latest_1h['SMA_20'] > latest_1h['SMA_50']

        # --- Automated Signal Logic (with new rules) ---
        sentiment_score = get_news_sentiment(instrument, sentiment_analyzer)
        reasons = []

        # --- BUY (CALL) Signal Conditions ---
        is_15m_bullish = latest_15m['SMA_20'] > latest_15m['SMA_50'] and latest_15m['RSI_14'] < 70
        volume_confirmed = latest_15m['volume'] > latest_15m['volume_avg_20']

        if is_15m_bullish and is_1h_bullish and volume_confirmed and sentiment_score > SENTIMENT_THRESHOLD:
            option_type = "CALL"
            reasons.append("15m/1h Bullish Trend")
            reasons.append("Volume Confirmation")
            reasons.append("Positive Sentiment")
            
            atr_val = latest_15m[f'ATRr_{ATR_PERIOD}']
            entry_price = latest_15m['close']
            stop_loss = entry_price - (atr_val * STOP_LOSS_MULTIPLIER)
            take_profit = entry_price + (atr_val * TAKE_PROFIT_MULTIPLIER)

            # --- Per-Trade Risk & Position Sizing ---
            account_size = 100000 # Example account size
            risk_per_trade_pct = 0.01 # 1% risk
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                continue # Avoid division by zero
            position_size = (account_size * risk_per_trade_pct) / risk_per_share

            instrument_signals.append({
                'option_type': option_type,
                'strike_price': get_atm_strike(entry_price, instrument),
                'underlying_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': round(position_size),
                'reason': ", ".join(reasons),
                'win_rate_p': win_rate,
                'win_loss_ratio_b': win_loss_ratio,
                'kelly_pct': kelly_pct
            })

        # --- SELL (PUT) Signal Conditions (Simplified for now) ---
        is_15m_bearish = latest_15m['SMA_20'] < latest_15m['SMA_50']
        if is_15m_bearish and not is_1h_bullish and sentiment_score < -SENTIMENT_THRESHOLD:
            # (Position sizing for PUTs would be similar)
            pass # Not fully implemented as per user request focus on BUY

        # --- Add common data and append to master list ---
        for sig in instrument_signals:
            sig['timestamp'] = latest_15m['timestamp']
            sig['instrument'] = instrument
            all_signals_list.append(sig)

    if not all_signals_list:
        print("No new signals generated after applying all rules.")
        return pd.DataFrame()
        
    final_signals_df = pd.DataFrame(all_signals_list)
    print(f"Generated {len(final_signals_df)} total signals after applying rules.")
    return final_signals_df

def generate_trade_package(signal):
    """Formats a single signal into the detailed trade package for the Advisor sheet."""
    instrument = signal['instrument'].replace('.NS', '')
    entry_price = signal['underlying_price']
    stop_loss = signal['stop_loss']
    take_profit = signal['take_profit']
    position_size = signal['position_size']
    win_rate = signal['win_rate_p']
    win_loss_ratio = signal['win_loss_ratio_b']
    
    trade_package = [
        ["Signal:", f"STRONG BUY {instrument}", "Entry:", f"<= {entry_price:.2f}"],
        ["Reason:", signal['reason'], "Target:", f"{take_profit:.2f} (1.5x ATR)"],
        ["Risk:", f"Stop Loss: {stop_loss:.2f}", "Hold Time:", "2-4 hours"],
        ["Kelly Calc:", f"Win Rate (p)={win_rate:.0%}, Win/Loss (b)={win_loss_ratio:.1f}", "Position Size:", f"{position_size} Shares"],
        ["Capital:", "Risk per Trade: ₹1000 (1% of ₹100k)"]
    ]
    return trade_package

def write_to_sheets(spreadsheet, price_df, signals_df):
    """Writes price data, signals, and the detailed trade package to their respective sheets."""
    # --- Get Worksheet Objects ---
    # (Assuming these worksheets exist, for brevity)
    data_worksheet = spreadsheet.worksheet(DATA_WORKSHEET_NAME)
    signals_worksheet = spreadsheet.worksheet(SIGNALS_WORKSHEET_NAME)
    advisor_worksheet = spreadsheet.worksheet("Advisor")

    # --- Write Price Data ---
    if not price_df.empty:
        print(f"Writing {len(price_df)} rows to '{DATA_WORKSHEET_NAME}'...")
        price_df_str = price_df.copy()
        price_df_str['timestamp'] = price_df_str['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        price_df_str.fillna('', inplace=True)
        price_data_to_write = [price_df_str.columns.tolist()] + price_df_str.values.tolist()
        data_worksheet.clear()
        data_worksheet.update(price_data_to_write, value_input_option='USER_ENTERED')
        print("Price data written successfully.")

    # --- Write Signal Data ---
    if not signals_df.empty:
        print(f"Writing {len(signals_df)} rows to '{SIGNALS_WORKSHEET_NAME}'...")
        signals_df_str = signals_df.copy()
        signals_df_str['timestamp'] = signals_df_str['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        signals_df_str.fillna('', inplace=True)
        signal_data_to_write = [signals_df_str.columns.tolist()] + signals_df_str.values.tolist()
        signals_worksheet.clear()
        signals_worksheet.update(signal_data_to_write, value_input_option='USER_ENTERED')
        print("Signal data written successfully.")
    else:
        signals_worksheet.clear()

    # --- Write Advisor Trade Package ---
    advisor_worksheet.clear()
    if not signals_df.empty:
        # For now, display the first signal as the trade package
        main_signal = signals_df.iloc[0]
        trade_package_data = generate_trade_package(main_signal)
        
        print("Writing detailed trade package to 'Advisor' sheet...")
        advisor_worksheet.update('A1', trade_package_data, value_input_option='USER_ENTERED')
        print("Trade package written successfully.")
    else:
        print("No signals to generate a trade package. Clearing Advisor sheet.")
        advisor_worksheet.update('A1', [["No valid trading signals found after applying all rules."]], value_input_option='USER_ENTERED')


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

    # Step 4: Collect new data for multiple timeframes
    price_data_dict = run_data_collection()
    
    # Step 5: Calculate all indicators for all instruments and timeframes
    price_data_dict["15m"] = calculate_indicators(price_data_dict["15m"])
    price_data_dict["1h"] = calculate_indicators(price_data_dict["1h"])
    
    # Step 6: Generate signals using data, manual controls, historical performance, and sentiment
    signals_df = generate_signals(price_data_dict, manual_controls_df, trade_log_df, sentiment_analyzer)
    
    # Step 7: Write both data and signals to the sheets
    write_to_sheets(spreadsheet, price_data_dict["15m"], signals_df)

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting trading signal process...")
    try:
        main()
        print("\n✅ Process completed successfully.")
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        sys.exit(1) # Exit with a non-zero code to fail the GitHub Action