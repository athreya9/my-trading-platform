# process_data.py
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting api/process_data.py")

# A single, combined script for GitHub Actions.
# It fetches data, generates signals, and updates Google Sheets.
from flask import Blueprint, request, jsonify
from kiteconnect import KiteConnect
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
import pytz
import joblib
from functools import wraps, lru_cache
import os
import yfinance as yf
from dotenv import load_dotenv

import logging
import re
import sys
from . import config
from .ai_analysis_engine import AIAnalysisEngine, send_ai_powered_alert
from .alert_manager import AlertManager
from .data_collector import DataCollector
from .ai_training_pipeline import AITrainingPipeline
from .market_calendar import should_run_trading_bot

alert_manager = AlertManager()
import requests
from .firestore_utils import (
    init_json_storage as init_firestore_client, 
    get_db, 
    write_data_to_firestore, 
    check_bot_status, 
    read_manual_controls, 
    read_trade_log,
    get_dashboard_data
)



def save_dataframe_to_json(df, filename):
    """Save DataFrame to JSON file"""
    os.makedirs('data', exist_ok=True)
    filepath = f"data/{filename}.json"
    
    # Convert DataFrame to records
    records = df.to_dict('records')
    
    # Read existing data
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    
    # Append new data with timestamp
    for record in records:
        record['timestamp'] = datetime.now().isoformat()
        existing_data.append(record)
    
    # Save back to file
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"✅ Data saved to {filepath}")

def save_to_json(data_type, data):
    """Save single data point to JSON"""
    os.makedirs('data', exist_ok=True)
    filename = f"data/{data_type}.json"
    
    # Read existing data
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    
    # Append new data
    data_with_timestamp = {
        **data,
        'timestamp': datetime.now().isoformat()
    }
    existing_data.append(data_with_timestamp)
    
    # Save back to file
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"✅ Data saved to {filename}")

# --- Defensive Import for feedparser ---
# This prevents the entire application from crashing on startup if 'feedparser' is not installed.
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    # This warning will appear in the logs on Google Cloud Run.
    logging.warning("`feedparser` library not found. News sentiment analysis will be disabled. To enable, run 'pip install feedparser' and add it to requirements.txt.")

# --- NEW: Custom Exception for Graceful Halting ---
class BotHaltedException(Exception):
    """Custom exception to indicate the bot was halted by user control."""
    pass

# --- In-memory cache for the dashboard endpoint ---
dashboard_cache = {
    "data": None,
    "timestamp": None,
}
CACHE_LIFETIME_SECONDS = 60  # Cache data for 60 seconds
cache_lock = threading.Lock()

# --- Logging Configuration ---
# Use a custom formatter to ensure all log times are in UTC for consistency
formatter = logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s')
formatter.converter = time.gmtime  # Use UTC for asctime

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Data collection settings are now imported from config.py

# Signal generation settings
SIGNAL_HEADERS = ['timestamp', 'instrument', 'option_type', 'strike_price', 'underlying_price', 'stop_loss', 'take_profit', 'position_size', 'reason', 'sentiment_score', 'kelly_pct', 'win_rate_p', 'win_loss_ratio_b', 'quality_score']


# Risk Management settings
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = 2.0  # e.g., 2 * ATR below entry price


# --- AI Model Configuration ---
# The confidence level the AI must have to generate a signal.
# Based on your training script, 0.80 (80%) is a good starting point for high precision.

AI_CONFIDENCE_THRESHOLD = 0.80

# Path to the trained model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trading_model.pkl')

def get_ai_model():
    """
    Lazily loads the AI model and its associated scaler from the file system.
    This function is NOT cached to ensure that if the model file is updated
    (e.g., by a retraining job), the new model is loaded on the next run.
    The OS file cache provides sufficient performance.
    """
    try:
        # --- CRITICAL FIX: Load both the model and the scaler ---
        saved_objects = joblib.load(MODEL_PATH)
        model = saved_objects['model']
        scaler = saved_objects['scaler']
        logger.info(f"Successfully loaded AI model and scaler from {MODEL_PATH}")
        return model, scaler
    except FileNotFoundError:
        logger.warning(f"AI model file '{os.path.basename(MODEL_PATH)}' not found. AI-based signals will be disabled.")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load AI model from {MODEL_PATH} due to an error: {e}. AI-based signals will be disabled.")
        return None, None

TAKE_PROFIT_MULTIPLIER = 4.0 # e.g., 4 * ATR above entry price (for a 1:2 risk/reward ratio)
MAX_RISK_PER_TRADE = 0.01  # Golden Rule: 1% of capital
ACCOUNT_SIZE = 100000.0 # Example account size

# Microstructure settings
REALIZED_VOL_WINDOW = 20 # e.g., 20 periods for volatility calculation
KELLY_CRITERION_CAP = 0.20 # Maximum percentage of capital to risk, as per Kelly Criterion

# --- New Market Internals Configuration ---
VIX_THRESHOLD = 20 # VIX level above which market is considered fearful/volatile

# --- New Price Action Configuration ---
# Defines a swing point as a candle higher/lower than its immediate neighbors.
SWING_POINT_LOOKBACK = 1

# NLP Sentiment Analysis settings
SENTIMENT_THRESHOLD = 0 # Keyword score must be positive to be considered.

# --- Kite Connect to yfinance Symbol Mapping ---
# This helps translate yfinance index names to Kite's trading symbols
YFINANCE_TO_KITE_MAP = {
    '^NSEI': 'NIFTY 50',
    '^INDIAVIX': 'INDIA VIX',
    '^CNXIT': 'NIFTY IT',
    '^NSEBANK': 'NIFTY BANK',
    '^CNXAUTO': 'NIFTY AUTO'
}

# --- Main Functions ---

def read_manual_controls(db):
    """Reads manual override settings from the 'manual_controls.json' file."""
    logger.info("Reading data from 'manual_controls.json'...")
    try:
        with open('data/manual_controls.json', 'r') as f:
            controls = json.load(f)

        if not controls:
            logger.info("No manual controls found or file is empty.")
            return pd.DataFrame()

        # The JSON is expected to be a dictionary where keys are instrument names
        df = pd.DataFrame.from_dict(controls, orient='index')
        df.index.name = 'instrument'
        df.reset_index(inplace=True)

        # Set instrument as index for easy lookup
        df.set_index('instrument', inplace=True)
        logger.info("Manual controls loaded successfully from JSON.")
        return df
    except FileNotFoundError:
        logger.warning("'manual_controls.json' not found. No manual controls will be applied.")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not read manual controls from JSON: {e}")
        return pd.DataFrame()

def read_trade_log(db):
    """Reads the historical trade log from the 'trade_log.json' file."""
    logger.info("Reading data from 'trade_log.json'...")
    try:
        with open('data/trade_log.json', 'r') as f:
            trades = json.load(f)

        if not trades:
            logger.info("Trade log is empty. Cannot calculate Kelly Criterion.")
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        # The JSON file should have a 'P/L' or 'p_l' column
        if 'P/L' not in df.columns:
            if 'p_l' in df.columns:
                df.rename(columns={'p_l': 'P/L'}, inplace=True)
            else:
                logger.warning("'P/L' or 'p_l' column not found in 'trade_log.json'. Cannot calculate performance.")
                return pd.DataFrame()

        # Convert profit to numeric, coercing errors to NaN and then dropping them
        df['profit'] = pd.to_numeric(df['P/L'], errors='coerce')
        df.dropna(subset=['profit'], inplace=True)

        logger.info(f"Successfully read {len(df)} trades from the log.")
        return df
    except FileNotFoundError:
        logger.warning("'trade_log.json' not found. Cannot calculate Kelly Criterion.")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not read trade log from JSON: {e}")
        return pd.DataFrame()

def check_bot_status(db):
    """Checks the 'bot_control.json' file for a 'running' status."""
    logger.info("Checking bot operational status from 'bot_control.json'...")
    try:
        with open('data/bot_control.json', 'r') as f:
            status_data = json.load(f)
        
        # The data is a list of dicts, get the last one
        if isinstance(status_data, list) and status_data:
            latest_status = status_data[-1]
            status = latest_status.get('status')
            logger.info(f"Bot status from JSON: '{status}'")
            if status and status.lower() == 'running':
                return True
        
        logger.warning("Bot status is not 'running' or file is empty/invalid. Halting execution.")
        return False
    except FileNotFoundError:
        logger.warning("'bot_control.json' not found. Assuming bot is stopped for safety.")
        return False
    except Exception as e:
        logger.error(f"Could not read bot status from JSON: {e}. Halting for safety.")
        return False

def calculate_kelly_criterion(trades_df):
    """Calculates the Kelly Criterion percentage, win rate, and win/loss ratio."""
    if trades_df.empty or len(trades_df) < 20:
        logger.info("Not enough historical trades (< 20) to calculate Kelly Criterion.")
        return np.nan, np.nan, np.nan

    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]

    win_rate = len(winning_trades) / len(trades_df)

    # If there are no losing trades, the ratio is technically infinite.
    # If there are no winning trades, the ratio is 0.
    if len(losing_trades) > 0 and len(winning_trades) > 0:
        average_win = winning_trades['profit'].mean()
        average_loss = abs(losing_trades['profit'].mean())
        win_loss_ratio = average_win / average_loss
    else:
        win_loss_ratio = 0

    # Kelly Criterion: K% = W – [(1 – W) / R]
    if win_loss_ratio > 0:
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    else:
        kelly_pct = 0

    # Use Half-Kelly for safety
    kelly_pct *= 0.5
    
    kelly_pct = max(0, min(kelly_pct, KELLY_CRITERION_CAP))
    
    logger.info(f"Win Rate: {win_rate:.2%}, Win/Loss Ratio: {win_loss_ratio:.2f}, Calculated Kelly Criterion: {kelly_pct:.2%}")
    return kelly_pct, win_rate, win_loss_ratio

def connect_to_kite():
    """Initializes the Kite Connect client using credentials from environment variables."""
    logger.info("Attempting to authenticate with Kite Connect...")
    api_key = os.getenv('KITE_API_KEY', '').strip().strip('"\'')
    access_token = os.getenv('KITE_ACCESS_TOKEN', '').strip().strip('"\'')

    if not api_key or not access_token:
        raise ValueError("KITE_API_KEY or KITE_ACCESS_TOKEN environment variables not found.")
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Verify connection
        profile = kite.profile()
        logger.info(f"Kite Connect authentication successful for user: {profile.get('user_id')}")
        return kite
    except Exception as e:
        logger.error(f"Failed to connect to Kite Connect: {e}", exc_info=True)
        raise Exception(f"Error connecting to Kite Connect: {e}")

@lru_cache(maxsize=1) # Cache the result to avoid repeated API calls within the same run
def get_instrument_map(kite):
    """Fetches all instruments from Kite and creates a symbol-to-token map."""
    logger.info("Fetching instrument list from Kite Connect...")
    try:
        instruments = kite.instruments("NSE") # We are focusing on the NSE segment
        instrument_df = pd.DataFrame(instruments)
        
        # Create a map for faster lookups: { 'TRADINGSYMBOL': instrument_token }
        # e.g., { 'RELIANCE': 256265, 'NIFTY 50': 256265, ... }
        symbol_to_token_map = pd.Series(instrument_df.instrument_token.values, index=instrument_df.tradingsymbol).to_dict()
        logger.info(f"Successfully created instrument map with {len(symbol_to_token_map)} entries.")
        return symbol_to_token_map
    except Exception as e:
        logger.error(f"Failed to fetch or process instrument list: {e}", exc_info=True)
        raise

def fetch_historical_data(kite, instrument_token, from_date, to_date, interval, original_symbol):
    """Fetches historical data from Kite Connect for a given instrument token."""
    logger.info(f"Fetching {interval} data for {original_symbol} (Token: {instrument_token})...")
    try:
        records = kite.historical_data(instrument_token, from_date, to_date, interval)
        if not records:
            logger.warning(f"No data downloaded for {original_symbol} at {interval} interval.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Rename columns to match the rest of the script's expectations
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        df['instrument'] = original_symbol
        return df[['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.warning(f"Could not fetch Kite data for {original_symbol}: {e}")
        return pd.DataFrame()

def fetch_historical_data_yfinance(symbols, interval, period):
    """Fetches historical data from Yahoo Finance for testing outside market hours."""
    logger.info(f"Fetching {interval} data for {len(symbols)} symbols from yfinance (period: {period})...")
    try:
        df = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by='ticker',
            threads=True
        )
        if df.empty:
            logger.warning("yfinance returned no data.")
            return pd.DataFrame()

        # If multiple tickers, stack to get a 'instrument' column
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0).rename_axis(['timestamp', 'instrument']).reset_index()
        else: # If only one ticker, it's a single index df
            df['instrument'] = symbols[0]
            df = df.reset_index()

        # Rename columns to be consistent with Kite data
        df.rename(columns={
            'Timestamp': 'timestamp', 'Date': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }, inplace=True, errors='ignore')

        # Convert timestamp to timezone-aware (yfinance returns timezone-aware for intraday)
        if df['timestamp'].dt.tz is None:
             df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
        else:
             df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')

        logger.info(f"Successfully fetched {len(df)} rows from yfinance.")
        return df[['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logger.warning(f"Could not fetch yfinance data: {e}")
        return pd.DataFrame()

def fetch_option_chain(kite, underlying_instrument):
    """Fetches and structures the option chain for a given underlying instrument."""
    logger.info(f"Fetching and structuring option chain for {underlying_instrument}...")
    try:
        # Get all instruments for the NFO exchange
        nfo_instruments = kite.instruments('NFO')
        nfo_df = pd.DataFrame(nfo_instruments)

        # Filter for options of the underlying instrument
        options_df = nfo_df[(nfo_df['name'] == underlying_instrument) & (nfo_df['segment'] == 'NFO-OPT')].copy()

        if options_df.empty:
            logger.warning(f"No options found for {underlying_instrument}")
            return None

        # Find the nearest expiry date
        options_df['expiry'] = pd.to_datetime(options_df['expiry'])
        nearest_expiry = options_df['expiry'].min()
        options_df = options_df[options_df['expiry'] == nearest_expiry]

        # Get the LTP of the underlying instrument to find the ATM strike
        # Use the correct tradingsymbol for the quote API
        underlying_tradingsymbol_for_quote = YFINANCE_TO_KITE_MAP.get('^NSEI') if underlying_instrument == 'NIFTY' else underlying_instrument
        quote = kite.quote([f"NSE:{underlying_tradingsymbol_for_quote}"])
        if not quote or f"NSE:{underlying_tradingsymbol_for_quote}" not in quote:
            logger.warning(f"Could not fetch quote for NSE:{underlying_tradingsymbol_for_quote}. Cannot determine ATM strike.")
            return None
        ltp = quote[underlying_tradingsymbol_for_quote]['last_price']
        atm_strike = round(ltp / 50) * 50

        # Filter for a range of strike prices around the ATM strike
        strike_range = 10
        min_strike = atm_strike - (strike_range * 50)
        max_strike = atm_strike + (strike_range * 50)
        options_df = options_df[(options_df['strike'] >= min_strike) & (options_df['strike'] <= max_strike)]

        # Separate call and put options
        calls_df = options_df[options_df['instrument_type'] == 'CE']
        puts_df = options_df[options_df['instrument_type'] == 'PE']

        # Fetch quotes for the filtered options
        call_instruments = calls_df['tradingsymbol'].tolist()
        put_instruments = puts_df['tradingsymbol'].tolist()
        option_quotes = kite.quote(call_instruments + put_instruments)

        # Create the structured option chain
        option_chain_data = []
        for strike in sorted(options_df['strike'].unique()):
            call_instrument = calls_df[calls_df['strike'] == strike]
            put_instrument = puts_df[puts_df['strike'] == strike]

            if not call_instrument.empty and not put_instrument.empty:
                call_symbol = call_instrument.iloc[0]['tradingsymbol']
                put_symbol = put_instrument.iloc[0]['tradingsymbol']

                call_quote = option_quotes.get(call_symbol, {})
                put_quote = option_quotes.get(put_symbol, {})

                option_chain_data.append({
                    'strike': strike,
                    'call_ltp': call_quote.get('last_price', 0),
                    'call_oi': call_quote.get('oi', 0),
                    'put_ltp': put_quote.get('last_price', 0),
                    'put_oi': put_quote.get('oi', 0),
                })

        option_chain_df = pd.DataFrame(option_chain_data)
        logger.info(f"Successfully built option chain for {underlying_instrument} with {len(option_chain_df)} strikes.")
        return option_chain_df

    except Exception as e:
        logger.error(f"Failed to fetch and structure option chain for {underlying_instrument}: {e}", exc_info=True)
        return None


def run_data_collection():
    """Fetches data for all symbols and timeframes using yfinance."""
    
    logger.warning("Using yfinance for historical data.")
    data_frames = {"15m": None, "30m": None, "1h": None}
    
    yfinance_params = {
        "15m": {"interval": "15m", "period": "5d"},
        "30m": {"interval": "30m", "period": "10d"},
        "1h": {"interval": "60m", "period": "60d"}
    }
    
    for tf, params in yfinance_params.items():
        data_frames[tf] = fetch_historical_data_yfinance(config.SYMBOLS, params['interval'], params['period'])
    
    combined_df_15m = data_frames["15m"]
    combined_df_30m = data_frames["30m"]
    combined_df_1h = data_frames["1h"]

    if combined_df_15m is None or combined_df_15m.empty:
        logger.warning("No 15m data was fetched for any symbol.")
        return {"15m": pd.DataFrame(), "30m": pd.DataFrame(), "1h": pd.DataFrame()}

    logger.info(f"Processed {len(combined_df_15m)} rows (15m), {len(combined_df_30m)} rows (30m), {len(combined_df_1h)} rows (1h).")
    return {"15m": combined_df_15m, "30m": combined_df_30m, "1h": combined_df_1h}

def calculate_indicators(price_df):
    """
    Calculates all technical indicators for all instruments using a compatible,
    robust method that avoids the outdated `ta.Strategy`.
    """
    logger.info("Calculating indicators for all instruments...")
    # --- Data Cleaning and Preparation ---
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    # Ensure all key columns are numeric, coercing errors to NaN
    for col in ['open', 'high', 'low', 'close', 'volume']:
        price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Drop any rows where the close price is missing, as it's essential for all indicators
    price_df.dropna(subset=['close'], inplace=True)
    price_df.sort_values(['instrument', 'timestamp'], inplace=True)

    # Define a single function to apply all indicators to a group (a single instrument's data)
    def apply_all_indicators(group):
        try:
            group = group.copy()

            # --- Standard TA-Lib Indicators ---
            # Note: Using pandas_ta's direct methods is clean and efficient.
            group.ta.sma(length=20, append=True)
            group.ta.sma(length=50, append=True)
            group.ta.rsi(length=14, append=True)
            group.ta.macd(fast=12, slow=26, signal=9, append=True)
            group.ta.atr(length=ATR_PERIOD, append=True)
            group['volume_avg_20'] = group['volume'].rolling(window=20).mean()

            # --- New Microstructure Indicators ---
            # 1. Realized Volatility (rolling standard deviation of log returns)
            group['log_return'] = np.log(group['close'] / group['close'].shift(1))
            group['realized_vol'] = group['log_return'].rolling(window=REALIZED_VOL_WINDOW).std()

            # 2. VWAP (Volume-Weighted Average Price) - calculated per day
            def calculate_daily_vwap(daily_group):
                # Fill NaN volumes with 0 to prevent issues in cumulative sum
                daily_group['volume'] = daily_group['volume'].fillna(0)
                cum_vol = daily_group['volume'].cumsum()
                # Avoid division by zero; use close price as VWAP if volume is zero.
                vwap_calc = (daily_group['close'] * daily_group['volume']).cumsum() / cum_vol.replace(0, np.nan)
                # If VWAP is NaN (e.g., at the start), fill with the current close price
                daily_group['vwap'] = vwap_calc.fillna(daily_group['close'])
                return daily_group

            # Apply VWAP calculation daily. Using group_keys=False to avoid extra index level.
            if not group.empty:
                group = group.groupby(group['timestamp'].dt.date, group_keys=False).apply(calculate_daily_vwap)

            return group
        except Exception as e:
            instrument_name = group['instrument'].iloc[0] if not group.empty else "Unknown"
            logger.error(f"Could not calculate indicators for instrument '{instrument_name}'. Error: {e}", exc_info=True)
            # Return the original group; it will be filtered out by later logic that requires indicator columns.
            return group

    # Use groupby().apply() to run the indicator calculations for each instrument
    price_df = price_df.groupby('instrument', group_keys=False).apply(apply_all_indicators)
    
    logger.info("Indicators calculated successfully for all instruments.")
    return price_df

def apply_price_action_indicators(group):
    """
    Detects Fair Value Gaps (FVGs) and Market Structure Shifts (BOS/CHoCH).
    This function is designed to be used within a pandas groupby().apply().
    """
    group = group.copy()

    # 1. Fair Value Gaps (FVG) Detection
    # A bullish FVG is where the low of candle[i-2] is above the high of candle[i].
    # The gap is the space on candle[i-1].
    bull_fvg_mask = group['low'].shift(2) > group['high']
    group['fvg_bull_top'] = pd.Series(np.where(bull_fvg_mask, group['low'].shift(2), np.nan), index=group.index).shift(1)
    group['fvg_bull_bottom'] = pd.Series(np.where(bull_fvg_mask, group['high'], np.nan), index=group.index).shift(1)

    # A bearish FVG is where the high of candle[i-2] is below the low of candle[i].
    bear_fvg_mask = group['high'].shift(2) < group['low']
    group['fvg_bear_top'] = pd.Series(np.where(bear_fvg_mask, group['high'].shift(2), np.nan), index=group.index).shift(1)
    group['fvg_bear_bottom'] = pd.Series(np.where(bear_fvg_mask, group['low'], np.nan), index=group.index).shift(1)

    # 2. Market Structure (3-candle Swing Points)
    # A swing high is a peak: high[i-1] is higher than its neighbors.
    is_swing_high = (group['high'].shift(SWING_POINT_LOOKBACK) > group['high'].shift(SWING_POINT_LOOKBACK * 2)) & \
                    (group['high'].shift(SWING_POINT_LOOKBACK) > group['high'])
    # A swing low is a trough: low[i-1] is lower than its neighbors.
    is_swing_low = (group['low'].shift(SWING_POINT_LOOKBACK) < group['low'].shift(SWING_POINT_LOOKBACK * 2)) & \
                   (group['low'].shift(SWING_POINT_LOOKBACK) < group['low'])

    group['swing_high_price'] = np.where(is_swing_high, group['high'].shift(SWING_POINT_LOOKBACK), np.nan)
    group['swing_low_price'] = np.where(is_swing_low, group['low'].shift(SWING_POINT_LOOKBACK), np.nan)

    # Forward-fill to get the last known swing points at every candle
    group['last_swing_high'] = group['swing_high_price'].ffill()
    group['last_swing_low'] = group['swing_low_price'].ffill()

    # 3. Market Structure Shift Detection (BOS/CHoCH)
    # We use the existing SMA trend to determine if a break is a BOS or CHoCH.
    is_uptrend = group['SMA_20'] > group['SMA_50']

    # Bullish BOS: In an uptrend, close breaks above last swing high.
    bull_bos = is_uptrend & (group['close'] > group['last_swing_high'].shift(1))
    # Bearish CHoCH: In an uptrend, close breaks below last swing low (potential reversal).
    bear_choch = is_uptrend & (group['close'] < group['last_swing_low'].shift(1))
    # Bearish BOS: In a downtrend, close breaks below last swing low.
    bear_bos = ~is_uptrend & (group['close'] < group['last_swing_low'].shift(1))
    # Bullish CHoCH: In a downtrend, close breaks above last swing high (potential reversal).
    bull_choch = ~is_uptrend & (group['close'] > group['last_swing_high'].shift(1))

    group['bos'] = np.select([bull_bos, bear_bos], [1, -1], default=0)
    group['choch'] = np.select([bull_choch, bear_choch], [1, -1], default=0)

    # 4. Order Block Identification
    # A potential bullish OB is a down-candle that forms a swing low.
    # is_swing_low is True at index `i` if `i-1` is a swing low.
    is_prev_down_candle = group['close'].shift(1) < group['open'].shift(1)
    potential_bull_ob = is_swing_low & is_prev_down_candle

    # A potential bearish OB is an up-candle that forms a swing high.
    is_prev_up_candle = group['close'].shift(1) > group['open'].shift(1)
    potential_bear_ob = is_swing_high & is_prev_up_candle

    # Mark the zones of these potential OBs. The values are from the previous candle (i-1).
    group['bull_ob_top'] = np.where(potential_bull_ob, group['open'].shift(1), np.nan)
    group['bull_ob_bottom'] = np.where(potential_bull_ob, group['low'].shift(1), np.nan)
    group['bear_ob_top'] = np.where(potential_bear_ob, group['high'].shift(1), np.nan)
    group['bear_ob_bottom'] = np.where(potential_bear_ob, group['open'].shift(1), np.nan)

    # Forward-fill these zones so we know where the last one was.
    group['last_bull_ob_top'] = group['bull_ob_top'].ffill()
    group['last_bull_ob_bottom'] = group['bull_ob_bottom'].ffill()

    return group

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

def calculate_risk_reward_ratio(entry, stop, target):
    """Calculates the risk/reward ratio."""
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return 0
    return reward / risk

def calculate_confidence_score(signal, latest_15m):
    """Calculates a confidence score for a signal based on multiple factors."""
    score = 0.0
    # Base score on the quality of the price action pattern
    score += signal.get('quality_score', 1) * 20  # Max 60 for a high-quality pattern
    if latest_15m.get('RSI_14', 50) < 65: score += 15 # Reward signals that are not overbought
    if signal.get('sentiment_score', 0) > 0: score += 25 # Reward signals with positive news sentiment
    return min(100, int(score)) # Return an integer score capped at 100

def calculate_position_size(entry_price, stop_loss):
    """
    Calculates position size based on the Golden Rule (max 1% risk per trade).
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return 0

    # Golden Rule: Never risk more than 1% of capital per trade.
    risk_amount = ACCOUNT_SIZE * MAX_RISK_PER_TRADE
    position_size = risk_amount / risk_per_share
    return round(position_size)

def should_enter_trade(signal_params, market_conditions):
    """
    Performs a series of multi-layer safety checks before entering a trade.
    """
    # MULTI-LAYER SAFETY CHECKS
    checks = {
        'confidence_is_high': signal_params.get('confidence_score', 0) > 70,
        'market_trend_is_bullish': market_conditions.get('sentiment') == 'BULLISH',
        'volatility_is_low': not market_conditions.get('is_vix_high', False),
        'volume_is_confirmed': signal_params.get('volume_confirmed', False),
        'rsi_is_not_overbought': signal_params.get('rsi', 100) < 70
    }
    
    if not all(checks.values()):
        failed_checks = [key for key, value in checks.items() if not value]
        logger.info(f"Trade for {signal_params.get('instrument')} REJECTED. Failed checks: {failed_checks}")
        return False
        
    return True



def fetch_news_from_rss(ticker):
    """Fetches news headlines from a Google News RSS feed."""
    # Sanitize ticker for URL and create a search query
    query = ticker.replace('.NS', '') + " stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        # Check if the library was imported successfully before trying to use it.
        if not FEEDPARSER_AVAILABLE:
            return []

        feed = feedparser.parse(url)
        # Get top 10 headlines for a good sample size
        headlines = [entry.title for entry in feed.entries[:10]]
        return headlines
    except Exception as e:
        logger.warning(f"Failed to fetch RSS feed for {ticker}: {e}")
        return []

def analyze_sentiment(ticker):
    """Safe sentiment analysis without huge AI models"""
    try:
        news_headlines = fetch_news_from_rss(ticker)
        if not news_headlines:
            return 0
            
        sentiment_score = 0
        positive_words = ['profit', 'growth', 'deal', 'expansion', 'beat', 'record', 'hike', 'strong', 'upgrade', 'buy']
        negative_words = ['loss', 'decline', 'cut', 'miss', 'investigation', 'probe', 'fall', 'weak', 'downgrade', 'sell']
        
        for headline in news_headlines:
            headline_lower = headline.lower()
            if any(word in headline_lower for word in positive_words): sentiment_score += 1
            elif any(word in headline_lower for word in negative_words): sentiment_score -= 1
        
        logger.info(f"Keyword sentiment score for {ticker}: {sentiment_score}")
        return sentiment_score
    except Exception as e:
        logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
        return 0  # Neutral on failure

def analyze_market_internals(price_data_dict):
    """
    Analyzes the broader market context using VIX and sectoral indices.
    Returns a dictionary summarizing the market state.
    """
    logger.info("Analyzing market internals (VIX, Sector Performance)...")
    market_context = {
        'sentiment': 'NEUTRAL',
        'vix_level': None,
        'is_vix_high': False,
        'leading_sector': None
    }
    
    # We need 1-hour data for trend analysis
    df_1h = price_data_dict.get("1h")
    if df_1h is None or df_1h.empty:
        logger.warning("Not enough 1h data to analyze market internals.")
        return market_context

    # 1. Analyze VIX
    vix_symbol = config.MARKET_BREADTH_SYMBOLS.get("VIX")
    if vix_symbol:
        vix_data = df_1h[df_1h['instrument'] == vix_symbol].copy()
        if not vix_data.empty:
            latest_vix = vix_data.iloc[-1]
            market_context['vix_level'] = latest_vix['close']
            if latest_vix['close'] > VIX_THRESHOLD:
                market_context['is_vix_high'] = True
                market_context['sentiment'] = 'CAUTIOUS'
                logger.info(f"Market sentiment is CAUTIOUS: VIX is high at {latest_vix['close']:.2f}")

    # 2. Analyze Sector Performance (e.g., find the strongest sector over the last 5 periods)
    sector_performance = {}
    for name, symbol in config.MARKET_BREADTH_SYMBOLS.items():
        if "NIFTY" in name: # Only check sectors, not VIX
            sector_data = df_1h[df_1h['instrument'] == symbol]
            if len(sector_data) >= 5:
                # Calculate 5-period return as a proxy for momentum
                roc = ((sector_data['close'].iloc[-1] / sector_data['close'].iloc[-5]) - 1) * 100
                sector_performance[name] = roc
    
    if sector_performance:
        leading_sector = max(sector_performance, key=sector_performance.get)
        market_context['leading_sector'] = leading_sector
        logger.info(f"Leading sector by momentum: {leading_sector} ({sector_performance[leading_sector]:.2f}%)")

    # Overall sentiment logic (can be expanded)
    if market_context['sentiment'] != 'CAUTIOUS':
        # Check overall market trend (^NSEI)
        nsei_data = df_1h[df_1h['instrument'] == '^NSEI'].dropna(subset=['SMA_20', 'SMA_50'])
        if not nsei_data.empty:
             latest_nsei = nsei_data.iloc[-1]
             if latest_nsei['SMA_20'] > latest_nsei['SMA_50']:
                 market_context['sentiment'] = 'BULLISH'
             else:
                 market_context['sentiment'] = 'BEARISH'
             logger.info(f"Market sentiment based on NIFTY trend: {market_context['sentiment']}")

    return market_context

def generate_intelligent_signals(price_data_dict, market_context, news_data):
    """Generate signals with AI analysis and smart alerting"""
    
    # 1. Run AI analysis
    ai_engine = AIAnalysisEngine()
    analysis = ai_engine.analyze_trading_opportunity(
        price_data_dict, market_context, news_data
    )
    
    # 2. Generate intelligent signal
    signal = ai_engine.generate_intelligent_signal(analysis)
    
    # 3. Manage alert
    should_send, reason = alert_manager.manage_alert(signal)

    if should_send:
        logger.info(f"✅ Alert to be sent: {signal.get('symbol')} - {reason}")
        send_ai_powered_alert(signal, analysis)
        return signal
    else:
        logger.info(f"⏸️  Alert filtered: {signal.get('symbol')} - {reason}")
        return None

def generate_advisor_output(signal):
    """Formats the top signal into a single row for the Advisor_Output sheet."""
    stock_name = signal['instrument'].replace('.NS', '')
    confidence = signal.get('confidence_score', 0)
    reasons = signal.get('reason', 'N/A') # Get the full reason string
    entry_price = signal.get('underlying_price', 0)
    stop_loss = signal.get('stop_loss', 0)
    take_profit = signal.get('take_profit', 0)
    
    recommendation = f"BUY {stock_name} ({signal['option_type']})"
    confidence_str = f"{int(confidence)}%"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # This list directly matches the new "Advisor_Output" tab structure.
    # Using f-strings to format numbers to 2 decimal places for clarity.
    # Return a dictionary for Firestore
    advisor_data = {
        "recommendation": recommendation,
        "confidence": confidence_str,
        "entry_price": f"{entry_price:.2f}",
        "stop_loss": f"{stop_loss:.2f}",
        "take_profit": f"{take_profit:.2f}",
        "reasons": reasons,
        "timestamp": timestamp_str
    }
    return advisor_data

def write_to_firestore(db, price_df, signals_df):
    """Writes all processed data to JSON files instead of Firestore."""
    logger.info("--- Starting JSON Storage Update Process ---")
    
    try:
        # 1. Write Price Data to JSON
        if not price_df.empty:
            save_dataframe_to_json(price_df, "price_data")
            logger.info(f"✅ Saved price data for {len(price_df['instrument'].unique())} instruments to JSON.")
        else:
            raise ValueError("Attempted to write to storage, but the provided price dataframe was empty.")

        # 2. Write signals to JSON
        if not signals_df.empty:
            save_dataframe_to_json(signals_df, "signals")
            logger.info(f"✅ Saved {len(signals_df)} signals to JSON.")
            
            # Generate and save advisor output
            signals_df_sorted = signals_df.sort_values(by=['confidence_score'], ascending=False)
            top_signal = signals_df_sorted.iloc[0].to_dict()
            advisor_data = generate_advisor_output(top_signal)
            
            # Save advisor output
            save_to_json("advisor_output", advisor_data)
            logger.info(f"✅ Saved top signal to advisor_output: {advisor_data['recommendation']}")
            
            # Send Telegram Notification for the top signal
            batch.set(advisor_ref, advisor_data)
        logger.info(f"Staged top signal to Advisor_Output: {advisor_data['recommendation']}")
    else:
        logger.info("No signals to generate advice. Updating Advisor_Output with status.")
        no_signal_data = {
            "recommendation": "No high-confidence signals found.",
            "confidence": "0%",
            "reasons": "Market conditions not met.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        batch.set(advisor_ref, no_signal_data)

        # 3. Update bot control timestamp in JSON
        bot_status_data = {
            "status": "running",
            "last_updated": datetime.now().isoformat()
        }
        save_to_json("bot_control", bot_status_data)

        logger.info("--- JSON Storage Update Process Completed ---")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to JSON storage: {e}", exc_info=True)
        return False



def main(force_run=False):
    """Main function that runs the entire process."""

    # 1. Check market status
    market_status = should_run_trading_bot()
    
    if market_status == "holiday" and not force_run:
        print(" Market holiday - no operations")
        return {"status": "holiday"}
    
    # 2. Always run data collection (if trading day)
    if market_status != "holiday" or force_run:
        data_collector = DataCollector()
        data_collector.collect_training_data([
            '^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TCS.NS'
        ])
    
    # 3. Run AI training weekly
    if datetime.now().weekday() == 0 and (market_status != "holiday" or force_run):  # Monday
        training_pipeline = AITrainingPipeline()
        training_pipeline.retrain_model()

    # 4. Only run trading during market hours
    if market_status == "full_trading" or force_run:
        try:
            from .firestore_utils import init_json_storage, get_db # LAZY IMPORT
            logger.info("--- Trading Signal Process Started ---")
            
            # Step 1: Connect to Firestore
            init_json_storage()
            db = get_db()  # This returns our dummy DB object for compatibility
            
            # Check if the bot is enabled in Firestore before proceeding.
            if not check_bot_status(db) and not force_run:
                raise BotHaltedException("Bot execution halted by user control in 'Bot_Control' sheet.")

            # Step 2: Conditionally connect to Kite and fetch instrument data
            kite = None
            if market_status == "full_trading":
                kite = connect_to_kite()

            # Step 4: Collect external data (market data, events)
            price_data_dict = run_data_collection()
            
            if price_data_dict.get("15m") is None or price_data_dict["15m"].empty:
                logger.warning("No data was collected from the source. Skipping indicator calculation, signal generation, and sheet writing.")
                return {"status": "success", "message": "No data collected from source."}

            # Step 5: Calculate all indicators for all instruments and timeframes
            if not price_data_dict["15m"].empty:
                price_data_dict["15m"] = calculate_indicators(price_data_dict["15m"])
                price_data_dict["15m"] = price_data_dict["15m"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
            if not price_data_dict["30m"].empty:
                price_data_dict["30m"] = calculate_indicators(price_data_dict["30m"])
                price_data_dict["30m"] = price_data_dict["30m"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
            if not price_data_dict["1h"].empty:
                price_data_dict["1h"] = calculate_indicators(price_data_dict["1h"])
                price_data_dict["1h"] = price_data_dict["1h"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
            
            # Step 5.5: Analyze Market Internals
            market_context = analyze_market_internals(price_data_dict)
            
            # Step 6: Generate signals using data, controls, performance, sentiment, and the new market context
            signals_df = generate_intelligent_signals(price_data_dict, market_context, news_data={})
            
            # Step 7: Write both data and signals to the sheets
            write_to_firestore(db, price_data_dict["15m"], signals_df)
            
            if not price_data_dict["15m"].empty:
                save_dataframe_to_json(price_data_dict["15m"], "price_data")
            else:
                logger.info("Price data is empty. Skipping JSON backup for price data.")
            if signals_df is not None and not signals_df.empty:
                save_dataframe_to_json(pd.DataFrame([signals_df]), "signals")
            else:
                logger.info("Signals data is empty. Skipping JSON backup for signals.")


            logger.info("--- Trading Signal Process Completed Successfully ---")
            return {"status": "success", "message": "Trading bot executed successfully."}

        except Exception as e:
            logger.error(f"A critical error occurred in the main process:", exc_info=True)
            raise
    else:
        return {"status": "data_collection_only", "message": "Market closed - data collected"}





# --- Blueprint Definition ---
process_data_bp = Blueprint('process_data', __name__)

# --- Script Execution ---
@process_data_bp.route("/run", methods=["GET"])
def run_bot():
    """
    HTTP endpoint to trigger the trading bot's main logic.
    """
    logger.info("Received request to run the trading bot.")
    force_run = request.args.get('force', 'false').lower() == 'true'
    if force_run:
        logger.warning("'force=true' parameter detected. Bypassing market hours check for this run.")

    try:
        result = main(force_run=force_run)
        http_status = 200
        if result.get("status") == "error":
            http_status = 500
        return jsonify(result), http_status
    except BotHaltedException as e:
        # This is a controlled, graceful exit, not an error.
        # Return 200 OK to prevent the GitHub Actions job from failing.
        logger.info(str(e))
        return jsonify({"status": "halted", "message": str(e)}), 200
    except Exception as e:
        logger.error(f"Error executing trading bot: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- NEW: API Endpoint for Frontend Dashboard ---
@process_data_bp.route("/dashboard", methods=["GET"])
def get_dashboard_data_endpoint():
    """
    Provides dashboard data from JSON files instead of Firestore.
    """
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    if not force_refresh and dashboard_cache["timestamp"] and \
       (datetime.now() - dashboard_cache["timestamp"]).total_seconds() < CACHE_LIFETIME_SECONDS:
        logger.info("Returning dashboard data from cache.")
        return jsonify(dashboard_cache["data"])

    with cache_lock:
        if not force_refresh and dashboard_cache["timestamp"] and \
           (datetime.now() - dashboard_cache["timestamp"]).total_seconds() < CACHE_LIFETIME_SECONDS:
            logger.info("Returning dashboard data from cache (after lock).")
            return jsonify(dashboard_cache["data"])

        logger.info("Fetching fresh dashboard data from JSON files.")
        
        try:
            def read_json_file(filename, default=[]):
                """Helper function to read JSON files safely."""
                filepath = f"data/{filename}.json"
                try:
                    with open(filepath, 'r') as f:
                        return json.load(f)
                except FileNotFoundError:
                    return default
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")
                    return default

            # Read all data from JSON files
            dashboard_data = {
                "advisorOutput": read_json_file("advisor_output", [{}]),
                "signals": read_json_file("signals", []),
                "botControl": read_json_file("bot_control", [{}]),
                "priceData": {},
                "tradeLog": read_json_file("trade_log", []),
                "lastRefreshed": datetime.now(pytz.utc).isoformat(),
            }

            # Read price data for each symbol
            for symbol in config.WATCHLIST_SYMBOLS:
                # Use a simplified filename pattern
                clean_symbol = symbol.replace('.NS', '').replace('^', '')
                price_data = read_json_file(f"price_data_{clean_symbol}", [])
                if price_data:
                    # Get latest 200 records
                    dashboard_data["priceData"][symbol] = price_data[-200:]

            # Update cache
            dashboard_cache["data"] = dashboard_data
            dashboard_cache["timestamp"] = datetime.now()
            logger.info("Dashboard cache updated from JSON files.")

            return jsonify(dashboard_data), 200

        except Exception as e:
            error_message = f"Error fetching dashboard data from JSON files: {str(e)}"
            logger.error(f"Error in dashboard endpoint: {e}", exc_info=True)
            return jsonify({"status": "error", "message": error_message}), 500
