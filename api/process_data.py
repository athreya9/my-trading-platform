# process_data.py
# A single, combined script for GitHub Actions.
# It fetches data, generates signals, and updates Google Sheets.

from flask import Blueprint, request, jsonify
import gspread
from kiteconnect import KiteConnect
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import time
from datetime import datetime, timedelta
import pytz
import joblib
from functools import wraps, lru_cache
import os
from dotenv import load_dotenv

# Load environment variables from .env file for local development.
# This will not override environment variables set in the GitHub Actions runner.
load_dotenv()

# --- NEW: Import for Secret Manager ---
try:
    from google.cloud import secretmanager
    GCP_SECRET_MANAGER_AVAILABLE = True
except ImportError:
    GCP_SECRET_MANAGER_AVAILABLE = False

import logging
import re
import sys
from api import config
import requests
import feedparser

process_data_bp = Blueprint('process_data', __name__)

# --- Logging Configuration ---
# Use a custom formatter to ensure all log times are in UTC for consistency
formatter = logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s')
formatter.converter = time.gmtime  # Use UTC for asctime

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Sheet & Symbol Configuration ---
SHEET_NAME = "Algo Trading Dashboard" # The name of your Google Sheet
DATA_WORKSHEET_NAME = "Price_Data"
SIGNALS_WORKSHEET_NAME = "Signals"

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

# Load the AI model once when the script starts.
try:
    AI_MODEL = joblib.load(MODEL_PATH)
    logger.info(f"Successfully loaded AI model from {MODEL_PATH}")
except FileNotFoundError:
    logger.warning(f"AI model '{os.path.basename(MODEL_PATH)}' not found. AI-based signals will be disabled.")
    AI_MODEL = None

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

def retry(tries=3, delay=5, backoff=2, logger=logger):
    """
    A retry decorator with exponential backoff.
    Catches common network-related exceptions for gspread and yfinance.
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
                        logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs) # Last attempt, if it fails, it fails
        return f_retry
    return deco_retry

@retry()
def enhance_sheet_structure(sheet):
    """Ensures all essential tabs exist in the Google Sheet with the correct headers."""
    logger.info("Verifying and enhancing Google Sheet structure...")
    
    try:
        existing_titles = [ws.title for ws in sheet.worksheets()]
        logger.info(f"Found existing tabs: {existing_titles}")
        
        # --- Define ALL essential tabs and their required headers ---
        essential_tabs = {
            "Advisor_Output": [["Recommendation", "Confidence", "Reasons", "Timestamp"]],
            "Signals": [["Action", "Symbol", "Price", "Confidence", "Reasons", "Timestamp"]],
            "Bot_Control": [["Parameter", "Value"], ["status", "running"], ["mode", "EMERGENCY"], ["last_updated", "never"]],
            "Price_Data": [["Symbol", "Price", "Volume", "Change", "Timestamp"]],
            "Trade_Log": [["Date", "Instrument", "Action", "Quantity", "Entry", "Exit", "P/L"]]
        }
        
        for tab_name, headers in essential_tabs.items():
            if tab_name not in existing_titles:
                logger.info(f"Tab '{tab_name}' not found. Creating it...")
                worksheet = sheet.add_worksheet(title=tab_name, rows="1000", cols="20")
                worksheet.update(range_name='A1', values=headers, value_input_option='USER_ENTERED')
                logger.info(f"Created and structured '{tab_name}'.")
            else:
                logger.info(f"Tab '{tab_name}' already exists.")
        
        logger.info("Google Sheet structure is verified and up-to-date.")
    except Exception as e:
        logger.error(f"An error occurred during sheet structure verification: {e}", exc_info=True)
        raise # Re-raise the exception to stop the main process

@retry()
def read_manual_controls(spreadsheet):
    """Reads manual override settings from the 'Manual Control' sheet."""
    logger.info("Reading data from 'Manual Control' sheet...")
    try:
        worksheet = spreadsheet.worksheet("Manual Control")
        records = worksheet.get_all_records()
        if not records:
            logger.info("No manual controls found or sheet is empty.")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # Ensure key columns exist, even if empty
        for col in ['instrument', 'limit_price', 'hold_status']:
            if col not in df.columns:
                df[col] = None
        
        df.set_index('instrument', inplace=True)
        logger.info("Manual controls loaded successfully.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        logger.warning("'Manual Control' worksheet not found. Skipping manual overrides.")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not read manual controls: {e}")
        return pd.DataFrame()

@retry()
def read_trade_log(spreadsheet):
    """Reads the historical trade log from the 'Trade Log' sheet."""
    logger.info("Reading data from 'Trade Log' sheet...")
    try:
        worksheet = spreadsheet.worksheet("Trade Log")
        records = worksheet.get_all_records()
        if not records:
            logger.info("Trade log is empty. Cannot calculate Kelly Criterion.")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # Convert profit to numeric, coercing errors to NaN and then dropping them
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
        df.dropna(subset=['profit'], inplace=True)
        
        logger.info(f"Successfully read {len(df)} trades from the log.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        logger.warning("'Trade Log' worksheet not found. Cannot calculate Kelly Criterion.")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not read trade log: {e}")
        return pd.DataFrame()

@retry()
def check_bot_status(spreadsheet):
    """Checks the 'Bot_Control' tab for a 'running' status."""
    logger.info("Checking bot operational status...")
    try:
        worksheet = spreadsheet.worksheet("Bot_Control")
        # Read the status from cell B2, which holds the value ("running" or "stopped")
        status = worksheet.acell('B2').value
        logger.info(f"Bot status from sheet ('Bot_Control'!B2): '{status}'")
        if status and status.strip().strip('.').lower() == 'running':
            return True
        else:
            logger.warning(f"Bot status is '{status}'. Halting execution as per Bot_Control sheet.")
            return False
    except Exception as e:
        logger.error(f"Could not read bot status: {e}. Halting for safety.")
        return False

def calculate_kelly_criterion(trades_df):
    """Calculates the Kelly Criterion percentage, win rate, and win/loss ratio."""
    if trades_df.empty or len(trades_df) < 20:
        logger.info("Not enough historical trades (< 20) to calculate Kelly Criterion.")
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
    
    logger.info(f"Win Rate: {win_rate:.2%}, Win/Loss Ratio: {win_loss_ratio:.2f}, Calculated Kelly Criterion: {kelly_pct:.2%}")
    return kelly_pct, win_rate, win_loss_ratio

@retry()
def get_gcp_secret(secret_id, project_id, version_id="latest"):
    """Fetches a secret from Google Secret Manager."""
    if not GCP_SECRET_MANAGER_AVAILABLE:
        logger.warning("google-cloud-secret-manager library not installed. Cannot fetch secrets from GCP.")
        return None

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        logger.info(f"Successfully fetched latest version of secret '{secret_id}'.")
        return payload
    except Exception as e:
        # Log the error but return None to allow fallback to environment variables
        logger.error(f"Failed to access secret '{secret_id}' in project '{project_id}': {e}")
        return None


@retry()
def connect_to_google_sheets():
    """Connects to Google Sheets using credentials from an environment variable."""
    logger.info("Attempting to authenticate with Google Sheets...")
    creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
    if not creds_json_str:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS environment variable not found. Ensure it's set in GitHub Actions secrets.")

    try:
        logger.info("Authenticating with Google Sheets via environment variable.")
        creds_dict = json.loads(creds_json_str)
        client = gspread.service_account_from_dict(creds_dict)
        spreadsheet = client.open(SHEET_NAME)
        logger.info(f"Successfully connected to Google Sheet: '{SHEET_NAME}'")
        return spreadsheet
    except Exception as e:
        raise Exception(f"Error connecting to Google Sheet: {e}")

def connect_to_kite():
    """Initializes the Kite Connect client using credentials from environment variables."""
    logger.info("Attempting to authenticate with Kite Connect...")
    api_key = os.getenv('KITE_API_KEY', '').strip().strip('"\'')
    access_token = None

    # --- START: MODIFICATION to fetch token from Secret Manager ---
    # In a GCP environment (like Cloud Run), fetch the access token from Secret Manager.
    # This ensures the latest token is always used, even in long-running instances.
    # The GCP_PROJECT env var is automatically set by Cloud Run.
    gcp_project_id = os.getenv('GCP_PROJECT')
    if gcp_project_id:
        logger.info(f"GCP environment detected (Project: {gcp_project_id}). Fetching KITE_ACCESS_TOKEN from Secret Manager.")
        access_token = get_gcp_secret("KITE_ACCESS_TOKEN", gcp_project_id)

    # Fallback to environment variable if not in GCP or if Secret Manager fetch fails.
    if not access_token:
        logger.info("Falling back to KITE_ACCESS_TOKEN from environment variable for local development or as a backup.")
        access_token = os.getenv('KITE_ACCESS_TOKEN', '').strip().strip('"\'')
    # --- END: MODIFICATION ---

    # --- START: Added for debugging authentication issues ---
    logger.info(f"DEBUG: API Key (first 4 chars): '{api_key[:4]}...'")
    logger.info(f"DEBUG: Access Token (first 4 chars): '{access_token[:4] if access_token else 'None'}...'")
    logger.info(f"DEBUG: API Key length: {len(api_key)}")
    logger.info(f"DEBUG: Access Token length: {len(access_token) if access_token else 0}")
    # --- END: Added for debugging ---

    if not api_key or not access_token:
        raise ValueError("KITE_API_KEY or KITE_ACCESS_TOKEN could not be obtained.")
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

@retry()
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


@retry()
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


def run_data_collection(kite, instrument_map):
    """Fetches data for all symbols and timeframes and returns a dictionary of DataFrames."""
    data_frames = {"15m": [], "30m": [], "1h": []}
    to_date = datetime.now()
    
    for symbol in config.SYMBOLS:
        # Translate yfinance symbol to Kite tradingsymbol
        kite_symbol = YFINANCE_TO_KITE_MAP.get(symbol, symbol.replace('.NS', ''))
        instrument_token = instrument_map.get(kite_symbol)

        if not instrument_token:
            logger.warning(f"Could not find instrument token for symbol '{symbol}' (Kite: '{kite_symbol}'). Skipping.")
            continue

        # --- Fetch data for all required timeframes ---
        from_date_15m = to_date - timedelta(days=5)
        df_15m = fetch_historical_data(kite, instrument_token, from_date_15m, to_date, "15minute", symbol)
        if not df_15m.empty:
            data_frames["15m"].append(df_15m)

        from_date_30m = to_date - timedelta(days=10)
        df_30m = fetch_historical_data(kite, instrument_token, from_date_30m, to_date, "30minute", symbol)
        if not df_30m.empty:
            data_frames["30m"].append(df_30m)

        from_date_1h = to_date - timedelta(days=60)
        df_1h = fetch_historical_data(kite, instrument_token, from_date_1h, to_date, "60minute", symbol)
        if not df_1h.empty:
            data_frames["1h"].append(df_1h)

    if not data_frames["15m"]:
        raise Exception("No 15m data was fetched for any symbol. Halting process.")

    # Combine the lists of dataframes into single dataframes
    combined_df_15m = pd.concat(data_frames["15m"], ignore_index=True) if data_frames["15m"] else pd.DataFrame()
    combined_df_30m = pd.concat(data_frames["30m"], ignore_index=True) if data_frames["30m"] else pd.DataFrame()
    combined_df_1h = pd.concat(data_frames["1h"], ignore_index=True) if data_frames["1h"] else pd.DataFrame()

    # --- NEW: Fetch LIVE data and merge it to prevent using stale historical data ---
    logger.info("Fetching live quote data to merge with historical data...")
    
    # Create a list of Kite-formatted instrument names for the quote API
    kite_symbols_for_quote = [f"NSE:{YFINANCE_TO_KITE_MAP.get(s, s.replace('.NS', ''))}" for s in config.SYMBOLS]

    if not kite_symbols_for_quote:
        logger.warning("No symbols to fetch live quotes for.")
    else:
        try:
            live_quotes = kite.quote(kite_symbols_for_quote)

            # Create a reverse map from Kite API symbol back to our internal symbol
            kite_to_internal_map = {f"NSE:{YFINANCE_TO_KITE_MAP.get(s, s.replace('.NS', ''))}": s for s in config.SYMBOLS}

            # Update the last row of each instrument in the dataframes with live data
            for df in [combined_df_15m, combined_df_30m, combined_df_1h]:
                if df.empty: continue
                
                last_indices = df.groupby('instrument').tail(1).index
                for idx in last_indices:
                    internal_symbol = df.loc[idx, 'instrument']
                    kite_api_symbol = next((k for k, v in kite_to_internal_map.items() if v == internal_symbol), None)
                    
                    if kite_api_symbol and kite_api_symbol in live_quotes:
                        quote = live_quotes[kite_api_symbol]
                        ltp = quote['last_price']
                        
                        df.loc[idx, 'close'] = ltp
                        df.loc[idx, 'high'] = max(df.loc[idx, 'high'], ltp)
                        df.loc[idx, 'low'] = min(df.loc[idx, 'low'], ltp)
                        df.loc[idx, 'timestamp'] = datetime.now(pytz.timezone('Asia/Kolkata'))
            logger.info("Successfully merged live quote data into historical dataframes.")
        except Exception as e:
            logger.error(f"Could not fetch or merge live quotes: {e}")

    logger.info(f"Processed {len(combined_df_15m)} rows (15m), {len(combined_df_30m)} rows (30m), {len(combined_df_1h)} rows (1h).")
    return {"15m": combined_df_15m, "30m": combined_df_30m, "1h": combined_df_1h}

def calculate_indicators(price_df):
    """
    Calculates all technical indicators (SMA, RSI, MACD, ATR) for all instruments
    using efficient, vectorized operations.
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

@retry()
def get_news_sentiment(instrument, analyzer):
    """Fetches news for an instrument and returns an aggregated sentiment score."""
    # If sentiment analysis is disabled (analyzer is None), return neutral score immediately.
    logger.info("Sentiment analysis is disabled. Skipping news fetch.")
    return 0.0

def fetch_news_from_rss(ticker):
    """Fetches news headlines from a Google News RSS feed."""
    # Sanitize ticker for URL and create a search query
    query = ticker.replace('.NS', '') + " stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
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

def generate_signals(price_data_dict, manual_controls_df, trade_log_df, market_context, economic_events):
    """Generates trading signals for all instruments, applying a suite of validation and risk rules."""
    logger.info("Generating signals with advanced rule validation...")
    
    # --- Setup & Market Context Filter ---
    kelly_pct, win_rate, win_loss_ratio = calculate_kelly_criterion(trade_log_df)
    all_potential_signals = []
    price_df_15m = price_data_dict['15m']
    price_df_30m = price_data_dict['30m']
    price_df_1h = price_data_dict['1h']
    if price_df_30m.empty:
        logger.warning("30m data not available. Multi-timeframe confluence check will be skipped.")
    if price_df_1h.empty:
        logger.warning("1h data not available. Multi-timeframe confluence check will be skipped for all signals.")

    if market_context.get('is_vix_high', False):
        logger.info(f"Market Context Alert: VIX is above {VIX_THRESHOLD}. Avoiding new aggressive long positions.")
    if market_context.get('sentiment') == 'BEARISH':
        logger.info("Market Context Alert: Overall market trend is Bearish. Long signals will be suppressed.")
    
    is_high_impact_event_today = any(event.get('impact') == 'high' for event in economic_events)
    if is_high_impact_event_today:
        logger.warning("High-impact economic event scheduled today. Trading will be more conservative.")

    # --- Iterate over each instrument's 15-minute data ---
    for instrument, group_15m in price_df_15m.groupby('instrument'):
        # Ensure we have enough data for lookbacks and that price action features were calculated
        required_cols = ['SMA_50', 'RSI_14', f'ATRr_{ATR_PERIOD}', 'volume_avg_20', 'fvg_bull_bottom', 'choch', 'last_bull_ob_top', 'last_bull_ob_bottom']
        group_15m = group_15m.copy().dropna(subset=required_cols)
        if len(group_15m) < 3: # Need at least 3 rows for pattern detection lookbacks
            continue

        instrument_signals = []

        # Skip signal generation for the market internal instruments themselves
        if instrument in config.MARKET_BREADTH_SYMBOLS.values():
            continue

        latest_15m = group_15m.iloc[-1]

        # --- AI-DRIVEN SIGNAL GENERATION (PRIMARY) ---
        # This is now the main signal generator. Rule-based signals can act as a fallback.
        ai_signal_generated = False
        if AI_MODEL is not None:
            # These features MUST match the ones used in `prepare_training_data.py`
            feature_columns = [
                'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'ATRr_14',
                'volume_avg_20', 'realized_vol', 'vwap', 'bos', 'choch',
                'last_bull_ob_top', 'last_bull_ob_bottom'
            ]
            # Ensure all required feature columns are present and not NaN
            if all(col in latest_15m and pd.notna(latest_15m[col]) for col in feature_columns):
                # Prepare features for the model
                features = latest_15m[feature_columns].values.reshape(1, -1)
                
                # Get prediction probability for the 'BUY' class (1)
                buy_probability = AI_MODEL.predict_proba(features)[0][1]

                if buy_probability >= AI_CONFIDENCE_THRESHOLD:
                    logger.info(f"AI SIGNAL for {instrument}: BUY with {buy_probability:.2%} confidence.")
                    
                    signal_params = {
                        'instrument': instrument,
                        'reason': f'AI Prediction ({buy_probability:.0%})',
                        'quality_score': 4, # Highest quality score for AI signals
                        'rsi': latest_15m['RSI_14'],
                        'volume_confirmed': True, # AI model implicitly learns volume patterns
                        'sentiment_score': analyze_sentiment(instrument),
                    }
                    signal_params['confidence_score'] = int(buy_probability * 100)

                    # Use existing safety checks before finalizing the signal
                    if should_enter_trade(signal_params, market_context):
                        entry_price = latest_15m['close']
                        atr_val = latest_15m[f'ATRr_{ATR_PERIOD}']
                        stop_loss = entry_price - (atr_val * STOP_LOSS_MULTIPLIER)
                        take_profit = entry_price + (atr_val * TAKE_PROFIT_MULTIPLIER)
                        
                        instrument_signals.append({
                            'option_type': 'CALL', 'strike_price': get_atm_strike(entry_price, instrument),
                            'underlying_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                            'position_size': calculate_position_size(entry_price, stop_loss),
                            'reason': signal_params['reason'], 'confidence_score': signal_params['confidence_score'],
                            'win_rate_p': win_rate, 'win_loss_ratio_b': win_loss_ratio, 'kelly_pct': kelly_pct
                        })
                        ai_signal_generated = True

        # --- Manual Override Logic ---
        manual_override_triggered = False
        if not manual_controls_df.empty and instrument in manual_controls_df.index:
            control = manual_controls_df.loc[instrument]
            if str(control['hold_status']).upper() == 'HOLD':
                logger.info(f"'{instrument}' is on HOLD. Skipping automated signal.")
                continue
            elif str(control['hold_status']).upper() == 'SELL':
                logger.info(f"'{instrument}' has manual SELL override.")
                instrument_signals.append({'option_type': 'PUT (MANUAL)', 'strike_price': get_atm_strike(latest_15m['close'], instrument), 'underlying_price': latest_15m['close']})
                manual_override_triggered = True

        if manual_override_triggered:
            # Add common data and append to master list
            for sig in instrument_signals:
                sig['timestamp'] = latest_15m['timestamp']
                sig['instrument'] = instrument
                all_potential_signals.append(sig)
            continue

        # --- FALLBACK: Rule-Based Signal Generation ---
        # This logic only runs if the AI did not generate a high-confidence signal.
        if not ai_signal_generated:
            # --- Rule 1: Volatility Filter ---
            atr_percentage = (latest_15m[f'ATRr_{ATR_PERIOD}'] / latest_15m['close']) * 100
            if atr_percentage > 3.0:
                logger.info(f"Skipping {instrument}: High volatility detected (ATR is {atr_percentage:.2f}% of price).")
                continue

            # --- Rule 2: Multi-Timeframe Confluence ---
            # Default to True, will be set to False if any timeframe disagrees.
            is_30m_bullish = True
            is_1h_bullish = True
            
            if not price_df_30m.empty:
                group_30m = price_df_30m[price_df_30m['instrument'] == instrument].copy().dropna(subset=['SMA_20', 'SMA_50'])
                if not group_30m.empty:
                    is_30m_bullish = group_30m.iloc[-1]['SMA_20'] > group_30m.iloc[-1]['SMA_50']
                else:
                    is_30m_bullish = False # Not enough data to confirm

            if not price_df_1h.empty:
                group_1h = price_df_1h[price_df_1h['instrument'] == instrument].copy().dropna(subset=['SMA_20', 'SMA_50'])
                if not group_1h.empty:
                    is_1h_bullish = group_1h.iloc[-1]['SMA_20'] > group_1h.iloc[-1]['SMA_50']
                else:
                    is_1h_bullish = False # Not enough data to confirm

            # The final trend check: all timeframes must agree
            is_15m_bullish = latest_15m['SMA_20'] > latest_15m['SMA_50'] and latest_15m['RSI_14'] < 70
            all_timeframes_bullish = is_15m_bullish and is_30m_bullish and is_1h_bullish

            # --- Automated Signal Logic (with new rules) ---
            sentiment_score = analyze_sentiment(instrument)
            signal_generated = False
            reasons = []

            if is_high_impact_event_today:
                logger.info(f"Skipping signal generation for {instrument} due to scheduled high-impact economic event.")
                continue

            # --- New VWAP Filter ---
            price_above_vwap = latest_15m['close'] > latest_15m['vwap']

            if not signal_generated and all_timeframes_bullish and price_above_vwap:
                signal_params = {
                    'instrument': instrument,
                    'reason': 'Bullish Trend',
                    'quality_score': 1,
                    'rsi': latest_15m['RSI_14'],
                    'volume_confirmed': latest_15m['volume'] > (latest_15m['volume_avg_20'] * 1.2), # 120% of avg volume
                    'sentiment_score': sentiment_score,
                }
                signal_params['confidence_score'] = calculate_confidence_score(signal_params, latest_15m)

                if should_enter_trade(signal_params, market_context):
                    logger.info(f"All safety checks passed for {instrument}. Generating BUY signal.")
                    reasons = [signal_params['reason'], "Passed all safety checks"]
                    option_type = "CALL"
                    
                    atr_val = latest_15m[f'ATRr_{ATR_PERIOD}']
                    entry_price = latest_15m['close']
                    stop_loss = entry_price - (atr_val * STOP_LOSS_MULTIPLIER)
                    take_profit = entry_price + (atr_val * TAKE_PROFIT_MULTIPLIER)
                    position_size = calculate_position_size(entry_price, stop_loss)

                    instrument_signals.append({
                        'option_type': option_type, 'strike_price': get_atm_strike(entry_price, instrument),
                        'underlying_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                        'position_size': round(position_size), 'reason': ", ".join(reasons),
                        'sentiment_score': sentiment_score,
                        'win_rate_p': win_rate,
                        'win_loss_ratio_b': win_loss_ratio,
                        'kelly_pct': kelly_pct,
                        'quality_score': signal_params['quality_score'],
                        'confidence_score': signal_params['confidence_score']
                    })
                    signal_generated = True

            # --- NEW: Price Action (FVG + CHoCH) Signal Logic ---
            if not signal_generated and all_timeframes_bullish:
                # ... (This logic can be kept as a secondary, rule-based check)
                pass

            # --- NEW: Order Block Entry Signal Logic ---
            if not signal_generated and all_timeframes_bullish:
                # ... (This logic can be kept as a secondary, rule-based check)
                pass

        # --- Add common data and append to master list ---
        for sig in instrument_signals:
            sig['timestamp'] = latest_15m['timestamp']
            sig['instrument'] = instrument
            all_potential_signals.append(sig)

    if not all_potential_signals:
        logger.info("No new signals generated after applying all rules.")
        return pd.DataFrame()
        
    # --- New Code: Filter and Rank Signals ---
    high_confidence_trades = []
    for signal in all_potential_signals:
        high_confidence_trades.append(signal) # All signals here have already passed the safety checks

    # --- Now, sort the shortlist by confidence, take the top 5 ---
    high_confidence_trades.sort(key=lambda x: x['confidence_score'], reverse=True)
    final_recommended_trades = high_confidence_trades[:5] # TOP 5 TRADES

    if not final_recommended_trades:
        logger.info("No high-confidence signals found after filtering.")
        return pd.DataFrame()

    final_signals_df = pd.DataFrame(final_recommended_trades)
    logger.info(f"Generated {len(final_signals_df)} high-confidence signals after filtering.")
    return final_signals_df

def generate_advisor_output(signal):
    """Formats the top signal into a single row for the Advisor_Output sheet."""
    stock_name = signal['instrument'].replace('.NS', '')
    confidence = signal.get('confidence_score', 0)
    reasons = signal.get('reason', 'N/A') # Get the full reason string
    
    recommendation = f"BUY {stock_name} ({signal['option_type']})"
    confidence_str = f"{confidence:.0f}%"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # This list directly matches the new "Advisor_Output" tab structure
    # ["Recommendation", "Confidence", "Reasons", "Timestamp"]
    advisor_data = [recommendation, confidence_str, reasons, timestamp_str]
    return advisor_data

@retry(logger=logger)
def send_telegram_notification(message):
    """Sends a message to a Telegram chat using a bot, with Markdown formatting."""
    logger.info("Attempting to send Telegram notification...")
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown' # Use Markdown for bold, italics, etc.
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status() # Raises an exception for 4xx/5xx status codes
        logger.info("Telegram notification sent successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        # The @retry decorator will handle re-attempts on failure.
        raise

@retry()
def write_to_sheets(spreadsheet, price_df, signals_df):
    """Writes price data, signals, and the final advice to their respective sheets."""
    logger.info("--- Starting Google Sheet Update Process ---")
    try:
        # Get all required worksheets, assuming they exist after running the fix script.
        price_worksheet = spreadsheet.worksheet("Price_Data")
        signals_worksheet = spreadsheet.worksheet("Signals")
        advisor_worksheet = spreadsheet.worksheet("Advisor_Output")
        bot_control_worksheet = spreadsheet.worksheet("Bot_Control")
        
        # Clear all sheets at the beginning of the update process
        price_worksheet.clear()
        signals_worksheet.clear()
        advisor_worksheet.clear()
    except gspread.exceptions.WorksheetNotFound as e:
        logger.error(f"A required worksheet is missing: {e}. Please run emergency_fix.py to set up the sheet structure.")
        raise

    # --- Write Price Data ---
    if not price_df.empty:
        logger.info(f"Preparing to write {len(price_df)} rows to 'Price_Data' sheet...")
        # Select and format columns for the sheet.
        price_data_to_write = price_df[['instrument', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        price_data_to_write.rename(columns={'instrument': 'Symbol', 'timestamp': 'Timestamp'}, inplace=True)
        
        # Ensure timestamp is a string for writing
        price_data_to_write['Timestamp'] = price_data_to_write['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info("Writing to 'Price_Data' sheet...")
        price_worksheet.update(range_name='A1', values=[price_data_to_write.columns.values.tolist()] + price_data_to_write.fillna('').values.tolist(), value_input_option='USER_ENTERED')
        logger.info("Price data written successfully.")
    else:
        logger.info("No price data to write.")

    # --- Write Signal Data ---
    if not signals_df.empty:
        logger.info(f"Preparing to write {len(signals_df)} rows to 'Signals'...")
        # Select and rename columns to match the new sheet structure
        signals_to_write = signals_df[['option_type', 'instrument', 'underlying_price', 'confidence_score', 'reason', 'timestamp']].copy()
        signals_to_write.rename(columns={'option_type': 'Action', 'instrument': 'Symbol', 'underlying_price': 'Price', 'confidence_score': 'Confidence', 'reason': 'Reasons', 'timestamp': 'Timestamp'}, inplace=True)
        signals_to_write['Timestamp'] = signals_to_write['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info("Writing to 'Signals' sheet...")
        signals_worksheet.update(range_name='A1', values=[signals_to_write.columns.values.tolist()] + signals_to_write.values.tolist(), value_input_option='USER_ENTERED')
        logger.info("Signal data written successfully.")
    else:
        logger.info("No signals to write.")

    # --- Write Final Advisor Output ---
    logger.info("Preparing to write to 'Advisor_Output' sheet...")
    advisor_header = [["Recommendation", "Confidence", "Reasons", "Timestamp"]]
    if not signals_df.empty:
        # Rank signals to find the best opportunity
        signals_df_sorted = signals_df.sort_values(by=['confidence_score', 'sentiment_score'], ascending=[False, False])
        top_signal = signals_df_sorted.iloc[0]
        
        # Generate the single-row advisor data
        advisor_row = generate_advisor_output(top_signal)
        
        logger.info(f"Writing top signal to Advisor_Output: {advisor_row}")
        advisor_worksheet.update('A1', advisor_header + [advisor_row], value_input_option='USER_ENTERED')
        logger.info("Advisor output written successfully.")

        # --- NEW: Send Telegram Notification for the top signal ---
        # advisor_row is a list: [recommendation, confidence_str, reasons, timestamp_str]
        notification_message = (
            f"📈 *New Trading Signal*\n\n"
            f"*Action:* {advisor_row[0]}\n"
            f"*Confidence:* {advisor_row[1]}\n"
            f"*Reason:* {advisor_row[2]}\n\n"
            f"_{advisor_row[3]} UTC_"
        )
        send_telegram_notification(notification_message)
    else:
        logger.info("No signals to generate advice. Clearing and updating Advisor_Output sheet with status.")
        # Append a "no signal" status row
        no_signal_row = ["No high-confidence signals found.", "0%", "Market conditions not met.", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        advisor_worksheet.update('A1', advisor_header + [no_signal_row], value_input_option='USER_ENTERED')

        # --- NEW: Send a status update to Telegram if no signal is found ---
        # This only runs on the first 15 minutes of the hour to avoid spam.
        if datetime.now().minute < 15:
            notification_message = (
                f"✅ *Bot Status Update*\n\nNo new high-confidence signals were found that met all criteria."
            )
            send_telegram_notification(notification_message)

    # --- NEW: Update Bot Control Timestamp ---
    try:
        timestamp_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')
        bot_control_worksheet.update_acell('B4', timestamp_str) # Update the correct cell for the timestamp
        logger.info("Successfully updated 'last_updated' timestamp in Bot_Control sheet.")
    except Exception as e:
        logger.warning(f"Could not update Bot_Control timestamp: {e}")

    logger.info("--- Sheet Update Process Completed ---")

def should_run():
    """
    Checks if the Indian stock market is open, considering weekends and holidays.
    (Mon-Fri, 9:15 AM - 3:30 PM IST, excluding holidays from config)
    """
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    today_str = now.strftime('%Y-%m-%d')
    
    # Check if today is a market holiday
    if today_str in config.MARKET_HOLIDAYS:
        logger.info(f"Market is closed today for a holiday: {today_str}")
        return False

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def fetch_economic_events():
    """
    Fetches high-impact economic events for the day.
    (Placeholder function - can be replaced with a real API call)
    """
    logger.info("Fetching economic calendar events...")
    # In a real implementation, you would call an API here.
    # Example: return requests.get("https://api.economiccalendar.com/events").json()
    # For now, we return a hardcoded example for demonstration.
    today_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
    
    # Example: High-impact US inflation data release at 6 PM IST.
    example_events = [{'date': today_str, 'time': '18:00', 'currency': 'USD', 'event': 'CPI m/m', 'impact': 'high'}]
    return example_events

def main(force_run=False):
    """Main function that runs the entire process."""
    try:
        logger.info("--- Trading Signal Process Started ---")
        
        # --- Market Hours Check ---
        if not force_run and not should_run():
            logger.info("Market is closed and 'force_run' is false. Exiting process.")
            # We can still update the sheet with a "Market Closed" message if desired
            return

        # Step 1: Connect to services and prepare data map
        kite = connect_to_kite()
        instrument_map = get_instrument_map(kite)

        # --- NEW: Fetch Option Chain Data ---
        option_chain_df = fetch_option_chain(kite, 'NIFTY')
        if option_chain_df is not None:
            # For now, just print the first 5 rows of the dataframe to verify
            logger.info("--- Option Chain Data (first 5 rows) ---")
            logger.info(option_chain_df.head())
            logger.info("-----------------------------------------")

        spreadsheet = connect_to_google_sheets()
        
        # Step 1.5: Ensure Google Sheet structure is correct
        enhance_sheet_structure(spreadsheet)
        
        # --- NEW: Bot Control Check ---
        # Check if the bot is enabled in the Google Sheet before proceeding.
        if not check_bot_status(spreadsheet):
            sys.exit(0) # Exit gracefully if bot is stopped

        # Step 2: Read supporting data from sheets
        manual_controls_df = read_manual_controls(spreadsheet)

        # Step 3: Read historical trade log to calculate performance stats
        trade_log_df = read_trade_log(spreadsheet)
        
        # Sentiment analysis model is no longer needed.

        # Step 4: Collect external data (market data, events)
        economic_events = fetch_economic_events()
        price_data_dict = run_data_collection(kite, instrument_map)
        
        # Step 5: Calculate all indicators for all instruments and timeframes
        if not price_data_dict["15m"].empty:
            price_data_dict["15m"] = calculate_indicators(price_data_dict["15m"])
            # Apply price action indicators after main indicators are calculated
            price_data_dict["15m"] = price_data_dict["15m"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
        if not price_data_dict["30m"].empty:
            price_data_dict["30m"] = calculate_indicators(price_data_dict["30m"])
            price_data_dict["30m"] = price_data_dict["30m"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
        if not price_data_dict["1h"].empty:
            price_data_dict["1h"] = calculate_indicators(price_data_dict["1h"])
            # Also apply to 1h data for completeness, though we primarily use it for trend context
            price_data_dict["1h"] = price_data_dict["1h"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
        
        # Step 5.5: Analyze Market Internals
        market_context = analyze_market_internals(price_data_dict)
        
        # Step 6: Generate signals using data, controls, performance, sentiment, and the new market context
        signals_df = generate_signals(price_data_dict, manual_controls_df, trade_log_df, market_context, economic_events)
        
        # Step 7: Write both data and signals to the sheets
        write_to_sheets(spreadsheet, price_data_dict["15m"], signals_df)

        logger.info("--- Trading Signal Process Completed Successfully ---")

    except Exception as e:
        # This will catch any error and log it, preventing a silent crash.
        logger.error("A critical error occurred in the main process:", exc_info=True)

# --- Script Execution ---
@process_data_bp.route('/run', methods=['GET'])
def run_bot():
    """
    HTTP endpoint to trigger the trading bot's main logic.
    """
    logger.info("Received request to run the trading bot.")
    force_run = request.args.get('force', 'false').lower() == 'true'
    if force_run:
        logger.warning("'force=true' parameter detected. Bypassing market hours check for this run.")

    try:
        # Call the existing main function
        main(force_run=force_run)
        return jsonify({"status": "success", "message": "Trading bot executed successfully."}), 200
    except Exception as e:
        logger.error(f"Error executing trading bot: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
