# process_data.py
# A single, combined script for GitHub Actions.
# It fetches data, generates signals, and updates Google Sheets.

import gspread
from kiteconnect import KiteConnect
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import json
from transformers import pipeline
import torch
import time
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import logging
import sys

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Sheet & Symbol Configuration ---
SHEET_NAME = "Algo Trading Dashboard" # The name of your Google Sheet
DATA_WORKSHEET_NAME = "Price Data"
SIGNALS_WORKSHEET_NAME = "Signals"

# Data collection settings
WATCHLIST_SYMBOLS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']
MARKET_BREADTH_SYMBOLS = {
    "VIX": "^INDIAVIX",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO"
}
# Combine the main index (^NSEI), watchlist, and market breadth symbols
SYMBOLS = ['^NSEI'] + WATCHLIST_SYMBOLS + list(MARKET_BREADTH_SYMBOLS.values())

# Signal generation settings
SIGNAL_HEADERS = ['timestamp', 'instrument', 'option_type', 'strike_price', 'underlying_price', 'stop_loss', 'take_profit', 'position_size', 'reason', 'sentiment_score', 'kelly_pct', 'win_rate_p', 'win_loss_ratio_b', 'quality_score']

# Risk Management settings
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = 2.0  # e.g., 2 * ATR below entry price
TAKE_PROFIT_MULTIPLIER = 4.0 # e.g., 4 * ATR above entry price (for a 1:2 risk/reward ratio)

# Microstructure settings
REALIZED_VOL_WINDOW = 20 # e.g., 20 periods for volatility calculation
KELLY_CRITERION_CAP = 0.20 # Maximum percentage of capital to risk, as per Kelly Criterion

# --- New Market Internals Configuration ---
VIX_THRESHOLD = 20 # VIX level above which market is considered fearful/volatile

# --- New Price Action Configuration ---
# Defines a swing point as a candle higher/lower than its immediate neighbors.
SWING_POINT_LOOKBACK = 1

# NLP Sentiment Analysis settings
NLP_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_THRESHOLD = 0.1 # Minimum positive/negative sentiment score to influence a signal

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
    access_token = os.getenv('KITE_ACCESS_TOKEN', '').strip().strip('"\'')
    if not api_key or not access_token:
        raise ValueError("KITE_API_KEY or KITE_ACCESS_TOKEN environment variables not found. Ensure they are set in GitHub secrets.")
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # Optional: Verify connection by fetching profile to ensure token is valid
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

def run_data_collection(kite, instrument_map):
    """Fetches data for all symbols and timeframes and returns a dictionary of DataFrames."""
    data_frames = {"15m": [], "1h": []}
    to_date = datetime.now()
    
    for symbol in SYMBOLS:
        # Translate yfinance symbol to Kite tradingsymbol
        kite_symbol = YFINANCE_TO_KITE_MAP.get(symbol, symbol.replace('.NS', ''))
        instrument_token = instrument_map.get(kite_symbol)

        if not instrument_token:
            logger.warning(f"Could not find instrument token for symbol '{symbol}' (Kite: '{kite_symbol}'). Skipping.")
            continue

        # --- Fetch 15-minute data ---
        from_date_15m = to_date - timedelta(days=5)
        # Kite interval names are different from yfinance
        df_15m = fetch_historical_data(kite, instrument_token, from_date_15m, to_date, "15minute", symbol)
        if not df_15m.empty:
            data_frames["15m"].append(df_15m)
            
        # --- Fetch 1-hour data ---
        from_date_1h = to_date - timedelta(days=60)
        df_1h = fetch_historical_data(kite, instrument_token, from_date_1h, to_date, "60minute", symbol)
        if not df_1h.empty:
            data_frames["1h"].append(df_1h)

    if not data_frames["15m"]:
        raise Exception("No 15m data was fetched for any symbol. Halting process.")

    # Combine the lists of dataframes into single dataframes
    if data_frames["15m"]:
        combined_df_15m = pd.concat(data_frames["15m"], ignore_index=True)
    else:
        combined_df_15m = pd.DataFrame()
        
    if data_frames["1h"]:
        combined_df_1h = pd.concat(data_frames["1h"], ignore_index=True)
    else:
        combined_df_1h = pd.DataFrame()
    
    logger.info(f"Successfully fetched {len(combined_df_15m)} rows of 15m data and {len(combined_df_1h)} rows of 1h data.")
    
    return {"15m": combined_df_15m, "1h": combined_df_1h}

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
    group['fvg_bull_top'] = np.where(bull_fvg_mask, group['low'].shift(2), np.nan).shift(1)
    group['fvg_bull_bottom'] = np.where(bull_fvg_mask, group['high'], np.nan).shift(1)

    # A bearish FVG is where the high of candle[i-2] is below the low of candle[i].
    bear_fvg_mask = group['high'].shift(2) < group['low']
    group['fvg_bear_top'] = np.where(bear_fvg_mask, group['high'].shift(2), np.nan).shift(1)
    group['fvg_bear_bottom'] = np.where(bear_fvg_mask, group['low'], np.nan).shift(1)

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

def get_stock_info(symbol):
    """Fetches basic stock information like sector. Caches results to avoid repeated calls."""
    # A simple in-memory cache to avoid hitting the API for the same symbol repeatedly
    if 'stock_info_cache' not in get_stock_info.__dict__:
        get_stock_info.stock_info_cache = {}
    
    if symbol in get_stock_info.stock_info_cache:
        return get_stock_info.stock_info_cache[symbol]

    try:
        logger.info(f"Fetching yfinance info for {symbol}...")
        ticker = yf.Ticker(symbol)
        # The .info dict can be slow; we only need the sector
        info = {'sector': ticker.info.get('sector', 'N/A')}
        get_stock_info.stock_info_cache[symbol] = info
        return info
    except Exception as e:
        logger.warning(f"Could not fetch yfinance info for {symbol}: {e}")
        # Return a default value and cache it to prevent retries on the same failing symbol
        info = {'sector': 'N/A'}
        get_stock_info.stock_info_cache[symbol] = info
        return info

def calculate_risk_reward_ratio(entry, stop, target):
    """Calculates the risk/reward ratio."""
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return 0
    return reward / risk

def calculate_confidence_score(signal, latest_15m, latest_1h):
    """Calculates a confidence score for a signal based on multiple factors."""
    score = 0.0
    # Base score on the quality of the price action pattern
    score += signal.get('quality_score', 1) * 20  # Max 60 for a high-quality pattern
    if latest_15m.get('RSI_14', 50) < 65: score += 15 # Reward signals that are not overbought
    if signal.get('sentiment_score', 0) > 0.1: score += 25 # Reward signals with positive news sentiment
    return min(100, int(score)) # Return an integer score capped at 100


@retry()
def get_news_sentiment(instrument, analyzer):
    """Fetches news for an instrument and returns an aggregated sentiment score."""
    logger.info(f"Fetching and analyzing sentiment for {instrument}...")
    try:
        # Fetch news using yfinance's built-in news feature
        ticker_news = yf.Ticker(instrument).news
        if not ticker_news:
            logger.info(f"No news found for {instrument}.")
            return 0.0 # Return neutral sentiment if no news

        # Use .get() to safely access 'title', which may not always be present
        headlines = [news.get('title') for news in ticker_news[:8]]
        # Filter out any None values if a headline was missing a title
        headlines = [h for h in headlines if h]
        if not headlines:
            logger.info(f"No valid headlines with titles found for {instrument}.")
            return 0.0
        
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
        logger.info(f"Average sentiment score for {instrument}: {avg_score:.3f}")
        return avg_score

    except Exception as e:
        logger.warning(f"Could not get sentiment for {instrument}: {e}")
        return 0.0 # Return neutral on error

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
    vix_symbol = MARKET_BREADTH_SYMBOLS["VIX"]
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
    for name, symbol in MARKET_BREADTH_SYMBOLS.items():
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

def generate_signals(price_data_dict, manual_controls_df, trade_log_df, sentiment_analyzer, market_context):
    """Generates trading signals for all instruments, applying a suite of validation and risk rules."""
    logger.info("Generating signals with advanced rule validation...")
    
    # --- Setup & Market Context Filter ---
    kelly_pct, win_rate, win_loss_ratio = calculate_kelly_criterion(trade_log_df)
    all_potential_signals = []
    price_df_15m = price_data_dict['15m']
    price_df_1h = price_data_dict['1h']
    if market_context.get('is_vix_high', False):
        logger.info(f"Market Context Alert: VIX is above {VIX_THRESHOLD}. Avoiding new aggressive long positions.")
    if market_context.get('sentiment') == 'BEARISH':
        logger.info("Market Context Alert: Overall market trend is Bearish. Long signals will be suppressed.")

    # --- Get Sector Information for all Watchlist Symbols ---
    sector_info = {symbol: get_stock_info(symbol) for symbol in WATCHLIST_SYMBOLS}

    # --- Iterate over each instrument's 15-minute data ---
    for instrument, group_15m in price_df_15m.groupby('instrument'):
        # Ensure we have enough data for lookbacks and that price action features were calculated
        required_cols = ['SMA_50', 'RSI_14', f'ATRr_{ATR_PERIOD}', 'volume_avg_20', 'fvg_bull_bottom', 'choch', 'last_bull_ob_top', 'last_bull_ob_bottom']
        group_15m = group_15m.copy().dropna(subset=required_cols)
        if len(group_15m) < 3: # Need at least 3 rows for pattern detection lookbacks
            continue

        instrument_signals = []

        # Skip signal generation for the market internal instruments themselves
        if instrument in MARKET_BREADTH_SYMBOLS.values():
            continue

        latest_15m = group_15m.iloc[-1]

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

        # --- Rule 1: Volatility Filter ---
        atr_percentage = (latest_15m[f'ATRr_{ATR_PERIOD}'] / latest_15m['close']) * 100
        if atr_percentage > 3.0:
            logger.info(f"Skipping {instrument}: High volatility detected (ATR is {atr_percentage:.2f}% of price).")
            continue

        # --- Rule 2: Multi-Timeframe Confluence ---
        group_1h = price_df_1h[price_df_1h['instrument'] == instrument].copy()
        group_1h = group_1h.dropna(subset=['SMA_20', 'SMA_50'])
        if group_1h.empty:
            logger.info(f"Skipping {instrument}: Not enough 1-hour data for trend analysis.")
            continue
        latest_1h = group_1h.iloc[-1]
        is_1h_bullish = latest_1h['SMA_20'] > latest_1h['SMA_50']

        # --- Automated Signal Logic (with new rules) ---
        sentiment_score = get_news_sentiment(instrument, sentiment_analyzer)
        signal_generated = False
        reasons = []

        # --- BUY (CALL) Signal Conditions ---
        is_15m_bullish = latest_15m['SMA_20'] > latest_15m['SMA_50'] and latest_15m['RSI_14'] < 70
        volume_confirmed = latest_15m['volume'] > latest_15m['volume_avg_20']

        # --- New VWAP Filter ---
        price_above_vwap = latest_15m['close'] > latest_15m['vwap']

        if not signal_generated and is_15m_bullish and is_1h_bullish and volume_confirmed and price_above_vwap and sentiment_score > SENTIMENT_THRESHOLD:
            # CONTEXT CHECK: Do not open long positions if VIX is too high or market is bearish
            if market_context.get('is_vix_high', False) or market_context.get('sentiment') == 'BEARISH':
                logger.info(f"Skipping BUY for {instrument}: Market context is unfavorable (High VIX or Bearish Trend).")
                continue

            option_type = "CALL"
            reasons.append("15m/1h Bullish Trend")
            reasons.append("Volume Confirmation")
            reasons.append("Price > VWAP")
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
                'sentiment_score': sentiment_score,
                'win_rate_p': win_rate,
                'win_loss_ratio_b': win_loss_ratio,
                'kelly_pct': kelly_pct,
                'quality_score': 1 # Base quality score for trend signal
            })
            signal_generated = True

        # --- NEW: Price Action (FVG + CHoCH) Signal Logic ---
        if not signal_generated and is_1h_bullish:
            # Look for a reaction to a recent Bullish FVG
            # Condition: Price dipped into a recent FVG and is now moving out of it.
            reacted_to_fvg = False
            recent_fvgs = group_15m[~group_15m['fvg_bull_bottom'].isna()].tail(3)
            if not recent_fvgs.empty:
                last_fvg = recent_fvgs.iloc[-1]
                prev_low = group_15m['low'].iloc[-2]
                if prev_low <= last_fvg['fvg_bull_top'] and latest_15m['close'] > last_fvg['fvg_bull_bottom']:
                    reacted_to_fvg = True
                    reasons.append(f"Reacted to Bullish FVG @ {last_fvg['fvg_bull_bottom']:.2f}")

            # Check for a recent bullish Change of Character (CHoCH)
            bullish_choch_occured = (group_15m['choch'].tail(3) == 1).any()

            if reacted_to_fvg and bullish_choch_occured:
                # CONTEXT CHECK: Do not open long positions if VIX is too high or market is bearish
                if market_context.get('is_vix_high', False) or market_context.get('sentiment') == 'BEARISH':
                    logger.info(f"Skipping FVG+CHoCH BUY for {instrument}: Market context is unfavorable.")
                    continue

                option_type = "CALL"
                reasons.append("1h Bullish Trend")
                reasons.append("Bullish CHoCH after FVG pullback")

                atr_val = latest_15m[f'ATRr_{ATR_PERIOD}']
                entry_price = latest_15m['close']
                stop_loss = entry_price - (atr_val * STOP_LOSS_MULTIPLIER)
                take_profit = entry_price + (atr_val * TAKE_PROFIT_MULTIPLIER)
                position_size = 10 # Simplified for now

                instrument_signals.append({
                    'option_type': option_type,
                    'strike_price': get_atm_strike(entry_price, instrument),
                    'underlying_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': round(position_size),
                    'reason': ", ".join(reasons),
                    'sentiment_score': sentiment_score,
                    'win_rate_p': win_rate,
                    'win_loss_ratio_b': win_loss_ratio,
                    'kelly_pct': kelly_pct,
                    'quality_score': 3 # High quality score for FVG+CHoCH pattern
                })

        # --- NEW: Order Block Entry Signal Logic ---
        if not signal_generated and is_1h_bullish:
            # Check if price has recently entered a bullish order block zone
            last_ob_top = latest_15m.get('last_bull_ob_top')
            last_ob_bottom = latest_15m.get('last_bull_ob_bottom')

            if pd.notna(last_ob_top) and pd.notna(last_ob_bottom):
                # Condition: The previous candle's low was inside the OB, and the current candle is closing above it.
                prev_low = group_15m['low'].iloc[-2]
                
                entered_ob = prev_low <= last_ob_top and prev_low >= last_ob_bottom
                exiting_ob_bullishly = latest_15m['close'] > last_ob_top

                if entered_ob and exiting_ob_bullishly:
                    # CONTEXT CHECK
                    if market_context.get('is_vix_high', False) or market_context.get('sentiment') == 'BEARISH':
                        logger.info(f"Skipping Order Block BUY for {instrument}: Market context is unfavorable.")
                        continue

                    option_type = "CALL"
                    reasons.append("1h Bullish Trend")
                    reasons.append(f"Reclaimed Bullish OB @ {last_ob_bottom:.2f}")

                    entry_price = latest_15m['close']
                    # Place stop loss just below the low of the order block for a defined risk
                    stop_loss = last_ob_bottom
                    
                    if entry_price <= stop_loss: continue

                    # Calculate take profit based on a fixed risk/reward from this tight stop
                    risk_amount = entry_price - stop_loss
                    take_profit = entry_price + (risk_amount * TAKE_PROFIT_MULTIPLIER)
                    
                    # Position Sizing
                    account_size = 100000 # Example
                    risk_per_trade_pct = 0.01 # 1% risk
                    position_size = (account_size * risk_per_trade_pct) / risk_amount if risk_amount > 0 else 0

                    instrument_signals.append({
                        'option_type': option_type, 'strike_price': get_atm_strike(entry_price, instrument),
                        'underlying_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                        'position_size': round(position_size), 'reason': ", ".join(reasons),
                        'sentiment_score': sentiment_score, 'win_rate_p': win_rate,
                        'win_loss_ratio_b': win_loss_ratio, 'kelly_pct': kelly_pct,
                        'quality_score': 3 # High quality score for Order Block pattern
                    })
                    signal_generated = True

        # --- SELL (PUT) Signal Conditions (Simplified for now) ---
        is_15m_bearish = latest_15m['SMA_20'] < latest_15m['SMA_50']
        if is_15m_bearish and not is_1h_bullish and sentiment_score < -SENTIMENT_THRESHOLD:
            # (Position sizing for PUTs would be similar)
            pass # Not fully implemented as per user request focus on BUY

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
    sector_allocation = {sector_info[s]['sector']: 0 for s in WATCHLIST_SYMBOLS}

    for signal in all_potential_signals:
        stock_sector = sector_info.get(signal['instrument'], {}).get('sector', 'N/A')
        signal['risk_reward_ratio'] = calculate_risk_reward_ratio(signal['underlying_price'], signal['stop_loss'], signal['take_profit'])
        signal['confidence_score'] = calculate_confidence_score(signal, price_df_15m[price_df_15m['instrument'] == signal['instrument']].iloc[-1], price_df_1h[price_df_1h['instrument'] == signal['instrument']].iloc[-1])

        # THE FILTER: Only add trades that pass all checks
        if (signal['confidence_score'] >= 75 and 
            signal['risk_reward_ratio'] >= 1.5 and 
            market_context['sentiment'] == "BULLISH" and 
            (stock_sector == 'N/A' or sector_allocation.get(stock_sector, 2) < 2)): # Max 2 trades per sector

            high_confidence_trades.append(signal)
            if stock_sector != 'N/A':
                sector_allocation[stock_sector] += 1

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
    """Formats the top signal into a natural language string for the advisor output."""
    stock_name = signal['instrument'].replace('.NS', '')
    action = f"BUY {signal['option_type']}"
    reason = signal['reason']
    entry_price = f"~{signal['underlying_price']:.2f}"
    stop_loss = f"{signal['stop_loss']:.2f}"
    target_price = f"{signal['take_profit']:.2f}"
    confidence = signal.get('confidence_score', 0)

    advice = f"""💡 TODAY'S TOP OPPORTUNITY:
   Stock: {stock_name}
   Action: {action}
   Reason: {reason}
   Entry: {entry_price}
   Stop Loss: {stop_loss}
   Target: {target_price}
   Confidence: {confidence:.0f}%"""

    return advice, confidence

def get_or_create_worksheet(spreadsheet, name, rows="1000", cols="20"):
    """Gets a worksheet by name, creating it if it doesn't exist."""
    try:
        return spreadsheet.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        logger.info(f"Creating '{name}' worksheet.")
        return spreadsheet.add_worksheet(title=name, rows=str(rows), cols=str(cols))

@retry()
def write_to_sheets(spreadsheet, price_df, signals_df):
    """Writes price data, signals, and the final advice to their respective sheets."""
    logger.info("--- Starting Sheet Update Process ---")
    data_worksheet = get_or_create_worksheet(spreadsheet, DATA_WORKSHEET_NAME, rows=1000, cols=20)
    signals_worksheet = get_or_create_worksheet(spreadsheet, SIGNALS_WORKSHEET_NAME, rows=100, cols=20)
    # New dedicated sheet for the final, polished output
    advisor_output_worksheet = get_or_create_worksheet(spreadsheet, "Advisor_Output", rows=10, cols=5)

    # --- Write Price Data ---
    if not price_df.empty:
        logger.info(f"Preparing to write {len(price_df)} rows to '{DATA_WORKSHEET_NAME}'...")
        price_df_str = price_df.copy()
        price_df_str['timestamp'] = price_df_str['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        price_df_str.fillna('', inplace=True)
        price_data_to_write = [price_df_str.columns.tolist()] + price_df_str.values.tolist()
        logger.info("Clearing Price Data sheet...")
        data_worksheet.clear()
        logger.info("Updating Price Data sheet...")
        data_worksheet.update('A1', price_data_to_write, value_input_option='USER_ENTERED')
        logger.info("Price data written successfully.")
    else:
        logger.info("No price data to write. Clearing old price data from sheet.")
        data_worksheet.clear()

    # --- Write Signal Data ---
    if not signals_df.empty:
        logger.info(f"Preparing to write {len(signals_df)} rows to '{SIGNALS_WORKSHEET_NAME}'...")
        signals_df_str = signals_df.copy()
        signals_df_str['timestamp'] = signals_df_str['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        signals_df_str.fillna('', inplace=True)
        signal_data_to_write = [signals_df_str.columns.tolist()] + signals_df_str.values.tolist()
        logger.info("Clearing Signals sheet...")
        signals_worksheet.clear()
        logger.info("Updating Signals sheet...")
        signals_worksheet.update('A1', signal_data_to_write, value_input_option='USER_ENTERED')
        logger.info("Signal data written successfully.")
    else:
        logger.info("No signals to write. Clearing old signals from sheet.")
        signals_worksheet.clear()

    # --- Write Final Advisor Output ---
    logger.info("Preparing to write to 'Advisor_Output' sheet...")
    advisor_output_worksheet.clear()
    if not signals_df.empty:
        # --- NEW: Rank signals to find the best opportunity ---
        # Sort by quality score (higher is better) and then by sentiment (higher is better) as a tie-breaker.
        signals_df_sorted = signals_df.sort_values(by=['confidence_score', 'sentiment_score'], ascending=[False, False])
        
        # Select the top opportunity after ranking
        top_signal = signals_df_sorted.iloc[0]
        
        # Generate the natural language advice and confidence score
        advice_string, confidence_score = generate_advisor_output(top_signal)
        
        logger.info("Updating Advisor_Output sheet with the top opportunity...")
        advisor_output_worksheet.update('A1', advice_string, value_input_option='USER_ENTERED')
        advisor_output_worksheet.update('B1', confidence_score, value_input_option='USER_ENTERED')
        logger.info("Advisor output written successfully.")
    else:
        logger.info("No signals to generate advice. Clearing and updating Advisor_Output sheet with status.")
        advisor_output_worksheet.update('A1', "No valid trading signals found after applying all rules.", value_input_option='USER_ENTERED')
    logger.info("--- Sheet Update Process Completed ---")

def main():
    """Main function that runs the entire process."""
    try:
        logger.info("--- Trading Signal Process Started ---")
        
        # Step 1: Connect to services and prepare data map
        kite = connect_to_kite()
        instrument_map = get_instrument_map(kite)
        spreadsheet = connect_to_google_sheets()
        
        # Step 2: Read supporting data from sheets
        manual_controls_df = read_manual_controls(spreadsheet)

        # Step 3: Read historical trade log to calculate performance stats
        trade_log_df = read_trade_log(spreadsheet)
        
        # Initialize the sentiment analysis pipeline once
        logger.info("Initializing sentiment analysis model (this may take a moment on first run)...")
        # Use GPU if available, otherwise CPU.
        device = 0 if torch.cuda.is_available() else -1
        sentiment_analyzer = pipeline("sentiment-analysis", model=NLP_MODEL_NAME, device=device)
        logger.info("Sentiment model initialized.")

        # Step 4: Collect new data for multiple timeframes
        price_data_dict = run_data_collection(kite, instrument_map)
        
        # Step 5: Calculate all indicators for all instruments and timeframes
        if not price_data_dict["15m"].empty:
            price_data_dict["15m"] = calculate_indicators(price_data_dict["15m"])
            # Apply price action indicators after main indicators are calculated
            price_data_dict["15m"] = price_data_dict["15m"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
        if not price_data_dict["1h"].empty:
            price_data_dict["1h"] = calculate_indicators(price_data_dict["1h"])
            # Also apply to 1h data for completeness, though we primarily use it for trend context
            price_data_dict["1h"] = price_data_dict["1h"].groupby('instrument', group_keys=False).apply(apply_price_action_indicators)
        
        # Step 5.5: Analyze Market Internals
        market_context = analyze_market_internals(price_data_dict)
        
        # Step 6: Generate signals using data, controls, performance, sentiment, and the new market context
        signals_df = generate_signals(price_data_dict, manual_controls_df, trade_log_df, sentiment_analyzer, market_context)
        
        # Step 7: Write both data and signals to the sheets
        write_to_sheets(spreadsheet, price_data_dict["15m"], signals_df)

        logger.info("--- Trading Signal Process Completed Successfully ---")

    except Exception as e:
        # This will catch any error and log it, preventing a silent crash.
        logger.error("A critical error occurred in the main process:", exc_info=True)
        sys.exit(1) # Exit with a non-zero code to fail the GitHub Action

# --- Script Execution ---
if __name__ == "__main__":
    main()
