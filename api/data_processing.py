import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

ATR_PERIOD = 14
REALIZED_VOL_WINDOW = 20
SWING_POINT_LOOKBACK = 1

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
