#!/usr/bin/env python3
"""
Central configuration file for the trading bot.
"""

# --- Trading Mode Configuration ---
# Set to 'EMERGENCY' to use a minimal set of instruments and save API credits.
# Set to 'FULL' for normal operation with the complete watchlist.
CURRENT_MODE = 'EMERGENCY'

# --- Instrument Lists ---

# Full list for normal operation
FULL_WATCHLIST_SYMBOLS = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS',
    'BHARTIARTL.NS'
]
FULL_MARKET_BREADTH_SYMBOLS = {
    "VIX": "^INDIAVIX",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO"
}

# Minimal list for credit-saving/emergency mode
EMERGENCY_WATCHLIST_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS']
EMERGENCY_MARKET_BREADTH_SYMBOLS = {} # No extra indices in emergency mode

# --- Market Holiday Configuration ---
# List of market holidays in 'YYYY-MM-DD' format.
# The bot will not run on these days.
MARKET_HOLIDAYS = [
    '2025-02-26', # Mahashivratri
    '2025-03-14', # Holi
    '2025-03-31', # Eid-Ul-Fitr (Ramadan Eid)
    '2025-04-10', # Mahavir Jayanti
    '2025-04-14', # Dr. Baba Saheb Ambedkar Jayanti
    '2025-04-18', # Good Friday
    '2025-05-01', # Maharashtra Day
    '2025-08-15', # Independence Day
    '2025-08-27', # Ganesh Chaturthi
    '2025-10-02', # Mahatma Gandhi Jayanti/Dussehra
    '2025-10-21', # Diwali Laxmi Pujan
    '2025-10-22', # Diwali-Balipratipada
    '2025-11-05', # Prakash Gurpurb Sri Guru Nanak Dev
    '2025-12-25', # Christmas
]

# --- Export the correct configuration based on the current mode ---
WATCHLIST_SYMBOLS = FULL_WATCHLIST_SYMBOLS if CURRENT_MODE == 'FULL' else EMERGENCY_WATCHLIST_SYMBOLS
MARKET_BREADTH_SYMBOLS = FULL_MARKET_BREADTH_SYMBOLS if CURRENT_MODE == 'FULL' else EMERGENCY_MARKET_BREADTH_SYMBOLS

# Combine the main index (^NSEI), watchlist, and market breadth symbols
SYMBOLS = ['^NSEI'] + WATCHLIST_SYMBOLS + list(MARKET_BREADTH_SYMBOLS.values())

# --- Machine Learning Feature Configuration ---
# This list defines the exact features the AI model will be trained on and use for predictions.
# Centralizing it here prevents inconsistencies between training and live prediction.
ML_FEATURE_COLUMNS = [
    'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'ATRr_14',
    'volume_avg_20', 'realized_vol', 'vwap', 'bos', 'choch',
    'last_bull_ob_top', 'last_bull_ob_bottom'
]