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
    '2024-01-26', # Republic Day
    '2024-08-15', # Independence Day
    '2024-10-02', # Gandhi Jayanti
    '2024-12-25', # Christmas
]

# --- Export the correct configuration based on the current mode ---
WATCHLIST_SYMBOLS = FULL_WATCHLIST_SYMBOLS if CURRENT_MODE == 'FULL' else EMERGENCY_WATCHLIST_SYMBOLS
MARKET_BREADTH_SYMBOLS = FULL_MARKET_BREADTH_SYMBOLS if CURRENT_MODE == 'FULL' else EMERGENCY_MARKET_BREADTH_SYMBOLS

# Combine the main index (^NSEI), watchlist, and market breadth symbols
SYMBOLS = ['^NSEI'] + WATCHLIST_SYMBOLS + list(MARKET_BREADTH_SYMBOLS.values())