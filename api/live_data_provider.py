# api/live_data_provider.py
import yfinance as yf
from datetime import datetime

# Note: yfinance is not a true real-time data provider. 
# The data is delayed, and the option chain data might not be suitable for live trading.
# For a real trading application, you should use a broker's API or a dedicated market data API.

def get_live_quote(symbol):
    """Fetches the last price for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching live quote for {symbol}: {e}")
    return None

def get_option_chain(symbol):
    """Fetches the option chain for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        # Get the next expiry date
        expiries = ticker.options
        if not expiries:
            return None
        
        # For simplicity, we'll use the first available expiry
        # In a real application, you would select the desired expiry
        expiry_date = expiries[0]
        
        opt = ticker.option_chain(expiry_date)
        return opt
    except Exception as e:
        print(f"Error fetching option chain for {symbol}: {e}")
    return None

if __name__ == '__main__':
    # Example usage
    nifty_price = get_live_quote("^NSEI")
    if nifty_price:
        print(f"NIFTY Price: {nifty_price}")
    
    banknifty_price = get_live_quote("^NSEBANK")
    if banknifty_price:
        print(f"Bank NIFTY Price: {banknifty_price}")

    nifty_options = get_option_chain("^NSEI")
    if nifty_options:
        print("NIFTY Option Chain Calls:")
        print(nifty_options.calls.head())
        print("NIFTY Option Chain Puts:")
        print(nifty_options.puts.head())
