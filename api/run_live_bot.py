import json
from datetime import datetime
from api.kite_connect import get_access_token_with_totp, get_kite_connect_client
from api.ai_analysis_engine import AIAnalysisEngine, send_ai_powered_alert
from api.data_collector import DataCollector
from api.technical_indicators import get_technical_indicators

def get_instrument_token(kite, symbol):
    """Gets the instrument token for a given symbol."""
    # In a real application, you would fetch the full instrument list and find the token dynamically.
    # For now, we will hardcode the tokens for NIFTY 50 and Bank NIFTY.
    if symbol == "^NSEI":
        return 256265 # NIFTY 50
    elif symbol == "^NSEBANK":
        return 260105 # NIFTY BANK
    else:
        # Fallback to search for the instrument token
        instruments = kite.instruments("NSE")
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
    return None

def run_live_bot():
    """The main function to run the live trading bot."""
    print("Running live bot...")

    # --- 1. Authenticate with Kite ---
    try:
        access_token = get_access_token_with_totp()
        if not access_token:
            print("Could not get access token. Exiting.")
            return

        kite = get_kite_connect_client()
        kite.set_access_token(access_token)
        print("Successfully authenticated with Kite.")

    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    # --- 2. Fetch Live Data and Generate Signals ---
    print("Fetching live data and generating signals...")
    ai_engine = AIAnalysisEngine()
    collector = DataCollector()

    signals_to_save = []

    for instrument_name, symbol, kite_symbol in [("NIFTY 50", "^NSEI", "NIFTY 50"), ("Bank NIFTY", "^NSEBANK", "NIFTY BANK")]:
        historical_data = collector.fetch_historical_data(symbol, period="1y", interval="1d")
        if historical_data is None or historical_data.empty:
            print(f"Could not fetch historical data for {instrument_name}. Skipping.")
            continue

        # Calculate technical indicators
        kite_data = get_technical_indicators(historical_data)
        kite_data['symbol'] = symbol

        # Get live price from Kite
        instrument_token = get_instrument_token(kite, kite_symbol)
        if not instrument_token:
            print(f"Could not get instrument token for {instrument_name}. Skipping.")
            continue
        ltp_data = kite.ltp(instrument_token)
        live_price = ltp_data[str(instrument_token)]['last_price']
        kite_data['Close'] = live_price # Update the close price with the live price

        # Prepare dummy market context and news sentiment
        market_context = {}
        news_sentiment = {}

        # Get AI analysis and generate signal
        analysis = ai_engine.analyze_trading_opportunity(kite_data, market_context, news_sentiment)
        signal = ai_engine.generate_intelligent_signal(analysis)

        # Determine trend for frontend display
        if signal['action'] == 'BUY':
            trend = "UP"
        elif signal['action'] == 'AVOID':
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        # Send Telegram alert
        if signal['action'] != 'HOLD':
            send_ai_powered_alert(signal, analysis)

        signals_to_save.append({
            "instrument": instrument_name,
            "trend": trend,
            "signal": signal['action'],
            "confidence": signal['confidence'],
            "reasoning": signal['reasoning'],
            "technical_score": analysis['technical']['score'],
            "specific_instructions": signal['specific_instructions'],
            "profit_targets": signal['profit_targets'],
            "time_horizon": signal['time_horizon'],
            "exit_conditions": signal['exit_conditions'],
            "trail_stop_level": signal['trail_stop_level']
        })

    # --- 3. Update JSON Files ---
    print("Updating signals.json...")
    with open("data/signals.json", 'w') as f:
        json.dump(signals_to_save, f, indent=2)

    print("signals.json updated successfully.")

if __name__ == '__main__':
    run_live_bot()
