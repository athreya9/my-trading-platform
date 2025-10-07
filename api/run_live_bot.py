import json
from datetime import datetime, time
import os
from dotenv import load_dotenv
import logging
import pytz # Added import for pytz
from api.kite_connect import get_kite_connect_client
from api.ai_analysis_engine import AIAnalysisEngine
from api.data_collector import DataCollector
from api.technical_indicators import get_technical_indicators

from api.news_sentiment import fetch_news_sentiment
from automate_token_generation import get_automated_access_token
from api.options_signal_engine import generate_option_signal

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MODE = os.getenv("MODE", "dry_run")  # default to dry_run
print("MODE:", MODE) # Added debug print for MODE


def is_market_open():
    # Get current time in Asia/Kolkata timezone
    kolkata_timezone = pytz.timezone("Asia/Kolkata")
    now = datetime.now(kolkata_timezone)
    
    # Check if it's a weekday (Monday=0 to Friday=4)
    is_weekday = now.weekday() < 5 # 0-4 for Monday-Friday
    
    # Check if current time is within market hours (9:15 to 15:30 IST)
    is_within_market_hours = time(9, 15) <= now.time() <= time(15, 30)
    
    return is_weekday and is_within_market_hours

def tag_signal(signal):
    if signal.get("signal") in ["BUY", "SELL"] or signal.get("action") in ["BUY", "SELL"]:
        signal["status"] = "live"
    else:
        signal["status"] = "inactive"
    signal["generated_at"] = datetime.now().isoformat()
    return signal

def create_instrument_map(kite):
    """Fetches all instruments and creates a map for quick lookups."""
    logging.info("Fetching instrument list from Kite...")
    instruments = kite.instruments("NSE")
    instrument_map = {inst['tradingsymbol']: inst['instrument_token'] for inst in instruments}
    # Add indices manually as they are not in NSE instruments
    instrument_map["NIFTY 50"] = 256265
    instrument_map["NIFTY BANK"] = 260105
    instrument_map["SENSEX"] = 273929 # This is a BSE index, token might be different if using BSE exchange
    instrument_map["NIFTY FIN SERVICE"] = 257801
    logging.info("Instrument map created successfully.")
    return instrument_map

def get_instrument_token(instrument_map, symbol):
    """Gets the instrument token for a given symbol from the map."""
    return instrument_map.get(symbol)

def run_live_bot():
    """The main function to run the live trading bot."""
    logging.info("Starting live bot run...")

    if MODE == "live":
        if not is_market_open():
            logging.info("Market is closed. Skipping bot run.")
            return
        try:
            access_token = get_automated_access_token()
            if not access_token:
                print("Could not get access token. Exiting.")
                return

            kite = get_kite_connect_client()
            kite.set_access_token(access_token)
            logging.info("Successfully authenticated with Kite.")

        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return

        
        ai_engine = AIAnalysisEngine()
        collector = DataCollector()
        signals_to_save = []
 
        instrument_map = create_instrument_map(kite)

        instrument_list = [
            {"name": "NIFTY 50", "yahoo_symbol": "^NSEI", "kite_symbol": "NIFTY 50", "type": "index"},
            {"name": "Bank NIFTY", "yahoo_symbol": "^NSEBANK", "kite_symbol": "NIFTY BANK", "type": "index"},
            {"name": "SENSEX", "yahoo_symbol": "^BSESN", "kite_symbol": "SENSEX", "type": "index"},
            {"name": "FINNIFTY", "yahoo_symbol": "^CNXFIN", "kite_symbol": "NIFTY FIN SERVICE", "type": "index"},
            {"name": "Reliance Industries", "yahoo_symbol": "RELIANCE.NS", "kite_symbol": "RELIANCE", "type": "stock"},
            {"name": "HDFC Bank", "yahoo_symbol": "HDFCBANK.NS", "kite_symbol": "HDFCBANK", "type": "stock"},
            {"name": "ICICI Bank", "yahoo_symbol": "ICICIBANK.NS", "kite_symbol": "ICICIBANK", "type": "stock"},
            {"name": "Infosys", "yahoo_symbol": "INFY.NS", "kite_symbol": "INFY", "type": "stock"},
            {"name": "Tata Consultancy", "yahoo_symbol": "TCS.NS", "kite_symbol": "TCS", "type": "stock"},
            {"name": "Larsen & Toubro", "yahoo_symbol": "LT.NS", "kite_symbol": "LT", "type": "stock"},
            {"name": "Adani Enterprises", "yahoo_symbol": "ADANIENT.NS", "kite_symbol": "ADANIENT", "type": "stock"},
            {"name": "Adani Ports", "yahoo_symbol": "ADANIPORTS.NS", "kite_symbol": "ADANIPORTS", "type": "stock"},
            {"name": "Bajaj Finance", "yahoo_symbol": "BAJFINANCE.NS", "kite_symbol": "BAJFINANCE", "type": "stock"},
            {"name": "Maruti Suzuki", "yahoo_symbol": "MARUTI.NS", "kite_symbol": "MARUTI", "type": "stock"},
            {"name": "Tata Motors", "yahoo_symbol": "TATAMOTORS.NS", "kite_symbol": "TATAMOTORS", "type": "stock"},
            {"name": "Hindustan Unilever", "yahoo_symbol": "HINDUNILVR.NS", "kite_symbol": "HINDUNILVR", "type": "stock"},
            {"name": "State Bank of India", "yahoo_symbol": "SBIN.NS", "kite_symbol": "SBIN", "type": "stock"},
            {"name": "Axis Bank", "yahoo_symbol": "AXISBANK.NS", "kite_symbol": "AXISBANK", "type": "stock"},
            {"name": "JSW Steel", "yahoo_symbol": "JSWSTEEL.NS", "kite_symbol": "JSWSTEEL", "type": "stock"},
        ]
 
        for instrument in instrument_list:
            logging.info(f"--- Processing {instrument['name']} ---")
            # Re-enable news sentiment fetching
            sentiment_string = fetch_news_sentiment(instrument['name'])

            # Convert sentiment string to numerical score for options signal engine
            sentiment_score_numerical = 0
            if sentiment_string == "positive":
                sentiment_score_numerical = 1
            elif sentiment_string == "negative":
                sentiment_score_numerical = -1
 
            if instrument['type'] == 'index':
                option_signal = generate_option_signal(kite, instrument['kite_symbol'], sentiment_score_numerical)
                if option_signal:
                    option_signal = tag_signal(option_signal)
                    signals_to_save.append(option_signal)
                    
            else:
                historical_data = collector.fetch_historical_data(instrument['yahoo_symbol'], period="1y", interval="1d")
                if historical_data is None or historical_data.empty:
                    logging.warning(f"Could not fetch historical data for {instrument['name']}. Skipping.")
                    continue
 
                kite_data = get_technical_indicators(historical_data)
                kite_data['symbol'] = instrument['yahoo_symbol']
 
                instrument_token = get_instrument_token(instrument_map, instrument['kite_symbol'])
                if not instrument_token:
                    logging.warning(f"Could not get instrument token for {instrument['name']}. Skipping.")
                    continue
                
                ltp_data = kite.ltp(instrument_token)
                ltp_info = ltp_data.get(str(instrument_token))
                if not ltp_info:
                    logging.warning(f"Could not get LTP for {instrument['name']}. Skipping.")
                    continue
                live_price = ltp_info['last_price']
                kite_data['Close'] = live_price
 
                market_context = {}
 
                analysis = ai_engine.analyze_trading_opportunity(kite_data, market_context, instrument['name'])
                signal = ai_engine.generate_intelligent_signal(analysis)
                
                signal = tag_signal(signal)
                signal["instrument"] = instrument['name']
                signal["technical_score"] = analysis['technical']['score']
                signal["trend"] = "UP" if signal.get('action') == 'BUY' else "DOWN" if signal.get('action') == 'AVOID' else "NEUTRAL"
 
                
 
                # Rename 'action' to 'signal' for consistency in the final JSON
                if 'action' in signal:
                    signal['signal'] = signal.pop('action')
                signals_to_save.append(signal)
 
        logging.info("\n--- Generated Signals ---")
        logging.info(json.dumps(signals_to_save, indent=2))

        # Save all generated signals, not just live ones
        logging.info(f"\nSaving {len(signals_to_save)} generated signals to signals.json.")

        try:
            logging.info("Saving generated signals to signals.json...")
        try:
            logging.info("Updating signals.json...")
            with open("data/signals.json", 'w') as f:
                json.dump(signals_to_save, f, indent=2)
            logging.info("signals.json updated successfully.")
        except Exception as e:
            logging.error(f"❌ Error writing to signals.json: {e}")
    else:
        logging.info(f"Skipping signal generation — MODE={MODE}")

if __name__ == '__main__':
    run_live_bot()