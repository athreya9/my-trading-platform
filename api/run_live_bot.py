import json
from datetime import datetime, time
import os
from api.kite_connect import get_kite_connect_client
from api.ai_analysis_engine import AIAnalysisEngine, send_ai_powered_alert
from api.data_collector import DataCollector
from api.technical_indicators import get_technical_indicators
from api.accurate_telegram_alerts import AccurateTelegramAlerts
from api.news_sentiment import fetch_news_sentiment
from automate_token_generation import get_automated_access_token
from api.options_signal_engine import generate_option_signal, send_option_alert

MODE = os.getenv("MODE", "dry_run")  # default to dry_run

def is_market_open():
    now = datetime.now().time()
    return time(9, 15) <= now <= time(15, 30)

def get_instrument_token(kite, symbol):
    """Gets the instrument token for a given symbol."""
    if symbol == "NIFTY 50":
        return 256265
    elif symbol == "NIFTY BANK":
        return 260105
    elif symbol == "SENSEX":
        return 273929
    elif symbol == "NIFTY FIN SERVICE":
        return 257801
    else:
        instruments = kite.instruments("NSE")
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
    return None

def run_live_bot():
    """The main function to run the live trading bot."""
    print("Running live bot...")

    if MODE == "live":
        try:
            access_token = get_automated_access_token()
            if not access_token:
                print("Could not get access token. Exiting.")
                return

            kite = get_kite_connect_client()
            kite.set_access_token(access_token)
            print("Successfully authenticated with Kite.")

        except Exception as e:
            print(f"Authentication failed: {e}")
            return

        telegram_bot = AccurateTelegramAlerts(kite=kite)
        ai_engine = AIAnalysisEngine()
        collector = DataCollector()
        signals_to_save = []
 
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
            print(f"--- Processing {instrument['name']} ---")
            # Temporarily disable news sentiment fetching to bypass rate limit issues
            # sentiment = fetch_news_sentiment(instrument['name'])
            sentiment = {"score": 0, "summary": "News sentiment temporarily disabled."} # Default neutral sentiment
 
            if instrument['type'] == 'index':
                option_signal = generate_option_signal(kite, instrument['kite_symbol'], sentiment['score'])
                if option_signal:
                    option_signal["status"] = "live"
                    option_signal["generated_at"] = datetime.now().isoformat()
                    signals_to_save.append(option_signal)
                    send_option_alert(option_signal)
            else:
                historical_data = collector.fetch_historical_data(instrument['yahoo_symbol'], period="1y", interval="1d")
                if historical_data is None or historical_data.empty:
                    print(f"Could not fetch historical data for {instrument['name']}. Skipping.")
                    continue
 
                kite_data = get_technical_indicators(historical_data)
                kite_data['symbol'] = instrument['yahoo_symbol']
 
                instrument_token = get_instrument_token(kite, instrument['kite_symbol'])
                if not instrument_token:
                    print(f"Could not get instrument token for {instrument['name']}. Skipping.")
                    continue
                
                ltp_data = kite.ltp(instrument_token)
                ltp_info = ltp_data.get(str(instrument_token))
                if not ltp_info:
                    print(f"Could not get LTP for {instrument['name']}. Skipping.")
                    continue
                live_price = ltp_info['last_price']
                kite_data['Close'] = live_price
 
                market_context = {}
 
                analysis = ai_engine.analyze_trading_opportunity(kite_data, market_context, sentiment)
                signal = ai_engine.generate_intelligent_signal(analysis)
                
                # Determine status based on action for stock signals
                signal["status"] = "live" if signal['action'] in ['BUY', 'SELL'] else "inactive"
                signal["generated_at"] = datetime.now().isoformat()
                signal["instrument"] = instrument['name']
                signal["technical_score"] = analysis['technical']['score']
                signal["trend"] = "UP" if signal['action'] == 'BUY' else "DOWN" if signal['action'] == 'AVOID' else "NEUTRAL"
 
                if signal['action'] != 'HOLD':
                    send_ai_powered_alert(signal, analysis, telegram_bot)
 
                # Rename 'action' to 'signal' for consistency in the final JSON
                signal['signal'] = signal.pop('action')
                signals_to_save.append(signal)
 
        print("\n--- Generated Signals ---")
        print(json.dumps(signals_to_save, indent=2))

        try:
            print("\nUpdating signals.json...")
            with open("data/signals.json", 'w') as f:
                json.dump(signals_to_save, f, indent=2)
            print("signals.json updated successfully.")
        except Exception as e:
            print(f"❌ Error writing to signals.json: {e}")
    else:
        print(f"Skipping signal generation — MODE={MODE}, Market Open={is_market_open()}")

if __name__ == '__main__':
    run_live_bot()
