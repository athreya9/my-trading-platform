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
            ("NIFTY 50", "^NSEI", "NIFTY 50"), 
            ("Bank NIFTY", "^NSEBANK", "NIFTY BANK"),
            ("SENSEX", "^BSESN", "SENSEX"),
            ("FINNIFTY", "^CNXFIN", "NIFTY FIN SERVICE"),
            ("Reliance Industries", "RELIANCE.NS", "RELIANCE"),
            ("HDFC Bank", "HDFCBANK.NS", "HDFCBANK"),
            ("ICICI Bank", "ICICIBANK.NS", "ICICIBANK"),
            ("Infosys", "INFY.NS", "INFY"),
            ("Tata Consultancy", "TCS.NS", "TCS"),
            ("Larsen & Toubro", "LT.NS", "LT"),
            ("Adani Enterprises", "ADANIENT.NS", "ADANIENT"),
            ("Adani Ports", "ADANIPORTS.NS", "ADANIPORTS"),
            ("Bajaj Finance", "BAJFINANCE.NS", "BAJFINANCE"),
            ("Maruti Suzuki", "MARUTI.NS", "MARUTI"),
            ("Tata Motors", "TATAMOTORS.NS", "TATAMOTORS"),
            ("Hindustan Unilever", "HINDUNILVR.NS", "HINDUNILVR"),
            ("State Bank of India", "SBIN.NS", "SBIN"),
            ("Axis Bank", "AXISBANK.NS", "AXISBANK"),
            ("JSW Steel", "JSWSTEEL.NS", "JSWSTEEL"),
        ]

        for instrument_name, symbol, kite_symbol in instrument_list:
            print(f"--- Processing {instrument_name} ---")
            sentiment = fetch_news_sentiment(instrument_name)

            if "NIFTY" in instrument_name or "SENSEX" in instrument_name:
                option_signal = generate_option_signal(kite, kite_symbol, sentiment['score'])
                if option_signal:
                    option_signal["status"] = "live"
                    option_signal["generated_at"] = datetime.now().isoformat()
                    signals_to_save.append(option_signal)
                    send_option_alert(option_signal)
            else:
                historical_data = collector.fetch_historical_data(symbol, period="1y", interval="1d")
                if historical_data is None or historical_data.empty:
                    print(f"Could not fetch historical data for {instrument_name}. Skipping.")
                    continue

                kite_data = get_technical_indicators(historical_data)
                kite_data['symbol'] = symbol

                instrument_token = get_instrument_token(kite, kite_symbol)
                if not instrument_token:
                    print(f"Could not get instrument token for {instrument_name}. Skipping.")
                    continue
                
                ltp_data = kite.ltp(instrument_token)
                live_price = ltp_data[str(instrument_token)]['last_price']
                kite_data['Close'] = live_price

                market_context = {}

                analysis = ai_engine.analyze_trading_opportunity(kite_data, market_context, sentiment)
                signal = ai_engine.generate_intelligent_signal(analysis)
                
                signal["status"] = "live"
                signal["generated_at"] = datetime.now().isoformat()

                if signal['action'] != 'HOLD':
                    send_ai_powered_alert(signal, analysis, telegram_bot)

                signals_to_save.append({
                    "instrument": instrument_name,
                    "trend": "UP" if signal['action'] == 'BUY' else "DOWN" if signal['action'] == 'AVOID' else "NEUTRAL",
                    "signal": signal['action'],
                    "confidence": signal['confidence'],
                    "reasoning": signal['reasoning'],
                    "technical_score": analysis['technical']['score'],
                    "specific_instructions": signal['specific_instructions'],
                    "profit_targets": signal['profit_targets'],
                    "time_horizon": signal['time_horizon'],
                    "exit_conditions": signal['exit_conditions'],
                    "trail_stop_level": signal['trail_stop_level'],
                    "status": "live",
                    "generated_at": datetime.now().isoformat()
                })

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
