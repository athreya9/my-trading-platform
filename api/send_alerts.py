import json
from api.accurate_telegram_alerts import AccurateTelegramAlerts
from api.ai_analysis_engine import send_ai_powered_alert
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_alerts():
    """Reads signals.json and sends alerts for live signals."""
    try:
        with open("data/signals.json", 'r') as f:
            signals = json.load(f)
    except FileNotFoundError:
        logging.info("signals.json not found. No alerts to send.")
        return
    except json.JSONDecodeError:
        logging.error("Could not decode signals.json. No alerts to send.")
        return

    live_signals = [s for s in signals if s.get("status") == "live"]
    logging.info(f"Found {len(live_signals)} live signals to send.")

    if not live_signals:
        return

    telegram_bot = AccurateTelegramAlerts()

    for signal in live_signals:
        print("Attempting to send alert for:", signal["instrument"])
        # The send_ai_powered_alert function expects an 'analysis' object.
        # We will create a dummy analysis object for now.
        analysis = {
            'technical': {'score': signal.get('technical_score', 0), 'pattern': 'N/A', 'strengths': []},
            'market': {'sector_rotation': 'N/A', 'market_breadth': 'N/A', 'volatility_regime': 'N/A'},
            'sentiment': {'overall': 'N/A', 'score': 0},
            'risk': {'kelly_fraction': 0, 'var_95': 0, 'win_probability': 0}
        }
        try:
            send_ai_powered_alert(signal, analysis, telegram_bot)
            print("Telegram alert function executed successfully.")
        except Exception as e:
            print("Telegram alert failed:", str(e))

if __name__ == "__main__":
    send_alerts()
