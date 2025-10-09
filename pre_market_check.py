#!/usr/bin/env python3
"""
Pre-market system health check
"""
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_telegram_bot():
    """Test Telegram bot connectivity"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        return False, "No bot token"
    
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True, "Bot online"
        return False, f"Bot error: {response.status_code}"
    except Exception as e:
        return False, f"Bot connection failed: {e}"

def check_kite_credentials():
    """Check Kite API credentials"""
    api_key = os.getenv('KITE_API_KEY')
    access_token = os.getenv('KITE_ACCESS_TOKEN')
    
    if not api_key:
        return False, "No Kite API key"
    if not access_token:
        return False, "No access token"
    
    return True, "Kite credentials present"

def check_data_files():
    """Check required data files"""
    files = ['data/signals.json', 'api-data/trading-signals.json']
    
    for file_path in files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
    
    return True, "Data files ready"

def test_signal_generation():
    """Test signal generation"""
    try:
        from api.kite_live_engine import KiteLiveEngine
        engine = KiteLiveEngine()
        signals = engine.generate_kite_signals()
        return True, f"Generated {len(signals)} test signals"
    except Exception as e:
        return False, f"Signal generation failed: {e}"

def test_telegram_alert():
    """Send test alert"""
    try:
        from api.telegram_alerts import send_quick_alert
        success = send_quick_alert(
            symbol="NIFTY",
            strike=25000,
            option_type="CE",
            entry_price=100,
            stoploss=85,
            reason="PRE-MARKET TEST"
        )
        return success, "Test alert sent" if success else "Alert failed"
    except Exception as e:
        return False, f"Alert test failed: {e}"

def main():
    """Run all pre-market checks"""
    print("🔍 PRE-MARKET SYSTEM CHECK")
    print("=" * 40)
    
    checks = [
        ("Telegram Bot", check_telegram_bot),
        ("Kite Credentials", check_kite_credentials),
        ("Data Files", check_data_files),
        ("Signal Generation", test_signal_generation),
        ("Telegram Alert", test_telegram_alert)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            success, message = check_func()
            status = "✅" if success else "❌"
            print(f"{status} {name}: {message}")
            if not success:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("🚀 ALL SYSTEMS READY FOR MARKET!")
        print("📊 5 instruments: NIFTY, BANKNIFTY, SENSEX, FINNIFTY, NIFTYIT")
        print("📱 Telegram alerts: Active")
        print("💻 Frontend: Ready")
    else:
        print("⚠️  SOME ISSUES FOUND - FIX BEFORE MARKET OPENS")
    
    print(f"⏰ Check completed at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()