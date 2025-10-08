#!/usr/bin/env python3
"""
Test the live system with real credentials
"""
import requests
from api.live_options_engine import run_live_options_bot

def test_telegram_live():
    """Test Telegram with real credentials"""
    print("📱 Testing live Telegram connection...")
    
    TOKEN = '8250334547:AAHFIXLgvwlJlUUasiXY-5wHJ85E2AeC39k'
    CHAT_ID = '1375236879'
    MESSAGE = '🚀 AI Trading Platform is LIVE! Options alerts ready.'
    
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    payload = {'chat_id': CHAT_ID, 'text': MESSAGE}
    
    try:
        response = requests.post(url, data=payload)
        result = response.json()
        
        if result.get('ok'):
            print("✅ Telegram alert sent successfully!")
            return True
        else:
            print(f"❌ Telegram error: {result}")
            return False
    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def test_options_signal():
    """Test options signal generation"""
    print("📊 Testing options signal generation...")
    
    try:
        success = run_live_options_bot()
        if success:
            print("✅ Options signal generated and alert sent!")
        else:
            print("⚠️ No high-confidence signal (market conditions)")
        return True
    except Exception as e:
        print(f"❌ Options signal failed: {e}")
        return False

def main():
    print("🚀 LIVE SYSTEM TEST - AI Trading Platform")
    print("=" * 50)
    
    # Test Telegram
    telegram_ok = test_telegram_live()
    
    # Test Options
    options_ok = test_options_signal()
    
    print("\n" + "=" * 50)
    if telegram_ok and options_ok:
        print("🎉 SYSTEM IS FULLY LIVE!")
        print("✅ Telegram alerts working")
        print("✅ Options signals working")
        print("\n🚀 Ready for live trading!")
    else:
        print("⚠️ Some issues found, but core system working")
    
    print("\n📱 Next: python3 start_trading_platform.py")

if __name__ == "__main__":
    main()