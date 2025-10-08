#!/usr/bin/env python3
"""
Test the system right now - no Telegram required
"""
import sys
import os
sys.path.append('.')

def test_signal_generation():
    """Test if AI signals can be generated"""
    print("🔄 Testing signal generation...")
    
    try:
        from api.live_trading_bot import LiveTradingBot
        bot = LiveTradingBot()
        
        # Override market hours check for testing
        bot.is_market_open = lambda: True
        
        signals = bot.generate_trading_signals()
        
        if signals:
            print(f"✅ Generated {len(signals)} signals:")
            for symbol, data in signals.items():
                print(f"  {symbol}: {data['signal']} (confidence: {data['confidence']:.2f})")
            return True
        else:
            print("❌ No signals generated")
            return False
            
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def test_api_server():
    """Test if API server can start"""
    print("🌐 Testing API server...")
    
    try:
        import subprocess
        import time
        import requests
        
        # Start server in background
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'api.main:app', '--port', '8000'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(3)  # Wait for startup
        
        # Test endpoints
        try:
            response = requests.get('http://localhost:8000/api/health', timeout=5)
            if response.status_code == 200:
                print("✅ API server running")
                
                # Test signals endpoint
                signals_response = requests.get('http://localhost:8000/api/trading-signals', timeout=5)
                print(f"✅ Signals endpoint: {signals_response.status_code}")
                
                process.terminate()
                return True
            else:
                print(f"❌ API server error: {response.status_code}")
                process.terminate()
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ API server not responding")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_data_files():
    """Test if data files are being created"""
    print("📁 Testing data file creation...")
    
    try:
        # Run signal generation
        from api.live_trading_bot import LiveTradingBot
        bot = LiveTradingBot()
        bot.is_market_open = lambda: True
        
        signals = bot.generate_trading_signals()
        bot.save_signals(signals)
        
        # Check if files exist
        files_to_check = ['data/signals.json', 'data/bot_status.json']
        all_exist = True
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
                all_exist = False
                
        return all_exist
        
    except Exception as e:
        print(f"❌ Data file test failed: {e}")
        return False

def test_telegram_mock():
    """Test Telegram alert system with mock"""
    print("📱 Testing Telegram alert system (mock)...")
    
    try:
        # Mock the telegram sending
        from api.telegram_alerts import TelegramAlerts
        
        class MockTelegramAlerts(TelegramAlerts):
            def _send_telegram_message(self, message):
                print("📨 Mock Telegram message:")
                print(message[:200] + "..." if len(message) > 200 else message)
                return True
        
        mock_bot = MockTelegramAlerts()
        success = mock_bot.send_trade_alert("NIFTY", 25000, "CE", 180, 150, "Test Alert")
        
        if success:
            print("✅ Telegram alert system working (mock)")
            return True
        else:
            print("❌ Telegram alert system failed")
            return False
            
    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def run_complete_test():
    """Run all tests"""
    print("🚀 Testing AI Trading Platform - Live Test")
    print("=" * 50)
    
    tests = [
        ("Signal Generation", test_signal_generation),
        ("Data Files", test_data_files),
        ("Telegram Alerts (Mock)", test_telegram_mock),
        ("API Server", test_api_server)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 System is ready for live trading!")
        print("⚠️  Still need to configure Telegram credentials")
    else:
        print(f"\n❌ {total - passed} critical issues found")
        print("System NOT ready for live trading")
    
    return passed == total

if __name__ == "__main__":
    run_complete_test()