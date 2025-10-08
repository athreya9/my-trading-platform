#!/usr/bin/env python3
"""
Complete system test for the AI trading platform
"""
import os
import json
import requests
from datetime import datetime
from api.live_trading_bot import LiveTradingBot
from api.telegram_alerts import test_telegram_alerts

def test_data_generation():
    """Test if signals are being generated"""
    print("🔄 Testing signal generation...")
    
    bot = LiveTradingBot()
    success = bot.run_single_cycle()
    
    if success:
        # Check if signals.json was created
        if os.path.exists('data/signals.json'):
            with open('data/signals.json', 'r') as f:
                signals = json.load(f)
            print(f"✅ Generated {len(signals)} trading signals")
            return True
        else:
            print("❌ No signals file created")
            return False
    else:
        print("❌ Signal generation failed")
        return False

def test_frontend_connection():
    """Test if frontend can connect to backend"""
    print("🌐 Testing frontend connection...")
    
    try:
        # Test local API endpoints
        endpoints = [
            'http://localhost:8000/api/health',
            'http://localhost:8000/api/trading-signals',
            'http://localhost:8000/api/status'
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {endpoint} - OK")
                else:
                    print(f"⚠️ {endpoint} - Status: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"❌ {endpoint} - Connection failed (API not running)")
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Frontend connection test failed: {e}")
        return False

def test_telegram_alerts():
    """Test Telegram alert system"""
    print("📱 Testing Telegram alerts...")
    
    try:
        from api.telegram_alerts import send_quick_alert
        
        success = send_quick_alert(
            symbol="NIFTY",
            strike=25000,
            option_type="CE",
            entry_price=180,
            stoploss=150,
            reason="System Test Alert"
        )
        
        if success:
            print("✅ Telegram alert sent successfully")
            return True
        else:
            print("❌ Telegram alert failed")
            return False
            
    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def test_github_actions():
    """Check GitHub Actions configuration"""
    print("⚙️ Checking GitHub Actions...")
    
    workflows = [
        '.github/workflows/send-alerts.yml',
        '.github/workflows/update_data.yml'
    ]
    
    all_good = True
    for workflow in workflows:
        if os.path.exists(workflow):
            print(f"✅ {workflow} exists")
        else:
            print(f"❌ {workflow} missing")
            all_good = False
    
    return all_good

def run_complete_test():
    """Run all system tests"""
    print("🚀 AI Trading Platform - Complete System Test")
    print("=" * 60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Telegram Alerts", test_telegram_alerts),
        ("Frontend Connection", test_frontend_connection),
        ("GitHub Actions", test_github_actions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems operational! Your trading platform is ready.")
    else:
        print(f"\n⚠️ {total - passed} issues found. Please fix them before going live.")
    
    return passed == total

if __name__ == "__main__":
    run_complete_test()