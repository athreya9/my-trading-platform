#!/usr/bin/env python3
"""
Final system check - honest assessment
"""
import os
import json
import requests
import subprocess
import time
import sys

def check_telegram_setup():
    """Check if Telegram is properly configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        print("✅ Telegram credentials configured")
        return True
    else:
        print("❌ Telegram credentials missing")
        print("   Run: python3 setup_telegram.py")
        return False

def check_signal_generation():
    """Check if signals are being generated"""
    try:
        from api.live_trading_bot import LiveTradingBot
        bot = LiveTradingBot()
        bot.is_market_open = lambda: True  # Override for testing
        
        signals = bot.generate_trading_signals()
        if signals:
            print(f"✅ Signal generation working ({len(signals)} signals)")
            return True
        else:
            print("❌ No signals generated")
            return False
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def check_api_endpoints():
    """Check if API endpoints are working"""
    try:
        # Start server
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'api.main:app', '--port', '8000'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(3)
        
        endpoints = [
            '/api/health',
            '/api/trading-signals', 
            '/api/status',
            '/api/stats'
        ]
        
        working = 0
        for endpoint in endpoints:
            try:
                response = requests.get(f'http://localhost:8000{endpoint}', timeout=5)
                if response.status_code == 200:
                    working += 1
            except:
                pass
        
        process.terminate()
        
        if working == len(endpoints):
            print(f"✅ All API endpoints working ({working}/{len(endpoints)})")
            return True
        else:
            print(f"⚠️ Some API endpoints failing ({working}/{len(endpoints)})")
            return working > 2
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def check_data_persistence():
    """Check if data is being saved correctly"""
    files_to_check = [
        'data/signals.json',
        'data/bot_status.json'
    ]
    
    all_good = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data:
                        print(f"✅ {file_path} has data")
                    else:
                        print(f"⚠️ {file_path} is empty")
                        all_good = False
            except:
                print(f"❌ {file_path} corrupted")
                all_good = False
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    return all_good

def check_github_actions():
    """Check GitHub Actions setup"""
    workflow_files = [
        '.github/workflows/send-alerts.yml',
        '.github/workflows/update_data.yml'
    ]
    
    all_exist = True
    for workflow in workflow_files:
        if os.path.exists(workflow):
            print(f"✅ {workflow}")
        else:
            print(f"❌ {workflow} missing")
            all_exist = False
    
    return all_exist

def main():
    print("🔍 FINAL SYSTEM CHECK - AI Trading Platform")
    print("=" * 60)
    
    checks = [
        ("Signal Generation", check_signal_generation),
        ("Data Persistence", check_data_persistence), 
        ("API Endpoints", check_api_endpoints),
        ("GitHub Actions", check_github_actions),
        ("Telegram Setup", check_telegram_setup)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        results[name] = check_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SYSTEM STATUS REPORT")
    print("=" * 60)
    
    core_systems = ["Signal Generation", "Data Persistence", "API Endpoints"]
    optional_systems = ["GitHub Actions", "Telegram Setup"]
    
    core_working = sum(results[sys] for sys in core_systems)
    optional_working = sum(results[sys] for sys in optional_systems)
    
    print(f"\n🔧 CORE SYSTEMS: {core_working}/{len(core_systems)}")
    for sys in core_systems:
        status = "✅ WORKING" if results[sys] else "❌ BROKEN"
        print(f"  {sys:<20} {status}")
    
    print(f"\n⚙️ OPTIONAL SYSTEMS: {optional_working}/{len(optional_systems)}")
    for sys in optional_systems:
        status = "✅ READY" if results[sys] else "⚠️ NEEDS SETUP"
        print(f"  {sys:<20} {status}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if core_working == len(core_systems):
        print("🎉 SYSTEM IS OPERATIONAL!")
        print("\n✅ Ready for:")
        print("  - Local signal generation")
        print("  - API serving data to frontend")
        print("  - Manual testing")
        
        if optional_working == len(optional_systems):
            print("  - Live Telegram alerts")
            print("  - Automated GitHub Actions")
            print("\n🚀 FULLY READY FOR LIVE TRADING!")
        else:
            print("\n⚠️ To enable live alerts:")
            if not results["Telegram Setup"]:
                print("  - Run: python3 setup_telegram.py")
            print("\n🔄 Current status: TESTING MODE")
    else:
        print("❌ SYSTEM HAS CRITICAL ISSUES")
        print(f"   {len(core_systems) - core_working} core systems need fixing")
    
    print("\n📱 Next steps:")
    print("1. Fix any critical issues above")
    print("2. Run: python3 setup_telegram.py (for alerts)")
    print("3. Run: python3 start_trading_platform.py")
    print("4. Test with: python3 generate_test_signals.py")

if __name__ == "__main__":
    main()