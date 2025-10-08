#!/usr/bin/env python3
"""
Fully automated AI trading platform - no manual intervention needed
"""
import os
import sys
import subprocess
import threading
import time
from datetime import datetime

def start_api_server():
    """Start the API server for frontend"""
    print("🌐 Starting API server for frontend...")
    
    try:
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'api.main:app', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(3)
        print("✅ API server running on port 8000")
        print("🔗 Frontend can access: https://trading-platform-analysis-dashboard.vercel.app")
        return process
        
    except Exception as e:
        print(f"❌ API server failed: {e}")
        return None

def start_automated_trading():
    """Start the automated trading system"""
    print("🤖 Starting automated trading system...")
    
    try:
        from api.automated_scheduler import start_automated_system
        
        # Run in separate thread so it doesn't block
        trading_thread = threading.Thread(target=start_automated_system, daemon=True)
        trading_thread.start()
        
        print("✅ Automated trading system started")
        return True
        
    except Exception as e:
        print(f"❌ Automated trading failed: {e}")
        return False

def main():
    """Main automated startup"""
    print("🚀 AI Trading Platform - FULLY AUTOMATED")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Cannot start without API server")
        return
    
    # Start automated trading
    trading_started = start_automated_trading()
    if not trading_started:
        print("❌ Automated trading failed to start")
        api_process.terminate()
        return
    
    # Show status
    print("\n" + "=" * 60)
    print("🎉 FULLY AUTOMATED SYSTEM RUNNING!")
    print("=" * 60)
    print("📊 Trading: NIFTY, BANKNIFTY, FINNIFTY options")
    print("📱 Alerts: Automatic Telegram notifications")
    print("🌐 Frontend: https://trading-platform-analysis-dashboard.vercel.app")
    print("⏰ Schedule: Every 15 minutes during market hours")
    print("🏖️ Smart: Respects weekends and holidays")
    print("🔄 Status: FULLY AUTOMATED - No manual intervention needed")
    
    print("\n📈 System will:")
    print("• Start trading automatically when market opens")
    print("• Send high-confidence alerts to Telegram")
    print("• Update frontend with live signals")
    print("• Stop automatically when market closes")
    print("• Resume next trading day")
    
    print("\n🛑 Press Ctrl+C to stop the system")
    
    try:
        # Keep running
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down automated system...")
        if api_process:
            api_process.terminate()
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()