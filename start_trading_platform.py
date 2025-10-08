#!/usr/bin/env python3
"""
Complete startup script for the AI trading platform
"""
import os
import sys
import subprocess
import time
import threading
from datetime import datetime

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'requests', 'pandas', 
        'yfinance', 'python-dotenv', 'schedule'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
    else:
        print("✅ All dependencies installed")

def check_environment():
    """Check environment variables"""
    print("🔧 Checking environment configuration...")
    
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Please run: python setup_telegram.py")
        return False
    else:
        print("✅ Environment configured")
        return True

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend API server...")
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, '-m', 'uvicorn', 
            'api.main:app', 
            '--host', '0.0.0.0', 
            '--port', '8000',
            '--reload'
        ]
        
        process = subprocess.Popen(cmd)
        time.sleep(3)  # Give server time to start
        
        # Test if server is running
        import requests
        try:
            response = requests.get('http://localhost:8000/api/health', timeout=5)
            if response.status_code == 200:
                print("✅ Backend API server started successfully")
                return process
            else:
                print("❌ Backend API server not responding")
                return None
        except:
            print("❌ Backend API server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_trading_bot():
    """Start the trading bot in a separate thread"""
    print("🤖 Starting trading bot...")
    
    def run_bot():
        import schedule
        from api.live_options_engine import run_live_options_bot as run_live_bot
        
        # Schedule bot to run every 15 minutes during market hours
        schedule.every(15).minutes.do(run_live_bot)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    print("✅ Trading bot started (runs every 15 minutes)")

def main():
    """Main startup function"""
    print("🎯 AI Trading Platform Startup")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Check dependencies
    check_dependencies()
    
    # Step 2: Check environment
    if not check_environment():
        print("\n❌ Environment setup required. Exiting.")
        return
    
    # Step 3: Create data directory
    os.makedirs('data', exist_ok=True)
    print("✅ Data directory ready")
    
    # Step 4: Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend. Exiting.")
        return
    
    # Step 5: Start trading bot
    start_trading_bot()
    
    # Step 6: Show status
    print("\n" + "=" * 50)
    print("🎉 AI Trading Platform Started Successfully!")
    print("=" * 50)
    print("📊 Dashboard: http://localhost:3000")
    print("🔗 API: http://localhost:8000")
    print("📱 Telegram alerts: Enabled")
    print("🤖 Trading bot: Running")
    print("\nPress Ctrl+C to stop the platform")
    
    try:
        # Keep the main process running
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down AI Trading Platform...")
        if backend_process:
            backend_process.terminate()
        print("✅ Platform stopped successfully")

if __name__ == "__main__":
    main()