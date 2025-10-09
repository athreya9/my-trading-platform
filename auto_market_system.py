#!/usr/bin/env python3
"""
AUTO MARKET SYSTEM - Starts/stops based on Indian market hours
"""
import os
import time
import subprocess
import logging
from datetime import datetime, timedelta
import pytz
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoMarketSystem:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '8250334547:AAHFIXLgvwlJlUUasiXY-5wHJ85E2AeC39k')
        self.admin_id = "1375236879"
        self.ist = pytz.timezone('Asia/Kolkata')
        self.trading_process = None
        self.system_running = False
        
    def get_ist_time(self):
        """Get current IST time"""
        return datetime.now(self.ist)
    
    def is_market_open(self):
        """Check if Indian market is open"""
        now = self.get_ist_time()
        
        # Check if weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Market hours: 9:15 AM - 3:30 PM IST
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def notify_admin(self, message):
        """Send notification to admin"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.admin_id, 
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=10)
            logger.info(f"Admin notified: {message}")
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")
    
    def start_trading_system(self):
        """Start the stable trading system"""
        if self.system_running:
            return True
            
        try:
            # Kill any existing processes
            subprocess.run(['pkill', '-f', 'stable_system.py'], capture_output=True)
            time.sleep(2)
            
            # Start new process
            self.trading_process = subprocess.Popen([
                'python3', 'stable_system.py'
            ], cwd=os.getcwd())
            
            self.system_running = True
            
            ist_time = self.get_ist_time().strftime('%H:%M:%S')
            message = f"""🚀 <b>MARKET OPENED</b>

⏰ <b>Time:</b> {ist_time} IST
📊 <b>Status:</b> Trading system started
🤖 <b>Signals:</b> Every 10 minutes
📱 <b>Channel:</b> @DATradingSignals

✅ System will auto-stop at 3:30 PM"""
            
            self.notify_admin(message)
            logger.info("✅ Trading system started for market hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            self.notify_admin(f"❌ Failed to start trading system: {str(e)}")
            return False
    
    def stop_trading_system(self):
        """Stop the trading system"""
        if not self.system_running:
            return True
            
        try:
            # Kill trading processes
            subprocess.run(['pkill', '-f', 'stable_system.py'], capture_output=True)
            
            if self.trading_process:
                self.trading_process.terminate()
                self.trading_process = None
            
            self.system_running = False
            
            ist_time = self.get_ist_time().strftime('%H:%M:%S')
            message = f"""🛑 <b>MARKET CLOSED</b>

⏰ <b>Time:</b> {ist_time} IST
📊 <b>Status:</b> Trading system stopped
🔄 <b>Next Start:</b> Tomorrow 9:15 AM

✅ System will auto-start when market opens"""
            
            self.notify_admin(message)
            logger.info("🛑 Trading system stopped - market closed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop trading system: {e}")
            return False
    
    def get_next_market_open(self):
        """Get next market open time"""
        now = self.get_ist_time()
        
        # If today is weekday and before market open
        if now.weekday() < 5 and now.hour < 9:
            return now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # Otherwise, next weekday at 9:15 AM
        days_ahead = 1
        if now.weekday() == 4:  # Friday
            days_ahead = 3  # Skip to Monday
        elif now.weekday() == 5:  # Saturday
            days_ahead = 2  # Skip to Monday
        
        next_open = now + timedelta(days=days_ahead)
        return next_open.replace(hour=9, minute=15, second=0, microsecond=0)
    
    def run_auto_system(self):
        """Main auto system loop"""
        logger.info("🤖 AUTO MARKET SYSTEM STARTED")
        
        # Send startup notification
        ist_time = self.get_ist_time().strftime('%H:%M:%S')
        next_open = self.get_next_market_open()
        
        startup_msg = f"""🤖 <b>AUTO SYSTEM STARTED</b>

⏰ <b>Current Time:</b> {ist_time} IST
📊 <b>Market Status:</b> {'OPEN' if self.is_market_open() else 'CLOSED'}
🔄 <b>Next Market Open:</b> {next_open.strftime('%d %b %Y, %H:%M')}

✅ System will auto-start/stop with market hours
🚀 No manual intervention needed"""
        
        self.notify_admin(startup_msg)
        
        while True:
            try:
                current_market_status = self.is_market_open()
                
                if current_market_status and not self.system_running:
                    # Market opened, start system
                    logger.info("📈 Market opened - starting trading system")
                    self.start_trading_system()
                    
                elif not current_market_status and self.system_running:
                    # Market closed, stop system
                    logger.info("📉 Market closed - stopping trading system")
                    self.stop_trading_system()
                
                # Check every 60 seconds
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Auto system interrupted")
                if self.system_running:
                    self.stop_trading_system()
                break
            except Exception as e:
                logger.error(f"Auto system error: {e}")
                time.sleep(60)
        
        logger.info("Auto system stopped")

def main():
    """Entry point"""
    auto_system = AutoMarketSystem()
    auto_system.run_auto_system()

if __name__ == "__main__":
    main()