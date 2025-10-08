#!/usr/bin/env python3
import schedule
import time
import logging
from datetime import datetime, timedelta
from .multi_instrument_engine import run_multi_instrument_bot
from .config import MARKET_HOLIDAYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedScheduler:
    def __init__(self):
        self.is_running = False
        
    def is_market_day(self):
        """Check if today is a trading day"""
        today = datetime.now()
        
        # Weekend check
        if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Holiday check
        today_str = today.strftime('%Y-%m-%d')
        if today_str in MARKET_HOLIDAYS:
            return False
            
        return True
    
    def is_market_hours(self):
        """Check if market is currently open"""
        if not self.is_market_day():
            return False
            
        now = datetime.now()
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        if not self.is_market_hours():
            logger.info("Market closed - skipping trading cycle")
            return
            
        logger.info("🚀 Running automated trading cycle...")
        try:
            success = run_multi_instrument_bot()
            if success:
                logger.info("✅ Trading cycle completed successfully")
            else:
                logger.info("⚠️ No high-confidence signals generated")
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
    
    def start_automated_trading(self):
        """Start the automated trading system"""
        logger.info("🤖 Starting Automated AI Trading System")
        logger.info("=" * 50)
        
        # Schedule trading during market hours (every 15 minutes)
        schedule.every(15).minutes.do(self.run_trading_cycle)
        
        # Status update every hour
        schedule.every().hour.do(self.log_status)
        
        self.is_running = True
        logger.info("✅ Automated scheduler started")
        logger.info("📊 Will trade: NIFTY, BANKNIFTY, FINNIFTY options")
        logger.info("⏰ Frequency: Every 15 minutes during market hours")
        logger.info("🏖️ Respects weekends and holidays")
        
        # Run immediately if market is open
        if self.is_market_hours():
            logger.info("🔥 Market is open - running initial cycle")
            self.run_trading_cycle()
        
        # Keep running
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def log_status(self):
        """Log current system status"""
        status = "TRADING" if self.is_market_hours() else "WAITING"
        logger.info(f"📈 System Status: {status} | Market Day: {self.is_market_day()}")
    
    def stop(self):
        """Stop the automated system"""
        self.is_running = False
        logger.info("🛑 Automated trading system stopped")

# Global scheduler instance
scheduler = AutomatedScheduler()

def start_automated_system():
    """Start the fully automated trading system"""
    scheduler.start_automated_trading()

if __name__ == "__main__":
    start_automated_system()