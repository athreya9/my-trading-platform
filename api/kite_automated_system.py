#!/usr/bin/env python3
import schedule
import time
import logging
from datetime import datetime
from .kite_live_engine import run_kite_live_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_market_open():
    """Check if market is open"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    
    market_start = now.replace(hour=9, minute=15)
    market_end = now.replace(hour=15, minute=30)
    return market_start <= now <= market_end

def automated_kite_cycle():
    """Run automated Kite trading cycle"""
    if not is_market_open():
        logger.info("Market closed - waiting")
        return
    
    logger.info("🔴 KITE AUTOMATED CYCLE - Checking for signals...")
    
    try:
        success = run_kite_live_system()
        if success:
            logger.info("✅ Kite signals generated and alerts sent")
        else:
            logger.info("⚪ No Kite signals (waiting for momentum)")
    except Exception as e:
        logger.error(f"❌ Kite cycle failed: {e}")

def start_kite_automated_system():
    """Start the Kite automated system"""
    logger.info("🚀 KITE AUTOMATED TRADING SYSTEM STARTED")
    logger.info("=" * 50)
    logger.info("📊 Real Kite live data & accurate premiums")
    logger.info("🎯 TOTP auto-login for fresh tokens")
    logger.info("📱 Automatic Telegram alerts")
    logger.info("⏰ Every 10 minutes during market hours")
    
    # Schedule every 10 minutes
    schedule.every(10).minutes.do(automated_kite_cycle)
    
    # Run immediately if market is open
    if is_market_open():
        logger.info("🔥 Market is open - running initial Kite check")
        automated_kite_cycle()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_kite_automated_system()