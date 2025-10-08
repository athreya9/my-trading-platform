#!/usr/bin/env python3
"""
Fully automated accurate trading system
"""
import schedule
import time
import logging
from datetime import datetime
from api.kite_live_engine import run_kite_live_system

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

def automated_trading_cycle():
    """Run automated trading cycle"""
    if not is_market_open():
        logger.info("Market closed - waiting")
        return
    
    logger.info("🔴 AUTOMATED CYCLE - Checking for tradeable signals...")
    
    try:
        success = run_kite_live_system()
        if success:
            logger.info("✅ High-confidence signal sent to Telegram")
        else:
            logger.info("⚪ No tradeable signal (waiting for momentum)")
    except Exception as e:
        logger.error(f"❌ Cycle failed: {e}")

def start_automated_system():
    """Start the fully automated system"""
    logger.info("🚀 AUTOMATED AI TRADING SYSTEM STARTED")
    logger.info("=" * 50)
    logger.info("📊 Real Kite live data & accurate premiums")
    logger.info("🎯 TOTP auto-login + high-confidence signals")
    logger.info("📱 Automatic Telegram alerts")
    logger.info("⏰ Every 10 minutes during market hours")
    logger.info("🔄 Fully automated - no manual intervention")
    
    # Schedule every 10 minutes for more responsive trading
    schedule.every(10).minutes.do(automated_trading_cycle)
    
    # Run immediately if market is open
    if is_market_open():
        logger.info("🔥 Market is open - running initial check")
        automated_trading_cycle()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_automated_system()