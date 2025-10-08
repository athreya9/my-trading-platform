#!/usr/bin/env python3
"""
Complete live trading system with REAL data
"""
import schedule
import time
import logging
from datetime import datetime
from api.real_options_data import send_real_alert
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_market_open():
    """Check if market is open"""
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        return False
    
    market_start = now.replace(hour=9, minute=15)
    market_end = now.replace(hour=15, minute=30)
    
    return market_start <= now <= market_end

def run_live_cycle():
    """Run one live trading cycle"""
    if not is_market_open():
        logger.info("Market closed - skipping")
        return
    
    logger.info("🔴 LIVE TRADING CYCLE STARTED")
    
    try:
        # Generate and send REAL alert
        success = send_real_alert()
        
        if success:
            logger.info("✅ Live cycle completed - real alert sent")
        else:
            logger.info("⚠️ No tradeable signal generated")
            
        # Update frontend data
        from update_frontend_config import create_frontend_data
        create_frontend_data()
        
    except Exception as e:
        logger.error(f"❌ Live cycle failed: {e}")

def start_live_system():
    """Start the live trading system"""
    logger.info("🚀 STARTING LIVE AI TRADING SYSTEM")
    logger.info("=" * 50)
    logger.info("📊 Real market data")
    logger.info("💰 Real option premiums") 
    logger.info("📱 Live Telegram alerts")
    logger.info("⏰ Every 15 minutes during market hours")
    
    # Schedule every 15 minutes
    schedule.every(15).minutes.do(run_live_cycle)
    
    # Run immediately if market is open
    if is_market_open():
        logger.info("🔥 Market is open - running initial cycle")
        run_live_cycle()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_live_system()