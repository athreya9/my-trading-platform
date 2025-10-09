#!/usr/bin/env python3
"""
Fully automated accurate trading system
"""
import schedule
import time
import logging
import subprocess
import sys
import os
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

def start_subscription_bot():
    """Start subscription bot in background"""
    try:
        subprocess.Popen([sys.executable, "api/simple_bot.py"], 
                        cwd=os.getcwd())
        logger.info("✅ Subscription bot started")
    except Exception as e:
        logger.error(f"❌ Failed to start subscription bot: {e}")

def run_pre_market_check():
    """Run pre-market system check"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, "pre_market_check.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            logger.info("✅ Pre-market check passed")
        else:
            logger.warning(f"⚠️ Pre-market check issues: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Pre-market check failed: {e}")

def start_automated_system():
    """Start the fully automated system"""
    logger.info("🚀 AUTOMATED AI TRADING SYSTEM STARTED")
    logger.info("=" * 50)
    logger.info("📊 Real Kite live data & accurate premiums")
    logger.info("🎯 TOTP auto-login + high-confidence signals")
    logger.info("📱 Automatic Telegram alerts")
    logger.info("💳 Subscription bot active")
    logger.info("⏰ Every 10 minutes during market hours")
    logger.info("🔄 Fully automated - no manual intervention")
    
    # Run pre-market check
    run_pre_market_check()
    
    # Start subscription bot
    start_subscription_bot()
    
    # Schedule every 10 minutes for more responsive trading
    schedule.every(10).minutes.do(automated_trading_cycle)
    
    # Run immediately if market is open
    if is_market_open():
        logger.info("🔥 Market is open - running initial check")
        automated_trading_cycle()
    else:
        logger.info("🕰 Market closed - system ready for 9:15 AM start")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_automated_system()