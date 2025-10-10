#!/usr/bin/env python3
"""
Auto Token Monitor - Runs every hour to check token expiry
Add to cron: 0 * * * * cd "/Users/datta/Desktop/My trading platform" && python3 auto_token_monitor.py
"""

import sys
import os
sys.path.append('./api')

from telegram_bot_commands import TradingTelegramBot
import datetime

def monitor_token():
    """Monitor token and send alerts if needed"""
    bot = TradingTelegramBot()
    status = bot.check_token_status()
    
    if not status["valid"]:
        # Token is invalid - send immediate alert
        message = """🚨 **URGENT: KITE TOKEN EXPIRED**

❌ Your Kite token has expired and trading is stopped.

🔧 **Quick Fix:**
1. Send `/generate_token` to this bot
2. Follow the instructions to renew

⚠️ **Trading will resume once token is renewed**"""
        
        bot.send_message(bot.chat_id, message, parse_mode='Markdown')
        print("❌ Token expired - Alert sent")
        return False
    
    else:
        # Token is valid - check if expiring soon
        expires_at = datetime.datetime.strptime(status["expires_at"], "%Y-%m-%d %H:%M:%S")
        now = datetime.datetime.now()
        time_left = expires_at - now
        
        # Alert if less than 2 hours left
        if time_left.total_seconds() < 7200:  # 2 hours
            message = f"""⚠️ **KITE TOKEN EXPIRING SOON**

⏰ **Time Left:** {status['time_left']}
📅 **Expires:** {status['expires_at']}

🔧 **Renew Now:**
Send `/generate_token` to this bot

💡 **Tip:** Renew before 7:30 AM to avoid interruption"""
            
            bot.send_message(bot.chat_id, message, parse_mode='Markdown')
            print(f"⚠️ Token expiring soon - Alert sent (Time left: {status['time_left']})")
            return True
        
        else:
            print(f"✅ Token valid - Time left: {status['time_left']}")
            return True

if __name__ == "__main__":
    monitor_token()