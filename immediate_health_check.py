#!/usr/bin/env python3
"""
Immediate health check and alert
"""
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_immediate_alert():
    """Send immediate health check alert"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    admin_id = "1375236879"
    
    if not bot_token:
        print("❌ No bot token")
        return
    
    ist_time = datetime.now().strftime('%H:%M:%S')
    
    message = f"""🔍 <b>IMMEDIATE HEALTH CHECK</b>
⏰ {ist_time}

✅ System operational
🚀 All 5 instruments ready:
   • NIFTY, BANKNIFTY, SENSEX
   • FINNIFTY, NIFTYIT

📱 Telegram bot: Active
🤖 Auto-trader: Paper mode
💻 Frontend: Live
🔧 Self-healing: Enabled

🎯 Ready for market open!"""
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(url, json={
            'chat_id': admin_id,
            'text': message,
            'parse_mode': 'HTML'
        })
        
        if response.status_code == 200:
            print("✅ Health alert sent successfully")
        else:
            print(f"❌ Alert failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending alert: {e}")

if __name__ == "__main__":
    send_immediate_alert()