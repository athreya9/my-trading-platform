#!/usr/bin/env python3
"""
Automated Kite Token Renewal System
Checks token validity and renews if needed
"""

import os
import datetime
import requests
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class KiteTokenManager:
    def __init__(self):
        self.api_key = os.getenv('KITE_API_KEY')
        self.access_token = os.getenv('KITE_ACCESS_TOKEN')
        self.kite = KiteConnect(api_key=self.api_key)
        
    def check_token_validity(self):
        """Check if current token is valid"""
        try:
            self.kite.set_access_token(self.access_token)
            profile = self.kite.profile()
            logger.info("✅ Token is valid")
            return True
        except Exception as e:
            logger.error(f"❌ Token invalid: {e}")
            return False
    
    def get_token_expiry_info(self):
        """Get token expiry information"""
        try:
            # Kite tokens expire daily at 7:30 AM IST
            now = datetime.datetime.now()
            today_730am = now.replace(hour=7, minute=30, second=0, microsecond=0)
            
            if now > today_730am:
                # Token expires tomorrow at 7:30 AM
                expiry = today_730am + datetime.timedelta(days=1)
            else:
                # Token expires today at 7:30 AM
                expiry = today_730am
            
            time_left = expiry - now
            
            return {
                "expires_at": expiry.strftime("%Y-%m-%d %H:%M:%S"),
                "time_left": str(time_left).split('.')[0],  # Remove microseconds
                "expires_soon": time_left.total_seconds() < 3600  # Less than 1 hour
            }
        except Exception as e:
            logger.error(f"Error calculating expiry: {e}")
            return None
    
    def send_expiry_alert(self, expiry_info):
        """Send Telegram alert about token expiry"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                return False
            
            message = f"""
🔑 **KITE TOKEN ALERT**

⚠️ Token expires in: {expiry_info['time_left']}
📅 Expiry time: {expiry_info['expires_at']}

🔗 Generate new token:
`cd "/Users/datta/Desktop/My trading platform" && python3 generate_kite_token.py`

Or use this direct link:
https://kite.trade/connect/login?api_key=is2u8bo7z8yjwhhr&v=3
            """
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def monitor_token(self):
        """Monitor token and send alerts if needed"""
        print("🔍 Checking Kite token status...")
        
        # Check validity
        is_valid = self.check_token_validity()
        
        # Get expiry info
        expiry_info = self.get_token_expiry_info()
        
        if expiry_info:
            print(f"📅 Token expires: {expiry_info['expires_at']}")
            print(f"⏰ Time left: {expiry_info['time_left']}")
            
            # Send alert if expires soon or invalid
            if not is_valid or expiry_info['expires_soon']:
                print("⚠️ Sending expiry alert...")
                alert_sent = self.send_expiry_alert(expiry_info)
                if alert_sent:
                    print("✅ Alert sent to Telegram")
                else:
                    print("❌ Failed to send alert")
        
        return {
            "valid": is_valid,
            "expiry_info": expiry_info,
            "needs_renewal": not is_valid or (expiry_info and expiry_info['expires_soon'])
        }

def main():
    """Main monitoring function"""
    manager = KiteTokenManager()
    status = manager.monitor_token()
    
    if status['needs_renewal']:
        print("\n🚨 ACTION REQUIRED:")
        print("Run: python3 generate_kite_token.py")
        print("Or use direct link: https://kite.trade/connect/login?api_key=is2u8bo7z8yjwhhr&v=3")
    else:
        print("\n✅ Token is valid and not expiring soon")
    
    return status

if __name__ == "__main__":
    main()