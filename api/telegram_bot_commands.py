#!/usr/bin/env python3
"""
Telegram Bot with /start, /subscribe, and token management commands
"""
import os
import requests
import pyotp
import datetime
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key

load_dotenv()

class TradingTelegramBot:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.api_key = os.getenv('KITE_API_KEY')
        self.api_secret = os.getenv('KITE_API_SECRET')
        self.totp_secret = os.getenv('KITE_TOTP_SECRET')
        
    def send_message(self, chat_id, text, parse_mode=None):
        """Send message to Telegram"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, data=payload)  # Use data instead of json
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def handle_start_command(self, chat_id):
        """Handle /start command"""
        welcome_message = """👋 Welcome to DA Trading Signals!

🚀 Get high-confidence trade alerts with clear entries, targets, and stoploss.
📡 Delivered directly to your Telegram.
🔗 Join the public channel: t.me/DATradingSignals

Type /subscribe to start receiving alerts."""
        return self.send_message(chat_id, welcome_message)
    
    def handle_subscribe_command(self, chat_id):
        """Handle /subscribe command"""
        subscription_message = """
📥 *Subscription Details*

To receive premium trade alerts, subscribe for just ₹499/month.

💳 *Payment Method*: UPI  
📲 *UPI ID*: datrade@ybl  
🧾 After payment, send a screenshot here to activate your access.

✅ Once confirmed, you'll start receiving premium trading alerts directly in this chat.

Thank you for trading with DA Trading Signals!
"""
        return self.send_message(chat_id, subscription_message, parse_mode='Markdown')
    
    def check_token_status(self):
        """Check current Kite token status"""
        try:
            access_token = os.getenv('KITE_ACCESS_TOKEN')
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(access_token)
            
            profile = kite.profile()
            
            # Calculate expiry
            now = datetime.datetime.now()
            today_730am = now.replace(hour=7, minute=30, second=0, microsecond=0)
            if now > today_730am:
                expiry = today_730am + datetime.timedelta(days=1)
            else:
                expiry = today_730am
            
            time_left = expiry - now
            
            return {
                "valid": True,
                "user": profile.get("user_name", "Unknown"),
                "expires_at": expiry.strftime("%Y-%m-%d %H:%M:%S"),
                "time_left": str(time_left).split('.')[0],
                "token": access_token[:20] + "..."
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def generate_login_info(self):
        """Generate Kite login link with TOTP"""
        try:
            kite = KiteConnect(api_key=self.api_key)
            login_url = kite.login_url()
            
            # Generate current TOTP
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()
            
            return {
                "login_url": login_url,
                "totp": current_totp
            }
        except Exception as e:
            return {"error": str(e)}
    
    def process_request_token(self, request_token):
        """Process request token and generate access token"""
        try:
            kite = KiteConnect(api_key=self.api_key)
            data = kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            
            # Save to .env
            set_key('.env', 'KITE_ACCESS_TOKEN', access_token)
            
            # Save to access_token.txt
            with open('access_token.txt', 'w') as f:
                f.write(access_token)
            
            return {"success": True, "token": access_token}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def handle_token_status_command(self, chat_id):
        """Handle /token_status command"""
        status = self.check_token_status()
        
        if status["valid"]:
            message = f"""🔑 *KITE TOKEN STATUS*

✅ *Status*: Valid
👤 *User*: {status['user']}
🔑 *Token*: {status['token']}
📅 *Expires*: {status['expires_at']}
⏰ *Time Left*: {status['time_left']}"""
        else:
            message = f"""🔑 *KITE TOKEN STATUS*

❌ *Status*: Invalid
🚨 *Error*: {status.get('error', 'Unknown error')}

Use /generate\_token to create a new one."""
        
        return self.send_message(chat_id, message, parse_mode='Markdown')
    
    def handle_generate_token_command(self, chat_id):
        """Handle /generate_token command"""
        login_info = self.generate_login_info()
        
        if "error" in login_info:
            message = f"❌ *Error generating login link*: {login_info['error']}"
        else:
            message = f"""🔑 *GENERATE NEW TOKEN*

📋 *Steps*:
1\. [Login to Kite]({login_info['login_url']})
2\. Enter your credentials
3\. Use TOTP: `{login_info['totp']}`
4\. Copy request\_token from URL
5\. Reply: `/renew_token YOUR_REQUEST_TOKEN`

⏰ *TOTP expires in 30 seconds*
🔄 *Get new TOTP*: /generate\_token"""
        
        return self.send_message(chat_id, message, parse_mode='Markdown')
    
    def handle_renew_token_command(self, chat_id, request_token):
        """Handle /renew_token command"""
        if not request_token:
            message = "❌ *Usage*: `/renew_token YOUR_REQUEST_TOKEN`"
            return self.send_message(chat_id, message, parse_mode='Markdown')
        
        result = self.process_request_token(request_token)
        
        if result["success"]:
            # Verify the new token
            status = self.check_token_status()
            message = f"""✅ *TOKEN RENEWED SUCCESSFULLY*

🔑 *New Token*: {result['token'][:20]}\.\.\.
👤 *User*: {status.get('user', 'Unknown')}
📅 *Expires*: {status.get('expires_at', 'Unknown')}
💾 *Saved*: \.env and access\_token\.txt

🚀 *Trading system is ready\!*"""
        else:
            message = f"❌ *Token renewal failed*: {result['error']}"
        
        return self.send_message(chat_id, message, parse_mode='Markdown')
    
    def handle_help_command(self, chat_id):
        """Handle /help command"""
        help_message = """🤖 *TRADING BOT COMMANDS*

*General:*
/start \- Welcome message
/subscribe \- Subscription details
/help \- Show this help

*Token Management:*
/token\_status \- Check current token status
/generate\_token \- Get login link and TOTP
/renew\_token TOKEN \- Renew with request token

*Example Token Renewal:*
1\. `/generate_token`
2\. Login with provided link and TOTP
3\. `/renew_token 3RsQdHdYtyxs9Vx563HV4g4IfeRt9gYG`"""
        return self.send_message(chat_id, help_message, parse_mode='Markdown')
    
    def handle_message(self, message_text, chat_id):
        """Handle incoming messages and route to appropriate commands"""
        # Only respond to authorized chat
        if str(chat_id) != self.chat_id:
            return False
        
        text = message_text.strip()
        
        if text == '/start':
            return self.handle_start_command(chat_id)
        elif text == '/subscribe':
            return self.handle_subscribe_command(chat_id)
        elif text == '/token_status':
            return self.handle_token_status_command(chat_id)
        elif text == '/generate_token':
            return self.handle_generate_token_command(chat_id)
        elif text.startswith('/renew_token '):
            request_token = text.replace('/renew_token ', '').strip()
            return self.handle_renew_token_command(chat_id, request_token)
        elif text == '/help':
            return self.handle_help_command(chat_id)
        else:
            return False
    
    def setup_webhook_handler(self):
        """Setup webhook to handle commands"""
        # This would be used with Flask/FastAPI webhook
        pass

# Test the bot commands
if __name__ == "__main__":
    import sys
    
    bot = TradingTelegramBot()
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            bot.handle_token_status_command(chat_id)
        elif command == "generate":
            bot.handle_generate_token_command(chat_id)
        elif command == "renew" and len(sys.argv) > 2:
            bot.handle_renew_token_command(chat_id, sys.argv[2])
        else:
            print("Usage: python3 telegram_bot_commands.py [status|generate|renew TOKEN]")
    else:
        print("Testing basic commands...")
        bot.handle_start_command(chat_id)
        bot.handle_help_command(chat_id)