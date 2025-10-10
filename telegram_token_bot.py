#!/usr/bin/env python3
"""
Telegram Bot for Remote Kite Token Management
Commands: /token_status, /generate_token, /renew_token
"""

import os
import pyotp
import requests
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import datetime
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class TelegramTokenBot:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.api_key = os.getenv('KITE_API_KEY')
        self.api_secret = os.getenv('KITE_API_SECRET')
        self.totp_secret = os.getenv('KITE_TOTP_SECRET')
        
    def send_message(self, text):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def check_token_status(self):
        """Check current token status"""
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
    
    def generate_login_link(self):
        """Generate Kite login link with TOTP"""
        try:
            kite = KiteConnect(api_key=self.api_key)
            login_url = kite.login_url()
            
            # Generate current TOTP
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()
            
            return {
                "login_url": login_url,
                "totp": current_totp,
                "api_key": self.api_key
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
    
    def handle_token_status(self):
        """Handle /token_status command"""
        status = self.check_token_status()
        
        if status["valid"]:
            message = f"""
🔑 **KITE TOKEN STATUS**

✅ **Status**: Valid
👤 **User**: {status['user']}
🔑 **Token**: {status['token']}
📅 **Expires**: {status['expires_at']}
⏰ **Time Left**: {status['time_left']}
            """
        else:
            message = f"""
🔑 **KITE TOKEN STATUS**

❌ **Status**: Invalid
🚨 **Error**: {status.get('error', 'Unknown error')}

Use /generate_token to create a new one.
            """
        
        self.send_message(message)
    
    def handle_generate_token(self):
        """Handle /generate_token command"""
        login_info = self.generate_login_link()
        
        if "error" in login_info:
            message = f"❌ **Error generating login link**: {login_info['error']}"
        else:
            message = f"""
🔑 **GENERATE NEW TOKEN**

📋 **Steps**:
1. Click: [Login to Kite]({login_info['login_url']})
2. Enter your credentials
3. Use TOTP: `{login_info['totp']}`
4. Copy request_token from URL
5. Reply with: `/renew_token YOUR_REQUEST_TOKEN`

⏰ **TOTP expires in 30 seconds**
🔄 **Get new TOTP**: /generate_token
            """
        
        self.send_message(message)
    
    def handle_renew_token(self, request_token):
        """Handle /renew_token command"""
        if not request_token:
            self.send_message("❌ **Usage**: `/renew_token YOUR_REQUEST_TOKEN`")
            return
        
        result = self.process_request_token(request_token)
        
        if result["success"]:
            # Verify the new token
            status = self.check_token_status()
            message = f"""
✅ **TOKEN RENEWED SUCCESSFULLY**

🔑 **New Token**: {result['token'][:20]}...
👤 **User**: {status.get('user', 'Unknown')}
📅 **Expires**: {status.get('expires_at', 'Unknown')}
💾 **Saved**: .env and access_token.txt

🚀 **Trading system is ready!**
            """
        else:
            message = f"❌ **Token renewal failed**: {result['error']}"
        
        self.send_message(message)

def setup_webhook_handler():
    """Setup webhook handler for Telegram commands"""
    bot = TelegramTokenBot()
    
    # This would be called by your webhook endpoint
    def handle_webhook(update):
        try:
            message = update.get('message', {})
            text = message.get('text', '')
            chat_id = message.get('chat', {}).get('id')
            
            # Only respond to authorized chat
            if str(chat_id) != bot.chat_id:
                return
            
            if text == '/token_status':
                bot.handle_token_status()
            elif text == '/generate_token':
                bot.handle_generate_token()
            elif text.startswith('/renew_token '):
                request_token = text.replace('/renew_token ', '').strip()
                bot.handle_renew_token(request_token)
            elif text == '/help':
                help_message = """
🤖 **KITE TOKEN BOT COMMANDS**

/token_status - Check current token status
/generate_token - Get login link and TOTP
/renew_token TOKEN - Renew with request token
/help - Show this help

**Example**:
1. `/generate_token`
2. Login with provided link and TOTP
3. `/renew_token 3RsQdHdYtyxs9Vx563HV4g4IfeRt9gYG`
                """
                bot.send_message(help_message)
        
        except Exception as e:
            logger.error(f"Webhook error: {e}")
    
    return handle_webhook

# Standalone command functions for direct use
def cmd_token_status():
    """Command line: Check token status"""
    bot = TelegramTokenBot()
    bot.handle_token_status()

def cmd_generate_token():
    """Command line: Generate token"""
    bot = TelegramTokenBot()
    bot.handle_generate_token()

def cmd_renew_token(request_token):
    """Command line: Renew token"""
    bot = TelegramTokenBot()
    bot.handle_renew_token(request_token)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 telegram_token_bot.py status")
        print("  python3 telegram_token_bot.py generate")
        print("  python3 telegram_token_bot.py renew REQUEST_TOKEN")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "status":
        cmd_token_status()
    elif command == "generate":
        cmd_generate_token()
    elif command == "renew" and len(sys.argv) > 2:
        cmd_renew_token(sys.argv[2])
    else:
        print("Invalid command")