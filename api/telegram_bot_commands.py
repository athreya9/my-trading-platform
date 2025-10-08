#!/usr/bin/env python3
"""
Telegram Bot with /start and /subscribe commands
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TradingTelegramBot:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
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
    
    def setup_webhook_handler(self):
        """Setup webhook to handle commands"""
        # This would be used with Flask/FastAPI webhook
        pass

# Test the bot commands
if __name__ == "__main__":
    bot = TradingTelegramBot()
    
    # Test with your chat ID
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print("Testing /start command...")
    bot.handle_start_command(chat_id)
    
    print("Testing /subscribe command...")
    bot.handle_subscribe_command(chat_id)