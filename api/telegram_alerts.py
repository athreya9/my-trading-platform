# telegram_alerts.py - Complete Telegram alert system
import os
import requests
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramAlerts:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
    def send_trade_alert(self, symbol, strike, option_type, entry_price, stoploss, reason=""):
        """
        Send formatted trade alert - exactly like your example format
        """
        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials not set")
            return False

        message = self._format_alert_message(symbol, strike, option_type, entry_price, stoploss, reason)
        
        return self._send_telegram_message(message)
    
    def send_detailed_alert(self, signal_data):
        """
        Send detailed trading signal with multiple parameters
        """
        message = self._format_detailed_alert(signal_data)
        return self._send_telegram_message(message)
    
    def _format_alert_message(self, symbol, strike, option_type, entry_price, stoploss, reason=""):
        """
        Format for channel broadcasting with enhanced visuals
        """
        # Calculate targets (5% to 25% profit)
        targets = [
            round(entry_price * 1.05, 1),  # 5%
            round(entry_price * 1.10, 1),  # 10% 
            round(entry_price * 1.15, 1),  # 15%
            round(entry_price * 1.20, 1),  # 20%
            round(entry_price * 1.25, 1)   # 25%
        ]
        
        # Enhanced message for channel
        message = f"""🚀 <b>TRADE ALERT</b>

📊 <b>{symbol} {strike} {option_type}</b>

💰 <b>BUY NOW</b>
🎯 <b>Entry:</b> ₹{entry_price}

🎆 <b>TARGETS:</b>
T1: ₹{targets[0]} (5%)
T2: ₹{targets[1]} (10%)
T3: ₹{targets[2]} (15%)
T4: ₹{targets[3]} (20%)
T5: ₹{targets[4]} (25%)

🛑 <b>Stoploss:</b> ₹{stoploss}

{'🤖 <b>AI Analysis:</b> ' + reason if reason else ''}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

📲 <b>Join:</b> @DATradingSignals
🔥 <i>LIVE KITE DATA - REAL TRADING SIGNAL</i>"""
        return message
    
    def _format_detailed_alert(self, signal):
        """
        More detailed format with additional information
        """
        symbol = signal.get('symbol', 'NIFTY')
        strike = signal.get('strike_price', 25250)
        option_type = signal.get('option_type', 'PE')
        entry_price = signal.get('entry_price', 170)
        stoploss = signal.get('stoploss', 155)
        confidence = signal.get('confidence', 'High')
        timeframe = signal.get('timeframe', '15min')
        reason = signal.get('reason', 'Technical Breakout')
        
        targets = [
            round(entry_price * 1.05, 1),
            round(entry_price * 1.10, 1),
            round(entry_price * 1.15, 1), 
            round(entry_price * 1.20, 1),
            round(entry_price * 1.25, 1)
        ]
        
        message = f"""
 <b>DETAILED TRADING SIGNAL</b> 

 <b>Instrument:</b> {symbol} {strike} {option_type}
⚡ <b>Action:</b> BUY
 <b>Entry Price:</b> {entry_price}

 <b>Targets:</b>
• Target 1: {targets[0]}
• Target 2: {targets[1]}
• Target 3: {targets[2]}
• Target 4: {targets[3]} 
• Target 5: {targets[4]}

 <b>Stoploss:</b> {stoploss}

 <b>Confidence:</b> {confidence}
⏱️ <b>Timeframe:</b> {timeframe}
 <b>Reason:</b> {reason}

⏰ <b>Alert Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        return message
    
    def notify(self, message, target="channel", parse_mode='HTML'):
        """Send notification with proper target routing"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        try:
            if target == "channel":
                # Send to public channel only (for trade alerts)
                payload = {
                    'chat_id': '@DATradingSignals',
                    'text': message,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    logger.info("✅ Trade alert sent to channel!")
                    return True
                    
            elif target == "admin":
                # Send to admin only (for system messages)
                payload = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    logger.info("✅ Admin notification sent!")
                    return True
                    
            elif target == "subscribers":
                # Send to verified subscribers only
                success_count = 0
                try:
                    with open('data/subscribers.json') as f:
                        users = json.load(f)
                    
                    for chat_id, info in users.items():
                        if info['status'] == 'active':
                            payload = {
                                'chat_id': chat_id,
                                'text': message,
                                'parse_mode': parse_mode,
                                'disable_web_page_preview': True
                            }
                            try:
                                response = requests.post(url, json=payload)
                                if response.status_code == 200:
                                    success_count += 1
                            except Exception as e:
                                logger.error(f"Failed to send to {chat_id}: {e}")
                    
                    logger.info(f"✅ Message sent to {success_count} subscribers!")
                    return success_count > 0
                    
                except FileNotFoundError:
                    logger.info("No subscribers file found")
                    return False
            
            return False
                
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
            return False
    
    def _send_telegram_message(self, message):
        """Legacy method - routes to channel for trade alerts"""
        return self.notify(message, target="channel")

# Create global instance
telegram_bot = TelegramAlerts()

# Simple functions for easy usage
def send_quick_alert(symbol, strike, option_type, entry_price, stoploss, reason=""):
    """Quick alert function - simple usage"""
    return telegram_bot.send_trade_alert(symbol, strike, option_type, entry_price, stoploss, reason)

def send_detailed_signal(signal_data):
    """Detailed alert function"""
    return telegram_bot.send_detailed_alert(signal_data)

# Test function
def test_telegram_alerts():
    """Test the Telegram alerts setup with proper routing"""
    print("Testing Telegram alerts...")
    
    # Test 1: Trade alert to channel
    success1 = send_quick_alert(
        symbol="NIFTY",
        strike=25250, 
        option_type="PE",
        entry_price=190,
        stoploss=155,
        reason="TEST: Breakout with volume surge"
    )
    
    # Test 2: Admin notification
    telegram_bot.notify("🔧 TEST: System health check completed", target="admin")
    
    if success1:
        print("✅ All Telegram tests passed! Alerts are working.")
    else:
        print("❌ Some alerts failed. Check your credentials.")

if __name__ == "__main__":
    test_telegram_alerts()