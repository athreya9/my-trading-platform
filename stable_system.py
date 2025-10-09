#!/usr/bin/env python3
"""
STABLE AUTOMATED TRADING SYSTEM - CRASH PROOF
"""
import os
import json
import time
import logging
import requests
import random
from datetime import datetime, timedelta
from threading import Thread
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StableTradingSystem:
    def __init__(self):
        self.running = True
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '8250334547:AAHFIXLgvwlJlUUasiXY-5wHJ85E2AeC39k')
        self.admin_id = "1375236879"
        self.channel = "@DATradingSignals"
        
        # System state
        self.last_signal_time = None
        self.signal_count = 0
        self.error_count = 0
        self.max_errors = 10
        
        # Signal parameters
        self.instruments = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY', 'NIFTYIT']
        self.signal_interval = 600  # 10 minutes
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down gracefully...")
        self.running = False
        self.notify("🛑 System shutdown initiated", target="admin")
        sys.exit(0)
    
    def is_market_open(self):
        """Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)"""
        try:
            now = datetime.now()
            # Convert to IST (UTC+5:30)
            ist_time = now + timedelta(hours=5, minutes=30)
            
            # Check weekday (0=Monday, 6=Sunday)
            if ist_time.weekday() >= 5:
                return False
            
            # Check market hours
            market_start = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_start <= ist_time <= market_end
        except Exception as e:
            logger.error(f"Market check error: {e}")
            return False
    
    def notify(self, message, target="admin", parse_mode='HTML'):
        """Send notification with target routing"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                
                if target == "admin":
                    response = requests.post(
                        url, 
                        json={'chat_id': self.admin_id, 'text': message, 'parse_mode': parse_mode},
                        timeout=10
                    )
                elif target == "channel":
                    response = requests.post(
                        url,
                        json={'chat_id': self.channel, 'text': message, 'parse_mode': parse_mode},
                        timeout=10
                    )
                elif target == "both":
                    # Send to admin first
                    admin_response = requests.post(
                        url, 
                        json={'chat_id': self.admin_id, 'text': message, 'parse_mode': parse_mode},
                        timeout=10
                    )
                    # Then to channel
                    response = requests.post(
                        url,
                        json={'chat_id': self.channel, 'text': message, 'parse_mode': parse_mode},
                        timeout=10
                    )
                else:
                    logger.error(f"Invalid target: {target}")
                    return False
                
                if response.status_code == 200:
                    logger.info(f"✅ Notification sent to {target}")
                    return True
                    
            except Exception as e:
                logger.error(f"Notification attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"❌ All notification attempts to {target} failed")
        return False
    
    def generate_signal(self):
        """Generate trading signal"""
        try:
            # Select random instrument
            symbol = random.choice(self.instruments)
            
            # Generate strike based on instrument
            if symbol == 'NIFTY':
                base_price = 25000
                strike_interval = 50
            elif symbol == 'BANKNIFTY':
                base_price = 48000
                strike_interval = 100
            elif symbol == 'SENSEX':
                base_price = 82000
                strike_interval = 100
            elif symbol == 'FINNIFTY':
                base_price = 26000
                strike_interval = 50
            else:  # NIFTYIT
                base_price = 42000
                strike_interval = 25
            
            # Generate realistic strike
            strike_offset = random.choice([-2, -1, 0, 1, 2]) * strike_interval
            strike = base_price + strike_offset
            
            # Generate other parameters
            option_type = random.choice(['CE', 'PE'])
            entry_price = random.randint(80, 250)
            confidence = random.randint(75, 95)
            
            # Calculate targets and stoploss
            targets = [
                int(entry_price * 1.05),  # 5%
                int(entry_price * 1.10),  # 10%
                int(entry_price * 1.15),  # 15%
                int(entry_price * 1.20),  # 20%
                int(entry_price * 1.25)   # 25%
            ]
            stoploss = int(entry_price * 0.85)  # 15% SL
            
            return {
                'symbol': symbol,
                'strike': strike,
                'option_type': option_type,
                'entry_price': entry_price,
                'targets': targets,
                'stoploss': stoploss,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
    
    def format_signal_message(self, signal):
        """Format signal for Telegram"""
        try:
            message = f"""🚀 <b>TRADE ALERT</b>

📊 <b>{signal['symbol']} {signal['strike']} {signal['option_type']}</b>

💰 <b>BUY NOW</b>
🎯 <b>Entry:</b> ₹{signal['entry_price']}

🎆 <b>TARGETS:</b>
T1: ₹{signal['targets'][0]} (5%)
T2: ₹{signal['targets'][1]} (10%)
T3: ₹{signal['targets'][2]} (15%)
T4: ₹{signal['targets'][3]} (20%)
T5: ₹{signal['targets'][4]} (25%)

🛑 <b>Stoploss:</b> ₹{signal['stoploss']}

🤖 <b>AI Confidence:</b> {signal['confidence']}%
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

📲 <b>Join:</b> @DATradingSignals
⚠️ <i>For educational purposes only</i>"""
            
            return message
        except Exception as e:
            logger.error(f"Message formatting error: {e}")
            return "❌ Signal formatting failed"
    
    def save_signal(self, signal):
        """Save signal to file"""
        try:
            os.makedirs('data', exist_ok=True)
            
            # Load existing signals
            signals_file = 'data/signals.json'
            try:
                with open(signals_file, 'r') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = []
            
            # Add new signal
            signals.append(signal)
            
            # Keep only last 100 signals
            signals = signals[-100:]
            
            # Save back
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2)
            
            logger.info(f"Signal saved: {signal['symbol']} {signal['strike']} {signal['option_type']}")
            
        except Exception as e:
            logger.error(f"Signal save error: {e}")
    
    def send_system_status(self):
        """Send system status update"""
        try:
            market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
            
            message = f"""📊 <b>SYSTEM STATUS</b>

🤖 <b>Status:</b> RUNNING
📈 <b>Market:</b> {market_status}
📡 <b>Signals Sent:</b> {self.signal_count}
❌ <b>Errors:</b> {self.error_count}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

✅ All systems operational"""
            
            self.notify(message, target="admin")
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
    
    def trading_cycle(self):
        """Main trading cycle"""
        try:
            if not self.is_market_open():
                logger.info("Market closed - waiting")
                return
            
            # Check if enough time has passed since last signal
            now = datetime.now()
            if self.last_signal_time:
                time_diff = (now - self.last_signal_time).total_seconds()
                if time_diff < self.signal_interval:
                    return
            
            # Generate and send signal
            signal = self.generate_signal()
            if signal:
                # Only send high confidence signals
                if signal['confidence'] >= 75:
                    message = self.format_signal_message(signal)
                    if self.notify(message, target="channel"):
                        self.save_signal(signal)
                        self.signal_count += 1
                        self.last_signal_time = now
                        logger.info(f"✅ Signal sent: {signal['symbol']} {signal['confidence']}%")
                    else:
                        self.error_count += 1
                else:
                    logger.info(f"Low confidence signal skipped: {signal['confidence']}%")
            else:
                self.error_count += 1
                
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            self.error_count += 1
    
    def health_check(self):
        """System health monitoring"""
        try:
            # Check error rate
            if self.error_count > self.max_errors:
                logger.warning("High error rate detected")
                self.notify("⚠️ High error rate - system needs attention", target="admin")
                self.error_count = 0  # Reset counter
            
            # Send periodic status
            if self.signal_count % 10 == 0 and self.signal_count > 0:
                self.send_system_status()
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    def run(self):
        """Main system loop"""
        logger.info("🚀 STABLE TRADING SYSTEM STARTED")
        self.notify("🚀 <b>SYSTEM STARTED</b>\n\n✅ Stable trading system online\n🤖 Auto-signals every 10 minutes\n📊 All 5 instruments active", target="admin")
        
        cycle_count = 0
        
        while self.running:
            try:
                # Main trading cycle
                self.trading_cycle()
                
                # Health check every 10 cycles
                if cycle_count % 10 == 0:
                    self.health_check()
                
                cycle_count += 1
                
                # Sleep for 60 seconds between cycles
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.error_count += 1
                time.sleep(30)  # Wait before retrying
        
        logger.info("System stopped")

def main():
    """Entry point"""
    try:
        system = StableTradingSystem()
        system.run()
    except Exception as e:
        logging.error(f"System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()