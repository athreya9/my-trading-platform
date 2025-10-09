#!/usr/bin/env python3
"""
REAL MARKET AUTOMATED SYSTEM - Uses live data, runs without fail
"""
import os
import time
import logging
import requests
import yfinance as yf
from datetime import datetime
import pytz
import json
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMarketSystem:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '8250334547:AAHFIXLgvwlJlUUasiXY-5wHJ85E2AeC39k')
        self.admin_id = "1375236879"
        self.channel = "@DATradingSignals"
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Real market symbols
        self.symbols = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'FINNIFTY': 'NIFTYFIN.NS',
            'NIFTYIT': 'NIFTYIT.NS'
        }
        
        self.running = True
        self.last_signal_time = None
        
    def get_ist_time(self):
        """Get current IST time"""
        return datetime.now(self.ist)
    
    def is_market_open(self):
        """Check if market is open"""
        now = self.get_ist_time()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def get_real_market_data(self, symbol):
        """Get real market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbols[symbol])
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
            
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[0]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            volume = data['Volume'].iloc[-1]
            
            return {
                'symbol': symbol,
                'price': round(current_price, 2),
                'change_pct': round(change_pct, 2),
                'volume': int(volume),
                'high': round(data['High'].max(), 2),
                'low': round(data['Low'].min(), 2)
            }
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def calculate_option_details(self, spot_price, symbol):
        """Calculate realistic option details"""
        try:
            # ATM strike calculation
            if symbol in ['SENSEX', 'BANKNIFTY']:
                strike_interval = 100
                atm_strike = round(spot_price / strike_interval) * strike_interval
            elif symbol == 'NIFTYIT':
                strike_interval = 25
                atm_strike = round(spot_price / strike_interval) * strike_interval
            else:
                strike_interval = 50
                atm_strike = round(spot_price / strike_interval) * strike_interval
            
            # Option premium calculation (simplified but realistic)
            distance_from_atm = abs(spot_price - atm_strike)
            time_value = 30  # Days to expiry assumption
            
            # Base premium calculation
            if distance_from_atm <= strike_interval:
                base_premium = spot_price * 0.02  # 2% for ATM
            else:
                base_premium = max(10, spot_price * 0.01 - distance_from_atm * 0.1)
            
            # Add volatility component
            volatility_premium = base_premium * 0.3
            total_premium = round(base_premium + volatility_premium, 1)
            
            return {
                'strike': atm_strike,
                'premium': max(10, total_premium),
                'strike_interval': strike_interval
            }
        except Exception as e:
            logger.error(f"Option calculation error: {e}")
            return None
    
    def generate_real_signal(self):
        """Generate signal based on real market data"""
        try:
            # Get data for all symbols
            market_data = {}
            for symbol in self.symbols.keys():
                data = self.get_real_market_data(symbol)
                if data:
                    market_data[symbol] = data
            
            if not market_data:
                logger.error("No market data available")
                return None
            
            # Find symbol with highest momentum
            best_symbol = None
            best_momentum = 0
            
            for symbol, data in market_data.items():
                momentum = abs(data['change_pct'])
                if momentum > best_momentum and momentum > 0.5:  # At least 0.5% movement
                    best_momentum = momentum
                    best_symbol = symbol
            
            if not best_symbol:
                logger.info("No significant momentum found")
                return None
            
            data = market_data[best_symbol]
            option_details = self.calculate_option_details(data['price'], best_symbol)
            
            if not option_details:
                return None
            
            # Determine option type based on market direction
            option_type = "CE" if data['change_pct'] > 0 else "PE"
            
            # Calculate confidence based on momentum and volume
            confidence = min(95, 70 + (best_momentum * 5))
            
            return {
                'symbol': best_symbol,
                'spot_price': data['price'],
                'change_pct': data['change_pct'],
                'strike': option_details['strike'],
                'option_type': option_type,
                'entry_price': option_details['premium'],
                'confidence': round(confidence),
                'volume': data['volume'],
                'high': data['high'],
                'low': data['low'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
    
    def format_real_signal(self, signal):
        """Format signal with real market data"""
        try:
            targets = [
                round(signal['entry_price'] * 1.05, 1),
                round(signal['entry_price'] * 1.10, 1),
                round(signal['entry_price'] * 1.15, 1),
                round(signal['entry_price'] * 1.20, 1),
                round(signal['entry_price'] * 1.25, 1)
            ]
            stoploss = round(signal['entry_price'] * 0.85, 1)
            
            direction = "📈 BULLISH" if signal['change_pct'] > 0 else "📉 BEARISH"
            
            message = f"""🚀 <b>LIVE TRADE ALERT</b>

📊 <b>{signal['symbol']} {signal['strike']} {signal['option_type']}</b>

💰 <b>BUY NOW</b>
🎯 <b>Entry:</b> ₹{signal['entry_price']}

🎆 <b>TARGETS:</b>
T1: ₹{targets[0]} (5%)
T2: ₹{targets[1]} (10%)
T3: ₹{targets[2]} (15%)
T4: ₹{targets[3]} (20%)
T5: ₹{targets[4]} (25%)

🛑 <b>Stoploss:</b> ₹{stoploss}

📈 <b>Spot:</b> ₹{signal['spot_price']} ({signal['change_pct']:+.2f}%)
🤖 <b>AI Confidence:</b> {signal['confidence']}%
{direction}

⏰ <b>Time:</b> {self.get_ist_time().strftime('%H:%M:%S')} IST

📲 <b>Join:</b> @DATradingSignals
⚠️ <i>For educational purposes only</i>"""
            
            return message
        except Exception as e:
            logger.error(f"Message formatting error: {e}")
            return None
    
    def send_notification(self, message, target="channel"):
        """Send notification with retry"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            if target == "channel":
                chat_id = self.channel
            else:
                chat_id = self.admin_id
            
            response = requests.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=10)
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Notification error: {e}")
            return False
    
    def save_signal(self, signal):
        """Save signal to file"""
        try:
            os.makedirs('data', exist_ok=True)
            
            signals_file = 'data/live_signals.json'
            try:
                with open(signals_file, 'r') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = []
            
            signals.append(signal)
            signals = signals[-50:]  # Keep last 50
            
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2)
                
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def run_forever(self):
        """Main system loop - runs forever"""
        logger.info("🚀 REAL MARKET SYSTEM STARTED")
        
        # Send startup notification
        startup_msg = f"""🚀 <b>REAL MARKET SYSTEM STARTED</b>

⏰ <b>Time:</b> {self.get_ist_time().strftime('%H:%M:%S')} IST
📊 <b>Data Source:</b> Live Yahoo Finance
🎯 <b>Signals:</b> Based on real momentum
🤖 <b>Frequency:</b> Every 10 minutes during market hours

✅ System will run 24/7 automatically"""
        
        self.send_notification(startup_msg, target="admin")
        
        signal_count = 0
        
        while self.running:
            try:
                if self.is_market_open():
                    # Check if 10 minutes passed since last signal
                    now = self.get_ist_time()
                    if (self.last_signal_time is None or 
                        (now - self.last_signal_time).total_seconds() >= 600):
                        
                        # Generate real signal
                        signal = self.generate_real_signal()
                        if signal and signal['confidence'] >= 75:
                            message = self.format_real_signal(signal)
                            if message:
                                # Send to channel
                                if self.send_notification(message, target="channel"):
                                    self.save_signal(signal)
                                    signal_count += 1
                                    self.last_signal_time = now
                                    logger.info(f"✅ Real signal sent: {signal['symbol']} {signal['confidence']}%")
                                    
                                    # Send admin update every 5 signals
                                    if signal_count % 5 == 0:
                                        admin_msg = f"📊 System Update: {signal_count} signals sent today"
                                        self.send_notification(admin_msg, target="admin")
                        else:
                            logger.info("No high-confidence signal generated")
                else:
                    logger.info("Market closed - waiting")
                
                # Sleep for 60 seconds
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("System interrupted")
                break
            except Exception as e:
                logger.error(f"System error: {e}")
                # Send error notification
                self.send_notification(f"⚠️ System error: {str(e)}", target="admin")
                time.sleep(60)  # Wait before retrying
        
        logger.info("System stopped")

def main():
    """Entry point"""
    system = RealMarketSystem()
    system.run_forever()

if __name__ == "__main__":
    main()