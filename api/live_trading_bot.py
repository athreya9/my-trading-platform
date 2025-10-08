#!/usr/bin/env python3
import json
import os
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .telegram_alerts import send_quick_alert
from .ai_analysis_engine import AIAnalysisEngine
from .data_collector import DataCollector
from .config import WATCHLIST_SYMBOLS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingBot:
    def __init__(self):
        self.ai_engine = AIAnalysisEngine()
        self.data_collector = DataCollector()
        self.signals_file = 'data/signals.json'
        self.bot_status_file = 'data/bot_status.json'
        
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def generate_trading_signals(self):
        """Generate AI-powered trading signals"""
        signals = {}
        
        for symbol in WATCHLIST_SYMBOLS:
            try:
                # Get historical data
                historical_data = self.data_collector.fetch_historical_data(
                    symbol, period="1mo", interval="1d"
                )
                
                if historical_data is None or historical_data.empty:
                    continue
                
                # Get AI signal
                signal = self.ai_engine.get_simple_trend_signal(historical_data)
                confidence = self.ai_engine.calculate_confidence(historical_data)
                
                # Get current price
                current_price = historical_data['Close'].iloc[-1]
                
                signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'current_price': float(current_price),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'AI_ENGINE'
                }
                
                logger.info(f"Generated signal for {symbol}: {signal} (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                
        return signals
    
    def should_send_alert(self, symbol, signal_data):
        """Determine if alert should be sent based on signal strength"""
        confidence = signal_data.get('confidence', 0)
        signal = signal_data.get('signal', 'HOLD')
        
        # Send alerts for high-confidence BUY/SELL signals
        return confidence > 0.7 and signal in ['BUY', 'SELL']
    
    def send_telegram_alert(self, symbol, signal_data):
        """Send Telegram alert for trading signal"""
        try:
            signal = signal_data['signal']
            price = signal_data['current_price']
            confidence = signal_data['confidence']
            
            # Calculate option parameters (simplified)
            if 'NIFTY' in symbol:
                strike = round(price / 50) * 50  # Round to nearest 50
                option_type = 'CE' if signal == 'BUY' else 'PE'
                entry_price = 150  # Estimated option premium
                stoploss = 120
            else:
                strike = round(price / 10) * 10
                option_type = 'CE' if signal == 'BUY' else 'PE'
                entry_price = 100
                stoploss = 80
            
            reason = f"AI Confidence: {confidence:.1%} - Technical Analysis"
            
            success = send_quick_alert(
                symbol=symbol.replace('.NS', ''),
                strike=strike,
                option_type=option_type,
                entry_price=entry_price,
                stoploss=stoploss,
                reason=reason
            )
            
            if success:
                logger.info(f"✅ Alert sent for {symbol}: {signal}")
            else:
                logger.error(f"❌ Failed to send alert for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert for {symbol}: {e}")
            return False
    
    def save_signals(self, signals):
        """Save signals to JSON file"""
        try:
            os.makedirs('data', exist_ok=True)
            
            # Convert to list format for frontend
            signals_list = []
            for symbol, data in signals.items():
                signals_list.append({
                    'symbol': symbol.replace('.NS', ''),
                    'signal': data['signal'],
                    'confidence': data['confidence'],
                    'price': data['current_price'],
                    'timestamp': data['timestamp'],
                    'source': data['source']
                })
            
            with open(self.signals_file, 'w') as f:
                json.dump(signals_list, f, indent=2)
                
            logger.info(f"Saved {len(signals_list)} signals to {self.signals_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
            return False
    
    def update_bot_status(self, status, reason=""):
        """Update bot status"""
        try:
            status_data = {
                'status': status,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Read existing status
            existing_status = []
            if os.path.exists(self.bot_status_file):
                with open(self.bot_status_file, 'r') as f:
                    existing_status = json.load(f)
            
            # Add new status
            existing_status.append(status_data)
            
            # Keep only last 50 entries
            existing_status = existing_status[-50:]
            
            with open(self.bot_status_file, 'w') as f:
                json.dump(existing_status, f, indent=2)
                
            logger.info(f"Bot status updated: {status}")
            
        except Exception as e:
            logger.error(f"Error updating bot status: {e}")
    
    def run_single_cycle(self):
        """Run one cycle of the trading bot"""
        try:
            if not self.is_market_open():
                logger.info("Market is closed. Skipping signal generation.")
                return False
            
            logger.info("Starting trading signal generation cycle...")
            
            # Generate signals
            signals = self.generate_trading_signals()
            
            if not signals:
                logger.warning("No signals generated")
                return False
            
            # Save signals
            self.save_signals(signals)
            
            # Send alerts for high-confidence signals
            alerts_sent = 0
            for symbol, signal_data in signals.items():
                if self.should_send_alert(symbol, signal_data):
                    if self.send_telegram_alert(symbol, signal_data):
                        alerts_sent += 1
            
            logger.info(f"Cycle completed. Generated {len(signals)} signals, sent {alerts_sent} alerts")
            self.update_bot_status("running", f"Generated {len(signals)} signals, sent {alerts_sent} alerts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trading bot cycle: {e}")
            self.update_bot_status("error", f"Error: {str(e)}")
            return False

def run_live_bot():
    """Main function to run the live trading bot"""
    bot = LiveTradingBot()
    return bot.run_single_cycle()

if __name__ == "__main__":
    run_live_bot()