#!/usr/bin/env python3
import os
import json
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .telegram_alerts import send_quick_alert
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LiveOptionsEngine:
    def __init__(self):
        self.nifty_symbol = "^NSEI"
        
    def get_nifty_price(self):
        """Get current NIFTY price"""
        try:
            nifty = yf.Ticker(self.nifty_symbol)
            data = nifty.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 25000  # Fallback
        except:
            return 25000
    
    def get_atm_strike(self, price):
        """Get ATM strike price (nearest 50)"""
        return round(price / 50) * 50
    
    def calculate_option_premium(self, strike, option_type, nifty_price):
        """Estimate option premium based on distance from ATM"""
        distance = abs(strike - nifty_price)
        
        if option_type == "CE":
            if strike <= nifty_price:  # ITM
                premium = 150 + (nifty_price - strike) * 0.8
            else:  # OTM
                premium = max(50, 200 - distance * 0.5)
        else:  # PE
            if strike >= nifty_price:  # ITM
                premium = 150 + (strike - nifty_price) * 0.8
            else:  # OTM
                premium = max(50, 200 - distance * 0.5)
        
        return round(premium, 1)
    
    def analyze_market_trend(self):
        """Analyze NIFTY trend for options direction"""
        try:
            nifty = yf.Ticker(self.nifty_symbol)
            data = nifty.history(period="5d", interval="15m")
            
            if len(data) < 20:
                return "NEUTRAL", 0.5
            
            # Simple trend analysis
            recent_close = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            # Price momentum
            price_change = (recent_close - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            
            confidence = 0.5
            trend = "NEUTRAL"
            
            if recent_close > sma_20 and price_change > 0.002:
                trend = "BULLISH"
                confidence = min(0.85, 0.6 + abs(price_change) * 10)
            elif recent_close < sma_20 and price_change < -0.002:
                trend = "BEARISH"
                confidence = min(0.85, 0.6 + abs(price_change) * 10)
            
            # Volume confirmation
            if current_volume > avg_volume * 1.2:
                confidence += 0.1
            
            return trend, min(confidence, 0.9)
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return "NEUTRAL", 0.5
    
    def generate_options_signal(self):
        """Generate live options trading signal"""
        try:
            # Get current market data
            nifty_price = self.get_nifty_price()
            trend, confidence = self.analyze_market_trend()
            
            if trend == "NEUTRAL" or confidence < 0.7:
                return None
            
            # Determine option type and strike
            atm_strike = self.get_atm_strike(nifty_price)
            
            if trend == "BULLISH":
                option_type = "CE"
                strike = atm_strike  # ATM for better premium
                reason = f"Bullish breakout detected. NIFTY trending up from {nifty_price:.0f}"
            else:
                option_type = "PE"
                strike = atm_strike
                reason = f"Bearish breakdown detected. NIFTY trending down from {nifty_price:.0f}"
            
            # Calculate premium and targets
            entry_price = self.calculate_option_premium(strike, option_type, nifty_price)
            stoploss = round(entry_price * 0.8, 1)  # 20% stop loss
            
            signal = {
                'symbol': 'NIFTY',
                'strike': strike,
                'option_type': option_type,
                'entry_price': entry_price,
                'stoploss': stoploss,
                'confidence': confidence,
                'reason': reason,
                'nifty_price': nifty_price,
                'trend': trend,
                'timestamp': datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Options signal generation error: {e}")
            return None
    
    def send_options_alert(self, signal):
        """Send formatted options alert"""
        if not signal:
            return False
        
        try:
            success = send_quick_alert(
                symbol=signal['symbol'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                entry_price=signal['entry_price'],
                stoploss=signal['stoploss'],
                reason=f"{signal['reason']} (Confidence: {signal['confidence']:.0%})"
            )
            
            if success:
                logger.info(f"✅ Options alert sent: {signal['symbol']} {signal['strike']} {signal['option_type']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Alert sending error: {e}")
            return False

def run_live_options_bot():
    """Main function for live options trading"""
    engine = LiveOptionsEngine()
    
    # Generate signal
    signal = engine.generate_options_signal()
    
    if signal:
        # Send alert
        alert_sent = engine.send_options_alert(signal)
        
        # Save signal for frontend
        signals_data = [signal]
        os.makedirs('data', exist_ok=True)
        with open('data/signals.json', 'w') as f:
            json.dump(signals_data, f, indent=2)
        
        logger.info(f"Generated options signal: {signal['symbol']} {signal['strike']} {signal['option_type']}")
        return True
    else:
        logger.info("No high-confidence options signal generated")
        return False

if __name__ == "__main__":
    run_live_options_bot()