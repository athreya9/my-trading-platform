#!/usr/bin/env python3
import os
import json
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from .telegram_alerts import send_quick_alert
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class MultiInstrumentEngine:
    def __init__(self):
        self.instruments = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'FINNIFTY': '^CNXFIN'
        }
        
    def get_price(self, symbol):
        """Get current price for any instrument"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        
        # Fallback prices
        fallbacks = {'^NSEI': 25000, '^NSEBANK': 52000, '^CNXFIN': 20000}
        return fallbacks.get(symbol, 25000)
    
    def get_atm_strike(self, price, instrument):
        """Get ATM strike based on instrument"""
        if 'NIFTY' in instrument and 'BANK' not in instrument:
            return round(price / 50) * 50
        elif 'BANK' in instrument:
            return round(price / 100) * 100
        else:
            return round(price / 50) * 50
    
    def analyze_trend(self, symbol):
        """Analyze trend for any instrument"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="15m")
            
            if len(data) < 20:
                return "NEUTRAL", 0.5
            
            recent_close = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            price_change = (recent_close - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            
            confidence = 0.5
            trend = "NEUTRAL"
            
            if recent_close > sma_20 and price_change > 0.003:
                trend = "BULLISH"
                confidence = min(0.85, 0.65 + abs(price_change) * 8)
            elif recent_close < sma_20 and price_change < -0.003:
                trend = "BEARISH"
                confidence = min(0.85, 0.65 + abs(price_change) * 8)
            
            return trend, confidence
            
        except Exception as e:
            logger.error(f"Trend analysis error for {symbol}: {e}")
            return "NEUTRAL", 0.5
    
    def calculate_premium(self, strike, option_type, current_price, instrument):
        """Calculate option premium based on instrument"""
        distance = abs(strike - current_price)
        
        # Base premium calculation
        if 'BANK' in instrument:
            base_premium = 200
            distance_factor = 0.6
        else:
            base_premium = 150
            distance_factor = 0.5
        
        if option_type == "CE":
            if strike <= current_price:  # ITM
                premium = base_premium + (current_price - strike) * 0.8
            else:  # OTM
                premium = max(50, base_premium - distance * distance_factor)
        else:  # PE
            if strike >= current_price:  # ITM
                premium = base_premium + (strike - current_price) * 0.8
            else:  # OTM
                premium = max(50, base_premium - distance * distance_factor)
        
        return round(premium, 1)
    
    def generate_signals(self):
        """Generate signals for all instruments"""
        signals = []
        
        for instrument_name, symbol in self.instruments.items():
            try:
                current_price = self.get_price(symbol)
                trend, confidence = self.analyze_trend(symbol)
                
                if confidence > 0.72:  # High confidence threshold
                    atm_strike = self.get_atm_strike(current_price, instrument_name)
                    
                    option_type = "CE" if trend == "BULLISH" else "PE"
                    entry_price = self.calculate_premium(atm_strike, option_type, current_price, instrument_name)
                    stoploss = round(entry_price * 0.75, 1)
                    
                    signal = {
                        'symbol': instrument_name,
                        'strike': atm_strike,
                        'option_type': option_type,
                        'entry_price': entry_price,
                        'stoploss': stoploss,
                        'confidence': confidence,
                        'current_price': current_price,
                        'trend': trend,
                        'reason': f"{trend.title()} breakout detected on {instrument_name}",
                        'timestamp': datetime.now().isoformat(),
                        'source': 'AI_ENGINE'
                    }
                    
                    signals.append(signal)
                    
                    # Send alert for very high confidence
                    if confidence > 0.8:
                        self.send_alert(signal)
                        
            except Exception as e:
                logger.error(f"Error generating signal for {instrument_name}: {e}")
        
        return signals
    
    def send_alert(self, signal):
        """Send Telegram alert"""
        try:
            send_quick_alert(
                symbol=signal['symbol'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                entry_price=signal['entry_price'],
                stoploss=signal['stoploss'],
                reason=f"{signal['reason']} (Confidence: {signal['confidence']:.0%})"
            )
            logger.info(f"✅ Alert sent: {signal['symbol']} {signal['strike']} {signal['option_type']}")
        except Exception as e:
            logger.error(f"Alert error: {e}")

def run_multi_instrument_bot():
    """Main function for multi-instrument trading"""
    engine = MultiInstrumentEngine()
    signals = engine.generate_signals()
    
    # Save all signals
    os.makedirs('data', exist_ok=True)
    with open('data/signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    logger.info(f"Generated {len(signals)} signals across all instruments")
    return len(signals) > 0

if __name__ == "__main__":
    run_multi_instrument_bot()