#!/usr/bin/env python3
import yfinance as yf
import json
import os
from datetime import datetime
from .telegram_alerts import send_quick_alert
import logging

logger = logging.getLogger(__name__)

class EnhancedMultiEngine:
    def __init__(self):
        self.instruments = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'FINNIFTY': '^CNXFIN',
            'MIDCPNIFTY': '^NSEMDCP50'  # Midcap Nifty
        }
        
    def get_market_data(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="5m")  # 5min for better signals
            if data.empty:
                return None
            
            current = float(data['Close'].iloc[-1])
            prev = float(data['Close'].iloc[-12]) if len(data) > 12 else current  # 1 hour ago
            change_pct = ((current - prev) / prev) * 100
            
            return {
                'price': current,
                'change_pct': change_pct,
                'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data else 0
            }
        except:
            return None
    
    def should_generate_signal(self, market_data):
        """Lower threshold for more signals"""
        if not market_data:
            return False, 0.5, "No data"
        
        change_pct = market_data['change_pct']
        
        # Lower threshold: 0.3% instead of 0.5%
        if abs(change_pct) > 0.3:
            confidence = min(0.9, 0.65 + abs(change_pct) * 0.15)
            direction = "BULLISH" if change_pct > 0 else "BEARISH"
            reason = f"{direction} momentum: {change_pct:+.2f}%"
            return True, confidence, reason
        
        return False, 0.5, f"Low momentum: {change_pct:+.2f}%"
    
    def calculate_premium(self, spot, strike, option_type, instrument):
        """Realistic premium calculation per instrument"""
        distance = abs(spot - strike)
        
        # Base premiums by instrument
        base_premiums = {
            'SENSEX': 200,
            'BANKNIFTY': 180,
            'NIFTY': 150,
            'FINNIFTY': 120,
            'MIDCPNIFTY': 100
        }
        
        base = base_premiums.get(instrument, 150)
        
        if option_type == "CE":
            if spot > strike:  # ITM
                premium = base + (spot - strike) * 0.7
            else:  # OTM
                premium = max(20, base - distance * 0.4)
        else:  # PE
            if spot < strike:  # ITM
                premium = base + (strike - spot) * 0.7
            else:  # OTM
                premium = max(20, base - distance * 0.4)
        
        return round(premium, 2)
    
    def get_strike_interval(self, instrument):
        """Strike intervals by instrument"""
        intervals = {
            'SENSEX': 100,
            'BANKNIFTY': 100,
            'NIFTY': 50,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 25
        }
        return intervals.get(instrument, 50)
    
    def generate_all_signals(self):
        """Generate signals for all instruments"""
        signals = []
        
        for instrument, symbol in self.instruments.items():
            try:
                market_data = self.get_market_data(symbol)
                should_signal, confidence, reason = self.should_generate_signal(market_data)
                
                if should_signal and confidence > 0.7:  # Lower threshold
                    spot = market_data['price']
                    interval = self.get_strike_interval(instrument)
                    atm_strike = round(spot / interval) * interval
                    
                    option_type = "CE" if market_data['change_pct'] > 0 else "PE"
                    premium = self.calculate_premium(spot, atm_strike, option_type, instrument)
                    stoploss = round(premium * 0.75, 2)
                    
                    signal = {
                        'symbol': instrument,
                        'strike': atm_strike,
                        'option_type': option_type,
                        'entry_price': premium,
                        'stoploss': stoploss,
                        'confidence': confidence,
                        'current_price': spot,
                        'change_pct': market_data['change_pct'],
                        'reason': f"LIVE: {reason} | {instrument} @ {spot:.2f}",
                        'timestamp': datetime.now().isoformat(),
                        'source': 'ENHANCED_LIVE'
                    }
                    
                    signals.append(signal)
                    logger.info(f"✅ Signal: {instrument} {atm_strike} {option_type} @ ₹{premium}")
                
            except Exception as e:
                logger.error(f"Error with {instrument}: {e}")
        
        return signals
    
    def send_signals(self):
        """Generate and send all signals"""
        signals = self.generate_all_signals()
        
        if not signals:
            logger.info("No signals generated")
            return False
        
        # Send alerts for high confidence signals
        alerts_sent = 0
        for signal in signals:
            if signal['confidence'] > 0.8:
                success = send_quick_alert(
                    symbol=signal['symbol'],
                    strike=signal['strike'],
                    option_type=signal['option_type'],
                    entry_price=signal['entry_price'],
                    stoploss=signal['stoploss'],
                    reason=signal['reason']
                )
                if success:
                    alerts_sent += 1
        
        # Save all signals for frontend
        os.makedirs('data', exist_ok=True)
        with open('data/signals.json', 'w') as f:
            json.dump(signals, f, indent=2)
        
        logger.info(f"Generated {len(signals)} signals, sent {alerts_sent} alerts")
        return len(signals) > 0

def run_enhanced_system():
    engine = EnhancedMultiEngine()
    return engine.send_signals()

if __name__ == "__main__":
    run_enhanced_system()