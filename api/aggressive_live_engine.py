#!/usr/bin/env python3
import yfinance as yf
import json
import os
from datetime import datetime
from .telegram_alerts import send_quick_alert
import logging

logger = logging.getLogger(__name__)

class AggressiveLiveEngine:
    def __init__(self):
        self.instruments = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'FINNIFTY': '^CNXFIN'
        }
        
    def get_live_data(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="5m")
            if data.empty:
                return None
            
            current = float(data['Close'].iloc[-1])
            prev = float(data['Close'].iloc[-6]) if len(data) > 6 else current  # 30 min ago
            change_pct = ((current - prev) / prev) * 100
            
            return {
                'price': current,
                'change_pct': change_pct,
                'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data else 0
            }
        except:
            return None
    
    def should_generate_signal(self, data):
        """VERY LOW threshold for more signals"""
        if not data:
            return False, 0.5, "No data"
        
        change_pct = data['change_pct']
        
        # VERY LOW threshold: 0.1% for testing
        if abs(change_pct) > 0.1:
            confidence = min(0.9, 0.6 + abs(change_pct) * 0.2)
            direction = "BULLISH" if change_pct > 0 else "BEARISH"
            reason = f"LIVE: {direction} move {change_pct:+.2f}%"
            return True, confidence, reason
        
        return False, 0.5, f"Very low momentum: {change_pct:+.2f}%"
    
    def calculate_premium(self, spot, strike, option_type, instrument):
        base_premiums = {
            'SENSEX': 250,
            'BANKNIFTY': 200,
            'NIFTY': 150,
            'FINNIFTY': 120
        }
        
        base = base_premiums.get(instrument, 150)
        distance = abs(spot - strike)
        
        if option_type == "CE":
            if spot > strike:
                premium = base + (spot - strike) * 0.7
            else:
                premium = max(20, base - distance * 0.3)
        else:
            if spot < strike:
                premium = base + (strike - spot) * 0.7
            else:
                premium = max(20, base - distance * 0.3)
        
        return round(premium, 2)
    
    def generate_aggressive_signals(self):
        signals = []
        
        for instrument, symbol in self.instruments.items():
            try:
                data = self.get_live_data(symbol)
                should_signal, confidence, reason = self.should_generate_signal(data)
                
                if should_signal and confidence > 0.6:  # Lower threshold
                    spot = data['price']
                    
                    if instrument in ['SENSEX', 'BANKNIFTY']:
                        atm_strike = round(spot / 100) * 100
                    else:
                        atm_strike = round(spot / 50) * 50
                    
                    option_type = "CE" if data['change_pct'] > 0 else "PE"
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
                        'change_pct': data['change_pct'],
                        'reason': reason,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'AGGRESSIVE_LIVE'
                    }
                    
                    signals.append(signal)
                    logger.info(f"✅ AGGRESSIVE Signal: {instrument} {atm_strike} {option_type} @ ₹{premium}")
                
            except Exception as e:
                logger.error(f"Error with {instrument}: {e}")
        
        return signals
    
    def send_aggressive_alerts(self):
        signals = self.generate_aggressive_signals()
        
        if not signals:
            logger.info("No aggressive signals generated")
            return False
        
        # Send ALL signals (not just high confidence)
        alerts_sent = 0
        for signal in signals:
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
        
        # Save for frontend
        os.makedirs('data', exist_ok=True)
        with open('data/signals.json', 'w') as f:
            json.dump(signals, f, indent=2)
        
        logger.info(f"Generated {len(signals)} signals, sent {alerts_sent} alerts")
        return len(signals) > 0

def run_aggressive_system():
    engine = AggressiveLiveEngine()
    return engine.send_aggressive_alerts()

if __name__ == "__main__":
    run_aggressive_system()