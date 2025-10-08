#!/usr/bin/env python3
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from .telegram_alerts import send_quick_alert
import logging

logger = logging.getLogger(__name__)

class AccurateLiveEngine:
    def __init__(self):
        self.instruments = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN'
        }
        
    def get_real_market_data(self, symbol):
        """Get real NIFTY data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2d", interval="1m")
            if data.empty:
                return None
            
            current = float(data['Close'].iloc[-1])
            prev_close = float(data['Close'].iloc[-20]) if len(data) > 20 else current
            change_pct = ((current - prev_close) / prev_close) * 100
            
            return {
                'price': current,
                'change_pct': change_pct,
                'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data else 0
            }
        except:
            return None
    
    def calculate_realistic_premium(self, spot, strike, option_type, days_to_expiry=7):
        """Calculate realistic option premium using Black-Scholes approximation"""
        try:
            import math
            
            # Market parameters
            risk_free_rate = 0.065  # 6.5%
            volatility = 0.18  # 18% for NIFTY
            
            # Time to expiry
            T = days_to_expiry / 365.0
            
            # Moneyness
            S = spot
            K = strike
            
            if option_type == "CE":
                # For Call options
                if S > K:  # ITM
                    intrinsic = S - K
                    time_value = max(10, 50 * math.sqrt(T) * volatility)
                    premium = intrinsic + time_value
                else:  # OTM
                    d1 = (math.log(S/K) + (risk_free_rate + 0.5*volatility**2)*T) / (volatility*math.sqrt(T))
                    d2 = d1 - volatility*math.sqrt(T)
                    
                    # Simplified Black-Scholes for OTM calls
                    premium = S * self.norm_cdf(d1) - K * math.exp(-risk_free_rate*T) * self.norm_cdf(d2)
                    premium = max(5, premium)
            else:  # PE
                if S < K:  # ITM
                    intrinsic = K - S
                    time_value = max(10, 50 * math.sqrt(T) * volatility)
                    premium = intrinsic + time_value
                else:  # OTM
                    premium = max(5, (K - S) * 0.1 + 20 * math.sqrt(T))
            
            return round(premium, 2)
            
        except:
            # Fallback calculation
            distance = abs(spot - strike)
            if option_type == "CE":
                if spot > strike:
                    return round(spot - strike + 30, 2)
                else:
                    return round(max(10, 80 - distance * 0.3), 2)
            else:
                if spot < strike:
                    return round(strike - spot + 30, 2)
                else:
                    return round(max(10, 80 - distance * 0.3), 2)
    
    def norm_cdf(self, x):
        """Cumulative distribution function for standard normal"""
        import math
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def analyze_market_momentum(self, nifty_data):
        """Analyze if there's tradeable momentum"""
        if not nifty_data:
            return False, 0.5, "No data"
        
        price = nifty_data['price']
        change_pct = nifty_data['change_pct']
        
        # Strong momentum criteria
        if abs(change_pct) > 0.5:  # More than 0.5% move
            confidence = min(0.9, 0.7 + abs(change_pct) * 0.1)
            direction = "BULLISH" if change_pct > 0 else "BEARISH"
            reason = f"Strong {direction.lower()} momentum: {change_pct:+.2f}%"
            return True, confidence, reason
        
        return False, 0.5, f"Low momentum: {change_pct:+.2f}%"
    
    def generate_accurate_signal(self):
        """Generate signal with accurate market data for all instruments"""
        signals = []
        
        for instrument, symbol in self.instruments.items():
            try:
                # Get real market data
                market_data = self.get_real_market_data(symbol)
                if not market_data:
                    continue
                
                # Check for tradeable momentum
                has_momentum, confidence, reason = self.analyze_market_momentum(market_data)
                
                if not has_momentum or confidence < 0.75:
                    logger.info(f"{instrument}: {reason}")
                    continue
                
                spot = market_data['price']
                
                # Different strike calculations for different instruments
                if instrument == 'SENSEX':
                    atm_strike = round(spot / 100) * 100  # SENSEX uses 100 point strikes
                elif instrument == 'BANKNIFTY':
                    atm_strike = round(spot / 100) * 100
                else:  # NIFTY
                    atm_strike = round(spot / 50) * 50
                
                # Determine option type based on momentum
                option_type = "CE" if market_data['change_pct'] > 0 else "PE"
                
                # Calculate REAL premium
                real_premium = self.calculate_realistic_premium(spot, atm_strike, option_type)
                stoploss = round(real_premium * 0.8, 2)
                
                signal = {
                    'symbol': instrument,
                    'strike': atm_strike,
                    'option_type': option_type,
                    'entry_price': real_premium,
                    'stoploss': stoploss,
                    'confidence': confidence,
                    'current_price': spot,
                    'change_pct': market_data['change_pct'],
                    'reason': f"LIVE: {reason} | {instrument} @ {spot:.2f}",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ACCURATE_LIVE'
                }
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error generating signal for {instrument}: {e}")
        
        return signals[0] if signals else None
    
    def send_accurate_alert(self):
        """Send alert with accurate data"""
        signal = self.generate_accurate_signal()
        
        if not signal:
            return False
        
        success = send_quick_alert(
            symbol=signal['symbol'],
            strike=signal['strike'],
            option_type=signal['option_type'],
            entry_price=signal['entry_price'],
            stoploss=signal['stoploss'],
            reason=signal['reason']
        )
        
        if success:
            logger.info(f"✅ ACCURATE alert sent: {signal['symbol']} {signal['strike']} {signal['option_type']} @ ₹{signal['entry_price']}")
            
            # Save for frontend
            import os
            os.makedirs('data', exist_ok=True)
            with open('data/signals.json', 'w') as f:
                json.dump([signal], f, indent=2)
        
        return success

def run_accurate_live_system():
    """Main function for accurate live trading"""
    engine = AccurateLiveEngine()
    return engine.send_accurate_alert()

if __name__ == "__main__":
    run_accurate_live_system()