#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from .telegram_alerts import send_quick_alert

load_dotenv()
logger = logging.getLogger(__name__)

class KiteLiveEngine:
    def __init__(self):
        self.api_key = os.getenv('KITE_API_KEY')
        self.access_token = os.getenv('KITE_ACCESS_TOKEN')
        self.kite = None
        self._connect_kite()
        
        self.instruments = {
            'NIFTY': 'NSE:NIFTY 50',
            'BANKNIFTY': 'NSE:NIFTY BANK', 
            'SENSEX': 'BSE:SENSEX',
            'FINNIFTY': 'NSE:NIFTY FIN SERVICE'
        }
    
    def _connect_kite(self):
        """Connect to Kite API with auto-token generation"""
        try:
            from kiteconnect import KiteConnect
            
            if not self.api_key:
                logger.error("KITE_API_KEY not found in environment")
                return False
            
            self.kite = KiteConnect(api_key=self.api_key)
            
            # Try existing token first
            if self.access_token:
                try:
                    self.kite.set_access_token(self.access_token)
                    # Test connection
                    self.kite.profile()
                    logger.info("✅ Connected to Kite API with existing token")
                    return True
                except:
                    logger.info("Existing token expired, generating new one...")
            
            # Generate fresh token using TOTP
            from .kite_totp_login import get_kite_token_with_totp
            fresh_token = get_kite_token_with_totp()
            
            if fresh_token:
                self.kite.set_access_token(fresh_token)
                self.access_token = fresh_token
                logger.info("✅ Connected to Kite API with fresh token")
                return True
            else:
                logger.error("Failed to generate fresh token")
                return False
                
        except ImportError:
            logger.error("Missing modules: pip install kiteconnect pyotp selenium")
            return False
        except Exception as e:
            logger.error(f"Kite connection failed: {e}")
            return False
    
    def get_live_price(self, instrument):
        """Get live price from Kite"""
        if not self.kite:
            logger.error("Kite not connected")
            return None
        
        try:
            quote = self.kite.quote([instrument])
            if instrument in quote:
                data = quote[instrument]
                return {
                    'price': data['last_price'],
                    'change_pct': data['net_change'] / data['ohlc']['close'] * 100 if data['ohlc']['close'] else 0,
                    'volume': data.get('volume', 0),
                    'ohlc': data['ohlc']
                }
        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
        
        return None
    
    def get_option_chain(self, underlying, expiry_date):
        """Get option chain from Kite"""
        if not self.kite:
            return None
        
        try:
            # This would need proper instrument tokens
            # For now, return None to use fallback calculation
            return None
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    def calculate_realistic_premium(self, spot, strike, option_type, instrument_name):
        """Calculate realistic premium using market data"""
        try:
            # Try real market data first
            from .accurate_option_pricing import get_real_option_premium
            real_premium = get_real_option_premium(instrument_name, strike, option_type)
            
            if real_premium and real_premium > 0:
                logger.info(f"✅ Real market premium: ₹{real_premium}")
                return real_premium
            
            # Fallback: Market-calibrated pricing
            from .market_calibrated_pricing import get_market_calibrated_premium
            calibrated_premium = get_market_calibrated_premium(spot, strike, option_type, instrument_name)
            logger.info(f"📊 Market-calibrated premium: ₹{calibrated_premium}")
            return calibrated_premium
            
        except Exception as e:
            logger.error(f"Premium calculation error: {e}")
            # Final fallback - realistic approximation
            distance = abs(spot - strike)
            base = {'NIFTY': 150, 'BANKNIFTY': 200, 'FINNIFTY': 180, 'SENSEX': 160}.get(instrument_name, 150)
            
            if option_type == "PE" and spot < strike:  # ITM PE
                return round((strike - spot) + base * 0.4, 2)
            elif option_type == "CE" and spot > strike:  # ITM CE
                return round((spot - strike) + base * 0.4, 2)
            else:  # OTM
                return round(max(15, base - distance * 0.8), 2)
    
    def analyze_momentum(self, price_data):
        """Analyze if there's tradeable momentum"""
        if not price_data:
            return False, 0.5, "No data"
        
        change_pct = price_data['change_pct']
        
        # Lower threshold for demo: 0.2%
        if abs(change_pct) > 0.2:
            confidence = min(0.9, 0.65 + abs(change_pct) * 0.15)
            direction = "BULLISH" if change_pct > 0 else "BEARISH"
            reason = f"KITE LIVE: {direction} momentum {change_pct:+.2f}%"
            return True, confidence, reason
        
        return False, 0.5, f"Low momentum: {change_pct:+.2f}%"
    
    def generate_kite_signals(self):
        """Generate signals using Kite live data"""
        signals = []
        
        for instrument_name, kite_symbol in self.instruments.items():
            try:
                # Get live price from Kite
                price_data = self.get_live_price(kite_symbol)
                if not price_data:
                    logger.info(f"No price data for {instrument_name}")
                    continue
                
                # Check momentum
                has_momentum, confidence, reason = self.analyze_momentum(price_data)
                
                if not has_momentum or confidence < 0.7:
                    logger.info(f"{instrument_name}: {reason}")
                    continue
                
                spot = price_data['price']
                
                # Calculate strike based on instrument
                if instrument_name in ['SENSEX', 'BANKNIFTY']:
                    atm_strike = round(spot / 100) * 100
                else:
                    atm_strike = round(spot / 50) * 50
                
                option_type = "CE" if price_data['change_pct'] > 0 else "PE"
                
                # Calculate realistic premium
                premium = self.calculate_realistic_premium(spot, atm_strike, option_type, instrument_name)
                stoploss = round(premium * 0.75, 2)
                
                signal = {
                    'symbol': instrument_name,
                    'strike': atm_strike,
                    'option_type': option_type,
                    'entry_price': premium,
                    'stoploss': stoploss,
                    'confidence': confidence,
                    'current_price': spot,
                    'change_pct': price_data['change_pct'],
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'KITE_LIVE'
                }
                
                signals.append(signal)
                logger.info(f"✅ KITE Signal: {instrument_name} {atm_strike} {option_type} @ ₹{premium}")
                
            except Exception as e:
                logger.error(f"Error generating signal for {instrument_name}: {e}")
        
        return signals
    
    def send_kite_alerts(self):
        """Generate and send Kite-based alerts"""
        signals = self.generate_kite_signals()
        
        if not signals:
            logger.info("No Kite signals generated")
            return False
        
        # Send alerts for high confidence signals
        alerts_sent = 0
        for signal in signals:
            if signal['confidence'] > 0.7:  # Lowered from 0.8 to 0.7
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
        
        # Save all signals for frontend (accumulate throughout the day)
        os.makedirs('data', exist_ok=True)
        
        # Load existing signals from today
        existing_signals = []
        try:
            with open('data/signals.json', 'r') as f:
                existing_signals = json.load(f)
                
            # Filter to keep only today's signals
            today = datetime.now().date()
            existing_signals = [s for s in existing_signals 
                             if datetime.fromisoformat(s['timestamp']).date() == today]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_signals = []
        
        # Add new signals to existing ones (avoid duplicates)
        for signal in signals:
            # Check if similar signal already exists (same symbol, strike, type within 1 hour)
            is_duplicate = False
            for existing in existing_signals:
                if (existing['symbol'] == signal['symbol'] and 
                    existing['strike'] == signal['strike'] and
                    existing['option_type'] == signal['option_type']):
                    # Check if within 1 hour
                    existing_time = datetime.fromisoformat(existing['timestamp'])
                    signal_time = datetime.fromisoformat(signal['timestamp'])
                    if abs((signal_time - existing_time).total_seconds()) < 3600:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                existing_signals.append(signal)
        
        # Sort by timestamp (newest first)
        existing_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Save accumulated signals
        with open('data/signals.json', 'w') as f:
            json.dump(existing_signals, f, indent=2)
        
        # Also update api-data for backend
        with open('api-data/trading-signals.json', 'w') as f:
            json.dump(existing_signals, f, indent=2)
        
        logger.info(f"Generated {len(signals)} Kite signals, sent {alerts_sent} alerts")
        return len(signals) > 0

def run_kite_live_system():
    """Main function for Kite live trading"""
    engine = KiteLiveEngine()
    return engine.send_kite_alerts()

if __name__ == "__main__":
    run_kite_live_system()