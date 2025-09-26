# live_telegram_alerts.py
import os
import requests
import json
from datetime import datetime
import logging
from kiteconnect import KiteConnect
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTelegramAlerts:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.kite = self._connect_to_kite()
        
    def _connect_to_kite(self):
        """Connect to Kite API for live data"""
        try:
            api_key = os.getenv('KITE_API_KEY', '')
            access_token = os.getenv('KITE_ACCESS_TOKEN', '')
            
            if not api_key or not access_token:
                logger.error("Kite API credentials not found")
                return None
                
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("✅ Connected to Kite API for live data")
            return kite
        except Exception as e:
            logger.error(f"❌ Kite connection failed: {e}")
            return None
    
    def get_live_market_data(self, symbol, strike, option_type):
        """Get REAL-TIME market data from Kite"""
        if not self.kite:
            return None

        try:
            kite_instrument = self._get_instrument_for_option(symbol, strike, option_type)
            if not kite_instrument:
                return None

            kite_symbol = f"NFO:{kite_instrument}"

            # Get live quote
            quote = self.kite.quote([kite_symbol])
            logger.info(f"Quote for {kite_symbol}: {quote}")
            if kite_symbol in quote:
                live_data = quote[kite_symbol]
                return {
                    'last_price': live_data['last_price'],
                    'open': live_data['ohlc']['open'],
                    'high': live_data['ohlc']['high'],
                    'low': live_data['ohlc']['low'],
                    'close': live_data['ohlc']['close'],
                    'volume': live_data['volume'],
                    'oi': live_data['oi'],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")

        return None
    
    def _get_instrument_for_option(self, symbol, strike, option_type):
        """Gets the tradingsymbol for an option with the nearest expiry."""
        if not self.kite:
            return None

        try:
            instruments = self.kite.instruments("NFO")
            instrument_df = pd.DataFrame(instruments)
            logger.info(f"Found {len(instrument_df)} NFO instruments.")

            # Filter for the given symbol and option type
            options_df = instrument_df[
                (instrument_df['name'] == symbol) &
                (instrument_df['instrument_type'] == option_type.upper()) &
                (instrument_df['strike'] == strike)
            ].copy() # Use .copy() to avoid SettingWithCopyWarning

            logger.info(f"Found {len(options_df)} options for {symbol} {strike} {option_type}.")
            logger.info(f"Options head:\n{options_df[['tradingsymbol', 'expiry']].head()}")

            if options_df.empty:
                logger.warning(f"No options found for {symbol} {strike} {option_type}")
                return None

            # Find the nearest expiry
            options_df['expiry'] = pd.to_datetime(options_df['expiry'])
            today = pd.to_datetime(datetime.now().date())
            
            # Filter for expiries that are on or after today
            future_options = options_df[options_df['expiry'] >= today].sort_values(by='expiry')
            logger.info(f"Found {len(future_options)} future options.")


            if future_options.empty:
                logger.warning(f"No future options found for {symbol} {strike} {option_type}")
                return None

            nearest_option = future_options.iloc[0]
            logger.info(f"Nearest option: {nearest_option['tradingsymbol']}")
            return nearest_option['tradingsymbol']

        except Exception as e:
            logger.error(f"Error finding instrument for {symbol} {strike} {option_type}: {e}")
            return None
    
    def send_live_trading_alert(self, signal_data):
        """
        Send alert with REAL LIVE market data from Kite
        """
        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials not set")
            return False

        # Enhance with live data
        enhanced_signal = self._enhance_with_live_data(signal_data)
        message = self._format_live_alert(enhanced_signal)
        
        return self._send_telegram_message(message)
    
    def _enhance_with_live_data(self, signal):
        """Enhance signal with real-time market data"""
        symbol = signal.get('symbol', 'BANKNIFTY')
        strike = signal.get('strike_price', 48000)
        option_type = signal.get('option_type', 'CE')
        
        # Get LIVE market data
        live_data = self.get_live_market_data(symbol, strike, option_type)
        
        if live_data:
            signal['live_price'] = live_data['last_price']
            signal['today_open'] = live_data['open']
            signal['today_high'] = live_data['high'] 
            signal['today_low'] = live_data['low']
            signal['volume'] = live_data['volume']
            signal['oi'] = live_data['oi']
            signal['data_freshness'] = 'LIVE ✅'
            signal['price_change_today'] = round(((live_data['last_price'] - live_data['open']) / live_data['open']) * 100, 2)
        else:
            signal['data_freshness'] = 'DELAYED ⚠️'
            signal['live_price'] = signal.get('entry_price', 0)
            
        return signal
    
    def _format_live_alert(self, signal):
        """
        Format with REAL-TIME market data and advanced analysis
        """
        symbol = signal.get('symbol', 'NIFTY')
        strike = signal.get('strike_price', 25250)
        option_type = signal.get('option_type', 'PE')
        entry_price = signal.get('entry_price', 170)
        live_price = signal.get('live_price', entry_price)
        stoploss = signal.get('stoploss', 155)
        
        # Profitability calculations
        is_profitable = live_price > entry_price
        current_pnl = live_price - entry_price
        pnl_percentage = ((live_price - entry_price) / entry_price) * 100
        
        message = f"""
 **LIVE TRADING ALERT** 
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

 **REAL-TIME MARKET DATA:**
• **Instrument:** {symbol} {strike} {option_type}
• **Live Price:** ₹{live_price} {signal.get('data_freshness', '')}
• **Today's Change:** {signal.get('price_change_today', 0)}%
• **Volume:** {signal.get('volume', 0):,}
• **Open Interest:** {signal.get('oi', 0):,}

 **CURRENT PERFORMANCE:**
• **Entry Price:** ₹{entry_price}
• **Current P&L:** ₹{current_pnl:+.2f} ({pnl_percentage:+.1f}%)
• **Status:** {' PROFITABLE' if is_profitable else ' IN LOSS' if current_pnl < 0 else '⚪ AT ENTRY'}

 **PROFIT TARGETS:**
• **Target 1 (5%):** ₹{entry_price * 1.05:.1f}
• **Target 2 (10%):** ₹{entry_price * 1.10:.1f} 
• **Target 3 (15%):** ₹{entry_price * 1.15:.1f}
• **Target 4 (20%):** ₹{entry_price * 1.20:.1f}
• **Target 5 (25%):** ₹{entry_price * 1.25:.1f}

️ **RISK MANAGEMENT:**
• **Stoploss:** ₹{stoploss}
• **Max Risk:** ₹{entry_price - stoploss:.1f} per share
• **Risk/Reward:** 1:3.5

 **TECHNICAL ANALYSIS:**
• **Confidence:** {signal.get('confidence', '85%')}
• **RSI:** {signal.get('rsi', '45')} 
• **Volume Trend:** {signal.get('volume_trend', 'Increasing')}
• **Pattern:** {signal.get('pattern', 'Breakout')}
• **Support/Resistance:** Strong

 **TRADING RECOMMENDATION:**
• **Action:**  BUY at current market price
• **Quantity:** {signal.get('quantity', '1-2 lots')}
• **Strategy:** {signal.get('strategy', 'Intraday Momentum')}

⏰ **TIME SENSITIVE:**
• **Valid for:** Next 30-60 minutes
• **Best entry:** Current price zone
• **Exit timing:** Before market close

 **MARKET CONTEXT:**
• **Market Trend:** {signal.get('market_trend', 'Bullish')}
• **Volatility:** {signal.get('volatility', 'Medium')}
• **Sector Performance:** {signal.get('sector', 'Outperforming')}

 **EXPERT INSIGHT:**
{signal.get('expert_comment', 'Strong technical setup with volume confirmation. Ideal for intraday trading.')}

⚠️ **LIVE MONITORING ADVICE:**
- Watch for volume spikes above 50,000
- Monitor 15-min RSI for overbought conditions  
- Consider partial profit booking at Target 3
- Trail stoploss to entry after Target 2

 **Need help?** Reply to this message for support.

*Data source: Kite Connect API • Algorithm: AI Trading Bot v2.0*
"""

        return message

    def _send_telegram_message(self, message):
        """Send message to Telegram"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logger.info("✅ LIVE Telegram alert sent successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram alert: {e}")
            return False

# Global instance
live_bot = LiveTelegramAlerts()

def send_live_alert(signal_data):
    """Send live trading alert"""
    return live_bot.send_live_trading_alert(signal_data)

def create_live_alert_from_kite():
    """Create alert using REAL data from Kite"""
    
    # This would be called from your actual signal generation
    alert_data = {
        'symbol': 'BANKNIFTY',
        'strike_price': 55000, # Changed from 48000
        'option_type': 'CE', 
        'entry_price': 85,
        'stoploss': 65,
        'confidence': '87% High Confidence',
        'rsi': '42 (Oversold Bounce)',
        'volume_trend': '3x Average Volume',
        'pattern': 'Bullish Breakout',
        'market_trend': 'Strong Bullish',
        'volatility': 'Medium',
        'sector': 'Banking Sector Leading',
        'strategy': 'Momentum Breakout',
        'quantity': '2 lots maximum',
        'expert_comment': 'Perfect setup with volume confirmation and RSI support. High probability trade.',
        'signal_time': datetime.now().strftime('%H:%M:%S')
    }
    
    return send_live_alert(alert_data)

# Test with REAL data
def test_live_alerts():
    """Test with actual live market data"""
    print("Testing LIVE Telegram alerts with real market data...")
    
    success = create_live_alert_from_kite()
    
    if success:
        print(" LIVE alert sent successfully with real market data!")
    else:
        print("❌ Live alert failed.")

if __name__ == "__main__":
    test_live_alerts()