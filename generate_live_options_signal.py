#!/usr/bin/env python3
"""
Generate a live options signal for testing
"""
import json
import os
from datetime import datetime
from api.telegram_alerts import send_quick_alert

def create_live_options_signal():
    """Create a realistic options signal"""
    
    # Simulate a high-confidence signal
    signal = {
        'symbol': 'NIFTY',
        'strike': 25000,
        'option_type': 'CE',
        'entry_price': 180,
        'stoploss': 150,
        'confidence': 0.87,
        'reason': 'AI Breakout Signal - Volume surge with RSI confirmation',
        'nifty_price': 24980,
        'trend': 'BULLISH',
        'timestamp': datetime.now().isoformat()
    }
    
    # Send Telegram alert
    alert_sent = send_quick_alert(
        symbol=signal['symbol'],
        strike=signal['strike'],
        option_type=signal['option_type'],
        entry_price=signal['entry_price'],
        stoploss=signal['stoploss'],
        reason=f"{signal['reason']} (Confidence: {signal['confidence']:.0%})"
    )
    
    # Save for frontend
    os.makedirs('data', exist_ok=True)
    with open('data/signals.json', 'w') as f:
        json.dump([signal], f, indent=2)
    
    if alert_sent:
        print("✅ Live options signal sent to Telegram!")
        print(f"📊 Signal: {signal['symbol']} {signal['strike']} {signal['option_type']}")
        print(f"💰 Entry: {signal['entry_price']}")
        print(f"🛑 Stoploss: {signal['stoploss']}")
        print(f"🎯 Confidence: {signal['confidence']:.0%}")
    else:
        print("❌ Failed to send alert")
    
    return signal

if __name__ == "__main__":
    create_live_options_signal()