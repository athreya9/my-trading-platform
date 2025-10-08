#!/usr/bin/env python3
"""
Generate test signals for all instruments
"""
import json
import os
from datetime import datetime
from api.telegram_alerts import send_quick_alert

def create_multi_instrument_signals():
    """Create signals for NIFTY, BANKNIFTY, FINNIFTY"""
    
    signals = [
        {
            'symbol': 'NIFTY',
            'strike': 25000,
            'option_type': 'CE',
            'entry_price': 180,
            'stoploss': 135,
            'confidence': 0.87,
            'current_price': 24980,
            'trend': 'BULLISH',
            'reason': 'Bullish breakout detected on NIFTY',
            'timestamp': datetime.now().isoformat(),
            'source': 'AI_ENGINE'
        },
        {
            'symbol': 'BANKNIFTY',
            'strike': 52000,
            'option_type': 'PE',
            'entry_price': 220,
            'stoploss': 165,
            'confidence': 0.82,
            'current_price': 52150,
            'trend': 'BEARISH',
            'reason': 'Bearish breakdown detected on BANKNIFTY',
            'timestamp': datetime.now().isoformat(),
            'source': 'AI_ENGINE'
        },
        {
            'symbol': 'FINNIFTY',
            'strike': 20000,
            'option_type': 'CE',
            'entry_price': 160,
            'stoploss': 120,
            'confidence': 0.79,
            'current_price': 19950,
            'trend': 'BULLISH',
            'reason': 'Bullish breakout detected on FINNIFTY',
            'timestamp': datetime.now().isoformat(),
            'source': 'AI_ENGINE'
        }
    ]
    
    # Save signals
    os.makedirs('data', exist_ok=True)
    with open('data/signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    # Send alerts for high confidence signals
    for signal in signals:
        if signal['confidence'] > 0.8:
            send_quick_alert(
                symbol=signal['symbol'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                entry_price=signal['entry_price'],
                stoploss=signal['stoploss'],
                reason=f"{signal['reason']} (Confidence: {signal['confidence']:.0%})"
            )
    
    print(f"✅ Generated {len(signals)} multi-instrument signals")
    for signal in signals:
        print(f"📊 {signal['symbol']} {signal['strike']} {signal['option_type']} - {signal['confidence']:.0%}")
    
    return signals

if __name__ == "__main__":
    create_multi_instrument_signals()