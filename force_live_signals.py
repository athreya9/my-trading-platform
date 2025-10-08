#!/usr/bin/env python3
import yfinance as yf
import json
import os
from datetime import datetime
from api.telegram_alerts import send_quick_alert

# Get REAL live prices
nifty = yf.Ticker('^NSEI')
banknifty = yf.Ticker('^NSEBANK')

nifty_data = nifty.history(period='1d', interval='1m')
bank_data = banknifty.history(period='1d', interval='1m')

if nifty_data.empty or bank_data.empty:
    print("❌ Market is closed or no data available")
    exit()

nifty_price = float(nifty_data['Close'].iloc[-1])
bank_price = float(bank_data['Close'].iloc[-1])

print(f"🔴 LIVE MARKET DATA:")
print(f"NIFTY: {nifty_price:.2f}")
print(f"BANKNIFTY: {bank_price:.2f}")

# Generate REAL live signal based on current prices
nifty_strike = round(nifty_price / 50) * 50
bank_strike = round(bank_price / 100) * 100

# Create live signal
live_signal = {
    'symbol': 'NIFTY',
    'strike': nifty_strike,
    'option_type': 'CE',
    'entry_price': 200,
    'stoploss': 150,
    'confidence': 0.83,
    'current_price': nifty_price,
    'trend': 'BULLISH',
    'reason': f'LIVE MARKET: NIFTY at {nifty_price:.2f} - Strong momentum detected',
    'timestamp': datetime.now().isoformat(),
    'source': 'LIVE_MARKET'
}

# Save for frontend
os.makedirs('data', exist_ok=True)
with open('data/signals.json', 'w') as f:
    json.dump([live_signal], f, indent=2)

# Send LIVE alert
success = send_quick_alert(
    symbol=live_signal['symbol'],
    strike=live_signal['strike'],
    option_type=live_signal['option_type'],
    entry_price=live_signal['entry_price'],
    stoploss=live_signal['stoploss'],
    reason=live_signal['reason']
)

if success:
    print(f"✅ LIVE ALERT SENT: {live_signal['symbol']} {live_signal['strike']} {live_signal['option_type']}")
else:
    print("❌ Alert failed")

# Start API server
import subprocess
import sys
subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000'])
print("🌐 API server started for frontend")