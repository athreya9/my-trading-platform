#!/usr/bin/env python3
"""
Real-time chart data for trading signals
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

def get_live_chart_data(symbol, period="1d", interval="5m"):
    """Get real-time chart data from Yahoo Finance"""
    try:
        # Symbol mapping for Indian indices
        symbol_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN', 
            'FINNIFTY': '^NSEI'  # Use NIFTY as proxy for FINNIFTY
        }
        
        ticker_symbol = symbol_map.get(symbol, symbol)
        ticker = yf.Ticker(ticker_symbol)
        
        # Get intraday data
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        # Format for chart
        chart_data = []
        for timestamp, row in data.iterrows():
            chart_data.append({
                'time': timestamp.strftime('%H:%M'),
                'price': round(row['Close'], 2),
                'volume': int(row['Volume']) if row['Volume'] > 0 else 0
            })
        
        return {
            'symbol': symbol,
            'data': chart_data[-50:],  # Last 50 data points
            'current_price': chart_data[-1]['price'] if chart_data else 0,
            'change': round(chart_data[-1]['price'] - chart_data[0]['price'], 2) if len(chart_data) > 1 else 0,
            'change_percent': round(((chart_data[-1]['price'] - chart_data[0]['price']) / chart_data[0]['price']) * 100, 2) if len(chart_data) > 1 else 0
        }
        
    except Exception as e:
        print(f"Error fetching chart data for {symbol}: {e}")
        return None

def get_option_chain_data(symbol, strike, option_type):
    """Get real option chain data"""
    try:
        import requests
        
        # NSE option chain API (simplified)
        if symbol == "NIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        elif symbol == "BANKNIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        else:
            return None
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Find specific option data
            for record in data.get('records', {}).get('data', []):
                if record.get('strikePrice') == strike:
                    option_data = record.get(option_type, {})
                    return {
                        'strike': strike,
                        'option_type': option_type,
                        'ltp': option_data.get('lastPrice', 0),
                        'change': option_data.get('change', 0),
                        'change_percent': option_data.get('pChange', 0),
                        'volume': option_data.get('totalTradedVolume', 0),
                        'oi': option_data.get('openInterest', 0),
                        'iv': option_data.get('impliedVolatility', 0)
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching option data: {e}")
        return None

if __name__ == "__main__":
    # Test real data
    chart = get_live_chart_data("NIFTY")
    if chart:
        print(f"NIFTY: ₹{chart['current_price']} ({chart['change_percent']:+.2f}%)")
        print(f"Data points: {len(chart['data'])}")