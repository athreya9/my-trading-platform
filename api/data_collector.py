# data_collector.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class DataCollector:
    def __init__(self):
        self.data_dir = "historical_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_historical_data(self, symbol, period="2y", interval="1d"):
        """Fetch historical data using yfinance for training"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Add metadata
            data['symbol'] = symbol
            data['data_source'] = 'yfinance'
            data['fetch_time'] = datetime.now().isoformat()
            
            return data
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def save_daily_data(self, symbol, data):
        """Save daily data for AI training"""
        filename = f"{self.data_dir}/{symbol}_daily.json"
        
        # Read existing data
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []
        
        # Append new data
        df_to_save = data.reset_index()
        df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        new_entry = {
            'date': datetime.now().isoformat(),
            'data': df_to_save.to_dict('records'),
            'symbol': symbol
        }
        existing_data.append(new_entry)
        
        # Save back
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"✅ Daily data saved for {symbol}")
    
    def collect_training_data(self, symbols):
        """Collect data for all symbols"""
        for symbol in symbols:
            print(f" Collecting data for {symbol}...")
            data = self.fetch_historical_data(symbol)
            if data is not None and not data.empty:
                self.save_daily_data(symbol, data)
                
                # Send Telegram alert for successful collection
                self.send_data_collection_alert(symbol, len(data))
    
    def send_data_collection_alert(self, symbol, record_count):
        """Send Telegram alert for data collection"""
        message = f"""
 **AI TRAINING DATA COLLECTED**

✅ **Symbol:** {symbol}
 **Records Collected:** {record_count}
 **Time Period:** 2 years daily data
 **Storage:** Historical database updated
 **Purpose:** AI model retraining

*Next AI training cycle will use this enhanced dataset*
"""
        # Use your existing Telegram function
        from .accurate_telegram_alerts import AccurateTelegramAlerts
        telegram_bot = AccurateTelegramAlerts()
        telegram_bot._send_telegram_message(message)