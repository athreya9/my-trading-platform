#!/usr/bin/env python3
"""
NSE Data Enrichment - For training only, NOT signal dispatch
"""
import requests
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NSEDataEnricher:
    def __init__(self):
        self.base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_historical_data(self, symbol, days=30):
        """Get historical OHLC for feature enrichment"""
        try:
            url = f"{self.base_url}/historical/cm/equity"
            params = {'symbol': symbol}
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
            
        except Exception as e:
            logger.error(f"NSE historical data error: {e}")
        
        return pd.DataFrame()
    
    def get_market_status(self):
        """Get market status for validation"""
        try:
            url = f"{self.base_url}/marketStatus"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"NSE market status error: {e}")
        
        return {}

def enrich_with_nse_data(kite_signals):
    """Enrich KITE signals with NSE data for training"""
    enricher = NSEDataEnricher()
    
    for signal in kite_signals:
        # Add historical volatility
        hist_data = enricher.get_historical_data(signal['symbol'])
        if not hist_data.empty:
            signal['historical_volatility'] = hist_data['close'].pct_change().std() * 100
        
        # Add market status validation
        market_status = enricher.get_market_status()
        signal['market_validated'] = market_status.get('marketState', [{}])[0].get('marketStatus') == 'Open'
    
    return kite_signals