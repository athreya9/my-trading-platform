#!/usr/bin/env python3
"""
Sentiment Data Enrichment - For training only, NOT signal dispatch
"""
import requests
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SentimentEnricher:
    def __init__(self):
        self.news_sources = {
            'google': 'https://news.google.com/rss/search?q={symbol}+stock&hl=en-IN&gl=IN&ceid=IN:en',
            'msn': 'https://www.msn.com/en-in/money/markets'
        }
    
    def get_market_sentiment(self, symbol):
        """Get sentiment score for training enrichment"""
        try:
            # Simple sentiment based on news volume
            url = self.news_sources['google'].format(symbol=symbol)
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # Count news mentions as basic sentiment
                news_count = response.text.count(symbol.lower())
                sentiment_score = min(1.0, news_count / 10)  # Normalize to 0-1
                
                return {
                    'sentiment_score': sentiment_score,
                    'news_volume': news_count,
                    'sentiment_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Sentiment data error: {e}")
        
        return {'sentiment_score': 0.5, 'news_volume': 0}
    
    def get_macro_indicators(self):
        """Get macro indicators for market context"""
        try:
            # VIX-like volatility indicator
            return {
                'market_volatility': 'normal',  # Could be enhanced with real VIX data
                'market_trend': 'neutral',
                'macro_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Macro indicators error: {e}")
            return {}

def enrich_with_sentiment(kite_signals):
    """Enrich KITE signals with sentiment for training"""
    enricher = SentimentEnricher()
    
    for signal in kite_signals:
        # Add sentiment data
        sentiment = enricher.get_market_sentiment(signal['symbol'])
        signal.update(sentiment)
        
        # Add macro context
        macro = enricher.get_macro_indicators()
        signal.update(macro)
    
    return kite_signals