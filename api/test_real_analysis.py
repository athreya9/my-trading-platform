# api/test_real_analysis.py
from .ai_analysis_engine import AIAnalysisEngine
import pandas as pd

# Test with sample real data
def test_with_real_data():
    engine = AIAnalysisEngine()
    
    sample_data = {
        'symbol': 'BANKNIFTY',
        'price': 54850,
        'rsi': 45,
        'sma_20': 54600,
        'sma_50': 54300,
        'volume': 125000,
        'volume_avg': 80000,
        'macd': 2.5,
        'macd_signal': 1.8,
        'atr': 180,
        'atr_avg': 200
    }
    
    analysis = engine._technical_analysis(sample_data)
    print(f"Technical Score: {analysis['score']}/100")
    print(f"Reason: {analysis['reason']}")

if __name__ == '__main__':
    test_with_real_data()