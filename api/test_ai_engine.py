# api/test_ai_engine.py
import unittest
from .ai_analysis_engine import AIAnalysisEngine, send_ai_powered_alert

class TestAIEngine(unittest.TestCase):
    def test_ai_engine(self):
        ai_engine = AIAnalysisEngine()

        kite_data = {'symbol': 'BANKNIFTY'}
        market_context = {}
        news_sentiment = {}

        analysis = ai_engine.analyze_trading_opportunity(
            kite_data, market_context, news_sentiment
        )

        signal = ai_engine.generate_intelligent_signal(analysis)

        if signal['confidence'] > 70:
            success = send_ai_powered_alert(signal, analysis)
            self.assertTrue(success)
        else:
            print("No high-confidence signal generated.")

if __name__ == '__main__':
    unittest.main()