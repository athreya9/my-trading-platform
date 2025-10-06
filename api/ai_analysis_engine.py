# api/ai_analysis_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os
from dotenv import load_dotenv
from .accurate_telegram_alerts import AccurateTelegramAlerts
from .news_sentiment import fetch_news_sentiment # Add this line

load_dotenv()

# Placeholder for the AI model
class DummyModel:
    def predict(self, data):
        return np.random.rand(1, 2)

class AIAnalysisEngine:
    def __init__(self):
        self.model = self.load_ai_model()

    def load_ai_model(self):
        # In a real scenario, this would load a trained model from a file
        return DummyModel()

    def analyze_trading_opportunity(self, kite_data, market_context, stock_name): # Changed news_sentiment to stock_name
        """
        Comprehensive AI analysis that provides INTELLIGENT suggestions
        """

        # 1. TECHNICAL ANALYSIS
        technical_analysis = self._technical_analysis(kite_data)

        # 2. FUNDAMENTAL/MARKET ANALYSIS
        market_score = self._market_analysis(market_context)

        # 3. NEWS SENTIMENT ANALYSIS
        # Fetch sentiment using the updated news_sentiment.py
        raw_sentiment = fetch_news_sentiment(stock_name) # Call the function
        sentiment_analysis_result = self._sentiment_analysis(raw_sentiment) # Pass the string result

        # 4. RISK ANALYSIS (Kelly Criterion, Position Sizing)
        risk_analysis = self._risk_analysis(kite_data)

        # 5. AI MODEL PREDICTION
        ai_prediction = self._ai_prediction(kite_data, technical_analysis['score'])

        # 6. COMBINE ALL FACTORS
        final_recommendation = self._combine_analysis(
            technical_analysis, market_score, sentiment_analysis_result, risk_analysis, ai_prediction
        )

        return final_recommendation

    def _technical_analysis(self, data):
        """Real technical analysis using your existing indicators"""
        score = 0
        strengths = []

        # Use your existing indicator calculations
        if data.get('rsi', 50) < 30:
            score += 20  # Oversold bounce
            strengths.append("RSI indicating oversold conditions")
        if data.get('sma_20', 0) > data.get('sma_50', 1):
            score += 25  # Trend alignment
            strengths.append("Positive trend alignment (SMA20 > SMA50)")
        if data.get('volume', 0) > data.get('volume_avg', 1) * 1.5:
            score += 15  # Volume confirmation
            strengths.append("Volume surge confirmation")
        if data.get('macd', 0) > data.get('macd_signal', 1):
            score += 20  # Momentum
            strengths.append("MACD indicating bullish momentum")
        if data.get('atr', 100) < data.get('atr_avg', 200):
            score += 10  # Low volatility
            strengths.append("Low volatility environment")

        return {'score': min(score, 100), 'reason': 'Multi-factor technical analysis', 'strengths': strengths, 'pattern': 'Rule-based'}

    def _market_analysis(self, context):
        """Analyze broader market conditions"""
        analysis = {
            'sector_rotation': self._detect_sector_rotation(context),
            'market_breadth': self._calculate_market_breadth(context),
            'volatility_regime': self._assess_volatility(context),
            'institutional_flow': self._detect_institutional_activity(context)
        }
        return analysis

    def _sentiment_analysis(self, sentiment_string): # Changed news_sentiment to sentiment_string
        """Processes the sentiment data from the news fetcher."""
        # Convert sentiment string to a numerical score for internal use
        sentiment_to_score = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }
        score = sentiment_to_score.get(sentiment_string, 0)

        # The 'overall' and 'summary' fields are no longer directly available from news_sentiment.py
        # We'll simplify this for now, or fetch summary separately if needed.
        # For now, we'll just return the string and the score.
        return {
            'overall': sentiment_string,
            'score': score,
            'summary': f"News sentiment is {sentiment_string}." # Simplified summary
        }

    def _risk_analysis(self, data):
        """Advanced risk management using Kelly Criterion"""
        # Calculate optimal position size
        win_rate = self._calculate_historical_win_rate(data.get('symbol'))
        avg_win, avg_loss = self._calculate_avg_win_loss(data.get('symbol'))

        # Kelly Criterion formula
        if avg_loss != 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / abs(avg_loss)
            kelly_fraction = max(0, min(kelly_fraction, 0.2))  # Cap at 20%
        else:
            kelly_fraction = 0.1  # Conservative default

        return {
            'kelly_fraction': kelly_fraction,
            'optimal_position_size': kelly_fraction * 100,  # % of capital
            'max_drawdown_risk': self._calculate_max_drawdown_risk(data),
            'var_95': self._calculate_var(data),
            'win_probability': win_rate * 100
        }

    def _ai_prediction(self, data, technical_score):
        """Placeholder for AI model prediction, based on technical score"""
        # Simple mapping from technical score to a pseudo-prediction
        return [0.0, technical_score / 100.0]

    def _combine_analysis(self, technical_analysis, market_score, sentiment_analysis, risk_analysis, ai_prediction):
        """Combine all analysis into a single recommendation"""
        sentiment_weight_map = {
            "positive": 1.2,
            "neutral": 1.0,
            "negative": 0.8
        }
        # Get the sentiment string from sentiment_analysis
        current_sentiment = sentiment_analysis.get('overall', 'neutral')
        applied_sentiment_weight = sentiment_weight_map.get(current_sentiment, 1.0)

        weighted_technical_score = technical_analysis['score'] * applied_sentiment_weight

        composite_score = (weighted_technical_score * 0.4) + (ai_prediction[1] * 100 * 0.4) + (sentiment_analysis['score'] * 10) # sentiment_analysis['score'] is now -1, 0, or 1

        reasoning = f"""{technical_analysis['reason']}. News sentiment is {sentiment_analysis['overall'].lower()}.
Recent headlines:
{sentiment_analysis['summary']}"""

        return {
            'composite_score': composite_score,
            'technical': technical_analysis,
            'market': market_score,
            'sentiment': sentiment_analysis,
            'risk': risk_analysis,
            'reasoning': reasoning
        }

    def generate_intelligent_signal(self, analysis_result):
        """
        Generate SMART trading signal with specific instructions
        """
        signal = {
            'action': 'HOLD',  # Default to safety
            'confidence': 0,
            'specific_instructions': ["No clear signal based on current analysis."],
            'risk_metrics': {},
            'profit_targets': {'description': 'N/A'},
            'time_horizon': 'N/A',
            'reasoning': analysis_result.get('reasoning', 'Market conditions are neutral.'),
            'exit_conditions': 'N/A',
            'trail_stop_level': 'N/A'
        }
        
        # Only recommend trade if AI confidence is high
        if analysis_result['composite_score'] > 75:
            signal['action'] = 'BUY'
            signal['confidence'] = analysis_result['composite_score']
            signal['time_horizon'] = '1-3 days'
            signal['exit_conditions'] = 'Price crosses below 20-period moving average'
            signal['trail_stop_level'] = 'Entry price after Target 1 is hit'
            
            # Specific instructions based on analysis
            if analysis_result['technical']['pattern'] == 'Breakout':
                signal['specific_instructions'] = ["Enter on breakout confirmation above resistance"]
            else:
                signal['specific_instructions'] = ["Enter at current market price."]
                
            # Risk-managed profit targets
            signal['profit_targets'] = self._calculate_smart_targets(analysis_result)
            
            # Position sizing based on Kelly Criterion
            signal['position_size'] = f"{analysis_result['risk']['optimal_position_size']:.2f}% of capital"
            
        elif analysis_result['composite_score'] < 30:
            signal['action'] = 'AVOID'
            signal['confidence'] = 100 - analysis_result['composite_score']
            signal['specific_instructions'] = ["Market conditions unfavorable for a new position."]
            signal['reasoning'] = "The AI model suggests avoiding new trades due to unfavorable market conditions."
            
        return signal

    # --- Placeholder helper functions ---
    def _check_multi_tf_bullish(self, data):
        return True
    
    def _analyze_volume_profile(self, data):
        return 0.8

    def _detect_price_patterns(self, data):
        return "Breakout"

    def _detect_sector_rotation(self, context):
        return "Rotation into Banks"

    def _calculate_market_breadth(self, context):
        return "Improving"

    def _assess_volatility(self, context):
        return "Medium"

    def _detect_institutional_activity(self, context):
        return "Accumulation"

    def _calculate_historical_win_rate(self, symbol):
        return 0.6

    def _calculate_avg_win_loss(self, symbol):
        return 100, 50

    def _calculate_max_drawdown_risk(self, data):
        return 0.1

    def _calculate_var(self, data):
        return 5.0

    def _calculate_smart_targets(self, analysis_result):
        return {
            'description': 'Target 1: 10%, Target 2: 20%',
            'targets': [1.1, 1.2]
        }

    def _calculate_sma(self, data, window):
        """Calculates the Simple Moving Average."""
        if len(data) < window:
            return None
        return data[-window:].mean()

    def get_simple_trend_signal(self, historical_data):
        """Generates a simple trend signal based on SMA crossover."""
        if historical_data is None or historical_data.empty:
            return "NEUTRAL"

        # Ensure we have enough data for SMAs
        if len(historical_data) < 50:
            return "NEUTRAL"

        # Calculate short and long term SMAs
        sma_20 = self._calculate_sma(historical_data['Close'], 20)
        sma_50 = self._calculate_sma(historical_data['Close'], 50)

        if sma_20 is None or sma_50 is None:
            return "NEUTRAL"

        if sma_20 > sma_50:
            return "UP"
        else:
            return "DOWN"


from .accurate_telegram_alerts import AccurateTelegramAlerts

def send_ai_powered_alert(signal, analysis, telegram_bot):
    """
    Send alert that shows WHY the AI is recommending this trade
    """
    if signal.get("status") != "live":
        return
    message = f"""
 **AI-POWERED TRADING SIGNAL** 

 **RECOMMENDATION:** {signal['action']}
 **CONFIDENCE LEVEL:** {signal['confidence']:.2f}%

 **AI ANALYSIS BREAKDOWN:**

 **Technical Score:** {analysis['technical']['score']}/100
• Patterns: {analysis['technical']['pattern']}
• Strengths: {', '.join(analysis['technical']['strengths'][:2])}

 **Market Context:** {analysis['market']['sector_rotation']}
• Breadth: {analysis['market']['market_breadth']}
• Volatility: {analysis['market']['volatility_regime']}

 **News Sentiment:** {analysis['sentiment']['overall']} (Score: {analysis['sentiment']['score']})

️ **RISK MANAGEMENT:**
• Kelly Criterion: {analysis['risk']['kelly_fraction']*100:.2f}% position
• Max Risk: {analysis['risk']['var_95']}% 
• Win Probability: {analysis['risk']['win_probability']:.2f}%

 **WHY THIS TRADE:**
{signal['reasoning']}

 **SMART TARGETS:**
{signal['profit_targets']['description']}

⏰ **EXPECTED DURATION:** {signal['time_horizon']}

⚠️ **AI MONITORING PARAMETERS:**
- Exit if: {signal['exit_conditions']}
- Trail stop at: {signal['trail_stop_level']}

*Generated by AI Trading Agent • {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    
    return telegram_bot._send_telegram_message(message)
