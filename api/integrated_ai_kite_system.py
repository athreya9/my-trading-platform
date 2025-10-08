#!/usr/bin/env python3
"""
Integrated AI + Kite System with Enhanced Risk Management
Combines your existing Kite live engine with the new enhanced AI model
"""
import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime
from .enhanced_risk_ai_model import EnhancedRiskAIModel
from .kite_live_engine import KiteLiveEngine
from .accurate_telegram_alerts import AccurateTelegramAlerts

class IntegratedAIKiteSystem:
    def __init__(self):
        self.kite_engine = KiteLiveEngine()
        self.ai_model = EnhancedRiskAIModel()
        self.telegram = AccurateTelegramAlerts()
        self.load_enhanced_model()
        
    def load_enhanced_model(self):
        """Load the enhanced AI model"""
        model_path = 'api/enhanced_trading_model.pkl'
        if os.path.exists(model_path):
            success = self.ai_model.load_model(model_path)
            if success:
                print("✅ Enhanced AI model loaded successfully")
            else:
                print("⚠️ Using fallback model")
        else:
            print("⚠️ Enhanced model not found, training new model...")
            self.train_model_if_needed()
    
    def train_model_if_needed(self):
        """Train model if it doesn't exist"""
        try:
            from .enhanced_risk_ai_model import main as train_model
            train_model()
            self.ai_model.load_model()
        except Exception as e:
            print(f"❌ Model training failed: {e}")
    
    def generate_ai_enhanced_signals(self):
        """Generate signals using both Kite data and AI analysis"""
        signals = []
        
        # Get live data from Kite
        instruments = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY']
        
        for instrument in instruments:
            try:
                # Get Kite live data
                kite_data = self.kite_engine.get_live_data(instrument)
                if not kite_data:
                    continue
                
                # Prepare features for AI model
                features = self.prepare_ai_features(kite_data, instrument)
                
                # Get AI prediction
                ai_signal = self.get_ai_prediction(features)
                
                # Combine with Kite analysis
                combined_signal = self.combine_kite_ai_analysis(kite_data, ai_signal, instrument)
                
                if combined_signal['confidence'] > 0.75:  # High confidence threshold
                    signals.append(combined_signal)
                    
            except Exception as e:
                print(f"Error processing {instrument}: {e}")
                continue
        
        return signals
    
    def prepare_ai_features(self, kite_data, instrument):
        """Convert Kite data to AI model features"""
        try:
            # Extract key features from Kite data
            features = {
                'close': kite_data.get('ltp', 0),
                'volume': kite_data.get('volume', 0),
                'change_pct': kite_data.get('net_change', 0) / kite_data.get('ltp', 1) * 100,
                'high': kite_data.get('ohlc', {}).get('high', 0),
                'low': kite_data.get('ohlc', {}).get('low', 0),
                'open': kite_data.get('ohlc', {}).get('open', 0)
            }
            
            # Calculate additional technical indicators
            # (In production, you'd have historical data for proper calculation)
            features.update({
                'rsi': min(max(50 + features['change_pct'] * 2, 0), 100),  # Simplified RSI
                'momentum': features['change_pct'],
                'volatility': abs(features['change_pct']),
                'volume_ratio': 1.0,  # Would need historical volume data
                'atr': (features['high'] - features['low']) / features['close'] * 100
            })
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return {}
    
    def get_ai_prediction(self, features):
        """Get prediction from enhanced AI model"""
        try:
            if not self.ai_model.models:
                return {'signal': 'HOLD', 'confidence': 0.5, 'reasoning': 'AI model not available'}
            
            # Convert features to array format expected by model
            feature_array = np.array([[
                features.get('close', 0),
                features.get('volume', 0),
                features.get('change_pct', 0),
                features.get('rsi', 50),
                features.get('momentum', 0),
                features.get('volatility', 0),
                features.get('volume_ratio', 1),
                features.get('atr', 0)
            ]])\n            
            # Scale features
            if hasattr(self.ai_model, 'scaler') and self.ai_model.scaler:
                # Pad or truncate to match training features
                if feature_array.shape[1] < self.ai_model.scaler.n_features_in_:
                    padding = np.zeros((1, self.ai_model.scaler.n_features_in_ - feature_array.shape[1]))
                    feature_array = np.hstack([feature_array, padding])
                elif feature_array.shape[1] > self.ai_model.scaler.n_features_in_:
                    feature_array = feature_array[:, :self.ai_model.scaler.n_features_in_]
                
                feature_array = self.ai_model.scaler.transform(feature_array)
            
            # Get ensemble prediction
            predictions = []
            for model_name, model in self.ai_model.models.items():
                try:
                    pred_proba = model.predict_proba(feature_array)[0, 1]
                    predictions.append(pred_proba)
                except:
                    predictions.append(0.5)  # Neutral if model fails
            
            # Average ensemble prediction
            avg_confidence = np.mean(predictions)
            
            # Determine signal
            if avg_confidence > 0.7:
                signal = 'BUY'
            elif avg_confidence < 0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'signal': signal,
                'confidence': avg_confidence,
                'reasoning': f'AI ensemble prediction: {avg_confidence:.3f}'
            }
            
        except Exception as e:
            print(f"AI prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'reasoning': f'AI error: {str(e)}'}
    
    def combine_kite_ai_analysis(self, kite_data, ai_signal, instrument):
        """Combine Kite technical analysis with AI prediction"""
        try:
            # Get Kite technical score
            kite_score = self.calculate_kite_technical_score(kite_data)
            
            # Combine scores (weighted average)
            kite_weight = 0.4
            ai_weight = 0.6
            
            combined_confidence = (kite_score * kite_weight) + (ai_signal['confidence'] * ai_weight)
            
            # Determine final signal
            if combined_confidence > 0.75 and ai_signal['signal'] == 'BUY':
                final_signal = 'BUY'
            elif combined_confidence < 0.25 and ai_signal['signal'] == 'SELL':
                final_signal = 'SELL'  
            else:
                final_signal = 'HOLD'
            
            # Calculate position sizing using AI risk management
            position_info = self.ai_model.calculate_position_sizing(
                combined_confidence, 
                account_balance=100000  # Default account size
            )
            
            return {
                'instrument': instrument,
                'signal': final_signal,
                'confidence': combined_confidence,
                'kite_score': kite_score,
                'ai_confidence': ai_signal['confidence'],
                'ai_reasoning': ai_signal['reasoning'],
                'position_sizing': position_info,
                'ltp': kite_data.get('ltp', 0),
                'change_pct': kite_data.get('net_change', 0) / kite_data.get('ltp', 1) * 100,
                'timestamp': datetime.now().isoformat(),
                'risk_metrics': {
                    'max_loss': position_info['max_loss_amount'],
                    'kelly_fraction': position_info['kelly_fraction'],
                    'position_size_pct': position_info['position_size_pct']
                }
            }
            
        except Exception as e:
            print(f"Error combining analysis: {e}")
            return None
    
    def calculate_kite_technical_score(self, kite_data):
        """Calculate technical score from Kite data"""
        try:
            score = 0.5  # Neutral baseline
            
            # Price momentum
            change_pct = kite_data.get('net_change', 0) / kite_data.get('ltp', 1) * 100
            if change_pct > 1:
                score += 0.2
            elif change_pct < -1:
                score -= 0.2
            
            # Volume (if available)
            volume = kite_data.get('volume', 0)
            if volume > 0:  # Above average volume
                score += 0.1
            
            # Volatility
            ohlc = kite_data.get('ohlc', {})
            if ohlc:
                daily_range = (ohlc.get('high', 0) - ohlc.get('low', 0)) / ohlc.get('open', 1)
                if daily_range > 0.02:  # High volatility
                    score += 0.1
            
            return max(0, min(score, 1))  # Clamp between 0 and 1
            
        except Exception as e:
            print(f"Error calculating technical score: {e}")
            return 0.5
    
    def send_enhanced_alert(self, signal):
        """Send enhanced alert with AI reasoning and risk metrics"""
        if signal['signal'] == 'HOLD':
            return
        
        message = f"""
🤖 **AI-ENHANCED TRADING SIGNAL**

📊 **{signal['instrument']}** | {signal['signal']}
💰 **LTP:** ₹{signal['ltp']:.2f} ({signal['change_pct']:+.2f}%)

🎯 **CONFIDENCE ANALYSIS:**
• Combined Score: {signal['confidence']:.1%}
• Kite Technical: {signal['kite_score']:.1%}  
• AI Prediction: {signal['ai_confidence']:.1%}

🧠 **AI REASONING:**
{signal['ai_reasoning']}

💼 **RISK MANAGEMENT:**
• Position Size: {signal['position_sizing']['position_size_pct']:.2f}% of capital
• Kelly Fraction: {signal['position_sizing']['kelly_fraction']:.3f}
• Max Loss: ₹{signal['position_sizing']['max_loss_amount']:.0f}

⚠️ **RISK METRICS:**
• Expected Max Loss: ₹{signal['risk_metrics']['max_loss']:.0f}
• Risk-Adjusted Size: {signal['risk_metrics']['position_size_pct']:.2f}%

⏰ **Generated:** {datetime.now().strftime('%H:%M:%S IST')}

*AI-Powered Trading • Risk-Managed Positions*
"""
        
        return self.telegram._send_telegram_message(message)
    
    def run_integrated_system(self):
        """Run the complete integrated system"""
        print("🚀 Starting Integrated AI + Kite Trading System...")
        
        try:
            # Generate AI-enhanced signals
            signals = self.generate_ai_enhanced_signals()
            
            if not signals:
                print("📊 No high-confidence signals generated")
                return
            
            print(f"✅ Generated {len(signals)} high-confidence signals")
            
            # Save signals to file
            with open('data/ai_enhanced_signals.json', 'w') as f:
                json.dump(signals, f, indent=2)
            
            # Send alerts for strong signals
            for signal in signals:
                if signal['confidence'] > 0.8:  # Very high confidence
                    self.send_enhanced_alert(signal)
                    print(f"📱 Alert sent for {signal['instrument']}: {signal['signal']}")
            
            print("🎯 Integrated system run completed successfully")
            
        except Exception as e:
            print(f"❌ System error: {e}")
            # Send error alert
            error_msg = f"🚨 **SYSTEM ERROR**\n\nIntegrated AI system encountered an error:\n{str(e)}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.telegram._send_telegram_message(error_msg)

def main():
    """Main function to run the integrated system"""
    system = IntegratedAIKiteSystem()
    system.run_integrated_system()

if __name__ == "__main__":
    main()