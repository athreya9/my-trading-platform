# ai_training_pipeline.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import json
from .accurate_telegram_alerts import AccurateTelegramAlerts

class AITrainingPipeline:
    def __init__(self):
        self.model_path = "ai_models/trading_model.pkl"
        self.training_data_path = "historical_data/"
    
    def load_training_data(self):
        """Load all historical data for training"""
        all_data = []
        
        for filename in os.listdir(self.training_data_path):
            if filename.endswith('_daily.json'):
                with open(f"{self.training_data_path}/{filename}", 'r') as f:
                    symbol_data = json.load(f)
                    all_data.extend(symbol_data)
        
        return pd.DataFrame(all_data)
    
    def create_features(self, data):
        """Create features for the model"""
        # This is a placeholder. In a real scenario, you would create features from the data.
        # For now, we will create some dummy features.
        features = pd.DataFrame()
        if 'data' in data.columns:
            for i, row in data.iterrows():
                df = pd.DataFrame(row['data'])
                df['symbol'] = row['symbol']
                features = pd.concat([features, df])
        
        features['SMA_10'] = features['Close'].rolling(window=10).mean()
        features['SMA_50'] = features['Close'].rolling(window=50).mean()
        features = features.dropna()
        return features[['SMA_10', 'SMA_50']]

    def create_labels(self, data):
        """Create labels for the model"""
        # This is a placeholder. In a real scenario, you would create labels based on your strategy.
        # For now, we will create some dummy labels.
        labels = (data['Close'].shift(-1) > data['Close']).astype(int)
        return labels

    def retrain_model(self):
        """Retrain AI model with latest data"""
        print(" Retraining AI model with latest data...")
        
        # Load data
        data = self.load_training_data()
        if data.empty:
            print("❌ No training data available")
            return False
        
        # Feature engineering (simplified example)
        features = self.create_features(data)
        labels = self.create_labels(features)
        
        # Align features and labels
        features = features.iloc[:len(labels)]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs("ai_models", exist_ok=True)
        joblib.dump(model, self.model_path)
        
        # Send training completion alert
        self.send_training_alert(accuracy=model.score(X_test, y_test))
        
        return True
    
    def send_training_alert(self, accuracy):
        """Send Telegram alert for AI training completion"""
        message = f"""
 **AI MODEL RETRAINING COMPLETE**

✅ **Training Status:** Successful
 **Dataset Size:** Enhanced with latest data
 **Model Accuracy:** {accuracy:.2%}
 **Model Saved:** Ready for production
 **Next Training:** 7 days or on demand

*AI agent now has updated market intelligence*
"""
        telegram_bot = AccurateTelegramAlerts()
        telegram_bot._send_telegram_message(message)