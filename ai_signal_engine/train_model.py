#!/usr/bin/env python3
"""
AI Signal Engine - Train on KITE trades only (No demo/fake data)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import argparse
from datetime import datetime
import logging
import sys
sys.path.append('../data_enrichment')
from nse_data import enrich_with_nse_data
from sentiment_data import enrich_with_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KiteAITrainer:
    def __init__(self, min_confidence=0.85):
        self.min_confidence = min_confidence
        self.model = None
        
    def load_data(self, path):
        """Load and filter KITE-only trades"""
        try:
            df = pd.read_csv(path)
            
            # Filter KITE trades only - NO DEMO/FAKE
            df = df[df["source"] == "KITE"]
            df = df[df["market_open"] == True]
            
            # Remove educational/demo trades
            df = df[~df["reason"].str.contains("demo|educational|fake", case=False, na=False)]
            
            logger.info(f"✅ Loaded {len(df)} KITE-only trades")
            return df
            
        except FileNotFoundError:
            logger.error(f"❌ Data file not found: {path}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Technical indicators
        df['momentum'] = df['change_pct'].abs()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Market timing features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_opening'] = (df['hour'] >= 9) & (df['hour'] <= 10)
        df['is_closing'] = (df['hour'] >= 14) & (df['hour'] <= 15)
        
        # Instrument encoding
        instrument_map = {'NIFTY': 1, 'BANKNIFTY': 2, 'SENSEX': 3, 'FINNIFTY': 4, 'NIFTYIT': 5}
        df['instrument_code'] = df['symbol'].map(instrument_map).fillna(0)
        
        return df
    
    def train_model(self, df):
        """Train AI model on KITE data with enrichment"""
        if len(df) < 50:
            logger.error("❌ Insufficient data for training")
            return None
        
        # Enrich with external data sources (NSE, sentiment)
        logger.info("📊 Enriching KITE data with NSE and sentiment data...")
        signals = df.to_dict('records')
        signals = enrich_with_nse_data(signals)
        signals = enrich_with_sentiment(signals)
        df = pd.DataFrame(signals)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Enhanced feature columns with enriched data
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'momentum', 'volume_ratio', 'price_position',
            'hour', 'is_opening', 'is_closing', 'instrument_code',
            'historical_volatility', 'sentiment_score', 'news_volume'
        ]
        
        X = df[feature_cols].fillna(0)
        y = df['outcome']  # 1 = profitable, 0 = loss
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Model trained - Accuracy: {accuracy:.2%}")
        logger.info(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return self.model
    
    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            logger.error("❌ No model to save")
            return False
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'min_confidence': self.min_confidence,
            'model_type': 'RandomForestClassifier',
            'data_source': 'KITE_ONLY'
        }
        
        metadata_path = path.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved: {path}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Train AI model on KITE trades')
    parser.add_argument('--source', default='data/kite_trades.csv', help='Source CSV file')
    parser.add_argument('--model_out', default='models/signal_model.pkl', help='Output model path')
    parser.add_argument('--min_confidence', type=float, default=0.85, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting AI training on KITE-only data")
    
    # Initialize trainer
    trainer = KiteAITrainer(min_confidence=args.min_confidence)
    
    # Load data
    df = trainer.load_data(args.source)
    if df.empty:
        logger.error("❌ No data to train on")
        return
    
    # Train model
    model = trainer.train_model(df)
    if model is None:
        logger.error("❌ Training failed")
        return
    
    # Save model
    if trainer.save_model(args.model_out):
        logger.info("✅ AI training completed successfully")
    else:
        logger.error("❌ Failed to save model")

if __name__ == "__main__":
    main()