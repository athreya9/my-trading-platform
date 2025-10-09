#!/usr/bin/env python3
"""
Automated AI Retraining - Runs via cron daily
"""
import os
import sys
import subprocess
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def automated_retrain():
    """Automated retraining process"""
    try:
        logger.info("🤖 Starting automated AI retraining...")
        
        # Change to project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_dir)
        
        # Run training script
        cmd = [
            'python3', 
            'ai_signal_engine/train_model.py',
            '--source', 'data/signal_log.json',
            '--model_out', 'models/signal_model.pkl',
            '--min_confidence', '0.85'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Automated retraining completed successfully")
            
            # Send success notification
            from api.telegram_alerts import telegram_bot
            telegram_bot.notify("🤖 AI Model Retrained Successfully", target="admin")
            
        else:
            logger.error(f"❌ Retraining failed: {result.stderr}")
            
            # Send failure notification
            from api.telegram_alerts import telegram_bot
            telegram_bot.notify(f"❌ AI Retraining Failed: {result.stderr[:100]}", target="admin")
    
    except Exception as e:
        logger.error(f"Automated retraining error: {e}")

if __name__ == "__main__":
    automated_retrain()