import json
import os
import logging
from datetime import datetime, timedelta

def send_alerts():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Read signals
        with open('data/signals.json', 'r') as f:
            signals = json.load(f)
        
        active_signals = []
        
        for symbol, data in signals.items():
            signal = data.get('signal', 'HOLD')
            timestamp = data.get('timestamp', '')
            source = data.get('source', 'unknown')
            
            # Check if signal is active and recent (last 1 hour)
            if signal in ['BUY', 'SELL']:
                try:
                    signal_time = datetime.fromisoformat(timestamp)
                    if datetime.now() - signal_time < timedelta(hours=1):
                        active_signals.append({
                            'symbol': symbol,
                            'signal': signal,
                            'timestamp': timestamp,
                            'source': source
                        })
                        logger.info(f"Active signal found: {symbol} - {signal}")
                except ValueError:
                    continue
        
        # Send alerts for active signals
        if active_signals:
            send_telegram_alerts(active_signals)  # Your existing telegram function
            update_frontend_data(active_signals)   # Your existing frontend update function
            logger.info(f"Sent alerts for {len(active_signals)} active signals")
        else:
            logger.info("No active signals to alert")
            
    except Exception as e:
        logger.error(f"Error in send_alerts: {str(e)}")

def send_telegram_alerts(active_signals):
    """Your existing telegram alert logic"""
    # Implement your telegram bot message sending here
    pass

def update_frontend_data(active_signals):
    """Update your frontend with trade data"""
    # Your existing logic to update frontend
    pass