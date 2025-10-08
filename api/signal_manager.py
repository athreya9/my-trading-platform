
import json
import os
from datetime import datetime, timedelta

class SignalManager:
    def __init__(self, signals_file='data/signals.json'):
        self.signals_file = signals_file
        self.ensure_directory()
    
    def ensure_directory(self):
        os.makedirs(os.path.dirname(self.signals_file), exist_ok=True)
    
    def add_manual_signal(self, symbol, signal, confidence=0.9):
        """Use this to inject manual BUY/SELL signals"""
        signals = self.load_signals()
        
        signals[symbol] = {
            'signal': signal,
            'timestamp': datetime.now().isoformat(),
            'source': 'MANUAL',
            'confidence': confidence
        }
        
        self.save_signals(signals)
        return True
    
    def load_signals(self):
        if os.path.exists(self.signals_file):
            with open(self.signals_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def save_signals(self, signals):
        with open(self.signals_file, 'w') as f:
            json.dump(signals, f, indent=2)
    
    def clear_old_signals(self, hours=24):
        """Clean up old signals"""
        signals = self.load_signals()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        updated_signals = {}
        for symbol, data in signals.items():
            try:
                signal_time = datetime.fromisoformat(data['timestamp'])
                if signal_time > cutoff:
                    updated_signals[symbol] = data
            except:
                continue
        
        self.save_signals(updated_signals)
