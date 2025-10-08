#!/usr/bin/env python3
"""
Generate test signals to verify the system is working
"""
import json
import os
from datetime import datetime

def generate_test_signals():
    """Generate sample trading signals for testing"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample signals
    test_signals = [
        {
            "symbol": "RELIANCE",
            "signal": "BUY",
            "confidence": 0.85,
            "price": 2450.50,
            "timestamp": datetime.now().isoformat(),
            "source": "AI_ENGINE"
        },
        {
            "symbol": "HDFCBANK", 
            "signal": "HOLD",
            "confidence": 0.65,
            "price": 1680.25,
            "timestamp": datetime.now().isoformat(),
            "source": "AI_ENGINE"
        },
        {
            "symbol": "TCS",
            "signal": "SELL",
            "confidence": 0.78,
            "price": 3920.75,
            "timestamp": datetime.now().isoformat(),
            "source": "AI_ENGINE"
        }
    ]
    
    # Save to signals.json
    with open('data/signals.json', 'w') as f:
        json.dump(test_signals, f, indent=2)
    
    print(f"✅ Generated {len(test_signals)} test signals")
    print("📁 Saved to data/signals.json")
    
    # Also update bot status
    bot_status = [{
        "status": "running",
        "reason": "Test signals generated",
        "timestamp": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }]
    
    with open('data/bot_status.json', 'w') as f:
        json.dump(bot_status, f, indent=2)
    
    print("✅ Updated bot status")
    
    return test_signals

if __name__ == "__main__":
    generate_test_signals()