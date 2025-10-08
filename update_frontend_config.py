#!/usr/bin/env python3
"""
Update frontend to use GitHub raw data (immediate fix)
"""
import json
import os

# Create the data structure your frontend expects
def create_frontend_data():
    """Create data files that frontend can read from GitHub"""
    
    # Read current signals
    if os.path.exists('data/signals.json'):
        with open('data/signals.json', 'r') as f:
            signals = json.load(f)
    else:
        signals = []
    
    # Create API-compatible responses
    api_responses = {
        'trading-signals.json': signals,
        'status.json': [
            {"name": "Trading Bot", "status": "connected"},
            {"name": "Backend", "status": "connected"},
            {"name": "Data Source", "status": "connected"}
        ],
        'trades.json': []
    }
    
    # Add trades based on signals
    if signals:
        for signal in signals:
            api_responses['trades.json'].append({
                "instrument": f"{signal['symbol']} {signal.get('strike', '')} {signal.get('option_type', '')}",
                "position_size": 1,
                "underlying_price": str(signal.get('entry_price', 0)),
                "P/L": "0.00"
            })
    
    # Save API response files
    os.makedirs('api-data', exist_ok=True)
    for filename, data in api_responses.items():
        with open(f'api-data/{filename}', 'w') as f:
            json.dump(data, f, indent=2)
    
    print("✅ Created frontend-compatible data files:")
    for filename in api_responses.keys():
        print(f"  - api-data/{filename}")
    
    return api_responses

if __name__ == "__main__":
    create_frontend_data()