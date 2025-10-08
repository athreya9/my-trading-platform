#!/usr/bin/env python3
"""
Quick solution: Create a simple backend that your Vercel frontend can access
"""
import json
import os
from datetime import datetime

def create_backend_data():
    """Create data that matches what your frontend expects"""
    
    # Get current signals
    signals_file = 'data/signals.json'
    if os.path.exists(signals_file):
        with open(signals_file, 'r') as f:
            signals = json.load(f)
    else:
        signals = []
    
    # Create status data that frontend expects
    status_data = [
        {"name": "Trading Bot", "status": "connected"},
        {"name": "Backend", "status": "connected"},
        {"name": "KITE API", "status": "disconnected"},
        {"name": "JSON Storage", "status": "connected"}
    ]
    
    # Create trades data
    trades_data = []
    if signals:
        for signal in signals:
            trades_data.append({
                "instrument": f"{signal['symbol']} {signal.get('strike', '')} {signal.get('option_type', '')}",
                "position_size": 1,
                "underlying_price": str(signal.get('entry_price', 0)),
                "P/L": "0.00"
            })
    
    print("📊 Backend data created:")
    print(f"- {len(signals)} signals")
    print(f"- {len(status_data)} status items")
    print(f"- {len(trades_data)} trades")
    
    return {
        'signals': signals,
        'status': status_data,
        'trades': trades_data
    }

def start_simple_backend():
    """Start a simple backend server"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI()
    
    # Allow your Vercel frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/api/trading-signals")
    def get_signals():
        data = create_backend_data()
        return data['signals']
    
    @app.get("/api/status")
    def get_status():
        data = create_backend_data()
        return data['status']
    
    @app.get("/api/trades")
    def get_trades():
        data = create_backend_data()
        return data['trades']
    
    @app.get("/api/health")
    def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    print("🚀 Starting simple backend for frontend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_simple_backend()