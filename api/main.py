#!/usr/bin/env python3
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from .firestore_utils import (
    init_json_storage as init_firestore_client, 
    get_db, 
    write_data_to_firestore, 
    check_bot_status, 
    read_manual_controls, 
    read_trade_log,
    get_dashboard_data
)

# --- Setup ---
logger = logging.getLogger("uvicorn")
app = FastAPI()

# --- CORS Configuration ---
# This is crucial to allow your frontend Cloud Run service to access this API.
origins = [
    "http://localhost:3000",  # For local Next.js development
    "https://trading-dashboard-app.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- JSON Storage Initialization ---
init_firestore_client()

# --- API Endpoints ---
@app.get("/api/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "json_storage_initialized": os.path.exists("data")}


@app.get("/api/trading-data")
def get_trading_data():
    """
    Fetches trading data from a json file.
    """
    db = get_db()
    data = read_collection(db, 'pnl_history')
    if not data:
        return {"message": "No data found in the collection.", "data": []}

    return {"data": data}

@app.get("/api/status")
def get_status():
    """Returns the connection status of various services."""
    return [
        {"name": "Frontend", "status": "connected"},
        {"name": "Backend", "status": "connected"},
        {"name": "KITE API", "status": "disconnected"},
        {"name": "JSON Storage", "status": "connected"},
    ]

@app.get("/api/stats")
def get_stats():
    """Returns key performance statistics for the dashboard."""
    db = get_db()
    dashboard_data = get_dashboard_data(db)
    return {
        "portfolioValue": {"value": f"${dashboard_data['current_portfolio_value']:.2f}", "change": ""},
        "dayPL": {"value": f"${dashboard_data['today_pnl']:.2f}", "change": f"{dashboard_data['total_trades']} trades", "status": "loss" if dashboard_data['today_pnl'] < 0 else "profit"},
        "activeTrades": {"value": f"{len(dashboard_data['open_positions'])}", "change": ""},
        "winRate": {"value": "0%", "change": "This month"},
    }

@app.get("/api/trades")
def get_trades():
    """Returns a list of recent trades."""
    db = get_db()
    return read_trade_log(db, limit=20)

@app.get("/api/performance")
def get_performance_data():
    """Returns data for the portfolio performance chart."""
    db = get_db()
    trades = read_trade_log(db)
    # create a fake performance chart
    return [
        {"name": f"Day {i+1}", "value": 100000 + (t['pnl'] if t else 0)} for i, t in enumerate(trades)
    ]
