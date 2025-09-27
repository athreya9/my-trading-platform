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

from .live_data_provider import get_live_quote, get_option_chain
from .ai_analysis_engine import AIAnalysisEngine
from .data_collector import DataCollector
import pandas as pd

# --- Helper Functions for Trading Signals ---

def get_lot_size(instrument):
    if instrument == "NIFTY":
        return 25
    elif instrument == "Bank NIFTY":
        return 15
    return 0

def get_atm_strike(price, strikes):
    """Finds the at-the-money strike price."""
    return strikes.iloc[(strikes - price).abs().argsort()[:1]].iloc[0]

def get_signal_for_instrument(instrument_name, symbol):
    """Gets the trading signal for a given instrument."""
    # 1. Fetch live and historical data
    live_price = get_live_quote(symbol)
    if not live_price:
        return None

    collector = DataCollector()
    historical_data = collector.fetch_historical_data(symbol, period="3mo", interval="1d")
    if historical_data is None or historical_data.empty:
        return None

    # 2. Get trend from AI engine
    ai_engine = AIAnalysisEngine()
    trend = ai_engine.get_simple_trend_signal(historical_data)

    # 3. Get option chain
    option_chain = get_option_chain(symbol)
    if not option_chain:
        return None

    # 4. Determine ATM strike and option details
    atm_strike = get_atm_strike(live_price, option_chain.calls['strike'])
    
    if trend == "UP":
        option_type = "CE"
        atm_option = option_chain.calls[option_chain.calls['strike'] == atm_strike]
    else:
        option_type = "PE"
        atm_option = option_chain.puts[option_chain.puts['strike'] == atm_strike]

    if atm_option.empty:
        return None

    return {
        "instrument": instrument_name,
        "price": live_price,
        "trend": trend,
        "signal": f"Buy {option_type}",
        "strikePrice": atm_strike,
        "premium": atm_option['lastPrice'].iloc[0],
        "expiry": option_chain.calls['expiryDate'].iloc[0],
        "lotSize": get_lot_size(instrument_name),
        "atm": atm_strike,
        "otm": atm_strike + 100 if trend == "UP" else atm_strike - 100,
        "itm": atm_strike - 100 if trend == "UP" else atm_strike + 100,
    }

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
        "portfolioValue": {"value": f"{dashboard_data['current_portfolio_value']:.2f}", "change": ""},
        "dayPL": {"value": f"{dashboard_data['today_pnl']:.2f}", "change": f"{dashboard_data['total_trades']} trades", "status": "loss" if dashboard_data['today_pnl'] < 0 else "profit"},
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

@app.get("/api/trading-signals")
def get_trading_signals():
    """Returns trading signals for NIFTY and Bank NIFTY."""
    nifty_signal = get_signal_for_instrument("NIFTY", "^NSEI")
    banknifty_signal = get_signal_for_instrument("Bank NIFTY", "^NSEBANK")

    signals = []
    if nifty_signal:
        signals.append(nifty_signal)
    if banknifty_signal:
        signals.append(banknifty_signal)

    if not signals:
        raise HTTPException(status_code=404, detail="Could not fetch trading signals.")

    return signals
