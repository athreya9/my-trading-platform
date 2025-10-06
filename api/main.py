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
    get_dashboard_data,
    update_bot_status, # Added import
    read_collection # Added import
)

# --- Setup ---
logger = logging.getLogger("uvicorn")
app = FastAPI()

# --- CORS Configuration ---
# This is crucial to allow your frontend Cloud Run service to access this API.
origins = [
    "http://localhost:3000",  # For local Next.js development
    "https://trading-dashboard-app.vercel.app/",
    "https://trading-platform-analysis-dashboard.vercel.app/", # Corrected live frontend URL
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
        "category": "Options" # Added category
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
    db = get_db()
    bot_status = "connected" if check_bot_status(db) else "disconnected" # Get actual bot status
    return [
        {"name": "Frontend", "status": "connected"},
        {"name": "Backend", "status": "connected"},
        {"name": "KITE API", "status": "disconnected"},
        {"name": "JSON Storage", "status": "connected"},
        {"name": "Trading Bot", "status": bot_status}, # Added Trading Bot status
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
    """Returns trading signals by reading from data/signals.json."""
    db = get_db()
    signals = read_collection(db, 'signals') # Read from signals.json
    
    if not signals:
        logger.info("No signals found in data/signals.json. Returning empty list.")
        return [] # Return empty list instead of 404

    return signals

@app.get("/api/bot/status")
def get_bot_status():
    """Returns the current status of the trading bot."""
    db = get_db()
    status = "running" if check_bot_status(db) else "paused"
    return {"status": status}

@app.post("/api/bot/start")
def start_bot():
    """Starts the trading bot."""
    logger.info("Received request to start bot.") # Added logging
    db = get_db()
    update_bot_status(db, "running", "Bot started via API")
    return {"message": "Bot started successfully."}

import yfinance as yf # Added import

# ... (rest of the file content) ...

@app.post("/api/bot/stop")
def stop_bot():
    """Stops the trading bot."""
    db = get_db()
    update_bot_status(db, "paused", "Bot stopped via API")
    return {"message": "Bot stopped successfully."}

from backtesting.backtest_engine import run_backtest # Added import
from backtesting.strategy_optimizer import optimize_strategy # Added import

# ... (rest of the file content) ...

@app.get("/api/chartData")
async def get_chart_data(symbol: str):
    """
    Fetches historical chart data for a given symbol from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch data for the last 7 days with 1-hour interval
        data = ticker.history(period="7d", interval="1h")

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No chart data found for {symbol}")

        # Format data for Chart.js
        timestamps = data.index.tolist()
        prices = data['Close'].tolist()

        chart_data = {
            "labels": [t.strftime("%Y-%m-%d %H:%M") for t in timestamps],
            "datasets": [
                {
                    "label": f"{symbol} Price",
                    "data": prices,
                    "borderColor": "#10b981",
                    "backgroundColor": "rgba(16,185,129,0.1)",
                },
            ],
        }
        return chart_data
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch chart data for {symbol}")

@app.post("/api/backtest")
async def trigger_backtest(symbol: str):
    """
    Triggers a backtest for a given symbol and returns the results.
    """
    logger.info(f"Received request to trigger backtest for {symbol}")
    try:
        trades = run_backtest(symbol)
        if not trades:
            raise HTTPException(status_code=404, detail=f"No trades generated during backtest for {symbol}")
        
        best_threshold, best_pnl = optimize_strategy(trades)

        return {
            "symbol": symbol,
            "total_trades": len(trades),
            "total_pnl": sum(t['pnl'] for t in trades),
            "optimized_threshold": best_threshold,
            "optimized_pnl": best_pnl,
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Error during backtest for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run backtest for {symbol}")
