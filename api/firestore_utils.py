import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def init_json_storage():
    """
    Initializes JSON storage by ensuring data directory exists.
    Replaces Firebase initialization.
    """
    try:
        os.makedirs('data', exist_ok=True)
        logger.info("✅ JSON storage initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize JSON storage: {e}", exc_info=True)
        raise

def get_db():
    """
    Returns a dummy DB object for compatibility.
    Replaces Firestore client.
    """
    logger.info("✅ Using JSON storage instead of Firestore.")
    return {"storage_type": "json"}

def write_data_to_firestore(db, collection_name, data):
    """
    Writes data to JSON files instead of Firestore.
    Each collection becomes a separate JSON file.
    """
    try:
        init_json_storage()
        filename = f"data/{collection_name}.json"
        
        # Read existing data
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []
        
        # Append new data with timestamps
        for item in data:
            item_with_timestamp = {
                **item,
                'saved_at': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            existing_data.append(item_with_timestamp)
        
        # Save back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"✅ Successfully wrote {len(data)} items to '{filename}'.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write to '{collection_name}': {e}", exc_info=True)
        raise

def read_collection(db, collection_name):
    """
    Reads data from JSON files instead of Firestore.
    """
    try:
        filename = f"data/{collection_name}.json"
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Successfully read {len(data)} items from '{filename}'.")
            return data
        except FileNotFoundError:
            logger.info(f" File '{filename}' not found. Returning empty list.")
            return []
            
    except Exception as e:
        logger.error(f"Failed to read from '{collection_name}': {e}", exc_info=True)
        raise

def check_bot_status(db):
    """
    Check bot status from JSON instead of Firestore.
    Returns True if bot should run, False if paused.
    """
    try:
        filename = "data/bot_status.json"
        
        try:
            with open(filename, 'r') as f:
                status_data = json.load(f)
            # Return the latest status
            if status_data:
                latest_status = status_data[-1]
                return latest_status.get('status', 'running') == 'running'
        except FileNotFoundError:
            # If no status file exists, assume bot should run
            return True
            
    except Exception as e:
        logger.error(f"Failed to check bot status: {e}")
        return True  # Default to running if there's an error

def update_bot_status(db, status, reason=""):
    """
    Update bot status in JSON file.
    """
    try:
        filename = "data/bot_status.json"
        
        # Read existing status
        try:
            with open(filename, 'r') as f:
                status_history = json.load(f)
        except FileNotFoundError:
            status_history = []
        
        # Add new status
        new_status = {
            'status': status,  # 'running' or 'paused'
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        status_history.append(new_status)
        
        # Save back
        with open(filename, 'w') as f:
            json.dump(status_history, f, indent=2)
        
        logger.info(f"✅ Bot status updated to '{status}': {reason}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update bot status: {e}")
        return False

def read_manual_controls(db):
    """
    Read manual controls from JSON instead of Firestore.
    """
    try:
        filename = "data/manual_controls.json"
        
        try:
            with open(filename, 'r') as f:
                controls = json.load(f)
            # Return the latest control settings
            if controls:
                return controls[-1]
        except FileNotFoundError:
            # Return default controls if file doesn't exist
            return {
                'manual_override': False,
                'max_trade_amount': 1000,
                'allowed_symbols': [],
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to read manual controls: {e}")
        return {
            'manual_override': False,
            'max_trade_amount': 1000,
            'allowed_symbols': []
        }

def read_trade_log(db, limit=50):
    """
    Read trade log from JSON instead of Firestore.
    """
    try:
        filename = "data/trades.json"
        
        try:
            with open(filename, 'r') as f:
                trades = json.load(f)
            # Return latest trades (reverse chronological)
            return trades[-limit:] if limit else trades
        except FileNotFoundError:
            return []
            
    except Exception as e:
        logger.error(f"Failed to read trade log: {e}")
        return []

def get_dashboard_data(db):
    """
    Get dashboard data from JSON files instead of Firestore.
    Returns aggregated data for the dashboard.
    """
    try:
        dashboard_data = {
            'total_trades': 0,
            'today_pnl': 0,
            'current_portfolio_value': 0,
            'open_positions': [],
            'recent_trades': [],
            'bot_status': 'running',
            'last_updated': datetime.now().isoformat()
        }
        
        # Read trades data
        try:
            with open('data/trades.json', 'r') as f:
                all_trades = json.load(f)
            
            if all_trades:
                dashboard_data['total_trades'] = len(all_trades)
                dashboard_data['recent_trades'] = all_trades[-10:]  # Last 10 trades
                
                # Calculate today's PnL (simplified - you can enhance this)
                today = datetime.now().date().isoformat()
                today_trades = [t for t in all_trades if t.get('timestamp', '').startswith(today)]
                dashboard_data['today_pnl'] = sum(t.get('pnl', 0) for t in today_trades)
        
        except FileNotFoundError:
            logger.info("No trades data found yet")
        
        # Read portfolio data
        try:
            with open('data/portfolio.json', 'r') as f:
                portfolio_data = json.load(f)
                dashboard_data['current_portfolio_value'] = portfolio_data.get('total_value', 0)
                dashboard_data['open_positions'] = portfolio_data.get('positions', [])
        except FileNotFoundError:
            logger.info("No portfolio data found yet")
        
        # Check bot status
        dashboard_data['bot_status'] = 'running' if check_bot_status(db) else 'paused'
        
        logger.info("✅ Dashboard data generated successfully")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        # Return basic structure even if there's an error
        return {
            'total_trades': 0,
            'today_pnl': 0,
            'current_portfolio_value': 0,
            'open_positions': [],
            'recent_trades': [],
            'bot_status': 'unknown',
            'last_updated': datetime.now().isoformat(),
            'error': str(e)
        }