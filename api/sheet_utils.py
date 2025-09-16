# api/sheet_utils.py
import gspread
import logging
import os
import json
from functools import wraps
import time
import pandas as pd

# This can be a shared logger or a new one.
logger = logging.getLogger(__name__)

def retry(tries=3, delay=5, backoff=2, logger=logger):
    """
    A retry decorator with exponential backoff.
    Catches common network-related exceptions for gspread.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)                
                except (gspread.exceptions.APIError, ConnectionError) as e:
                    msg = f"'{f.__name__}' failed with {e}. Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

def get_gspread_client():
    """
    Authenticates with Google Sheets and returns a gspread client.

    It uses a robust authentication strategy:
    1. Tries to use the GOOGLE_SHEETS_CREDENTIALS environment variable,
       which is ideal for production/CI environments (like GitHub Actions).
    2. If the environment variable is not found, it falls back to loading
       the 'credentials.json' file from the same directory. This is
       ideal for local development.
    """
    # --- Strategy 1: Use Environment Variable ---
    creds_json_str = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
    if creds_json_str:
        try:
            creds_dict = json.loads(creds_json_str)
            print("Authenticating with Google Sheets via environment variable.")
            gc = gspread.service_account_from_dict(creds_dict)
            return gc
        except json.JSONDecodeError:
            print("Error: GOOGLE_SHEETS_CREDENTIALS environment variable is not valid JSON.")
        except Exception as e:
            print(f"An error occurred during authentication with environment variable: {e}")

    # --- Strategy 2: Fallback to local credentials.json file ---
    try:
        # __file__ is the path to the current script (e.g., /path/to/api/sheet_utils.py)
        # os.path.dirname(__file__) gives the directory (e.g., /path/to/api)
        # os.path.join(...) creates the full path to credentials.json
        credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
        print(f"Authenticating with Google Sheets via file: {credentials_path}")
        gc = gspread.service_account(filename=credentials_path)
        return gc
    except FileNotFoundError:
        print(f"Warning: credentials.json not found at {credentials_path}.")
        print("And GOOGLE_SHEETS_CREDENTIALS environment variable was not set.")
    except Exception as e:
        print(f"An error occurred during authentication with file: {e}")

    print("Error: Could not authenticate with Google Sheets.")
    return None

@retry()
def connect_to_google_sheets(sheet_name):
<<<<<<< HEAD
    """
    Connects to Google Sheets using a robust authentication strategy and opens the specified sheet.
=======
    """Connects to Google Sheets using the new authentication function and opens a sheet."""
    logger.info("Attempting to connect to Google Sheets...")
    client = get_gspread_client()
    if not client:
        # The get_gspread_client function already prints detailed errors.
        # We can raise an exception to halt execution if no client is returned.
        raise Exception("Failed to get Google Sheets client. Halting execution.")
>>>>>>> feature/frontend-backend-setup

    Authentication Strategy:
    1. Tries to use the GOOGLE_SHEETS_CREDENTIALS environment variable (for production/CI).
    2. Falls back to loading 'credentials.json' from the same directory (for local development).
    """
    logger.info("Attempting to authenticate with Google Sheets...")
    client = None

    # --- Strategy 1: Use Environment Variable ---
    creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
    if creds_json_str:
        try:
            creds_dict = json.loads(creds_json_str)
            logger.info("Authenticating with Google Sheets via environment variable.")
            client = gspread.service_account_from_dict(creds_dict)
        except json.JSONDecodeError:
            logger.error("Error: GOOGLE_SHEETS_CREDENTIALS environment variable is not valid JSON.")
        except Exception as e:
            logger.error(f"An error occurred during authentication with environment variable: {e}")

    # --- Strategy 2: Fallback to local credentials.json file ---
    if not client:
        try:
            credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
            logger.info(f"Authenticating with Google Sheets via file: {credentials_path}")
            client = gspread.service_account(filename=credentials_path)
        except FileNotFoundError:
            logger.error(f"CRITICAL: 'credentials.json' not found at {credentials_path} and GOOGLE_SHEETS_CREDENTIALS env var was not set or failed.")
            raise
        except Exception as e:
            logger.error(f"An error occurred during authentication with file: {e}")
            raise

    if not client:
        raise Exception("Could not authenticate with Google Sheets using any method.")

    # --- Open Spreadsheet ---
    try:
<<<<<<< HEAD
        logger.info(f"Opening Google Sheet: '{sheet_name}'")
=======
>>>>>>> feature/frontend-backend-setup
        spreadsheet = client.open(sheet_name)
        logger.info(f"Successfully connected to Google Sheet: '{spreadsheet.title}'")
        return spreadsheet
    except gspread.exceptions.SpreadsheetNotFound:
        logger.error(f"Error: Spreadsheet '{sheet_name}' not found or not shared with the service account.")
        raise
    except Exception as e:
<<<<<<< HEAD
        raise Exception(f"Error opening Google Sheet '{sheet_name}'. Ensure the name is correct and the service account has access. Original error: {e}")
=======
        logger.error(f"An unexpected error occurred when opening the sheet: {e}")
        raise
>>>>>>> feature/frontend-backend-setup


@retry()
def enhance_sheet_structure(sheet):
    """Ensures all essential tabs exist in the Google Sheet with the correct headers."""
    logger.info("Verifying and enhancing Google Sheet structure...")
    
    try:
        existing_titles = [ws.title for ws in sheet.worksheets()]
        essential_tabs = {
            "Advisor_Output": [["Recommendation", "Confidence", "Entry Price", "Stop Loss", "Take Profit", "Reasons", "Timestamp"]],
            "Signals": [["Action", "Symbol", "Entry Price", "Stop Loss", "Take Profit", "Confidence", "Reasons", "Timestamp"]],
            "Bot_Control": [["Parameter", "Value"], ["status", "running"], ["mode", "EMERGENCY"], ["last_updated", "never"]],
            "Price_Data": [["Symbol", "Timestamp", "open", "high", "low", "close", "volume"]],
            "Historical_Data": [["Symbol", "Timestamp", "open", "high", "low", "close", "volume"]],
            "Trade_Log": [["Date", "Instrument", "Action", "Quantity", "Entry", "Exit", "P/L"]]
        }
        
        for tab_name, headers in essential_tabs.items():
            if tab_name not in existing_titles:
                logger.info(f"Tab '{tab_name}' not found. Creating it...")
                worksheet = sheet.add_worksheet(title=tab_name, rows="1000", cols="20")
                worksheet.update(range_name='A1', values=headers, value_input_option='USER_ENTERED')
                logger.info(f"Created and structured '{tab_name}'.")
        
        logger.info("Google Sheet structure is verified and up-to-date.")
    except Exception as e:
        logger.error(f"An error occurred during sheet structure verification: {e}", exc_info=True)
        raise

@retry()
def read_historical_data(spreadsheet):
    """
    Reads the 'Historical_Data' worksheet, which is the source for ML training.
    Raises a ValueError if the sheet is empty.
    """
    logger.info("Reading historical data from 'Historical_Data' tab...")
    try:
        worksheet = spreadsheet.worksheet("Historical_Data")
        # Use get_all_records for structured data.
        records = worksheet.get_all_records()
        if not records:
            raise ValueError("The worksheet 'Historical_Data' was found, but it is empty. It must be populated by a scheduled run before training can occur.")
        
        df = pd.DataFrame(records)
        logger.info(f"Successfully read {len(df)} rows from 'Historical_Data'.")
        return df
    except gspread.exceptions.WorksheetNotFound:
        logger.error("❌ Critical Error: 'Historical_Data' worksheet not found. Please run the 'setup-sheets' job to create it.")
        raise
    except Exception as e:
        logger.error(f"Could not read historical data: {e}", exc_info=True)
        raise

@retry()
def read_worksheet_data(spreadsheet, worksheet_name):
    """
    Reads all data from a given worksheet and returns it as a list of dicts.
    Returns an empty list if the sheet is not found or an error occurs.
    """
    logger.info(f"Reading all data from '{worksheet_name}' sheet...")
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        records = worksheet.get_all_records()
        logger.info(f"Successfully read {len(records)} records from '{worksheet_name}'.")
        return records
    except gspread.exceptions.WorksheetNotFound:
        logger.warning(f"'{worksheet_name}' worksheet not found. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Could not read data from '{worksheet_name}': {e}", exc_info=True)
        return []