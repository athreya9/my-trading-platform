# api/sheet_utils.py
import gspread
import logging
import os
import json
from functools import wraps
import time

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
    """Connects to Google Sheets using the new authentication function and opens a sheet."""
    logger.info("Attempting to connect to Google Sheets...")
    client = get_gspread_client()
    if not client:
        # The get_gspread_client function already prints detailed errors.
        # We can raise an exception to halt execution if no client is returned.
        raise Exception("Failed to get Google Sheets client. Halting execution.")

    try:
        spreadsheet = client.open(sheet_name)
        logger.info(f"Successfully connected to Google Sheet: '{sheet_name}'")
        return spreadsheet
    except gspread.exceptions.SpreadsheetNotFound:
        logger.error(f"Error: Spreadsheet '{sheet_name}' not found or not shared with the service account.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred when opening the sheet: {e}")
        raise


@retry()
def enhance_sheet_structure(sheet):
    """Ensures all essential tabs exist in the Google Sheet with the correct headers."""
    logger.info("Verifying and enhancing Google Sheet structure...")
    
    try:
        existing_titles = [ws.title for ws in sheet.worksheets()]
        essential_tabs = {
            "Advisor_Output": [["Recommendation", "Confidence", "Reasons", "Timestamp"]],
            "Signals": [["Action", "Symbol", "Price", "Confidence", "Reasons", "Timestamp"]],
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