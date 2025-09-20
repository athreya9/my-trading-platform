# api/sheet_utils.py
import base64
import gspread
import logging
import os
import json
from functools import wraps
import time
import pandas as pd
import tempfile

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
        temp_file_path = None
        try:
            # Attempt to decode from Base64 first (as before)
            try:
                decoded_creds = base64.b64decode(creds_json_str).decode('utf-8')
                creds_to_write = decoded_creds
            except (base64.binascii.Error, json.JSONDecodeError):
                # Fallback to direct JSON load if not Base64 or invalid JSON
                creds_to_write = creds_json_str

            # Write credentials to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                temp_file.write(creds_to_write)
                temp_file_path = temp_file.name

            gc = gspread.service_account(filename=temp_file_path)
            logger.info("Authenticating with Google Sheets via temporary file from environment variable.")
            return gc
        except Exception as e:
            logger.error(f"An error occurred during authentication with environment variable via temp file: {e}", exc_info=True)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path) # Clean up the temporary file

    # --- Strategy 2: Fallback to local credentials.json file ---
    try:
        credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
        logger.info(f"Authenticating with Google Sheets via file: {credentials_path}")
        gc = gspread.service_account(filename=credentials_path)
        return gc
    except FileNotFoundError:
        logger.warning(f"Warning: credentials.json not found at {credentials_path}.")
        logger.warning("And GOOGLE_SHEETS_CREDENTIALS environment variable was not set.")
    except Exception as e:
        logger.error(f"An error occurred during authentication with file: {e}", exc_info=True)

    logger.error("Error: Could not authenticate with Google Sheets.")
    return None

@retry()
def connect_to_google_sheets(sheet_name):
    """
    Connects to Google Sheets using a robust authentication strategy and opens the specified sheet.
    """
    logger.info("Attempting to connect to Google Sheets...")
    client = get_gspread_client()
    if not client:
        # The get_gspread_client function already prints detailed errors.
        # We can raise an exception to halt execution if no client is returned.
        raise Exception("Failed to get Google Sheets client. Halting execution.")

    # --- Open Spreadsheet ---
    try:
        logger.info(f"Opening Google Sheet: '{sheet_name}'")
        spreadsheet = client.open(sheet_name)
        logger.info(f"Successfully connected to Google Sheet: '{spreadsheet.title}'")
        return spreadsheet
    except gspread.exceptions.SpreadsheetNotFound:
        logger.error(f"Error: Spreadsheet '{sheet_name}' not found or not shared with the service account.")
        raise
    except Exception as e:
        raise Exception(f"Error opening Google Sheet '{sheet_name}'. Ensure the name is correct and the service account has access. Original error: {e}")


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