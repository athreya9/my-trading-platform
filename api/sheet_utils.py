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

@retry()
def connect_to_google_sheets(sheet_name):
    """Connects to Google Sheets using credentials from an environment variable."""
    logger.info("Attempting to authenticate with Google Sheets...")
    creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
    if not creds_json_str:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS environment variable not found. Ensure it's set in GitHub Actions secrets.")

    try:
        creds_dict = json.loads(creds_json_str)
        client = gspread.service_account_from_dict(creds_dict)
        spreadsheet = client.open(sheet_name)
        logger.info(f"Successfully connected to Google Sheet: '{sheet_name}'")
        return spreadsheet
    except Exception as e:
        raise Exception(f"Error connecting to Google Sheet: {e}")


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