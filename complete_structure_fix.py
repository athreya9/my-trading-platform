#!/usr/bin/env python3
"""
A script to perform a complete and robust overhaul of the Google Sheet structure.
"""
from dotenv import load_dotenv
import gspread
import json
import os
import sys

# Load environment variables from .env file for local development.
load_dotenv()


def enhance_sheet_structure():
    """Connects to Google Sheets and ensures all essential tabs exist with the correct headers."""
    print("⚙️  Verifying and Enhancing Google Sheet Structure...")
    
    try:
        # Connect to Google Sheets using modern, robust methods
        creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        if not creds_json:
            print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS secret is missing.", file=sys.stderr)
            sys.exit(1)
            
        credentials = json.loads(creds_json)
        gc = gspread.service_account_from_dict(credentials)
        sheet = gc.open("Algo Trading Dashboard")
        existing_titles = [ws.title for ws in sheet.worksheets()]
        print(f"Found existing tabs: {existing_titles}")
        
        # --- Define ALL essential tabs and their required headers ---
        essential_tabs = {
            "Advisor_Output": [["Recommendation", "Confidence", "Reasons", "Timestamp"]],
            "Signals": [["Action", "Symbol", "Price", "Confidence", "Reasons", "Timestamp"]],
            "Bot_Control": [["Parameter", "Value"], ["status", "running"], ["mode", "EMERGENCY"], ["last_updated", "never"]],
            "Price_Data": [["Symbol", "Price", "Volume", "Change", "Timestamp"]],
            "Trade_Log": [["Date", "Instrument", "Action", "Quantity", "Entry", "Exit", "P/L"]]
        }
        
        for tab_name, headers in essential_tabs.items():
            if tab_name not in existing_titles:
                print(f"Tab '{tab_name}' not found. Creating it...")
                worksheet = sheet.add_worksheet(title=tab_name, rows="1000", cols="20")
                worksheet.update(range_name='A1', values=headers, value_input_option='USER_ENTERED')
                print(f"✅ Created and structured '{tab_name}'.")
            else:
                print(f"✅ Tab '{tab_name}' already exists.")
        
        print("\n🎉 Google Sheet structure is verified and up-to-date.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    enhance_sheet_structure()