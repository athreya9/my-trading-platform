#!/usr/bin/env python3
"""
A utility script to ensure the Google Sheet has the correct structure.
"""
import gspread
from google.oauth2 import service_account
import json
import os
import sys

def fix_sheet_structure():
    """Connects to Google Sheets and creates essential tabs if they don't exist."""
    print("--- Starting Google Sheet Structure Fix ---")
    try:
        # Connect to Google Sheets
        print("Connecting to Google Sheets...")
        creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        if not creds_json:
            print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS secret is missing.", file=sys.stderr)
            sys.exit(1)

        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        gc = gspread.service_account(credentials=credentials)
        print("✅ Google Sheets connection successful.")
        
        # Open your sheet
        print("Opening 'Algo Trading Dashboard' spreadsheet...")
        sheet = gc.open("Algo Trading Dashboard")
        
        # Create essential tabs if they don't exist
        essential_tabs = ['Price Data', 'Advisor_Output', 'Signals', 'Trade Log']
        
        for tab_name in essential_tabs:
            try:
                sheet.worksheet(tab_name)
                print(f"✅ Tab '{tab_name}' already exists.")
            except gspread.WorksheetNotFound:
                sheet.add_worksheet(title=tab_name, rows="1000", cols="20")
                print(f"✅ Created tab: '{tab_name}'.")
        
        # Set up Advisor_Output with proper structure for frontend
        print("Initializing 'Advisor_Output' tab for frontend...")
        advisor_sheet = sheet.worksheet("Advisor_Output")
        initial_data = [["🎯 TRADING ADVISOR", ""], ["Status:", "System Initializing"], ["Last Updated:", "Loading..."], ["Recommendation:", "Analyzing market..."]]
        advisor_sheet.update(range_name='A1', values=initial_data, value_input_option='USER_ENTERED')
        print("✅ 'Advisor_Output' tab initialized successfully.")
        print("--- Sheet Structure Fix Completed ---")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    fix_sheet_structure()