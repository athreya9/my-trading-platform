#!/usr/bin/env python3
"""
A script to perform a complete and robust overhaul of the Google Sheet structure.
"""
import gspread
import json
import os
import sys
from google.oauth2 import service_account

def complete_sheet_fix():
    """Connects to Google Sheets, deletes all but the first tab, and recreates the required structure."""
    print("🚨 PERFORMING COMPLETE GOOGLE SHEET STRUCTURE FIX...")
    
    try:
        # Connect to Google Sheets using modern, robust methods
        creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        if not creds_json:
            print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS secret is missing.", file=sys.stderr)
            sys.exit(1)
            
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        gc = gspread.service_account(credentials=credentials)
        sheet = gc.open("Algo Trading Dashboard")
        
        # --- Clean Slate Protocol ---
        print("Clearing old sheet structure for a clean slate...")
        existing_worksheets = sheet.worksheets()
        # Iterate backwards to avoid index shifting issues while deleting
        for i in range(len(existing_worksheets) - 1, 0, -1):
            print(f"Deleting tab: {existing_worksheets[i].title}")
            sheet.del_worksheet(existing_worksheets[i])
        
        # Rename the first sheet to be our main Advisor_Output
        first_sheet = sheet.get_worksheet(0)
        first_sheet.update_title("Advisor_Output")
        print("✅ Renamed first sheet to 'Advisor_Output'")

        # --- Create ALL essential tabs with proper headers ---
        essential_tabs = {
            "Advisor_Output": [["Recommendation", "Confidence", "Reasons", "Timestamp"], ["Analyzing market...", "0%", "System initializing", "2024-01-15 09:00:00"]],
            "Price_Data": [["Symbol", "Price", "Volume", "Change", "Timestamp"]],
            "Signals": [["Action", "Symbol", "Price", "Confidence", "Reasons", "Timestamp"]],
            "Bot_Control": [["Parameter", "Value"], ["status", "stopped"], ["mode", "emergency"]],
            "Trade_Log": [["Date", "Instrument", "Action", "Quantity", "Entry", "Exit", "P/L"]]
        }
        
        for tab_name, data in essential_tabs.items():
            worksheet = sheet.worksheet(tab_name) if tab_name == "Advisor_Output" else sheet.add_worksheet(title=tab_name, rows="1000", cols="20")
            worksheet.clear()
            worksheet.update(range_name='A1', values=data, value_input_option='USER_ENTERED')
            print(f"✅ Structured '{tab_name}' with proper headers and initial data.")
        
        print("\n🎉 COMPLETE SHEET STRUCTURE FIX SUCCESSFUL! The system is ready.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    complete_sheet_fix()