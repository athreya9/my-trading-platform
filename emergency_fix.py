#!/usr/bin/env python3
"""
An emergency script to completely reset and restructure the Google Sheet.
"""
import gspread
import json
import os
import sys
from datetime import datetime
from google.oauth2 import service_account

def emergency_fix():
    """Connects to Google Sheets, deletes all but the first tab, and recreates the required structure."""
    print("🚨 PERFORMING EMERGENCY SHEET STRUCTURE FIX...")
    
    try:
        # Connect to Google Sheets using modern, robust methods
        creds_json = os.getenv('GSHEET_CREDENTIALS')
        if not creds_json:
            print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS secret is missing.", file=sys.stderr)
            sys.exit(1)
            
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        gc = gspread.service_account(credentials=credentials)
        sheet = gc.open("Algo Trading Dashboard")
        
        # DELETE ALL existing tabs except the first one to ensure a clean slate
        print("Clearing old sheet structure...")
        existing_worksheets = sheet.worksheets()
        # Iterate backwards to avoid index shifting issues while deleting
        for i in range(len(existing_worksheets) - 1, 0, -1):
            print(f"Deleting tab: {existing_worksheets[i].title}")
            sheet.del_worksheet(existing_worksheets[i])
        
        # Rename the first sheet to be our main Advisor_Output
        first_sheet = sheet.get_worksheet(0)
        first_sheet.update_title("Advisor_Output")
        print("✅ Renamed first sheet to 'Advisor_Output'")

        # CREATE/UPDATE ESSENTIAL TABS with proper structure
        tabs = {
            "Advisor_Output": [["🎯 ALGO TRADING ADVISOR", ""], ["LAST UPDATED", datetime.now().strftime("%Y-%m-%d %H:%M:%S")], ["STATUS", "🟢 SYSTEM ONLINE"], ["RECOMMENDATION", "Analyzing market data..."], ["CONFIDENCE", "85%"], ["EXPECTED HOLD TIME", "2-4 hours"]],
            "Price_Data": [["Timestamp", "Instrument", "Price", "Volume"]],
            "Signals": [["Timestamp", "Instrument", "Action", "Price", "Confidence"]],
            "Trade_Log": [["Date", "Instrument", "Action", "Quantity", "P/L"]]
        }
        
        for tab_name, headers in tabs.items():
            worksheet = sheet.worksheet(tab_name) if tab_name == "Advisor_Output" else sheet.add_worksheet(title=tab_name, rows="1000", cols="10")
            worksheet.clear()
            worksheet.update(range_name='A1', values=headers, value_input_option='USER_ENTERED')
            print(f"✅ Structured '{tab_name}' with proper headers")
        
        print("\n🎉 SHEET STRUCTURE FIXED! Backend now has an organized place to write data.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    emergency_fix()