#!/usr/bin/env python3
"""
Test frontend-backend connection by writing directly to the Advisor_Output sheet.
"""
import gspread
from google.oauth2 import service_account
import json
import os
import sys
from datetime import datetime

def main():
    """Connects to Google Sheets and writes a test message."""
    print("--- Starting Frontend Connection Test ---")
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

        # Test write to Advisor_Output
        print("Opening 'Algo Trading Dashboard' spreadsheet...")
        sheet = gc.open("Algo Trading Dashboard")
        advisor_sheet = sheet.worksheet("Advisor_Output")
        print("✅ Opened 'Advisor_Output' worksheet.")

        # Write a simple test message
        print("Writing test message...")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        test_message = [["🚀 TEST MESSAGE FROM BACKEND"], ["Frontend-Backend connection:", "SUCCESSFUL!"], ["Last updated:", current_time]]

        advisor_sheet.update(range_name='A1', values=test_message, value_input_option='USER_ENTERED')
        print("\n✅ Test message written to Advisor_Output. Check your frontend!")
        print("--- Test Completed Successfully ---")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()