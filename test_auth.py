# test_auth.py
import os
import sys
import json
from kiteconnect import KiteConnect
import re
import gspread

def test_kite_auth():
    """Tests Kite Connect authentication using environment variables."""
    print("--- Testing Kite Connect Authentication ---")
    
    # Get credentials from environment variables
    api_key = os.getenv('KITE_API_KEY', '').strip()
    access_token = os.getenv('KITE_ACCESS_TOKEN', '').strip()
    
    # DEBUG: Print what we actually received
    print(f"DEBUG - API Key: '{api_key}'")
    print(f"DEBUG - Access Token: '{access_token}'")
    print(f"DEBUG - API Key length: {len(api_key)}")
    print(f"DEBUG - Access Token length: {len(access_token)}")
    
    if not api_key or not access_token:
        print("❌ ERROR: KITE_API_KEY or KITE_ACCESS_TOKEN environment variables not found.")
        return False

    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"✅ SUCCESS: Kite authentication successful for user: {profile.get('user_id')}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Kite authentication failed: {e}")
        return False

def test_gsheet_auth():
    """Tests Google Sheets authentication using environment variables."""
    print("\n--- Testing Google Sheets Authentication ---")
    creds_json_str = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

    if not creds_json_str:
        print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS secret is missing.")
        return False

    try:
        creds_dict = json.loads(creds_json_str)
        gspread.service_account_from_dict(creds_dict)
        print("✅ SUCCESS: Google Sheets authentication successful.")
        return True
    except Exception as e:
        print(f"❌ ERROR: Google Sheets authentication failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting authentication tests...")
    
    kite_success = test_kite_auth()
    gsheet_success = test_gsheet_auth()

    if kite_success and gsheet_success:
        print("\n🎉 All authentication tests passed!")
        sys.exit(0)
    else:
        print("\n🔥 One or more authentication tests failed.")
        sys.exit(1)