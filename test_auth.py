# test_auth.py
import os
import sys
from dotenv import load_dotenv
from api.kite_connect import get_access_token_with_totp, get_kite_connect_client

# Load environment variables from .env file for local development.
load_dotenv()

def test_kite_totp_auth():
    """Tests Kite Connect authentication using TOTP."""
    print("--- Testing Kite Connect Authentication with TOTP ---")
    
    try:
        access_token = get_access_token_with_totp()
        if not access_token:
            print("❌ ERROR: Failed to get access token with TOTP.")
            return False

        kite = get_kite_connect_client()
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"✅ SUCCESS: Kite authentication successful for user: {profile.get('user_id')}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Kite authentication with TOTP failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting authentication tests...")
    
    kite_success = test_kite_totp_auth()

    if kite_success:
        print("\n🎉 All authentication tests passed!")
        sys.exit(0)
    else:
        print("\n🔥 One or more authentication tests failed.")
        sys.exit(1)
