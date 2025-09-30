# test_auth.py
import os
import sys
from dotenv import load_dotenv
from automate_token_generation import get_automated_access_token

# Load environment variables from .env file for local development.
load_dotenv()

def test_kite_automated_login():
    """Tests the automated Kite login process."""
    print("--- Testing Automated Kite Login ---")
    
    try:
        access_token = get_automated_access_token()
        
        if access_token:
            print(f"✅ SUCCESS: Successfully generated access token.")
            return True
        else:
            print("❌ ERROR: The script ran but did not return an access token.")
            return False

    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Starting authentication tests...")
    
    success = test_kite_automated_login()

    if success:
        print("
🎉 All authentication tests passed!")
        sys.exit(0)
    else:
        print("
🔥 One or more authentication tests failed.")
        sys.exit(1)
