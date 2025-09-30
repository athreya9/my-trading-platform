import os
import sys
from automate_token_generation import get_automated_access_token

def test_kite_automated_login():
    """Tests the automated Kite login process."""
    print("--- Testing Automated Kite Login ---")
    
    try:
        access_token = get_automated_access_token()
        
        if access_token:
            print("✅ SUCCESS: Successfully generated access token.")
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
        print("\n🎉 All authentication tests passed!")
        sys.exit(0)
    else:
        print("\n🔥 One or more authentication tests failed.")
        sys.exit(1)
