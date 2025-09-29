# test_auth.py
import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file for local development.
load_dotenv()

def test_kite_automated_login():
    """Tests the automated Kite login process."""
    print("--- Testing Automated Kite Login ---")
    
    try:
        # Run the automate_token_generation.py script as a subprocess
        result = subprocess.run(
            [sys.executable, "automate_token_generation.py"],
            capture_output=True,
            text=True,
            check=True
        )
        
        access_token = result.stdout.strip()
        
        if access_token:
            print(f"✅ SUCCESS: Successfully generated access token.")
            return True
        else:
            print("❌ ERROR: The script ran but did not return an access token.")
            print("Stderr:", result.stderr)
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: The automate_token_generation.py script failed with exit code {e.returncode}.")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
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