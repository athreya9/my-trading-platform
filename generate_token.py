# generate_token.py
import os
import sys
from kiteconnect import KiteConnect

# --- Instructions ---
# This script helps you generate a one-time access token for Kite Connect.
# You only need to run this once to get your access token.
#
# How to use:
# 1. Open this file and fill in your `api_key` and `api_secret` below.
# 2. Run the script from your terminal: `python generate_token.py`
# 3. It will print a URL. Copy and paste this URL into your web browser.
# 4. Log in to your Zerodha account.
# 5. After successful login, you will be redirected to your `redirect_url` (e.g., https://google.com).
#    The URL in your browser's address bar will contain a `request_token`.
#    Example: https://your_redirect_url.com/?request_token=YOUR_REQUEST_TOKEN_HERE&action=login&status=success
# 6. Copy the `request_token` value from the URL.
# 7. Paste it into the terminal when this script prompts you for it.
# 8. The script will then generate and print your `access_token`.
# 9. Copy this `access_token` and save it as a GitHub Secret named `KITE_ACCESS_TOKEN`.
# ---

# Step 1: Enter your API Key and Secret here (from your Zerodha Kite Connect console)
api_key = "qiuzgd7eavdehvre"
api_secret = "9tb2y2idkpsfhh64vypilvl9t5k8u42o"

if api_key == "your_api_key_here" or api_secret == "your_api_secret_here":
    print("Error: Please open `generate_token.py` and replace 'your_api_key_here' and 'your_api_secret_here' with your actual Kite API credentials.")
    sys.exit(1)

# Step 2: Initialize KiteConnect
kite = KiteConnect(api_key=api_key)

# Step 3: Generate the Login URL
print("="*80)
print("1. Paste this URL into your browser and log in to your Zerodha account:")
print(f"\n   {kite.login_url()}\n")
print("2. After logging in, you will be redirected. Copy the 'request_token' from the URL.")
print("   It will look like: https://your_redirect_url.com/?request_token=AbC123...&action=login")
print("="*80)

# Step 4: Get the request_token from the user
try:
    request_token = input("\n3. Paste your request_token here and press Enter: ")
except KeyboardInterrupt:
    print("\n\nOperation cancelled by user.")
    sys.exit(0)

# Step 5: Generate the session and access_token
try:
    print("\nGenerating session with the provided request_token...")
    data = kite.generate_session(request_token.strip(), api_secret=api_secret)
    access_token = data["access_token"]

    print("\n" + "="*80)
    print("✅ SUCCESS! Your access_token has been generated.")
    print(f"\nYour access_token is: {access_token}\n")
    print("IMPORTANT: Copy this access_token and save it as a new GitHub Secret.")
    print("Create a new secret named: KITE_ACCESS_TOKEN")
    print("="*80)

except Exception as e:
    print(f"\n❌ ERROR: Could not generate session. Please check your credentials and request_token.")
    print(f"   Kite API Error: {e}")
    sys.exit(1)