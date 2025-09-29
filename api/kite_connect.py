import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()

# --- Kite Connect Integration ---

# IMPORTANT: Store your API key and secret as environment variables.
# Do not hardcode them in your code.

API_KEY = os.environ.get("KITE_API_KEY")
API_SECRET = os.environ.get("KITE_API_SECRET")

def get_kite_connect_client():
    """Initializes the Kite Connect client."""
    if not API_KEY or not API_SECRET:
        raise ValueError("KITE_API_KEY and KITE_API_SECRET environment variables are not set.")
    
    kite = KiteConnect(api_key=API_KEY)
    return kite

def generate_login_url():
    """Generates the login URL for Kite Connect."""
    kite = get_kite_connect_client()
    login_url = kite.login_url()
    return login_url

def get_access_token(request_token):
    """Generates an access token from a request token."""
    kite = get_kite_connect_client()
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        return data["access_token"]
    except Exception as e:
        print(f"Error generating access token: {e}")
        return None
