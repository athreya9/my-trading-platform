import os
import logging
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# --- Kite Connect Integration ---

# Load environment variables from .env file.
# This makes the module self-contained and robust.
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")


def get_kite_connect_client():
    """Initializes the Kite Connect client."""
    if not API_KEY:
        logger.critical("KITE_API_KEY environment variable is not set. Cannot initialize client.")
        raise ValueError("KITE_API_KEY is not set.")
    
    kite = KiteConnect(api_key=API_KEY)
    return kite

def get_access_token(request_token):
    """Generates an access token from a request token."""
    if not API_SECRET:
        logger.critical("KITE_API_SECRET environment variable is not set. Cannot generate session.")
        raise ValueError("KITE_API_SECRET is not set.")
        
    kite = get_kite_connect_client()
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        return data["access_token"]
    except Exception as e:
        # The exception from kiteconnect is usually descriptive enough.
        logger.error(f"Error generating access token: {e}", exc_info=True)
        return None
