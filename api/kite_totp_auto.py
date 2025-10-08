#!/usr/bin/env python3
import os
import pyotp
import requests
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def auto_generate_kite_token():
    """Auto-generate Kite token using TOTP"""
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    user_id = os.getenv('KITE_USER_ID')
    password = os.getenv('KITE_PASSWORD')
    totp_secret = os.getenv('KITE_TOTP_SECRET')
    
    kite = KiteConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret)
    
    try:
        # Generate current TOTP
        current_totp = totp.now()
        
        # Create session for login
        session = requests.Session()
        
        # Step 1: Get login page
        login_url = kite.login_url()
        response = session.get(login_url)
        
        # Step 2: Login with credentials
        login_data = {
            'user_id': user_id,
            'password': password
        }
        
        login_response = session.post('https://kite.zerodha.com/api/login', data=login_data)
        
        if login_response.status_code == 200:
            # Step 3: Submit TOTP
            twofa_data = {
                'user_id': user_id,
                'request_token': login_response.json().get('data', {}).get('request_token', ''),
                'twofa_value': current_totp,
                'twofa_type': 'totp'
            }
            
            twofa_response = session.post('https://kite.zerodha.com/api/twofa', data=twofa_data)
            
            if twofa_response.status_code == 200:
                request_token = twofa_response.json().get('data', {}).get('request_token', '')
                
                if request_token:
                    # Generate access token
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    access_token = data["access_token"]
                    
                    # Save to .env
                    set_key('.env', 'KITE_ACCESS_TOKEN', access_token)
                    
                    logger.info("✅ Kite token auto-generated successfully")
                    return access_token
        
        logger.error("❌ Auto-login failed")
        return None
        
    except Exception as e:
        logger.error(f"Auto-login error: {e}")
        return None

if __name__ == "__main__":
    token = auto_generate_kite_token()
    if token:
        print(f"✅ Token generated: {token[:20]}...")
    else:
        print("❌ Token generation failed")