#!/usr/bin/env python3
import os
import pyotp
import requests
import time
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class KiteTOTPLogin:
    def __init__(self):
        self.api_key = os.getenv('KITE_API_KEY')
        self.api_secret = os.getenv('KITE_API_SECRET') 
        self.user_id = os.getenv('KITE_USER_ID')
        self.password = os.getenv('KITE_PASSWORD')
        self.totp_secret = os.getenv('KITE_TOTP_SECRET')
        self.kite = KiteConnect(api_key=self.api_key)
        
    def generate_totp(self):
        """Generate current TOTP code"""
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()
    
    def login_and_get_token(self):
        """Login using TOTP and get access token"""
        try:
            # Step 1: Get login URL
            login_url = self.kite.login_url()
            
            # Step 2: Simulate login (simplified approach)
            session = requests.Session()
            
            # Get login page
            response = session.get(login_url)
            
            # Login with credentials
            login_data = {
                'user_id': self.user_id,
                'password': self.password
            }
            
            login_response = session.post('https://kite.zerodha.com/api/login', data=login_data)
            
            if 'request_token' in login_response.text:
                # Extract request token from response
                import re
                token_match = re.search(r'request_token=([^&]+)', login_response.text)
                if token_match:
                    request_token = token_match.group(1)
                    
                    # Generate session with TOTP
                    totp_code = self.generate_totp()
                    
                    # Complete 2FA
                    twofa_data = {
                        'request_token': request_token,
                        'twofa_value': totp_code
                    }
                    
                    session.post('https://kite.zerodha.com/api/twofa', data=twofa_data)
                    
                    # Generate access token
                    data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                    access_token = data["access_token"]
                    
                    # Update .env file
                    self.update_env_token(access_token)
                    
                    logger.info("✅ TOTP login successful, token saved")
                    return access_token
            
            logger.error("❌ TOTP login failed")
            return None
            
        except Exception as e:
            logger.error(f"TOTP login error: {e}")
            return None
    
    def update_env_token(self, token):
        """Update .env file with new token"""
        try:
            with open('.env', 'r') as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.startswith('KITE_ACCESS_TOKEN='):
                    lines[i] = f'KITE_ACCESS_TOKEN={token}\n'
                    updated = True
                    break
            
            if not updated:
                lines.append(f'KITE_ACCESS_TOKEN={token}\n')
            
            with open('.env', 'w') as f:
                f.writelines(lines)
                
        except Exception as e:
            logger.error(f"Error updating .env: {e}")

def get_kite_token_with_totp():
    """Get fresh Kite token using TOTP"""
    login = KiteTOTPLogin()
    return login.login_and_get_token()

if __name__ == "__main__":
    get_kite_token_with_totp()