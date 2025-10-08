#!/usr/bin/env python3
import os
import pyotp
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def get_kite_access_token():
    """Simple manual token generation"""
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    totp_secret = os.getenv('KITE_TOTP_SECRET')
    
    kite = KiteConnect(api_key=api_key)
    
    # Generate TOTP
    totp = pyotp.TOTP(totp_secret)
    current_totp = totp.now()
    
    print(f"🔑 API Key: {api_key}")
    print(f"🔐 Current TOTP: {current_totp}")
    print(f"🌐 Login URL: {kite.login_url()}")
    print("\n📋 Manual Steps:")
    print("1. Open the login URL above")
    print("2. Login with your credentials")
    print(f"3. Use TOTP: {current_totp}")
    print("4. Copy the request_token from redirect URL")
    
    # Wait for manual input
    request_token = input("\n🔗 Enter request_token: ").strip()
    
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # Save to .env
            with open('.env', 'r') as f:
                content = f.read()
            
            if 'KITE_ACCESS_TOKEN=' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('KITE_ACCESS_TOKEN='):
                        lines[i] = f'KITE_ACCESS_TOKEN={access_token}'
                        break
                content = '\n'.join(lines)
            else:
                content += f'\nKITE_ACCESS_TOKEN={access_token}'
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print(f"✅ Access token saved: {access_token[:20]}...")
            return access_token
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    return None

if __name__ == "__main__":
    get_kite_access_token()