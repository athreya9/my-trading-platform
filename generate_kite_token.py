#!/usr/bin/env python3
"""
Simple Kite Token Generator
Run this daily or when token expires
"""

import os
import pyotp
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key

load_dotenv()

def generate_fresh_token():
    """Generate fresh Kite access token"""
    
    # Your credentials
    api_key = "is2u8bo7z8yjwhhr"
    api_secret = "lczq9vywhz57obbjwj4wgtakqaa2s609"
    totp_secret = "2W53IZK5OZBVTJNR6ABMRHGYCOPFHNVB"
    
    # Initialize Kite
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    print(f"🔗 Login URL: {login_url}")
    
    # Generate current TOTP
    totp = pyotp.TOTP(totp_secret)
    current_totp = totp.now()
    print(f"🔑 Current TOTP: {current_totp}")
    
    # Get request token from user
    print("\n📋 Steps:")
    print("1. Open the login URL above")
    print("2. Login with your credentials")
    print(f"3. Enter TOTP: {current_totp}")
    print("4. Copy the request_token from the redirected URL")
    
    request_token = input("\n🎯 Enter request_token: ").strip()
    
    if request_token:
        try:
            # Generate session
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # Save to .env
            set_key('.env', 'KITE_ACCESS_TOKEN', access_token)
            
            # Save to access_token.txt
            with open('access_token.txt', 'w') as f:
                f.write(access_token)
            
            print(f"\n✅ Token generated successfully!")
            print(f"🔑 Access Token: {access_token}")
            print(f"💾 Saved to .env and access_token.txt")
            
            return access_token
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    else:
        print("❌ No request token provided")
        return None

if __name__ == "__main__":
    print("🚀 Kite Token Generator")
    print("=" * 40)
    generate_fresh_token()