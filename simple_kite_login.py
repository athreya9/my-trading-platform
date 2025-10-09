#!/usr/bin/env python3
"""
Simple KITE Login - Manual token generation with TOTP
"""
import os
import pyotp
import logging
from kiteconnect import KiteConnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_totp():
    """Generate TOTP for Kite login"""
    totp_secret = '2W53IZK5OZBVTJNR6ABMRHGYCOPFHNVB'
    totp = pyotp.TOTP(totp_secret)
    return totp.now()

def manual_kite_setup():
    """Manual Kite setup with TOTP"""
    api_key = 'is2u8bo7z8yjwhhr'
    api_secret = 'lczq9vywhz57obbjwj4wgtakqaa2s609'
    
    # Generate TOTP
    twofa_code = generate_totp()
    
    print("🔐 KITE LOGIN SETUP")
    print("=" * 40)
    print(f"📱 TOTP Code: {twofa_code}")
    print(f"🔑 API Key: {api_key}")
    print()
    print("📋 MANUAL STEPS:")
    print("1. Go to: https://kite.trade/connect/login?api_key=" + api_key)
    print("2. Login with: QEM464 / @Sumanth$74724")
    print(f"3. Enter TOTP: {twofa_code}")
    print("4. Copy the request_token from URL after login")
    print()
    
    # Wait for user input
    request_token = input("📝 Paste request_token here: ").strip()
    
    if request_token:
        try:
            kite = KiteConnect(api_key=api_key)
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            print(f"✅ Access Token: {access_token}")
            
            # Update .env file
            update_env_token(access_token)
            
            # Test connection
            test_connection(access_token)
            
        except Exception as e:
            print(f"❌ Token generation failed: {e}")
    else:
        print("❌ No request token provided")

def update_env_token(access_token):
    """Update .env file with new token"""
    try:
        env_file = '.env'
        env_lines = []
        
        try:
            with open(env_file, 'r') as f:
                env_lines = f.readlines()
        except FileNotFoundError:
            pass
        
        # Update or add KITE_ACCESS_TOKEN
        token_updated = False
        for i, line in enumerate(env_lines):
            if line.startswith('KITE_ACCESS_TOKEN='):
                env_lines[i] = f'KITE_ACCESS_TOKEN={access_token}\n'
                token_updated = True
                break
        
        if not token_updated:
            env_lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
        
        with open(env_file, 'w') as f:
            f.writelines(env_lines)
        
        print("✅ Updated .env file")
        
    except Exception as e:
        print(f"❌ Failed to update .env: {e}")

def test_connection(access_token):
    """Test Kite connection"""
    try:
        kite = KiteConnect(api_key='is2u8bo7z8yjwhhr')
        kite.set_access_token(access_token)
        
        profile = kite.profile()
        print(f"✅ Connected: {profile['user_name']}")
        
        # Test live data
        quote = kite.quote(['NSE:NIFTY 50'])
        if 'NSE:NIFTY 50' in quote:
            price = quote['NSE:NIFTY 50']['last_price']
            print(f"✅ Live data: NIFTY at ₹{price}")
            return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def auto_totp_refresh():
    """Auto generate TOTP every 30 seconds"""
    import time
    
    print("🔄 Auto TOTP Generator (Press Ctrl+C to stop)")
    print("=" * 50)
    
    try:
        while True:
            twofa_code = generate_totp()
            current_time = time.strftime("%H:%M:%S")
            print(f"⏰ {current_time} | TOTP: {twofa_code}")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑 TOTP generator stopped")

if __name__ == "__main__":
    print("🚀 KITE SETUP OPTIONS")
    print("1. Manual token setup")
    print("2. Auto TOTP generator")
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "1":
        manual_kite_setup()
    elif choice == "2":
        auto_totp_refresh()
    else:
        print("❌ Invalid choice")