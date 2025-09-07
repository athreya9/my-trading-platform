#!/usr/bin/env python3
"""
Kite Connect Token Generator - FINAL WORKING VERSION
"""
import requests
import hashlib
 
# YOUR CREDENTIALS (UPDATED WITH FRESH TOKEN)
API_KEY = "qiuzgd7eavdehvre"
API_SECRET = "9tb2y2idkpsfhh64vypilvl9t5k8u42o"  # ✅ Your actual secret
REQUEST_TOKEN = "bmJHrW7CDk1yWmE5RM9XO15fS5zR9oIH"  # 🆕 FRESH TOKEN
 
def generate_checksum(api_key, request_token, api_secret):
    """Generate SHA256 checksum for Kite Connect API"""
    data = api_key + request_token + api_secret
    checksum = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return checksum

def generate_access_token():
    print("🚀 Generating your permanent access token...")
    print(f"🔑 API Key: {API_KEY}")
    print(f"🎫 Request Token: {REQUEST_TOKEN}")
    
    # Generate checksum
    checksum = generate_checksum(API_KEY, REQUEST_TOKEN, API_SECRET)
    print(f"🔐 Generated checksum: {checksum}")
    
    # Make the API request
    url = "https://api.kite.trade/session/token"
    
    data = {
        "api_key": API_KEY,
        "request_token": REQUEST_TOKEN,
        "checksum": checksum
    }
    
    print("\n📤 Sending request to Kite Connect API...")
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        print(f"📥 API Response: {result}")
        
        if response.status_code == 200 and "access_token" in result:
            access_token = result["access_token"]
            print("\n" + "="*60)
            print("✅ SUCCESS! PERMANENT ACCESS TOKEN GENERATED")
            print("="*60)
            print(f"🎯 Your Access Token: {access_token}")
            print("\n" + "="*60)
            print("📋 NEXT STEPS:")
            print("="*60)
            print("1. Add to GitHub Secrets:")
            print(f"   KITE_API_KEY = {API_KEY}")
            print(f"   KITE_ACCESS_TOKEN = {access_token}")
            print("\n2. Your Python script can now authenticate with:")
            print("   kite = KiteConnect(api_key=API_KEY)")
            print("   kite.set_access_token(ACCESS_TOKEN)")
            
            return access_token
            
        else:
            print(f"\n❌ API Error: {result}")
            if result.get('error_type') == 'InputException':
                print("💡 Check if API key is enabled in Zerodha Kite profile")
                    
    except Exception as e:
        print(f"\n❌ Failed to generate token: {e}")

if __name__ == "__main__":
    generate_access_token()
