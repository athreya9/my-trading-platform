#!/usr/bin/env python3
"""
DIRECT Kite Token Generator - No browser needed
"""
import requests
import webbrowser
import json

# Your credentials
API_KEY = "qiuzgd7eavdehvre"
API_SECRET = "9tb2y2idkpsfhh64vypilvl9t5k8u42o"

def generate_token_direct():
    # Step 1: Generate login URL
    login_url = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"
    print(f"\n1. Open this URL in your browser:")
    print(f"\n   {login_url}")
    print("\n2. Login with your Zerodha credentials")
    print("3. After login, you'll be redirected to a blank page")
    print("4. Copy the ENTIRE URL from your browser address bar")
    
    # Optional: auto-open the URL
    webbrowser.open(login_url)
    
    # Step 2: Get the request token from the redirected URL
    print("\n" + "="*60)
    redirected_url = input("Paste the entire URL after login: ").strip()
    
    # Extract request_token from URL
    if "request_token=" in redirected_url:
        request_token = redirected_url.split("request_token=")[1].split("&")[0]
        print(f"\n✓ Found request token: {request_token}")
    else:
        print("\n❌ No request_token found in the URL")
        return

    # Step 3: Generate session directly via API
    print("\n3. Generating access token...")
    url = "https://api.kite.trade/session/token"
    
    data = {
        "api_key": API_KEY,
        "request_token": request_token
    }
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        if response.status_code == 200 and "access_token" in result:
            access_token = result["access_token"]
            print("\n" + "="*60)
            print("✅ SUCCESS! Token generated via direct API")
            print("="*60)
            print(f"\nAccess Token: {access_token}")
            print(f"\nAPI Key: {API_KEY}")
            print("\nAdd these to your GitHub Secrets:")
            print(f"KITE_API_KEY: {API_KEY}")
            print(f"KITE_ACCESS_TOKEN: {access_token}")
            
        else:
            print(f"\n❌ API Error: {result}")
            
    except Exception as e:
        print(f"\n❌ Failed to generate token: {e}")

if __name__ == "__main__":
    generate_token_direct()
