#!/usr/bin/env python3
"""
Regenerate Kite Connect Access Token with NEW API Key
"""
from kiteconnect import KiteConnect
import webbrowser

# YOUR NEW CREDENTIALS (from Connect plan)
API_KEY = "is2u8bo7z8yjwhhr"  # ← YOUR NEW API KEY
API_SECRET = "lczq9vywhz57obbjwj4wgtakqaa2s609"  # ← YOUR NEW API SECRET

def generate_new_token():
    print(" Generating new access token with your CONNECT plan API key...")
    print(f"API Key: {API_KEY}")
    
    # Initialize KiteConnect with NEW API key
    kite = KiteConnect(api_key=API_KEY)
    
    # Generate login URL
    login_url = kite.login_url()
    print(f"\n1. Open this URL in your browser:")
    print(f"\n   {login_url}")
    print("\n2. Login with your Zerodha credentials")
    print("3. After login, copy the ENTIRE URL from the address bar")
    
    # Auto-open the browser
    webbrowser.open(login_url)
    
    # Get the request token from the redirected URL
    print("\n" + "="*60)
    redirected_url = input("4. Paste the redirected URL here: ").strip()
    
    if "request_token=" in redirected_url:
        request_token = redirected_url.split("request_token=")[1].split("&")[0]
        print(f"✅ Request token: {request_token}")
        
        # Generate session and get access token
        try:
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            access_token = data["access_token"]
            
            print("\n" + "="*60)
            print(" SUCCESS! NEW ACCESS TOKEN GENERATED")
            print("="*60)
            print(f"Access Token: {access_token}")
            print(f"API Key: {API_KEY}")
            print("\n Add these to your GitHub Secrets:")
            print(f"KITE_API_KEY: {API_KEY}")
            print(f"KITE_ACCESS_TOKEN: {access_token}")
            
            return access_token
            
        except Exception as e:
            print(f"❌ Error generating access token: {e}")
            return None
    else:
        print("❌ No request_token found in the URL")
        return None

if __name__ == "__main__":
    generate_new_token()
