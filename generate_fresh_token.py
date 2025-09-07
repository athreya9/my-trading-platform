#!/usr/bin/env python3
"""
Generate fresh token with new API key
"""
from kiteconnect import KiteConnect
import requests
import hashlib

# YOUR NEW CREDENTIALS
API_KEY = "is2u8bo7z8yjwhhr"
API_SECRET = "lczq9vywhz57obbjwj4wgtakqaa2s609"

# Generate fresh login URL
kite = KiteConnect(api_key=API_KEY)
login_url = kite.login_url()

print("1. Open this FRESH URL and login immediately:")
print(login_url)
print("\n2. After login, paste the redirected URL below:")

# Get fresh request token
redirected_url = input("Paste redirected URL: ").strip()

if "request_token=" in redirected_url:
    request_token = redirected_url.split("request_token=")[1].split("&")[0]
    print(f"✓ Fresh request token: {request_token}")
    
    # Generate access token with the CORRECT checksum method
    data_for_checksum = f"{API_KEY}{request_token}{API_SECRET}"
    checksum = hashlib.sha256(data_for_checksum.encode("utf-8")).hexdigest()
    
    response = requests.post("https://api.kite.trade/session/token", data={
        "api_key": API_KEY,
        "request_token": request_token,
        "checksum": checksum
    })
    
    result = response.json()
    print("API Response:", result)
    
    if 'access_token' in result:
        print(f"\n🎉 SUCCESS! New Access Token: {result['access_token']}")
        print("\nUpdate your GitHub KITE_ACCESS_TOKEN secret with this value!")
    else:
        print("❌ Failed to generate access token")

else:
    print("❌ No request_token found in the URL")