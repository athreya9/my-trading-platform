import os
import sys
import hashlib
import requests

def generate_checksum(api_key, request_token, api_secret):
    """Generate SHA256 checksum for Kite Connect API"""
    data = api_key + request_token + api_secret
    checksum = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return checksum

def generate_access_token():
    print(" Generating your permanent access token...")
    
    # --- DEBUGGING: Check if environment variables are loaded ---
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    request_token = os.getenv('REQUEST_TOKEN')
    
    if not all([api_key, api_secret, request_token]):
        print("❌ ERROR: One or more environment variables are not set.", file=sys.stderr)
        print(f"KITE_API_KEY: {api_key}", file=sys.stderr)
        print(f"KITE_API_SECRET: {api_secret}", file=sys.stderr)
        print(f"REQUEST_TOKEN: {request_token}", file=sys.stderr)
        sys.exit(1)
    # --- END DEBUGGING ---
    
    checksum = generate_checksum(api_key, request_token, api_secret)
    
    url = "https://api.kite.trade/session/token"
    payload = {
        "api_key": api_key,
        "request_token": request_token,
        "checksum": checksum
    }
    
    response = requests.post(url, data=payload)
    result = response.json()
    
    if response.status_code == 200 and "data" in result and "access_token" in result["data"]:
        access_token = result["data"]["access_token"]
        print(f"✅ SUCCESS! Your access token is: {access_token}")
        # Optionally, save the access token to a file or environment variable
        with open("access_token.txt", "w") as f:
            f.write(access_token)
        print("Saved access token to access_token.txt")
    else:
        print(f"❌ ERROR: Failed to generate access token: {result.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    generate_access_token()