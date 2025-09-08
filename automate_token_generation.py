#!/usr/bin/env python3
"""
Automates the Kite Connect login process to generate a daily access token.

This script uses Selenium to perform a headless browser login, handles 2FA using
pyotp, captures the `request_token` from the redirect, and then generates the
final `access_token`.

Required Environment Variables:
- KITE_API_KEY: Your Kite application API key.
- KITE_API_SECRET: Your Kite application API secret.
- KITE_USER_ID: Your Zerodha user ID.
- KITE_PASSWORD: Your Zerodha password.
- KITE_TOTP_SECRET: The secret key from your 2FA authenticator app.
"""
import os
import sys
import hashlib
import requests
import pyotp
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, parse_qs

def generate_access_token(api_key, api_secret, request_token):
    """Generates an access token using the request token."""
    data = api_key + request_token + api_secret
    checksum = hashlib.sha256(data.encode("utf-8")).hexdigest()
    
    url = "https://api.kite.trade/session/token"
    payload = {
        "api_key": api_key,
        "request_token": request_token,
        "checksum": checksum
    }
    
    response = requests.post(url, data=payload)
    result = response.json()
    
    if response.status_code == 200 and "data" in result and "access_token" in result["data"]:
        return result["data"]["access_token"]
    else:
        raise Exception(f"Failed to generate access token: {result.get('message', 'Unknown error')}")

def main():
    """
    Automates the Kite login process to fetch a request_token and then
    generates and prints the daily access_token.
    """
    driver = None
    try:
        # --- Get Credentials from Environment Variables ---
        api_key = os.environ['KITE_API_KEY']
        api_secret = os.environ['KITE_API_SECRET']
        user_id = os.environ['KITE_USER_ID']
        password = os.environ['KITE_PASSWORD']
        totp_secret = os.environ['KITE_TOTP_SECRET']
        
        print("--- Starting Automated Token Generation ---")

        # --- Configure Selenium WebDriver ---
        options = webdriver.ChromeOptions()
        options.add_argument("--headless") # Run in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        print("Setting up Chrome WebDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # --- Step 1: Initial Login ---
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        print(f"Navigating to login URL...")
        driver.get(login_url)

        wait = WebDriverWait(driver, 20)
        
        # Enter User ID and Password
        print("Entering User ID and Password...")
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(user_id)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # --- Step 2: Handle 2FA/TOTP ---
        print("Generating and entering TOTP...")
        totp = pyotp.TOTP(totp_secret)
        totp_input = wait.until(EC.visibility_of_element_located((By.ID, "totp")))
        totp_input.send_keys(totp.now())
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        # --- Step 3: Capture the Request Token ---
        print("Waiting for redirect to capture request_token...")
        wait.until(EC.url_contains("request_token"))
        
        redirect_url = driver.current_url
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' not in query_params:
            raise Exception("Login succeeded but could not find 'request_token' in the redirect URL.")
            
        request_token = query_params['request_token'][0]
        print(f"Successfully captured request_token.")

        # --- Step 4: Generate the Access Token ---
        print("Generating access_token...")
        access_token = generate_access_token(api_key, api_secret, request_token)
        
        print("\n" + "="*60)
        print("✅ SUCCESS! NEW ACCESS TOKEN GENERATED")
        print("="*60)
        
        # Print the token to stdout so it can be captured by the GitHub Action
        print(access_token)
        
    except Exception as e:
        print(f"\n❌ An error occurred during automated token generation: {e}", file=sys.stderr)
        if driver:
            driver.save_screenshot('error_screenshot.png')
            print("Saved screenshot to 'error_screenshot.png' for debugging.", file=sys.stderr)
        sys.exit(1)
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()