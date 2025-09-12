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
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
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
        
        print("--- Starting Automated Token Generation ---", file=sys.stderr)

        # --- Configure Selenium WebDriver ---
        options = webdriver.ChromeOptions()
        
        options.add_argument("--headless") # Run in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-pipe")
        
        print("Setting up Chrome WebDriver...", file=sys.stderr)
        service = Service(
            ChromeDriverManager().install(),
            log_path="chromedriver.log",
            service_args=["--verbose"]
        )
        driver = webdriver.Chrome(service=service, options=options)
        
        # --- Step 1: Initial Login ---
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        print(f"Navigating to login URL...", file=sys.stderr)
        driver.get(login_url)

        wait = WebDriverWait(driver, 60)
        
        # Enter User ID and Password
        print("Entering User ID and Password...", file=sys.stderr)
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(user_id)
        time.sleep(1) # Add a small delay
        # Use JavaScript to set the password to bypass potential interactability issues
        password_input = wait.until(EC.visibility_of_element_located((By.NAME, "password")))
        driver.execute_script("arguments[0].value = arguments[1];", password_input, password)

        # Wait for the submit button to be clickable before clicking
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))).click()
        
        # --- Step 2: Handle 2FA/PIN/TOTP (with iframe detection) ---
        # The 2FA page can be a PIN or TOTP page, and it might be inside an iframe.
        # This logic attempts to handle these variations robustly.
        print("Login submitted. Waiting for 2FA/PIN page...", file=sys.stderr)
        pin_input = None
        switched_to_iframe = False

        try:
            # Use a flexible XPath to find the 2FA/PIN input field by common IDs/names.
            twofa_input_selector = (By.XPATH, "//input[@id='userid' or @id='pin' or @name='totp' or @id='totp']")

            # First, try to find the input field in the main document with a short timeout.
            pin_input = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(twofa_input_selector)
            )
            print("2FA input found in main document.", file=sys.stderr)
        except TimeoutException:
            # If not found, assume it's inside an iframe.
            print("2FA input not in main document. Waiting for iframe...", file=sys.stderr)
            try:
                iframe = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "iframe"))
                )
                driver.switch_to.frame(iframe)
                switched_to_iframe = True
                print("Successfully switched to 2FA iframe.", file=sys.stderr)
                
                # Now find the input field inside the iframe.
                pin_input = wait.until(EC.visibility_of_element_located(twofa_input_selector))
                print("2FA input found in iframe.", file=sys.stderr)
            except TimeoutException:
                # If it's still not found, then we have a problem.
                raise Exception("Login failed: Could not find 2FA/PIN input in main document or inside an iframe.")

        # We found the input field, now enter the TOTP.
        print("Generating and entering TOTP...", file=sys.stderr)
        totp = pyotp.TOTP(totp_secret)
        pin_input.send_keys(totp.now())
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        if switched_to_iframe:
            driver.switch_to.default_content()

        # --- Step 3: Capture the Request Token ---
        print("Waiting for redirect to capture request_token...", file=sys.stderr)
        wait.until(EC.url_contains("request_token"))
        
        redirect_url = driver.current_url
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' not in query_params:
            raise Exception("Login succeeded but could not find 'request_token' in the redirect URL.")
            
        request_token = query_params['request_token'][0]
        print(f"Successfully captured request_token.", file=sys.stderr)

        # --- Step 4: Generate the Access Token ---
        print("Generating access_token...", file=sys.stderr)
        access_token = generate_access_token(api_key, api_secret, request_token)
        
        print("\n" + "="*60, file=sys.stderr)
        print("✅ SUCCESS! NEW ACCESS TOKEN GENERATED", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Print the token to stdout so it can be captured by the GitHub Action
        print(access_token)
        
    except Exception as e:
        print(f"\n❌ An error occurred during automated token generation: {e}", file=sys.stderr)
        if driver:
            driver.save_screenshot('error_screenshot.png')
            print("Saved screenshot to 'error_screenshot.png' for debugging.", file=sys.stderr)
            with open('error_page_source.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print("Saved page source to 'error_page_source.html' for debugging.", file=sys.stderr)
        sys.exit(1)
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()