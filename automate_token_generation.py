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

# --- Centralized Logging Setup ---
# This is a best practice to manage diagnostic output.
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
        
        logger.info("--- Starting Automated Token Generation ---")

        # --- Configure Selenium WebDriver ---
        options = webdriver.ChromeOptions()
        
        options.add_argument("--headless") # Run in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-pipe")
        
        logger.info("Setting up Chrome WebDriver...")
        service = Service(
            ChromeDriverManager().install(),
            log_path="chromedriver.log",
            service_args=["--verbose"]
        )
        driver = webdriver.Chrome(service=service, options=options)
        
        # --- Step 1: Initial Login ---
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        logger.info(f"Navigating to login URL...")
        driver.get(login_url)

        wait = WebDriverWait(driver, 60)
        
        # Enter User ID and Password
        logger.info("Entering User ID and Password...")
        user_id_input = wait.until(EC.visibility_of_element_located((By.ID, "userid")))
        user_id_input.clear()
        user_id_input.send_keys(user_id)

        time.sleep(0.5) # Small delay to mimic human behavior

        # Use a more robust, flexible selector for the password field to handle page changes.
        password_selector = (By.XPATH, "//input[@type='password' or @name='password' or @id='password']")
        password_input = wait.until(EC.visibility_of_element_located(password_selector))
        password_input.clear()
        password_input.send_keys(password) # Use send_keys for a more "human-like" interaction.

        # Wait for the submit button to be clickable before clicking
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))).click()
        
        # --- Add a check for immediate login errors on the first page ---
        try:
            # Wait for a very short time to see if an error message appears on the same page
            error_element = WebDriverWait(driver, 3).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "p.error, span.error"))
            )
            error_text = error_element.text.strip()
            raise Exception(f"Login failed on the first page. The website returned the error: '{error_text}'. This almost always means your KITE_USER_ID or KITE_PASSWORD secrets are incorrect. Please double-check them in your GitHub repository settings.")
        except TimeoutException:
            # No immediate error found, which is the expected good path. Continue.
            pass
        
        # --- Step 2: Handle 2FA/PIN/TOTP (with iframe detection) ---
        # The 2FA page can be a PIN or TOTP page, and it might be inside an iframe.
        # This logic attempts to handle these variations robustly.
        logger.info("Login submitted. Waiting for 2FA/PIN page...")
        pin_input = None
        switched_to_iframe = False

        try:
            # Use a flexible XPath to find the 2FA/PIN input field by common IDs/names.
            twofa_input_selector = (By.XPATH, "//input[@id='userid' or @id='pin' or @name='totp' or @id='totp']")

            # First, try to find the input field in the main document with a short timeout.
            pin_input = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(twofa_input_selector)
            )
            logger.info("2FA input found in main document.")
        except TimeoutException:
            # If not found, assume it's inside an iframe.
            logger.info("2FA input not in main document. Waiting for iframe...")
            try:
                iframe = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "iframe"))
                )
                driver.switch_to.frame(iframe)
                switched_to_iframe = True
                logger.info("Successfully switched to 2FA iframe.")
                
                # Now find the input field inside the iframe.
                pin_input = wait.until(EC.visibility_of_element_located(twofa_input_selector))
                logger.info("2FA input found in iframe.")
            except TimeoutException:
                # If it's still not found, then we have a problem.
                raise Exception("Login failed: Could not find 2FA/PIN input in main document or inside an iframe.")

        # We found the input field, now enter the TOTP.
        logger.info("Generating and entering TOTP...")
        totp = pyotp.TOTP(totp_secret)
        pin_input.send_keys(totp.now())
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        if switched_to_iframe:
            driver.switch_to.default_content()

        # --- Step 3: Capture the Request Token ---
        logger.info("Waiting for redirect to capture request_token or for an error message...")

        # Wait for either the successful redirect OR a visible error message on the page.
        # This prevents the script from timing out silently on a login failure.
        wait.until(
            EC.any_of(
                EC.url_contains("request_token"),
                EC.visibility_of_element_located((By.CSS_SELECTOR, "p.error, span.error"))
            )
        )

        # After the wait, check which condition was met.
        if "request_token" not in driver.current_url:
            # If the URL doesn't have the token, it means an error message appeared.
            error_element = driver.find_element(By.CSS_SELECTOR, "p.error, span.error")
            error_text = error_element.text.strip()
            raise Exception(f"Login failed after 2FA. The website returned the error: '{error_text}'. This often means your KITE_TOTP_SECRET is incorrect or your Zerodha account is set to use a PIN instead of TOTP.")
        
        logger.info("Redirect successful. Capturing request_token...")
        redirect_url = driver.current_url
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' not in query_params:
            raise Exception("Login succeeded but could not find 'request_token' in the redirect URL.")
            
        request_token = query_params['request_token'][0]
        logger.info(f"Successfully captured request_token.")

        # --- Step 4: Generate the Access Token ---
        logger.info("Generating access_token...")
        access_token = generate_access_token(api_key, api_secret, request_token)
        
        logger.info("\n" + "="*60)
        logger.info("✅ SUCCESS! NEW ACCESS TOKEN GENERATED")
        logger.info("="*60)
        
        # Print the token to stdout so it can be captured by the GitHub Action
        print(access_token)
        
    except Exception as e:
        logger.error(f"An error occurred during automated token generation: {e}", exc_info=True)
        if driver:
            driver.save_screenshot('error_screenshot.png')
            logger.info("Saved screenshot to 'error_screenshot.png' for debugging.")
            with open('error_page_source.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            logger.info("Saved page source to 'error_page_source.html' for debugging.")
        sys.exit(1)
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()