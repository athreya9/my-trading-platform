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

class url_or_iframe_with_token:
    """
    An expected condition for waiting for a URL to contain a substring,
    checking both the main URL and the src of any iframes. This is robust
    against redirects that happen inside a frame.
    """
    def __init__(self, substring):
        self.substring = substring
        self.captured_url = None

    def __call__(self, driver):
        try:
            # 1. Check the main browser URL
            if self.substring in driver.current_url:
                self.captured_url = driver.current_url
                print(f"DEBUG: Found token in main URL: {self.captured_url}", file=sys.stderr)
                return True
            
            # 2. Check the src of all iframes
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for frame in iframes:
                iframe_src = frame.get_attribute('src')
                if iframe_src and self.substring in iframe_src:
                    self.captured_url = iframe_src
                    print(f"DEBUG: Found token in iframe src: {self.captured_url}", file=sys.stderr)
                    return True
        except Exception:
            # Ignore exceptions like StaleElementReferenceException during page loads.
            return False
        return False

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
        # Use the new headless mode which is less detectable
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-pipe")
        # Add a common user-agent to avoid headless detection
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        # Disable automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # Further stealth options
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        
        print("Setting up Chrome WebDriver...", file=sys.stderr)
        service = Service(
            ChromeDriverManager().install(),
            log_path="chromedriver.log",
            service_args=["--verbose"]
        )
        driver = webdriver.Chrome(service=service, options=options)
        # Hide the "navigator.webdriver" flag to appear more human
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # --- Step 1: Initial Login ---
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        print(f"DEBUG: Navigating to URL: {login_url}", file=sys.stderr)  # <-- Add this line for debugging
        print(f"Navigating to login URL...", file=sys.stderr)
        driver.get(login_url)
        time.sleep(1) # Allow page to settle

        wait = WebDriverWait(driver, 60)
        
        # Enter User ID and Password
        print("Entering User ID and Password...", file=sys.stderr)
        # Wait for the user ID field to be visible before interacting
        wait.until(EC.visibility_of_element_located((By.ID, "userid"))).send_keys(user_id)
        # Wait for the password field to be visible and use its ID
        wait.until(EC.visibility_of_element_located((By.ID, "password"))).send_keys(password)

        # Find and click the submit button using JavaScript for robustness
        submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        driver.execute_script("arguments[0].click();", submit_button)
        
        # --- Step 2: Handle 2FA/TOTP and verify login success ---
        # This logic is designed to be robust against timing issues and iframes.
        pin_input = None
        try:
            # First, try a direct wait for the PIN input. This works if it's not in an iframe.
            print("Login submitted. Waiting for 2FA/PIN page to load...", file=sys.stderr)
            pin_input = wait.until(EC.element_to_be_clickable((By.ID, "pin")))
            print("Found PIN input directly in the main document.", file=sys.stderr)

        except TimeoutException:
            # If the direct wait fails, the element is likely inside an iframe.
            print("PIN input not found directly. Searching inside iframes...", file=sys.stderr)
            try:
                # Wait for at least one iframe to be present on the page.
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
                iframes = driver.find_elements(By.TAG_NAME, "iframe")
                print(f"Found {len(iframes)} iframe(s). Checking each for the PIN input.", file=sys.stderr)

                for index, frame_element in enumerate(iframes):
                    try:
                        print(f"Switching to iframe #{index}...", file=sys.stderr)
                        driver.switch_to.frame(frame_element)
                        # Look for the pin input inside this specific iframe with a shorter wait.
                        pin_input = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "pin")))
                        print(f"✅ Found PIN input inside iframe #{index}.", file=sys.stderr)
                        break  # Exit the loop once found
                    except TimeoutException:
                        print(f"PIN input not in iframe #{index}. Switching back.", file=sys.stderr)
                        driver.switch_to.default_content() # IMPORTANT: Switch back before next loop iteration
                        continue
            except TimeoutException:
                # This means no iframes were even found on the page after waiting.
                print("Search failed: No iframes were found on the page.", file=sys.stderr)

        # Final check: If pin_input is still None after all attempts, we must fail.
        if not pin_input:
            print("Login failed: Could not find PIN input in main document or inside any iframe.", file=sys.stderr)
            with open('error_page_source.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print("Saved page source to 'error_page_source.html' for debugging.", file=sys.stderr)
            raise Exception("The script timed out waiting for the 2FA/PIN input field. Please inspect the 'error_page_source.html' artifact.")

        # --- Now that pin_input is found, proceed with submission ---
        print("Generating and entering TOTP...", file=sys.stderr)
        totp = pyotp.TOTP(totp_secret)
        pin_input.send_keys(totp.now())
        time.sleep(1) # Brief pause
        totp_submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        driver.execute_script("arguments[0].click();", totp_submit_button)

        # --- Step 3: Capture the Request Token ---
        # IMPORTANT: Switch back to the main document before waiting for the redirect.
        driver.switch_to.default_content()
        print("Switched back to default content. Waiting for redirect...", file=sys.stderr)

        # Use the custom expected condition to safely wait for the token.
        print("Waiting for request_token in URL or iframe...", file=sys.stderr)
        token_condition = url_or_iframe_with_token("request_token")
        WebDriverWait(driver, 25).until(token_condition)
        
        redirect_url = token_condition.captured_url
        if not redirect_url:
                raise Exception("Condition passed but no URL was captured. This indicates a logic error.")

        print(f"Redirect successful. Captured URL for parsing: {redirect_url}", file=sys.stderr)

        # Parse the captured URL to extract the request token
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' not in query_params:
            raise Exception(f"Login succeeded but could not find 'request_token' in the redirect URL ({redirect_url}). Your frontend might be removing it too quickly.")
            
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
        print(f"\n❌ An error occurred during automated token generation.", file=sys.stderr)
        print(f"Exception Type: {type(e).__name__}", file=sys.stderr)
        print(f"Exception Details: {e}", file=sys.stderr)
        if driver:
            driver.save_screenshot('error_screenshot.png')
            print("Saved screenshot to 'error_screenshot.png' for debugging.", file=sys.stderr)
        sys.exit(1)
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()