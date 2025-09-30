
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

def get_automated_access_token():
    """
    Automates the Kite login process to fetch a request_token and then
    generates and returns the daily access_token.
    """
    driver = None
    try:
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

        password_selector = (By.XPATH, "//input[@type='password' or @name='password' or @id='password']")
        password_input = wait.until(EC.visibility_of_element_located(password_selector))
        password_input.clear()
        password_input.send_keys(password)

        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))).click()
        
        try:
            error_element = WebDriverWait(driver, 3).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "p.error, span.error"))
            )
            error_text = error_element.text.strip()
            raise Exception(f"Login failed on the first page. The website returned the error: '{error_text}'. This almost always means your KITE_USER_ID or KITE_PASSWORD secrets are incorrect. Please double-check them in your GitHub repository settings.")
        except TimeoutException:
            pass
        
        logger.info("Login submitted. Waiting for 2FA/PIN page...")
        pin_input = None
        switched_to_iframe = False

        try:
            twofa_input_selector = (By.XPATH, "//input[@id='userid' or @id='pin' or @name='totp' or @id='totp']")

            pin_input = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(twofa_input_selector)
            )
            logger.info("2FA input found in main document.")
        except TimeoutException:
            logger.info("2FA input not in main document. Waiting for iframe...")
            try:
                iframe = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "iframe"))
                )
                driver.switch_to.frame(iframe)
                switched_to_iframe = True
                logger.info("Successfully switched to 2FA iframe.")
                
                pin_input = wait.until(EC.visibility_of_element_located(twofa_input_selector))
                logger.info("2FA input found in iframe.")
            except TimeoutException:
                raise Exception("Login failed: Could not find 2FA/PIN input in main document or inside an iframe.")

        logger.info("Generating and entering TOTP...")
        totp = pyotp.TOTP(totp_secret)
        pin_input.send_keys(totp.now())
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        if switched_to_iframe:
            driver.switch_to.default_content()

        logger.info("Waiting for redirect to capture request_token or for an error message...")

        wait.until(
            EC.any_of(
                EC.url_contains("request_token"),
                EC.visibility_of_element_located((By.CSS_SELECTOR, "p.error, span.error"))
            )
        )

        if "request_token" not in driver.current_url:
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

        logger.info("Generating access_token...")
        access_token = generate_access_token(api_key, api_secret, request_token)
        
        logger.info("\n" + "="*60)
        logger.info("✅ SUCCESS! NEW ACCESS TOKEN GENERATED")
        logger.info("="*60)
        
        return access_token
        
    except Exception as e:
        logger.error(f"An error occurred during automated token generation: {e}", exc_info=True)
        if driver:
            driver.save_screenshot('error_screenshot.png')
            logger.info("Saved screenshot to 'error_screenshot.png' for debugging.")
            with open('error_page_source.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            logger.info("Saved page source to 'error_page_source.html' for debugging.")
        raise e
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    access_token = get_automated_access_token()
    if access_token:
        print(access_token)
    else:
        sys.exit(1)

