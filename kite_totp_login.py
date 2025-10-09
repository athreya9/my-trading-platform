#!/usr/bin/env python3
"""
KITE TOTP Login - Real account login with TOTP
"""
import os
import time
import logging
import requests
import pyotp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_kite_access_token():
    """Get fresh Kite access token using TOTP"""
    try:
        # Kite credentials
        api_key = 'is2u8bo7z8yjwhhr'
        api_secret = 'lczq9vywhz57obbjwj4wgtakqaa2s609'
        user_id = 'QEM464'
        password = '@Sumanth$74724'
        totp_secret = '2W53IZK5OZBVTJNR6ABMRHGYCOPFHNVB'
        
        # Generate TOTP
        totp = pyotp.TOTP(totp_secret)
        twofa_code = totp.now()
        
        logger.info(f"🔐 Generated TOTP: {twofa_code}")
        
        # Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        # Start browser
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)
        
        try:
            # Step 1: Go to Kite login
            login_url = f"https://kite.trade/connect/login?api_key={api_key}"
            driver.get(login_url)
            logger.info("📱 Opened Kite login page")
            
            # Step 2: Enter user ID
            user_input = wait.until(EC.presence_of_element_located((By.ID, "userid")))
            user_input.send_keys(user_id)
            
            # Step 3: Enter password
            password_input = driver.find_element(By.ID, "password")
            password_input.send_keys(password)
            
            # Step 4: Click login
            login_button = driver.find_element(By.CLASS_NAME, "button-orange")
            login_button.click()
            logger.info("🔑 Submitted login credentials")
            
            # Step 5: Wait for TOTP page and enter TOTP
            time.sleep(3)
            totp_input = wait.until(EC.presence_of_element_located((By.ID, "totp")))
            totp_input.send_keys(twofa_code)
            
            # Step 6: Submit TOTP
            totp_button = driver.find_element(By.CLASS_NAME, "button-orange")
            totp_button.click()
            logger.info(f"🔐 Submitted TOTP: {twofa_code}")
            
            # Step 7: Wait for redirect and extract request token
            time.sleep(5)
            current_url = driver.current_url
            
            if "request_token=" in current_url:
                request_token = current_url.split("request_token=")[1].split("&")[0]
                logger.info(f"✅ Got request token: {request_token[:20]}...")
                
                # Step 8: Generate access token
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=api_key)
                
                data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data["access_token"]
                
                logger.info(f"✅ Generated access token: {access_token[:20]}...")
                
                # Update .env file
                env_file = '.env'
                env_lines = []
                
                try:
                    with open(env_file, 'r') as f:
                        env_lines = f.readlines()
                except FileNotFoundError:
                    pass
                
                # Update or add KITE_ACCESS_TOKEN
                token_updated = False
                for i, line in enumerate(env_lines):
                    if line.startswith('KITE_ACCESS_TOKEN='):
                        env_lines[i] = f'KITE_ACCESS_TOKEN={access_token}\n'
                        token_updated = True
                        break
                
                if not token_updated:
                    env_lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
                
                with open(env_file, 'w') as f:
                    f.writelines(env_lines)
                
                logger.info("✅ Updated .env with new access token")
                return access_token
                
            else:
                logger.error("❌ No request token found in URL")
                return None
                
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"❌ TOTP login failed: {e}")
        return None

def test_kite_connection(access_token):
    """Test Kite connection with access token"""
    try:
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key='is2u8bo7z8yjwhhr')
        kite.set_access_token(access_token)
        
        profile = kite.profile()
        logger.info(f"✅ Kite connected: {profile['user_name']}")
        
        # Test live data
        quote = kite.quote(['NSE:NIFTY 50'])
        if 'NSE:NIFTY 50' in quote:
            price = quote['NSE:NIFTY 50']['last_price']
            logger.info(f"✅ Live data: NIFTY at ₹{price}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting KITE TOTP login...")
    
    access_token = get_kite_access_token()
    if access_token:
        if test_kite_connection(access_token):
            logger.info("🎉 KITE login successful - ready for trading!")
        else:
            logger.error("❌ Connection test failed")
    else:
        logger.error("❌ Failed to get access token")