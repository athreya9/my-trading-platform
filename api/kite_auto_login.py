#!/usr/bin/env python3
import os
import pyotp
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class KiteAutoLogin:
    def __init__(self):
        self.api_key = os.getenv('KITE_API_KEY')
        self.api_secret = os.getenv('KITE_API_SECRET')
        self.user_id = os.getenv('KITE_USER_ID')
        self.password = os.getenv('KITE_PASSWORD')
        self.totp_secret = os.getenv('KITE_TOTP_SECRET')
        self.kite = KiteConnect(api_key=self.api_key)
        
    def generate_totp(self):
        """Generate TOTP using secret"""
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()
    
    def get_access_token(self):
        """Auto-generate access token using TOTP"""
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # Get login URL
            login_url = self.kite.login_url()
            driver.get(login_url)
            
            # Fill login form
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "userid"))
            )
            
            driver.find_element(By.ID, "userid").send_keys(self.user_id)
            driver.find_element(By.ID, "password").send_keys(self.password)
            driver.find_element(By.CLASS_NAME, "button-orange").click()
            
            # Handle TOTP
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "totp"))
            )
            
            totp_code = self.generate_totp()
            driver.find_element(By.ID, "totp").send_keys(totp_code)
            driver.find_element(By.CLASS_NAME, "button-orange").click()
            
            # Get request token from redirect URL
            WebDriverWait(driver, 10).until(lambda d: "request_token" in d.current_url)
            
            request_token = driver.current_url.split("request_token=")[1].split("&")[0]
            driver.quit()
            
            # Generate access token
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            
            # Save to .env
            with open('.env', 'r') as f:
                env_content = f.read()
            
            if 'KITE_ACCESS_TOKEN=' in env_content:
                # Update existing
                lines = env_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('KITE_ACCESS_TOKEN='):
                        lines[i] = f'KITE_ACCESS_TOKEN={access_token}'
                        break
                env_content = '\n'.join(lines)
            else:
                # Add new
                env_content += f'\nKITE_ACCESS_TOKEN={access_token}'
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            logger.info("✅ Access token generated and saved")
            return access_token
            
        except Exception as e:
            logger.error(f"Auto-login failed: {e}")
            return None

def get_fresh_kite_token():
    """Get fresh Kite access token"""
    auto_login = KiteAutoLogin()
    return auto_login.get_access_token()

if __name__ == "__main__":
    get_fresh_kite_token()