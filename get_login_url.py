from kiteconnect import KiteConnect
import os

# Your API key is retrieved from your GitHub secret.
api_key = os.getenv('KITE_API_KEY', 'is2u8bo7z8yjwhhr')
kite = KiteConnect(api_key=api_key)
login_url = kite.login_url()
print(login_url)