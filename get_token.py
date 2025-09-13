from kiteconnect import KiteConnect

# Replace with your actual Kite API key
KITE_API_KEY = "2u8bo7z8yjwhhr"

kite = KiteConnect(api_key=KITE_API_KEY)
login_url = kite.login_url()
print(f"Please visit this URL to log in and get your request token:\n{login_url}")