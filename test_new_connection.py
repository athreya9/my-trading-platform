#!/usr/bin/env python3
from kiteconnect import KiteConnect

# YOUR NEW CREDENTIALS
API_KEY = "is2u8bo7z8yjwhhr"
ACCESS_TOKEN = "2l8KxKsnXNaN5E04zjakn5KiPI63CVgY"

# Test connection
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# This should now work with market data!
profile = kite.profile()
print(f"✅ Connected as: {profile['user_name']}")

# Test market data access
quote = kite.quote("NSE:RELIANCE")
print(f"✅ Market data working! RELIANCE LTP: {quote['NSE:RELIANCE']['last_price']}")

print("🎉 Your Connect plan is working perfectly!")