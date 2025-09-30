
import time
from api.run_live_bot import run_live_bot, is_market_open
import os

if __name__ == '__main__':
    while True:
        if is_market_open():
            os.environ['MODE'] = 'live'
            run_live_bot()
        else:
            print(f"Market is closed. Waiting for the market to open.")
        time.sleep(60)
