# market_calendar.py
import pandas as pd
from datetime import datetime, time
import holidays

class IndianMarketCalendar:
    def __init__(self):
        self.indian_holidays = holidays.India()
        self.market_open = time(9, 15)  # 9:15 AM IST
        self.market_close = time(15, 30) # 3:30 PM IST
    
    def is_market_open(self):
        """Check if Indian market is currently open"""
        now = datetime.now()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if holiday
        if now.date() in self.indian_holidays:
            return False
        
        # Check market hours
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def get_next_market_day(self):
        """Get the next trading day"""
        next_day = datetime.now() + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def is_trading_day(self, date):
        """Check if a specific date is a trading day"""
        if date.weekday() >= 5:  # Weekend
            return False
        if date.date() in self.indian_holidays:
            return False
        return True

# Integrated into your main bot
def should_run_trading_bot():
    """Enhanced market hours check"""
    calendar = IndianMarketCalendar()
    
    if not calendar.is_market_open():
        print("⏸️  Market closed - skipping trading execution")
        
        # But still run data collection if it's a trading day
        if calendar.is_trading_day(datetime.now()):
            print(" Running data collection only...")
            return "data_collection_only"
        else:
            return "holiday"
    
    return "full_trading"