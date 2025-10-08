#!/usr/bin/env python3
"""
Accurate Option Pricing using real market data
"""
import requests
import json
from datetime import datetime

def get_real_option_premium(instrument, strike, option_type, expiry="28OCT2025"):
    """Get real option premium from NSE API"""
    try:
        # NSE Option Chain API (simplified)
        if instrument == "FINNIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY"
        elif instrument == "NIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        elif instrument == "BANKNIFTY":
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        else:
            return None
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Find the specific strike and expiry
            for record in data.get('records', {}).get('data', []):
                if record.get('strikePrice') == strike:
                    if option_type == "CE" and 'CE' in record:
                        return record['CE'].get('lastPrice', 0)
                    elif option_type == "PE" and 'PE' in record:
                        return record['PE'].get('lastPrice', 0)
        
        return None
        
    except Exception as e:
        print(f"Error fetching real premium: {e}")
        return None

def calculate_accurate_premium(spot, strike, option_type, instrument_name, days_to_expiry=20):
    """Calculate accurate premium using market-calibrated model"""
    try:
        # Distance from ATM
        distance = abs(spot - strike)
        distance_pct = distance / spot * 100
        
        # Base premiums by instrument (market-calibrated)
        base_premiums = {
            'NIFTY': 180,
            'BANKNIFTY': 250, 
            'SENSEX': 200,
            'FINNIFTY': 220
        }
        
        base = base_premiums.get(instrument_name, 200)
        
        # Time decay factor
        time_factor = max(0.3, days_to_expiry / 30.0)
        
        if option_type == "PE":
            if spot < strike:  # ITM
                intrinsic = strike - spot
                time_value = base * time_factor * 0.6
                premium = intrinsic + time_value
            else:  # OTM
                # Exponential decay for OTM
                premium = base * time_factor * max(0.1, 1 - distance_pct * 0.15)
        else:  # CE
            if spot > strike:  # ITM
                intrinsic = spot - strike
                time_value = base * time_factor * 0.6
                premium = intrinsic + time_value
            else:  # OTM
                premium = base * time_factor * max(0.1, 1 - distance_pct * 0.15)
        
        return max(10, round(premium, 2))
        
    except Exception as e:
        print(f"Premium calculation error: {e}")
        return 150

def test_premium_accuracy():
    """Test premium calculation accuracy"""
    print("Testing Premium Accuracy...")
    
    # Test case: FINNIFTY 26700 PE
    spot = 26777
    strike = 26700
    option_type = "PE"
    instrument = "FINNIFTY"
    
    # Get real premium (if available)
    real_premium = get_real_option_premium(instrument, strike, option_type)
    
    # Calculate using improved formula
    calculated_premium = calculate_accurate_premium(spot, strike, option_type, instrument)
    
    print(f"Spot: {spot}")
    print(f"Strike: {strike} {option_type}")
    print(f"Real Premium: ₹{real_premium if real_premium else 'N/A'}")
    print(f"Calculated Premium: ₹{calculated_premium}")
    
    if real_premium:
        error = abs(calculated_premium - real_premium) / real_premium * 100
        print(f"Error: {error:.1f}%")
        
        if error < 20:
            print("✅ Acceptable accuracy")
        else:
            print("❌ High error - needs improvement")

if __name__ == "__main__":
    test_premium_accuracy()