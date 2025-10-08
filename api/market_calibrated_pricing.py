#!/usr/bin/env python3
"""
Market-calibrated option pricing based on real observations
"""

def get_market_calibrated_premium(spot, strike, option_type, instrument_name):
    """Get premium based on real market observations"""
    
    # Distance from spot
    distance = abs(spot - strike)
    distance_pct = distance / spot * 100
    
    # Market-observed premium patterns (₹)
    premium_tables = {
        'FINNIFTY': {
            'ATM': 220,      # At-the-money base
            'ITM_100': 280,  # 100 points ITM
            'OTM_100': 160,  # 100 points OTM
            'OTM_200': 120,  # 200 points OTM
            'OTM_300': 80    # 300+ points OTM
        },
        'NIFTY': {
            'ATM': 180,
            'ITM_100': 240,
            'OTM_100': 130,
            'OTM_200': 90,
            'OTM_300': 60
        },
        'BANKNIFTY': {
            'ATM': 280,
            'ITM_100': 350,
            'OTM_100': 200,
            'OTM_200': 150,
            'OTM_300': 100
        }
    }
    
    table = premium_tables.get(instrument_name, premium_tables['NIFTY'])
    
    # Determine premium based on moneyness
    if option_type == "PE":
        if spot < strike:  # ITM
            intrinsic = strike - spot
            if distance <= 50:
                time_value = table['ATM'] * 0.4
            elif distance <= 100:
                time_value = table['ITM_100'] * 0.3
            else:
                time_value = table['ITM_100'] * 0.2
            premium = intrinsic + time_value
        else:  # OTM
            if distance <= 100:
                premium = table['OTM_100']
            elif distance <= 200:
                premium = table['OTM_200']
            else:
                premium = table['OTM_300']
    else:  # CE
        if spot > strike:  # ITM
            intrinsic = spot - strike
            if distance <= 50:
                time_value = table['ATM'] * 0.4
            elif distance <= 100:
                time_value = table['ITM_100'] * 0.3
            else:
                time_value = table['ITM_100'] * 0.2
            premium = intrinsic + time_value
        else:  # OTM
            if distance <= 100:
                premium = table['OTM_100']
            elif distance <= 200:
                premium = table['OTM_200']
            else:
                premium = table['OTM_300']
    
    return round(premium, 2)

def test_calibrated_pricing():
    """Test the calibrated pricing"""
    print("Testing Market-Calibrated Pricing...")
    
    # Test case: FINNIFTY 26700 PE (spot=26777)
    spot = 26777
    strike = 26700
    option_type = "PE"
    instrument = "FINNIFTY"
    
    premium = get_market_calibrated_premium(spot, strike, option_type, instrument)
    
    print(f"Spot: {spot}")
    print(f"Strike: {strike} {option_type}")
    print(f"Distance: {abs(spot-strike)} points")
    print(f"Calculated Premium: ₹{premium}")
    print(f"Expected Range: ₹200-280 (market observation)")
    
    if 200 <= premium <= 280:
        print("✅ Within expected range")
    else:
        print("⚠️ Outside expected range")

if __name__ == "__main__":
    test_calibrated_pricing()