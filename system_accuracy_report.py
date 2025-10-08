#!/usr/bin/env python3
"""
System Accuracy Analysis Report
"""
import json
from datetime import datetime
from api.kite_live_engine import KiteLiveEngine
from api.accurate_option_pricing import get_real_option_premium
from api.market_calibrated_pricing import get_market_calibrated_premium

def generate_accuracy_report():
    """Generate comprehensive accuracy report"""
    print("🔍 SYSTEM ACCURACY ANALYSIS REPORT")
    print("=" * 50)
    
    # Initialize engine
    engine = KiteLiveEngine()
    
    # Test instruments
    test_cases = [
        {'instrument': 'FINNIFTY', 'strike': 26700, 'type': 'PE'},
        {'instrument': 'NIFTY', 'strike': 25000, 'type': 'PE'},
        {'instrument': 'BANKNIFTY', 'strike': 56000, 'type': 'CE'}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📊 Testing {case['instrument']} {case['strike']} {case['type']}")
        
        # Get live price
        kite_symbol = engine.instruments.get(case['instrument'])
        price_data = engine.get_live_price(kite_symbol)
        
        if not price_data:
            print("❌ No price data available")
            continue
            
        spot = price_data['price']
        
        # Get different premium calculations
        real_premium = get_real_option_premium(case['instrument'], case['strike'], case['type'])
        calibrated_premium = get_market_calibrated_premium(spot, case['strike'], case['type'], case['instrument'])
        system_premium = engine.calculate_realistic_premium(spot, case['strike'], case['type'], case['instrument'])
        
        # Analysis
        result = {
            'instrument': case['instrument'],
            'strike': case['strike'],
            'option_type': case['type'],
            'spot_price': spot,
            'change_pct': price_data['change_pct'],
            'real_premium': real_premium,
            'calibrated_premium': calibrated_premium,
            'system_premium': system_premium,
            'accuracy': None,
            'status': 'Unknown'
        }
        
        if real_premium and real_premium > 0:
            error = abs(system_premium - real_premium) / real_premium * 100
            result['accuracy'] = f"{100-error:.1f}%"
            result['error'] = f"{error:.1f}%"
            
            if error < 15:
                result['status'] = '✅ Excellent'
            elif error < 30:
                result['status'] = '⚠️ Acceptable'
            else:
                result['status'] = '❌ Poor'
        
        results.append(result)
        
        # Print results
        print(f"  Spot Price: ₹{spot:.2f} ({price_data['change_pct']:+.2f}%)")
        print(f"  Real Premium: ₹{real_premium if real_premium else 'N/A'}")
        print(f"  System Premium: ₹{system_premium}")
        print(f"  Accuracy: {result.get('accuracy', 'N/A')}")
        print(f"  Status: {result['status']}")
    
    # Overall assessment
    print(f"\n📈 OVERALL SYSTEM ASSESSMENT")
    print("=" * 30)
    
    accurate_count = sum(1 for r in results if r['status'] == '✅ Excellent')
    total_count = len([r for r in results if r['real_premium']])
    
    if total_count > 0:
        accuracy_rate = accurate_count / total_count * 100
        print(f"Accuracy Rate: {accuracy_rate:.1f}%")
        
        if accuracy_rate >= 70:
            print("🎯 System Status: PRODUCTION READY")
        elif accuracy_rate >= 50:
            print("⚠️ System Status: NEEDS CALIBRATION")
        else:
            print("❌ System Status: REQUIRES MAJOR FIXES")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    poor_results = [r for r in results if r['status'] == '❌ Poor']
    if poor_results:
        print("1. Premium calculation needs improvement")
        print("2. Consider using real-time option chain data")
        print("3. Calibrate volatility parameters")
    else:
        print("1. System accuracy is acceptable")
        print("2. Monitor for market condition changes")
    
    # Save report
    with open('data/accuracy_report.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_tests': len(results),
                'accurate_count': accurate_count,
                'accuracy_rate': accuracy_rate if total_count > 0 else 0
            }
        }, f, indent=2)
    
    print(f"\n📄 Report saved to: data/accuracy_report.json")

if __name__ == "__main__":
    generate_accuracy_report()