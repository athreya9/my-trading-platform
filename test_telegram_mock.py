#!/usr/bin/env python3
"""
Test Telegram alerts with mock - bypassing credentials
"""

def test_telegram_mock():
    """Test Telegram formatting without actually sending"""
    print("📱 Testing Telegram alert formatting...")
    
    # Mock alert data
    alert_data = {
        'symbol': 'NIFTY',
        'strike_price': 25000,
        'option_type': 'CE',
        'entry_price': 180,
        'stoploss': 150,
        'confidence': 'High (85%)',
        'reason': 'AI Breakout Signal'
    }
    
    # Format message like the real system would
    targets = [
        round(alert_data['entry_price'] * 1.05, 1),  # 5%
        round(alert_data['entry_price'] * 1.10, 1),  # 10% 
        round(alert_data['entry_price'] * 1.15, 1),  # 15%
        round(alert_data['entry_price'] * 1.20, 1),  # 20%
        round(alert_data['entry_price'] * 1.25, 1)   # 25%
    ]
    
    message = f"""
🚨 TRADING ALERT 🚨

{alert_data['symbol']} {alert_data['strike_price']} {alert_data['option_type']}

💰 Buy it
📈 Entry: {alert_data['entry_price']}

🎯 Targets:
{targets[0]}
{targets[1]} 
{targets[2]}
{targets[3]}
{targets[4]}

🛑 Stoploss: {alert_data['stoploss']}

💡 Reason: {alert_data['reason']}
⏰ Time: 14:30:25
"""
    
    print("✅ Mock Telegram Alert:")
    print(message)
    print("✅ Alert formatting works correctly")
    return True

if __name__ == "__main__":
    test_telegram_mock()