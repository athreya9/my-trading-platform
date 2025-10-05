import datetime
from api.accurate_telegram_alerts import AccurateTelegramAlerts

def get_upcoming_expiries(kite, symbol):
    """Gets the upcoming expiry dates for a given symbol."""
    instruments = kite.instruments("NFO")
    expiries = sorted(list(set([i['expiry'] for i in instruments if i['name'] == symbol])))
    return expiries

def fetch_options_chain(kite, symbol):
    """Fetches the option chain for a given symbol."""
    try:
        expiries = get_upcoming_expiries(kite, symbol)
        if not expiries:
            return None
        # For simplicity, we'll use the first available expiry
        expiry_date = expiries[0]
        
        # The get_option_chain method does not exist in kiteconnect, so we will get all instruments and filter
        instruments = kite.instruments("NFO")
        option_chain = [i for i in instruments if i['name'] == symbol and i['expiry'] == expiry_date]
        return option_chain, expiry_date
    except Exception as e:
        print(f"Error fetching option chain for {symbol}: {e}")
        return None, None

def select_strike(chain, spot_price, sentiment_score):
    """Selects the optimal strike price based on the sentiment."""
    atm_strike = round(spot_price / 100) * 100
    bias = "bullish" if sentiment_score > 2 else "bearish" if sentiment_score < -2 else "neutral"
    if bias == "bullish":
        return f"{atm_strike} CE"
    elif bias == "bearish":
        return f"{atm_strike} PE"
    else:
        return None

def get_premium(chain, strike_symbol):
    """Gets the premium for a given strike symbol."""
    for instrument in chain:
        if instrument['tradingsymbol'] == strike_symbol:
            return instrument['last_price']
    return 0

def calculate_trade_levels(premium):
    """Calculates entry, stoploss, and targets based on the premium."""
    entry = premium
    stoploss = round(entry * 0.9)
    targets = [round(entry * 1.1), round(entry * 1.2), round(entry * 1.3)]
    return entry, stoploss, targets

def kelly_position_size(win_prob, win_ratio):
    """Calculates the Kelly Criterion for position sizing."""
    if win_ratio == 0:
        return 0.01
    return max(0.01, min(0.05, (win_prob * win_ratio - (1 - win_prob)) / win_ratio))

def estimate_confidence(sentiment_score, premium):
    """Estimates the confidence score for a signal."""
    # Simple logic for now, can be enhanced
    confidence = 50 + (sentiment_score * 5)
    if premium < 100:
        confidence += 10
    elif premium > 500:
        confidence -= 10
    return min(100, max(0, confidence))

def generate_option_signal(kite, symbol, sentiment_score):
    """Generates a complete option trading signal."""
    try:
        ltp_data = kite.ltp(f"NSE:{symbol}")
        spot_price = ltp_data[f"NSE:{symbol}"]['last_price']
        chain, expiry = fetch_options_chain(kite, symbol)
        if not chain:
            return None

        strike_symbol_name = select_strike(chain, spot_price, sentiment_score)
        if not strike_symbol_name:
            return None

        # Find the full instrument details for the selected strike
        selected_instrument = None
        for instrument in chain:
            if instrument['tradingsymbol'].endswith(strike_symbol_name):
                selected_instrument = instrument
                break
        
        if not selected_instrument:
            return None

        premium = get_premium(chain, selected_instrument['tradingsymbol'])
        entry, stoploss, targets = calculate_trade_levels(premium)
        confidence = estimate_confidence(sentiment_score, premium)
        position_size = kelly_position_size(confidence / 100, 2.0) # Assuming a 2:1 win/loss ratio for now

        return {
            "symbol": selected_instrument['tradingsymbol'],
            "type": "Call Option" if "CE" in strike_symbol_name else "Put Option",
            "expiry": expiry.strftime("%Y-%m-%d"),
            "entry": entry,
            "stoploss": stoploss,
            "targets": targets,
            "confidence": confidence,
            "hold_till": (datetime.datetime.now() + datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            "kelly_position_size": position_size,
            "reasoning": f"Sentiment score: {sentiment_score}, premium: ₹{premium}",
            "action": "Buy"
        }
    except Exception as e:
        print(f"Error generating option signal for {symbol}: {e}")
        return None

def send_option_alert(signal):
    """Formats and sends an option trading alert to Telegram."""
    if signal.get("status") != "live":
        return
    message = f"""
{signal['symbol']}
Type: {signal['type']}
Buy @ ₹{signal['entry']}
Stoploss: ₹{signal['stoploss']}
Targets: {', '.join([f'₹{t}' for t in signal['targets']])}
Expiry: {signal['expiry']}
Hold till: {signal['hold_till']}
Confidence: {signal['confidence']}% 
Kelly Position Size: {round(signal['kelly_position_size'] * 100)}%
Reason: {signal['reasoning']}
"""
    telegram_bot = AccurateTelegramAlerts()
    telegram_bot._send_telegram_message(message)
