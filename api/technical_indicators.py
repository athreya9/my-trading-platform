import pandas as pd

def calculate_rsi(data, period=14):
    delta = data.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u_avg = u.ewm(com=period - 1, min_periods=period).mean()
    d_avg = d.ewm(com=period - 1, min_periods=period).mean()
    rs = u_avg / d_avg
    return 100 - 100 / (1 + rs)

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_atr(high, low, close, period=14):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def get_technical_indicators(data):
    """Calculates all the technical indicators needed by the AI engine."""
    indicators = {}
    try:
        if len(data) > 14:
            indicators['rsi'] = calculate_rsi(data['Close']).iloc[-1]
        else:
            indicators['rsi'] = 50 # Neutral RSI

        if len(data) > 20:
            indicators['sma_20'] = calculate_sma(data['Close'], 20).iloc[-1]
            indicators['volume_avg'] = data['Volume'].rolling(window=20).mean().iloc[-1]
        else:
            indicators['sma_20'] = data['Close'].iloc[-1]
            indicators['volume_avg'] = data['Volume'].mean()

        if len(data) > 50:
            indicators['sma_50'] = calculate_sma(data['Close'], 50).iloc[-1]
        else:
            indicators['sma_50'] = data['Close'].iloc[-1]

        indicators['volume'] = data['Volume'].iloc[-1]
        
        macd, signal_line = calculate_macd(data['Close'])
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        
        indicators['atr'] = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
        indicators['atr_avg'] = indicators['atr'] # Placeholder for average ATR

    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        # Return default values if calculation fails
        return {
            'rsi': 50,
            'sma_20': data['Close'].iloc[-1] if not data.empty else 0,
            'sma_50': data['Close'].iloc[-1] if not data.empty else 0,
            'volume': 0,
            'volume_avg': 0,
            'macd': 0,
            'macd_signal': 0,
            'atr': 0,
            'atr_avg': 0
        }
    return indicators
