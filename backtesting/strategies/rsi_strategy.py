import pandas as pd
import numpy as np
from backtesting.core import Strategy

def rsi(data, period=14):
    delta = data.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u_avg = u.ewm(com=period - 1, min_periods=period).mean()
    d_avg = d.ewm(com=period - 1, min_periods=period).mean()
    rs = u_avg / d_avg
    return 100 - 100 / (1 + rs)

class RSIStrategy(Strategy):
    """A simple RSI strategy."""
    def __init__(self, period=14, oversold_threshold=30, overbought_threshold=70):
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self, data):
        """Generates trading signals based on RSI."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Calculate RSI
        data['rsi'] = rsi(data['Close'], self.period)

        # Create signals
        signals['signal'] = np.where(data['rsi'] < self.oversold_threshold, 1.0, 0.0)
        signals['signal'] = np.where(data['rsi'] > self.overbought_threshold, -1.0, signals['signal'])

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()

        return signals
