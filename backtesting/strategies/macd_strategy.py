import pandas as pd
import numpy as np
from backtesting.core import Strategy

class MACDStrategy(Strategy):
    """A simple MACD crossover strategy."""
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def generate_signals(self, data):
        """Generates trading signals based on MACD crossover."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Calculate MACD
        short_ema = data['Close'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=self.long_window, adjust=False).mean()
        signals['macd'] = short_ema - long_ema
        signals['signal_line'] = signals['macd'].ewm(span=self.signal_window, adjust=False).mean()

        # Create signals
        signals['signal'] = np.where(signals['macd'] > signals['signal_line'], 1.0, 0.0)

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()

        return signals
