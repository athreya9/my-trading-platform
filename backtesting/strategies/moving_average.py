import pandas as pd
import numpy as np
from backtesting.core import Strategy

class MovingAverageCrossover(Strategy):
    """A simple moving average crossover strategy."""
    def __init__(self, short_window=40, long_window=100):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """Generates trading signals based on SMA crossover."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Calculate SMAs
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        # Create signals
        signals['signal'][self.short_window:] = \
            np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()

        return signals
