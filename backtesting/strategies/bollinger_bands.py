import pandas as pd
import numpy as np
from backtesting.core import Strategy

class BollingerBandsStrategy(Strategy):
    """A simple Bollinger Bands strategy."""
    def __init__(self, window=20, num_std_dev=2):
        self.window = window
        self.num_std_dev = num_std_dev

    def generate_signals(self, data):
        """Generates trading signals based on Bollinger Bands."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Calculate Bollinger Bands
        signals['sma'] = data['Close'].rolling(window=self.window).mean()
        signals['std_dev'] = data['Close'].rolling(window=self.window).std()
        signals['upper_band'] = signals['sma'] + (signals['std_dev'] * self.num_std_dev)
        signals['lower_band'] = signals['sma'] - (signals['std_dev'] * self.num_std_dev)

        # Create signals
        signals['signal'] = np.where(data['Close'] < signals['lower_band'], 1.0, 0.0)
        signals['signal'] = np.where(data['Close'] > signals['upper_band'], -1.0, signals['signal'])

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()

        return signals
