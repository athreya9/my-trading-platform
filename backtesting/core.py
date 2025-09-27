import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# --- Base Strategy Class ---
class Strategy:
    """Base class for all trading strategies."""
    def generate_signals(self, data):
        """Generates trading signals for the given data."""
        raise NotImplementedError("Should implement generate_signals()")

from backtesting.metrics import PerformanceMetrics

# --- Backtesting Engine ---
class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = {}
        self.strategies = {}
        sns.set_style("darkgrid")

    def register_strategy(self, name, strategy):
        """Register a trading strategy."""
        if isinstance(strategy, Strategy):
            self.strategies[name] = strategy
        else:
            raise ValueError("Strategy must be an instance of the Strategy class")

    def load_data(self, symbol, start_date, end_date, source='yfinance'):
        """Load historical data for backtesting"""
        if source == 'yfinance':
            return self._load_yfinance_data(symbol, start_date, end_date)
        elif source == 'kite':
            return self._load_kite_data(symbol, start_date, end_date)
    
    def _load_yfinance_data(self, symbol, start_date, end_date):
        """Load data from yfinance (free)"""
        import yfinance as yf
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None

    def _load_kite_data(self, symbol, start_date, end_date):
        """Placeholder for loading data from Kite Connect API."""
        print("Note: _load_kite_data is a placeholder. You need to implement your Kite API logic here.")
        # Example: Load from a local CSV file
        # try:
        #     df = pd.read_csv(f'{symbol}.csv', index_col='date', parse_dates=True)
        #     return df[(df.index >= start_date) & (df.index <= end_date)]
        # except FileNotFoundError:
        #     return None
        return None

    def run_backtest(self, strategy_name, symbol, start_date, end_date, source='yfinance'):
        """Run a backtest for a given strategy and symbol."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not registered.")

        data = self.load_data(symbol, start_date, end_date, source)
        if data is None or data.empty:
            print("Could not load data for backtest.")
            return

        strategy = self.strategies[strategy_name]
        signals = strategy.generate_signals(data)

        # --- Execute Trades ---
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[symbol] = signals['signal']
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
        
        # Adjust cash for trades, commission and slippage
        pos_diff = positions.diff()
        trade_costs = (pos_diff.abs().multiply(data['Close'], axis=0) * (self.commission + self.slippage)).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum() - trade_costs.cumsum()
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()

        # --- Generate Trade Log ---
        trades = []
        position = 0
        entry_price = 0
        entry_date = None

        for i in range(len(signals)):
            if signals['positions'].iloc[i] == 1: # Buy signal
                if position == 0:
                    position = 1
                    entry_price = data['Close'].iloc[i] * (1 + self.slippage)
                    entry_date = data.index[i]
            elif signals['positions'].iloc[i] == -1: # Sell signal
                if position == 1:
                    position = 0
                    exit_price = data['Close'].iloc[i] * (1 - self.slippage)
                    exit_date = data.index[i]
                    pnl = exit_price - entry_price - (entry_price * self.commission) - (exit_price * self.commission)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })

        trade_log = pd.DataFrame(trades)

        self.results[strategy_name] = {'portfolio': portfolio, 'trade_log': trade_log}
        print(f"Backtest completed for strategy: {strategy_name}")

    def calculate_performance_metrics(self, strategy_name):
        """Calculate performance metrics for a backtest."""
        if strategy_name not in self.results:
            raise ValueError(f"No backtest results found for strategy '{strategy_name}'.")

        results = self.results[strategy_name]
        portfolio = results['portfolio']
        trade_log = results['trade_log']
        
        return PerformanceMetrics.calculate_all_metrics(portfolio, trade_log)

    def plot_results(self, strategy_name):
        """Plot the results of a backtest."""
        if strategy_name not in self.results:
            raise ValueError(f"No backtest results found for strategy '{strategy_name}'.")

        portfolio = self.results[strategy_name]['portfolio']

        plt.figure(figsize=(12, 8))
        plt.title(f'Portfolio Value Over Time: {strategy_name}')
        plt.plot(portfolio['total'], label='Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()
