import numpy as np
import pandas as pd

class PerformanceMetrics:
    @staticmethod
    def calculate_all_metrics(portfolio, trade_log, confidence_level=0.95):
        """Calculate various return metrics"""
        portfolio['daily_return'] = portfolio['total'].pct_change()
        
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
        annual_return = portfolio['daily_return'].mean() * 252 * 100
        volatility = portfolio['daily_return'].std() * np.sqrt(252) * 100
        sharpe_ratio = portfolio['daily_return'].mean() / portfolio['daily_return'].std() * np.sqrt(252)
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(portfolio)
        var = PerformanceMetrics.calculate_var(portfolio, confidence_level)

        if not trade_log.empty:
            winning_trades = trade_log[trade_log['pnl'] > 0]
            losing_trades = trade_log[trade_log['pnl'] <= 0]
            total_trades = len(trade_log)
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profits = winning_trades['pnl'].sum()
            gross_losses = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
            avg_winning_trade = winning_trades['pnl'].mean()
            avg_losing_trade = losing_trades['pnl'].mean()
        else:
            win_rate = 0
            profit_factor = 0
            avg_winning_trade = 0
            avg_losing_trade = 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var': var,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade
        }
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(portfolio):
        """Calculate maximum drawdown"""
        portfolio['peak'] = portfolio['total'].cummax()
        portfolio['drawdown'] = (portfolio['total'] - portfolio['peak']) / portfolio['peak']
        return portfolio['drawdown'].min() * 100

    @staticmethod
    def calculate_var(portfolio, confidence_level=0.95):
        """Calculate Value at Risk (VaR)"""
        daily_returns = portfolio['daily_return'].dropna()
        return daily_returns.quantile(1 - confidence_level) * 100
