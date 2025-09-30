import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import seaborn as sns
from scipy import stats

class PerformanceAnalyzer:
    
    def __init__(self, results_df: pd.DataFrame, trades_df: pd.DataFrame, 
                 initial_capital: float = 100000):
        self.results_df = results_df
        self.trades_df = trades_df
        self.initial_capital = initial_capital
        
    def calculate_all_metrics(self) -> Dict:
        '''Calculate comprehensive performance metrics'''
        metrics = {}
        metrics.update(self._calculate_return_metrics())
        metrics.update(self._calculate_risk_metrics())
        return metrics
    
    def _calculate_return_metrics(self) -> Dict:
        '''Calculate return-based metrics'''
        final_value = self.results_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        trading_days = len(self.results_df)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'final_portfolio_value': final_value,
            'total_profit': final_value - self.initial_capital,
        }
    
    def calculate_pair_metrics(self) -> Dict:

        if self.trades_df.empty:
            return {}

        # Check if 'pair' column exists
        if 'pair' not in self.trades_df.columns:
            print("Warning: 'pair' column not found in trades_df")
            return {}

        pair_metrics = {}
        pairs = self.trades_df['pair'].unique()

        for pair in pairs:
            pair_trades = self.trades_df[self.trades_df['pair'] == pair].copy()

            # Get exit trades for this pair
            exit_trades = pair_trades[pair_trades['action'] == 'EXIT']

            if exit_trades.empty:
                continue

            # Calculate daily PnL series for this pair
            # Group by date and sum PnL
            if 'date' in exit_trades.columns:
                daily_pnl = exit_trades.groupby('date')['pnl'].sum()
            else:
                # If no date column, use the exit_trades directly
                daily_pnl = exit_trades['pnl']

            # Calculate cumulative PnL for drawdown calculation
            cumulative_pnl = daily_pnl.cumsum() if len(daily_pnl) > 1 else pd.Series([daily_pnl.sum()])

            # Max Drawdown calculation
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / running_max
            max_drawdown = drawdown.min()

            # Sharpe Ratio calculation
            if len(daily_pnl) > 1:
                daily_returns = daily_pnl / self.initial_capital  # Normalize by capital
                mean_return = daily_returns.mean()
                std_return = daily_returns.std()

                # Annualized Sharpe Ratio (assuming 252 trading days)
                sharpe_ratio = (mean_return * np.sqrt(252)) / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0

            # Additional pair statistics
            total_pnl = exit_trades['pnl'].sum()
            num_trades = len(exit_trades)
            winning_trades = len(exit_trades[exit_trades['pnl'] > 0])
            win_rate = winning_trades / num_trades if num_trades > 0 else 0

            pair_metrics[pair] = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_pnl': total_pnl,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100
            }

        return pair_metrics
    
    def print_pair_metrics(self):
            pair_metrics = self.calculate_pair_metrics()

            if not pair_metrics:
                return

            print("=" * 80)
            print("PER-PAIR PERFORMANCE METRICS")
            print("=" * 80)

            for pair, metrics in pair_metrics.items():
                print(f"\n{pair}:")
                print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
                print(f"  Max Drawdown:        $({metrics['max_drawdown_pct']:.2f}%)")
                print(f"  Total PnL:           ${metrics['total_pnl']:>8,.2f}")
                print(f"  Number of Trades:    {metrics['num_trades']:>8}")
                print(f"  Win Rate:            {metrics['win_rate_pct']:>8.2f}%")

            print("=" * 80)
    def _calculate_risk_metrics(self) -> Dict:
        '''Calculate risk-based metrics'''
        daily_returns = self.results_df['daily_return'].dropna()
        
        # Volatility
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (daily_returns.mean() * 252) / annualized_vol if annualized_vol > 0 else 0


        portfolio_values = self.results_df['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        annualized_return = self._calculate_return_metrics()['annualized_return']
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'volatility_daily': daily_vol,
            'volatility_annual': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
        }
    
    
    def print_summary(self):
        '''Print formatted performance summary'''
        metrics = self.calculate_all_metrics()
        
        print("="*70)
        print("PERFORMANCE SUMMARY - Pairs Trading Strategy")
        print("="*70)
        
        print("\nRETURN METRICS:")
        print(f"  Total Return:              {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return:         {metrics['annualized_return_pct']:.2f}%")
        print(f"  Final Portfolio Value:     ${metrics['final_portfolio_value']:,.2f}")
        print(f"  Total Profit:              ${metrics['total_profit']:,.2f}")
        
        print("\nRISK METRICS:")
        print(f"  Annual Volatility:         {metrics['volatility_annual']*100:.2f}%")
        print(f"  Sharpe Ratio:              {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:              {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:              {metrics['calmar_ratio']:.2f}")
        
        
        print("="*70)
