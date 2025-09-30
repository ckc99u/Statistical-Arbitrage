"""
Enhanced Plotting Utilities for Pairs Trading System
Implements all visualizations from AJeanis/Pairs-Trading repository
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
from typing import Optional, Tuple, List, Dict
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')

class Plotter:
    """Comprehensive plotting utilities for pairs trading analysis"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'primary': 'black',
            'secondary': 'royalblue',
            'tertiary': 'midnightblue',
            'buy': 'green',
            'sell': 'red',
            'neutral': 'darkgrey',
            'train': 'green',
            'test': 'red'
        }
    
    # ============================================================================
    # BASIC PERFORMANCE PLOTS
    # ============================================================================
    
    def plot_portfolio_performance(self, results_df: pd.DataFrame,
                                   initial_capital: float = 100000):
        """Plot portfolio value over time with drawdown"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                        height_ratios=[3, 1])
        
        # Portfolio value
        ax1.plot(results_df.index, results_df['portfolio_value'],
                color=self.colors['primary'], linewidth=2, label='Strategy')
        ax1.axhline(initial_capital, color=self.colors['neutral'], 
                   linestyle='dashed', linewidth=1, label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Value Over Time', fontsize=15, fontweight='bold')
        ax1.legend(prop={'size': 12})
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        running_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color=self.colors['sell'], alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, 
                color=self.colors['sell'], linewidth=1.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_testing_periods(self, price1_series: pd.Series,
                                      price2_series: pd.Series,
                                      train_end_date: str,
                                      symbol1: str = 'Asset1',
                                      symbol2: str = 'Asset2',
                                      train_coint_pval: float = None,
                                      test_coint_pval: float = None):
        """Plot price series with training/testing period split"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate nominal price difference
        diff = price1_series - price2_series
        
        # Plot price series
        ax.plot(price1_series.index, price1_series, 
               color=self.colors['primary'], linewidth=2, label=symbol1)
        ax.plot(price2_series.index, price2_series, 
               color=self.colors['secondary'], linewidth=2, label=symbol2)
        ax.plot(diff.index, diff, 
               color=self.colors['tertiary'], linewidth=2, 
               label='Nominal Price Difference')
        
        # Add vertical line at split
        train_end = pd.to_datetime(train_end_date)
        ax.axvline(train_end, color=self.colors['primary'], linewidth=2)
        
        # Highlight periods
        ax.axvspan(price1_series.index[0], train_end, 
                  color=self.colors['train'], alpha=0.2)
        ax.axvspan(train_end, price1_series.index[-1], 
                  color=self.colors['test'], alpha=0.2)
        
        # Text annotations
        mid_train = price1_series.index[0] + (train_end - price1_series.index[0]) / 2
        mid_test = train_end + (price1_series.index[-1] - train_end) / 2
        
        y_pos_high = price1_series.max() * 0.9
        y_pos_low = price1_series.max() * 0.1
        
        ax.text(mid_train, y_pos_high, 'training period', 
               fontsize=15, style='italic', ha='center')
        ax.text(mid_test, y_pos_high, 'testing period', 
               fontsize=15, style='italic', ha='center')
        
        # Cointegration p-values
        if train_coint_pval is not None:
            ax.text(mid_train, y_pos_low, 
                   f'Cointegration P-Value = {train_coint_pval:.3f}',
                   fontsize=15, style='italic', ha='center')
        if test_coint_pval is not None:
            ax.text(mid_test, y_pos_low, 
                   f'Cointegration P-Value = {test_coint_pval:.3f}',
                   fontsize=15, style='italic', ha='center')
        
        ax.set_ylabel('Price In U.S. Dollars', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(f'{symbol1} vs {symbol2}: Training/Testing Analysis',
                    fontsize=16, fontweight='bold')
        ax.legend(prop={'size': 12}, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # SPREAD AND Z-SCORE ANALYSIS
    # ============================================================================
    
    def plot_spread_analysis(self, spread: pd.Series, 
                            hedge_ratio: float,
                            intercept: float,
                            symbol1: str = 'Asset1',
                            symbol2: str = 'Asset2'):
        """Plot spread with mean and standard deviations"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                        height_ratios=[2, 1])
        
        # Spread plot
        ax1.plot(spread.index, spread.values, 
                color=self.colors['primary'], linewidth=2, label='Spread')
        
        # Mean and std bands
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        ax1.axhline(spread_mean, color=self.colors['neutral'], 
                   linestyle='--', linewidth=1.5, label='Mean')
        ax1.axhline(spread_mean + spread_std, color=self.colors['sell'], 
                   linestyle=':', linewidth=1, label='+1 Std')
        ax1.axhline(spread_mean - spread_std, color=self.colors['buy'], 
                   linestyle=':', linewidth=1, label='-1 Std')
        ax1.axhline(spread_mean + 2*spread_std, color=self.colors['sell'], 
                   linestyle=':', linewidth=1.5, label='+2 Std')
        ax1.axhline(spread_mean - 2*spread_std, color=self.colors['buy'], 
                   linestyle=':', linewidth=1.5, label='-2 Std')
        
        ax1.fill_between(spread.index, 
                        spread_mean - 2*spread_std, 
                        spread_mean + 2*spread_std,
                        color=self.colors['neutral'], alpha=0.1)
        
        ax1.set_ylabel('Spread Value', fontsize=12)
        ax1.set_title(f'Spread Analysis: {symbol1} - {hedge_ratio:.4f}*{symbol2} - {intercept:.4f}',
                     fontsize=15, fontweight='bold')
        ax1.legend(prop={'size': 10}, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of spread
        ax2.hist(spread.dropna().values, bins=50, 
                color=self.colors['secondary'], alpha=0.7, edgecolor='black')
        ax2.axvline(spread_mean, color=self.colors['sell'], 
                   linestyle='--', linewidth=2)
        ax2.set_xlabel('Spread Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Spread Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_zscore_signals(self, z_scores: pd.Series,
                           entry_threshold: float = 2.0,
                           exit_threshold: float = 0.5,
                           symbol1: str = 'Asset1',
                           symbol2: str = 'Asset2'):
        """Plot z-scores with entry/exit thresholds"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(z_scores.index, z_scores.values, 
               color=self.colors['primary'], linewidth=2, label='Z-Score')
        
        # Entry/exit thresholds
        ax.axhline(entry_threshold, color=self.colors['sell'], 
                  linestyle='--', linewidth=1.5, label=f'Entry (+{entry_threshold})')
        ax.axhline(-entry_threshold, color=self.colors['buy'], 
                  linestyle='--', linewidth=1.5, label=f'Entry (-{entry_threshold})')
        ax.axhline(exit_threshold, color=self.colors['neutral'], 
                  linestyle=':', linewidth=1, label=f'Exit (±{exit_threshold})')
        ax.axhline(-exit_threshold, color=self.colors['neutral'], 
                  linestyle=':', linewidth=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Shade entry zones
        ax.fill_between(z_scores.index, entry_threshold, 
                       z_scores.max(), color=self.colors['sell'], alpha=0.1)
        ax.fill_between(z_scores.index, -entry_threshold, 
                       z_scores.min(), color=self.colors['buy'], alpha=0.1)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Z-Score', fontsize=12)
        ax.set_title(f'Z-Score Trading Signals: {symbol1} vs {symbol2}',
                    fontsize=15, fontweight='bold')
        ax.legend(prop={'size': 12})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_price_ratio_with_signals(self, price1: pd.Series, 
                                      price2: pd.Series,
                                      z_scores: pd.Series,
                                      entry_threshold: float = 2.0,
                                      symbol1: str = 'Asset1',
                                      symbol2: str = 'Asset2'):
        """Plot price ratio with buy/sell signals based on z-scores"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate price ratio
        price_ratio = price1 / price2
        
        # Plot price ratio
        ax.plot(price_ratio.index, price_ratio.values,
               color=self.colors['primary'], linewidth=2, label='Price Ratio')
        
        # Generate buy/sell signals
        buy_signals = price_ratio.copy()
        buy_signals[z_scores > -entry_threshold] = np.nan
        
        sell_signals = price_ratio.copy()
        sell_signals[z_scores < entry_threshold] = np.nan
        
        # Plot signals
        ax.scatter(buy_signals.index, buy_signals.values,
                  color=self.colors['buy'], marker='^', s=100, 
                  label='Buy Signal', zorder=5)
        ax.scatter(sell_signals.index, sell_signals.values,
                  color=self.colors['sell'], marker='v', s=100, 
                  label='Sell Signal', zorder=5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price Ratio', fontsize=12)
        ax.set_title(f'Price Ratio with Trading Signals: {symbol1}/{symbol2}',
                    fontsize=15, fontweight='bold')
        ax.legend(prop={'size': 12}, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_buy_sell_signals(self, price1_series: pd.Series, 
                             price2_series: pd.Series,
                             buy_signals: pd.Series, 
                             sell_signals: pd.Series,
                             symbol1: str, 
                             symbol2: str):
        """Plot price series with buy/sell signals on both assets"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot prices
        ax.plot(price1_series.index, price1_series.values,
               color=self.colors['primary'], linewidth=2, label=symbol1)
        ax.plot(price2_series.index, price2_series.values,
               color=self.colors['secondary'], linewidth=2, label=symbol2)
        
        # Buy signals
        buy_mask = buy_signals != 0
        if buy_mask.any():
            ax.scatter(price1_series[buy_mask].index, price1_series[buy_mask].values,
                      color=self.colors['buy'], marker='^', s=100, 
                      label='Buy Signal', zorder=5)
            ax.scatter(price2_series[buy_mask].index, price2_series[buy_mask].values,
                      color=self.colors['buy'], marker='^', s=100, zorder=5)
        
        # Sell signals
        sell_mask = sell_signals != 0
        if sell_mask.any():
            ax.scatter(price1_series[sell_mask].index, price1_series[sell_mask].values,
                      color=self.colors['sell'], marker='v', s=100, 
                      label='Sell Signal', zorder=5)
            ax.scatter(price2_series[sell_mask].index, price2_series[sell_mask].values,
                      color=self.colors['sell'], marker='v', s=100, zorder=5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.set_title(f'Trading Signals: {symbol1} vs {symbol2}',
                    fontsize=15, fontweight='bold')
        ax.legend(prop={'size': 12}, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # ROLLING STATISTICS
    # ============================================================================
    
    def plot_rolling_zscore(self, spread: pd.Series,
                           window: int = 30,
                           entry_threshold: float = 2.0,
                           exit_threshold: float = 0.5):
        """Plot spread with rolling z-score"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                        sharex=True)
        
        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        rolling_zscore = (spread - rolling_mean) / rolling_std
        
        # Plot spread with rolling bands
        ax1.plot(spread.index, spread.values, 
                color=self.colors['primary'], linewidth=1.5, 
                label='Spread', alpha=0.7)
        ax1.plot(rolling_mean.index, rolling_mean.values,
                color=self.colors['neutral'], linewidth=2, label='Rolling Mean')
        ax1.fill_between(rolling_mean.index,
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        color=self.colors['secondary'], alpha=0.2, 
                        label='±2 Std')
        
        ax1.set_ylabel('Spread Value', fontsize=12)
        ax1.set_title(f'Spread with Rolling Statistics (Window={window})',
                     fontsize=15, fontweight='bold')
        ax1.legend(prop={'size': 10})
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling z-score
        ax2.plot(rolling_zscore.index, rolling_zscore.values,
                color=self.colors['primary'], linewidth=2, label='Rolling Z-Score')
        ax2.axhline(entry_threshold, color=self.colors['sell'], 
                   linestyle='--', linewidth=1.5)
        ax2.axhline(-entry_threshold, color=self.colors['buy'], 
                   linestyle='--', linewidth=1.5)
        ax2.axhline(exit_threshold, color=self.colors['neutral'], 
                   linestyle=':', linewidth=1)
        ax2.axhline(-exit_threshold, color=self.colors['neutral'], 
                   linestyle=':', linewidth=1)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.fill_between(rolling_zscore.index, entry_threshold, 
                        rolling_zscore.max(), color=self.colors['sell'], alpha=0.1)
        ax2.fill_between(rolling_zscore.index, -entry_threshold, 
                        rolling_zscore.min(), color=self.colors['buy'], alpha=0.1)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Z-Score', fontsize=12)
        ax2.set_title('Rolling Z-Score', fontsize=12, fontweight='bold')
        ax2.legend(prop={'size': 10})
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # PERFORMANCE ANALYSIS
    # ============================================================================
    
    def plot_cumulative_returns(self, results_df: pd.DataFrame,
                               benchmark_series: pd.Series = None):
        """Plot cumulative returns vs benchmark"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate cumulative returns
        cum_returns = (1 + results_df['daily_return']).cumprod() - 1
        
        ax.plot(cum_returns.index, cum_returns.values * 100,
               color=self.colors['primary'], linewidth=2, label='Strategy')
        
        # Add benchmark if provided
        if benchmark_series is not None:
            benchmark_returns = benchmark_series.pct_change().fillna(0)
            cum_benchmark = (1 + benchmark_returns).cumprod() - 1
            ax.plot(cum_benchmark.index, cum_benchmark.values * 100,
                   color=self.colors['neutral'], linewidth=2, 
                   linestyle='--', label='Benchmark')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Returns Over Time', fontsize=15, fontweight='bold')
        ax.legend(prop={'size': 12})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_sharpe(self, results_df: pd.DataFrame, window: int = 60):
        """Plot rolling Sharpe ratio"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        returns = results_df['daily_return'].dropna()
        
        # Calculate rolling Sharpe ratio (annualized)
        rolling_sharpe = (returns.rolling(window=window).mean() * 252) / \
                        (returns.rolling(window=window).std() * np.sqrt(252))
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
               color=self.colors['primary'], linewidth=2)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(1, color=self.colors['buy'], linestyle='--', 
                  linewidth=1, alpha=0.5, label='Sharpe = 1')
        ax.axhline(2, color=self.colors['buy'], linestyle='--', 
                  linewidth=1, alpha=0.7, label='Sharpe = 2')
        
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=(rolling_sharpe.values > 0),
                       color=self.colors['buy'], alpha=0.1)
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=(rolling_sharpe.values < 0),
                       color=self.colors['sell'], alpha=0.1)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
        ax.set_title(f'Rolling Sharpe Ratio ({window}-Day Window)',
                    fontsize=15, fontweight='bold')
        ax.legend(prop={'size': 12})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # COMPREHENSIVE DASHBOARD
    # ============================================================================
    
    def plot_comprehensive_dashboard(self, results_df: pd.DataFrame,
                                    trades_df: pd.DataFrame,
                                    initial_capital: float = 100000):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio value
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(results_df.index, results_df['portfolio_value'],
                color=self.colors['primary'], linewidth=2)
        ax1.axhline(initial_capital, color=self.colors['neutral'],
                   linestyle='--', linewidth=1)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax1.set_title('Portfolio Performance', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        running_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color=self.colors['sell'], alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values,
                color=self.colors['sell'], linewidth=1.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        returns = results_df['daily_return'].dropna() * 100
        ax3.hist(returns, bins=40, color=self.colors['secondary'],
                alpha=0.7, edgecolor='black')
        ax3.axvline(returns.mean(), color=self.colors['sell'],
                   linestyle='--', linewidth=2)
        ax3.set_xlabel('Daily Return (%)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative returns
        ax4 = fig.add_subplot(gs[2, 1])
        cum_returns = (1 + results_df['daily_return']).cumprod() - 1
        ax4.plot(cum_returns.index, cum_returns.values * 100,
                color=self.colors['primary'], linewidth=2)
        ax4.fill_between(cum_returns.index, cum_returns.values * 100, 0,
                        color=self.colors['secondary'], alpha=0.2)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax4.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe
        ax5 = fig.add_subplot(gs[2, 2])
        rolling_sharpe = (returns.rolling(window=60).mean() * 252) / \
                        (returns.rolling(window=60).std() * np.sqrt(252))
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values / 100,  # Adjust scale
                color=self.colors['primary'], linewidth=2)
        ax5.axhline(0, color='black', linewidth=0.5)
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Rolling Sharpe', fontsize=10)
        ax5.set_title('60-Day Rolling Sharpe', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        if not trades_df.empty and 'action' in trades_df.columns:
            exit_trades = trades_df[trades_df['action'] == 'EXIT']
            
            if not exit_trades.empty:
                # 6. Trade P&L
                ax6 = fig.add_subplot(gs[3, 0])
                colors = [self.colors['buy'] if x > 0 else self.colors['sell']
                         for x in exit_trades['pnl']]
                ax6.bar(range(len(exit_trades)), exit_trades['pnl'],
                       color=colors, alpha=0.7)
                ax6.axhline(0, color='black', linewidth=1)
                ax6.set_xlabel('Trade #', fontsize=10)
                ax6.set_ylabel('P&L ($)', fontsize=10)
                ax6.set_title('Trade P&L', fontsize=12, fontweight='bold')
                ax6.grid(True, alpha=0.3, axis='y')
                
                # 7. Cumulative trade P&L
                ax7 = fig.add_subplot(gs[3, 1])
                cumulative_pnl = exit_trades['pnl'].cumsum()
                ax7.plot(range(len(cumulative_pnl)), cumulative_pnl.values,
                        color=self.colors['primary'], linewidth=2)
                ax7.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                                color=self.colors['secondary'], alpha=0.3)
                ax7.set_xlabel('Trade #', fontsize=10)
                ax7.set_ylabel('Cumulative P&L ($)', fontsize=10)
                ax7.set_title('Cumulative Trade P&L', fontsize=12, fontweight='bold')
                ax7.grid(True, alpha=0.3)
                
                # 8. Win/Loss pie chart
                ax8 = fig.add_subplot(gs[3, 2])
                wins = len(exit_trades[exit_trades['pnl'] > 0])
                losses = len(exit_trades[exit_trades['pnl'] < 0])
                ax8.pie([wins, losses],
                       labels=[f'Wins\n({wins})', f'Losses\n({losses})'],
                       colors=[self.colors['buy'], self.colors['sell']],
                       autopct='%1.1f%%', startangle=90,
                       textprops={'fontsize': 10})
                ax8.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')
        
        fig.suptitle('Pairs Trading Strategy - Performance Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    # ============================================================================
    # CORRELATION AND COINTEGRATION ANALYSIS
    # ============================================================================
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                                 pairs_list: List[Tuple[str, str]] = None):
        """Plot correlation heatmap for assets"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        # Highlight pairs if provided
        if pairs_list:
            for symbol1, symbol2 in pairs_list:
                if symbol1 in corr_matrix.columns and symbol2 in corr_matrix.columns:
                    idx1 = corr_matrix.columns.get_loc(symbol1)
                    idx2 = corr_matrix.columns.get_loc(symbol2)
                    ax.add_patch(plt.Rectangle((idx1, idx2), 1, 1,
                                              fill=False, edgecolor='lime',
                                              linewidth=3))
                    ax.add_patch(plt.Rectangle((idx2, idx1), 1, 1,
                                              fill=False, edgecolor='lime',
                                              linewidth=3))
        
        ax.set_title('Asset Correlation Heatmap', fontsize=15, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def save_all_plots(self, results_df: pd.DataFrame, 
                      trades_df: pd.DataFrame,
                      pair_data: Dict = None,
                      output_dir: str = 'plots'):
        """Save all plots to directory"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving plots to {output_dir}/...")
        
        # Portfolio performance
        fig1 = self.plot_portfolio_performance(results_df)
        fig1.savefig(f'{output_dir}/01_portfolio_performance.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Returns distribution
        fig2 = self.plot_returns_distribution(results_df)
        fig2.savefig(f'{output_dir}/02_returns_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Cumulative returns
        fig3 = self.plot_cumulative_returns(results_df)
        fig3.savefig(f'{output_dir}/03_cumulative_returns.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        
        # Rolling Sharpe
        fig5 = self.plot_rolling_sharpe(results_df)
        fig5.savefig(f'{output_dir}/05_rolling_sharpe.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # Comprehensive dashboard
        fig6 = self.plot_comprehensive_dashboard(results_df, trades_df)
        fig6.savefig(f'{output_dir}/06_comprehensive_dashboard.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig6)
        
        print(f"Successfully saved plots to {output_dir}/")
