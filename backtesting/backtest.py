"""
Backtesting Engine with Performance Analytics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.trades = []
        self.initial_capital = getattr(config, 'initial_capital', 50000)

    def run_backtest(self, data: pd.DataFrame, pairs: list, signal_generator, risk_manager) -> dict:
        if data.empty or not pairs:
            return {'overall_performance': {}, 'pair_performance': {}, 'daily_results': pd.DataFrame(), 'trade_log': []}

        train_data, test_data = self._split_data(data)
        
        trained_pairs = self._train_models(train_data, pairs, signal_generator)
        results_df = self._run_out_of_sample_test(test_data, trained_pairs, signal_generator, risk_manager)
        overall_performance = self._calculate_performance_metrics(results_df)
        pair_performance = self._calculate_per_pair_metrics(self.trades, test_data.index)
        
        return {
            'overall_performance': overall_performance,
            'pair_performance': pair_performance,
            'daily_results': results_df,
            'trade_log': self.trades
        }


    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_start_date = pd.to_datetime(self.config.test_start_date)
        data.index = pd.to_datetime(data.index)

        train_data = data[data.index < test_start_date].copy()
        test_data = data[data.index >= test_start_date].copy()
        return train_data, test_data

    def _train_models(self, train_data: pd.DataFrame, pairs: List[Dict], signal_generator) -> List[Dict]:
        trained_pairs = []
        for i, pair in enumerate(pairs):
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            if symbol1 not in train_data.columns or symbol2 not in train_data.columns:
                continue
            
            training_success = signal_generator.train_models(train_data, symbol1, symbol2)
            
            pair_copy = pair.copy()
            pair_copy['lstm_trained'] = training_success
            trained_pairs.append(pair_copy)
        
        return trained_pairs

    def _run_out_of_sample_test(self, test_data: pd.DataFrame, pairs: List[Dict], signal_generator, risk_manager) -> pd.DataFrame:
        results = []
        portfolio_value = self.initial_capital
        active_positions = {}
        
        for date_idx, (date, row) in enumerate(test_data.iterrows()):
            daily_pnl = 0.0
            daily_trades = []
            
            for pair in pairs:
                symbol1 = pair['symbol1']
                symbol2 = pair['symbol2']
                pair_name = f"{symbol1}-{symbol2}"
                
                if symbol1 not in row.index or symbol2 not in row.index:
                    continue
                
                price1 = row[symbol1]
                price2 = row[symbol2]
                
                if pd.isna(price1) or pd.isna(price2) or price1 <= 0 or price2 <= 0:
                    continue
                
                hedge_ratio = pair.get('hedge_ratio', 1.0)
                signal_data = signal_generator.generate_signals(pair_name, price1, price2, hedge_ratio)
                signal = signal_data['signal']
                
                if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
                    if pair_name not in active_positions:
                        volatility = self._estimate_pair_volatility(test_data, symbol1, symbol2, date_idx)
                        signal_strength = abs(signal_data.get('z_score', 1.0))
                        position_size = risk_manager.calculate_position_size(signal_strength, volatility, portfolio_value)
                        
                        if risk_manager.check_risk_limits(pair_name, position_size):
                            active_positions[pair_name] = {
                                'signal': signal,
                                'size': position_size,
                                'entry_date': date,
                                'entry_price1': price1,
                                'entry_price2': price2,
                                'hedge_ratio': hedge_ratio
                            }
                            
                            transaction_cost = risk_manager.calculate_transaction_costs(position_size)
                            daily_pnl -= transaction_cost
                
                elif signal in ['CLOSE_POSITION', 'STOP_LOSS']:
                    if pair_name in active_positions:
                        position = active_positions[pair_name]
                        trade_pnl = self._calculate_trade_pnl(position, price1, price2)
                        daily_pnl += trade_pnl
                        
                        trade = {
                            'date': date,
                            'pair': pair_name,
                            'action': 'EXIT',
                            'signal': signal,
                            'size': position['size'],
                            'price1': price1,
                            'price2': price2,
                            'pnl': trade_pnl,
                            'hold_days': (date - position['entry_date']).days
                        }
                        
                        self.trades.append(trade)
                        del active_positions[pair_name]
                        
                        transaction_cost = risk_manager.calculate_transaction_costs(position['size'])
                        daily_pnl -= transaction_cost
            
            portfolio_value += daily_pnl
            
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'daily_return': daily_pnl / (portfolio_value - daily_pnl) if portfolio_value != daily_pnl else 0.0,
                'active_positions': len(active_positions),
                'trades_count': len(daily_trades)
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        return results_df

    def _estimate_pair_volatility(self, data: pd.DataFrame, symbol1: str, symbol2: str, current_idx: int, window: int = 30) -> float:
        start_idx = max(0, current_idx - window)
        recent_data = data.iloc[start_idx:current_idx]
        
        if len(recent_data) < 10 or symbol1 not in recent_data.columns or symbol2 not in recent_data.columns:
            return 0.1
        
        price1 = recent_data[symbol1].dropna()
        price2 = recent_data[symbol2].dropna()
        
        if len(price1) < 5 or len(price2) < 5:
            return 0.1
        
        spread = price1 - price2
        spread_returns = spread.pct_change().dropna()
        
        if len(spread_returns) < 2:
            return 0.1
        
        volatility = spread_returns.std() * np.sqrt(252)
        return max(0.01, min(1.0, volatility))

    def _calculate_trade_pnl(self, position: Dict, current_price1: float, current_price2: float) -> float:
        entry_spread = position['entry_price1'] - position['hedge_ratio'] * position['entry_price2']
        current_spread = current_price1 - position['hedge_ratio'] * current_price2
        spread_change = current_spread - entry_spread
        
        if position['signal'] == 'LONG_SPREAD':
            pnl = spread_change * abs(position['size'])
        else:
            pnl = -spread_change * abs(position['size'])
        
        return pnl
    def _calculate_per_pair_metrics(self, trade_log: list, all_dates: pd.DatetimeIndex) -> dict:

        if not trade_log:
            return {}

        # Group PnL by pair and date
        pnl_by_pair_date = defaultdict(lambda: defaultdict(float))
        for trade in trade_log:
            pair = trade.get('pair')
            date = trade.get('date')
            pnl = trade.get('pnl', 0.0)
            if pair and date:
                pnl_by_pair_date[pair][date] += pnl
        
        pair_performance = {}

        for pair, date_pnl_map in pnl_by_pair_date.items():
            # Create a daily PnL series for the pair, filling non-trade days with 0
            daily_pnl = pd.Series(date_pnl_map, name='pnl').reindex(all_dates, fill_value=0.0)
            
            # We don't need an initial capital assumption if we work with returns
            # A small non-zero base is needed to avoid division by zero
            # The choice of base doesn't affect Sharpe or Max DD
            base_value = 100_000 
            equity_curve = base_value + daily_pnl.cumsum()
            
            if equity_curve.empty:
                continue

            # Calculate daily returns from the pair's equity curve
            daily_returns = equity_curve.pct_change().fillna(0)
            
            # Calculate Metrics
            if len(daily_returns) < 2:
                continue
            
            # Annualized Sharpe Ratio
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
            # Maximum Drawdown
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Basic metrics (from your existing implementation)
            pnl_list = [pnl for date, pnl in date_pnl_map.items()]
            winning_trades = [pnl for pnl in pnl_list if pnl > 0]
            losing_trades = [pnl for pnl in pnl_list if pnl < 0]

            pair_performance[pair] = {
                'total_pnl': sum(pnl_list),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown, 3),
                'total_trades': len(pnl_list),
                'win_rate': len(winning_trades) / len(pnl_list) if pnl_list else 0.0,
                'profit_factor': abs(sum(winning_trades)) / abs(sum(losing_trades)) if losing_trades else float('inf')
            }
            
        return pair_performance
            
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        if results.empty:
            return {}
        
        portfolio_returns = results['daily_return'].dropna()
        portfolio_values = results['portfolio_value']
        
        if len(portfolio_returns) == 0:
            return {}
        
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        mean_daily_return = portfolio_returns.mean()
        std_daily_return = portfolio_returns.std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
        cagr = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
        volatility = std_daily_return * np.sqrt(252)
        
        cumulative_values = portfolio_values / self.initial_capital
        running_max = cumulative_values.expanding().max()
        drawdown = (cumulative_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio_values.iloc[-1]
        }
