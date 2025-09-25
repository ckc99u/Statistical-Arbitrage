"""
Backtesting Engine with Performance Analytics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.trades = []
        self.initial_capital = getattr(config, 'initial_capital', 1000000)
        self.train_test_split = getattr(config, 'train_test_split', 0.5)

    def run_backtest(self, data: pd.DataFrame, pairs: List[Dict], signal_generator, risk_manager) -> Dict:
        if data.empty or not pairs:
            return {'performance': {}, 'results': pd.DataFrame(), 'trades': []}
        
        train_data, test_data = self._split_data(data)
        trained_pairs = self._train_models(train_data, pairs, signal_generator)
        results = self._run_out_of_sample_test(test_data, trained_pairs, signal_generator, risk_manager)
        performance = self._calculate_performance_metrics(results)
        
        return {
            'performance': performance,
            'results': results,
            'trades': self.trades
        }

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_point = int(len(data) * self.train_test_split)
        train_data = data.iloc[:split_point].copy()
        test_data = data.iloc[split_point:].copy()
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

    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        if results.empty:
            return {}
        
        portfolio_returns = results['daily_return'].dropna()
        portfolio_values = results['portfolio_value']
        
        if len(portfolio_returns) == 0:
            return {}
        
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
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
