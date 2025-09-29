# backtest.py

"""

Backtesting Engine with Performance Analytics.

MODIFIED to support dynamic, time-varying hedge ratios and enhanced risk management.

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

    def run_backtest(self, data, pairs, signal_generator, risk_manager):
        if data.empty or not pairs:
            return {}, {}
        
        traindata, testdata = self.split_data(data)
        pair_performance = {}
        
        for p in pairs:
            symbol1 = p['symbol1']
            symbol2 = p['symbol2']
            self.trades = []
            pair_name = f"{symbol1}-{symbol2}"
            print(pair_name)
            price1_train = traindata[symbol1].dropna()
            price2_train = traindata[symbol2].dropna()
            success = signal_generator.fit(price1_train, price2_train)
            if not success:
                continue
            _ = self.run_out_of_sample_test(
                testdata, symbol1, symbol2, signal_generator, risk_manager
            )

            perf = self.calculate_per_pair_metrics(self.trades, testdata.index).get(pair_name, {})
            pair_performance[pair_name] = perf
            risk_manager.reset()

        return pair_performance


    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_start_date = pd.to_datetime(self.config.test_start_date)
        data.index = pd.to_datetime(data.index)
        train_data = data[data.index < test_start_date].copy()
        test_data = data[data.index >= test_start_date].copy()

        return train_data, test_data

    def train_models(self, train_data: pd.DataFrame, pairs: List[Dict], signal_generator) -> List[Dict]:
        trained_pairs = []
        for i, pair in enumerate(pairs):
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            logger.info(f"--- Training models for pair {i+1}/{len(pairs)}: {symbol1}-{symbol2} ---")
            if symbol1 not in train_data.columns or symbol2 not in train_data.columns:
                logger.warning(f"Skipping pair {symbol1}-{symbol2}: Data not available in training set.")
                continue

            price1_train = train_data[symbol1].dropna()
            price2_train = train_data[symbol2].dropna()
            training_success = signal_generator.fit(price1_train, price2_train)
            if training_success:
                pair_copy = pair.copy()
                trained_pairs.append(pair_copy)
                logger.info(f"Successfully trained models for {symbol1}-{symbol2}")
            else:
                logger.error(f"Failed to train models for {symbol1}-{symbol2}.")

        return trained_pairs


    def run_out_of_sample_test(self, test_data: pd.DataFrame, symbol1, symbol2, signal_generator, risk_manager) -> pd.DataFrame:
        results = []
        portfolio_value = self.initial_capital
        for date, row in test_data.iterrows():
            daily_pnl = 0.0
            daily_trades = []
            pair_name = f"{symbol1}-{symbol2}"
            if symbol1 not in row.index or symbol2 not in row.index:
                continue
            price1, price2 = row[symbol1], row[symbol2]
            if pd.isna(price1) or pd.isna(price2) or price1 == 0 or price2 == 0: 
                continue
            signal_strength, hedge_ratio, volatility = signal_generator.generate_signal(price1, price2)
            if True: ## maybe fo other risk check
                # Map raw signal strength to trade decisions
                if signal_strength > self.config.entry_threshold:
                    signal = 'LONG_SPREAD'
                elif signal_strength < -self.config.entry_threshold:
                    signal = 'SHORT_SPREAD'
                elif abs(signal_strength) < self.config.exit_threshold:
                    signal = 'CLOSE_POSITION'
                else:
                    signal = 'HOLD'
                current_prices = {'price1': price1, 'price2': price2}
                if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
                    
                    # Check if we don't already have a position
                    if pair_name not in risk_manager.position_tracker or risk_manager.position_tracker[pair_name]['size'] == 0:
                        position_size = risk_manager.calculate_position_size(signal_strength, volatility, portfolio_value)
                        result = risk_manager.update_position(
                            pair_name=pair_name,
                            signal=signal,
                            position_size=position_size,
                            prices=current_prices,
                            current_volatility=volatility,
                        )

                        self.trades.append({
                                'date': date, 'pair': pair_name, 'action': 'ENTRY', 'signal': signal,
                                'size': position_size, 'price1': price1, 'price2': price2, 'pnl': 0.0,
                                'hold_days': 0
                            })
                            
                        transaction_cost = risk_manager.calculate_transaction_costs(position_size)
                        daily_pnl -= transaction_cost


                # Handle position exit or check for stop loss
                elif signal == 'CLOSE_POSITION' or (pair_name in risk_manager.position_tracker and risk_manager.position_tracker[pair_name]['size'] != 0):
                    result = risk_manager.update_position(
                        pair_name=pair_name,
                        signal=signal,
                        position_size=0,
                        prices=current_prices,
                        current_volatility=volatility
                    )
                    exit_reason = result.get('exit_reason', signal)
                    # If position was closed (either by signal or stop loss)
                    if result['realized_pnl'] != 0:
                        trade_pnl = result['realized_pnl']
                        daily_pnl += trade_pnl
                        
                        # Find the entry trade to calculate hold days
                        entry_trade = None
                        for trade in reversed(self.trades):
                            if trade['pair'] == pair_name and trade['action'] == 'ENTRY':
                                entry_trade = trade
                                break
                        
                        hold_days = (date - entry_trade['date']).days if entry_trade else 0
                        
                        # Log the exit trade
                        self.trades.append({
                            'date': date, 'pair': pair_name, 'action': 'EXIT', 'signal': exit_reason,
                            'size': entry_trade['size'] if entry_trade else 0, 
                            'price1': price1, 'price2': price2, 'pnl': trade_pnl,
                            'hold_days': hold_days
                        })
                        
                        transaction_cost = risk_manager.calculate_transaction_costs(entry_trade['size'] if entry_trade else 0)
                        daily_pnl -= transaction_cost

            portfolio_value += daily_pnl

            results.append({
                'date': date, 'portfolio_value': portfolio_value, 'daily_pnl': daily_pnl,
                'daily_return': daily_pnl / (portfolio_value - daily_pnl) if (portfolio_value - daily_pnl) != 0 else 0.0,
                'active_positions': len([p for p in risk_manager.position_tracker.values() if p['size'] != 0]),
                'trades_count': len(daily_trades)
            })

        
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        return results_df

    def calculate_per_pair_metrics(self, trade_log: list, all_dates: pd.DatetimeIndex) -> dict:

        if not trade_log: 
            return {}

        pnl_by_pair_date = defaultdict(lambda: defaultdict(float))
        for trade in trade_log:
            print(trade)
            if trade.get('pair') and trade.get('date') and trade['action'] == 'EXIT':
                pnl_by_pair_date[trade['pair']][trade['date']] += trade.get('pnl', 0.0)

        pair_performance = {}

        for pair, date_pnl_map in pnl_by_pair_date.items():
            daily_pnl = pd.Series(date_pnl_map, name='pnl').reindex(all_dates, fill_value=0.0)
            equity_curve = self.initial_capital + daily_pnl.cumsum()

            if equity_curve.empty or len(daily_pnl) < 2:
                continue

            daily_returns = equity_curve.pct_change().fillna(0)
            mean_ret, std_ret = daily_returns.mean(), daily_returns.std()
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_dd = drawdown.min()

            pnl_list = list(date_pnl_map.values())
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]

            pair_performance[pair] = {
                'total_pnl': sum(pnl_list), 'sharpe_ratio': round(sharpe, 3), 'max_drawdown': round(max_dd, 3),
                'total_trades': len(pnl_list), 'win_rate': len(wins) / len(pnl_list) if pnl_list else 0.0,
                'profit_factor': abs(sum(wins)) / abs(sum(losses)) if losses else float('inf')
            }

        return pair_performance
