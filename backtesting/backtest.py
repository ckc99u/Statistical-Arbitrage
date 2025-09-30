"""
Enhanced Backtesting Engine following   Trading Strategy Methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from utils.signal_generator import SignalGenerator
logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, backtest_config, pairs_config, risk_config, transaction_cost_config):
            self.backtest_config = backtest_config
            self.pairs_config = pairs_config
            self.risk_config = risk_config

            self.results = []
            self.trades = []
            self.positions = {}
            # Use risk_config for initial capital
            self.portfolio_value = self.risk_config.initial_capital
            self.daily_returns = []
            self.transaction_costs = transaction_cost_config
            #  -specific tracking
            self.sector_exposure = defaultdict(float)
            self.pair_performance = {}
            self.cumulative_costs = {
                'commissions': 0.0,
                'slippage': 0.0,
                'total': 0.0}

    def run_backtest(self, data: pd.DataFrame, pairs: List[Dict],
                     signal_generator, risk_manager) -> Dict:
        if data.empty or not pairs:
            logger.error("No data or pairs provided")
            return {}

        train_data, test_data = self._split_data(data)
        logger.info(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")

        trained_pairs = self._train_pairs(train_data, pairs)
        if not trained_pairs:
            logger.error("No pairs successfully trained")
            return {}

        logger.info(f"Successfully trained {len(trained_pairs)} pairs")

        portfolio_results = self._run_trading_simulation(
            test_data, trained_pairs, risk_manager
        )

        performance_metrics = self._calculate_performance_metrics(portfolio_results)

        return {
            'portfolio_performance': performance_metrics,
            'individual_pairs': self.pair_performance,
            'trades': self.trades,
            'daily_results': portfolio_results,
            # 'transaction_costs': self.cumulative_costs,
            # 'cost_analysis': self._analyze_transaction_costs() 
        }

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_point = int(len(data) * 0.7)
        return data.iloc[:split_point].copy(), data.iloc[split_point:].copy()
    
    def calculate_transaction_costs(self, shares1: int, shares2: int, price1: float, price2: float) -> Dict:
        """Calculate commission and slippage costs for a pair trade"""
        
        # Calculate commissions for both legs
        commission1 = max(abs(shares1) * self.transaction_costs.commission_per_share, 
                        self.transaction_costs.min_commission)
        commission2 = max(abs(shares2) * self.transaction_costs.commission_per_share, 
                        self.transaction_costs.min_commission)
        total_commission = commission1 + commission2
        
        # Calculate slippage for both legs
        trade_value1 = abs(shares1) * price1
        trade_value2 = abs(shares2) * price2
        slippage1 = trade_value1 * (self.transaction_costs.slippage_bps / 10000)
        slippage2 = trade_value2 * (self.transaction_costs.slippage_bps / 10000)
        total_slippage = slippage1 + slippage2
        
        return {
            'commissions': total_commission,
            'slippage': total_slippage,
            'total': total_commission + total_slippage
        }
    def _train_pairs(self, train_data: pd.DataFrame, pairs: List[Dict]) -> List[Dict]:
        """Train signal generators for each pair."""
        trained_pairs = []
        for i, pair in enumerate(pairs):
            try:
                symbol1, symbol2 = pair['symbol1'], pair['symbol2']
                if symbol1 not in train_data.columns or symbol2 not in train_data.columns:
                    logger.warning(f"Skipping pair {symbol1}-{symbol2}: Missing data in training set.")
                    continue

                price1 = train_data[symbol1].dropna()
                price2 = train_data[symbol2].dropna()

                # FIX 3: Use the correct pairs_config object
                generator =  SignalGenerator(self.pairs_config)
                
                if  generator.fit(price1, price2, pair):
                    pair['signal_generator'] =  generator
                    trained_pairs.append(pair)
                else:
                    logger.warning(f"Failed to train pair: {symbol1}-{symbol2}")
            except Exception as e:
                logger.error(f"Error training pair {pair.get('symbol1', '')}-{pair.get('symbol2', '')}: {e}", exc_info=True)
        return trained_pairs

    def _run_trading_simulation(self, test_data: pd.DataFrame,
                                        trained_pairs: List[Dict], risk_manager) -> pd.DataFrame:
        
        results = []
        self.portfolio_value = self.risk_config.initial_capital
        
        for date_idx, (date, row) in enumerate(test_data.iterrows()):
            daily_pnl = 0.0
            daily_trades = []
            active_positions = 0
            
            # Process each pair
            for pair in trained_pairs:
                try:
                    symbol1, symbol2 = pair['symbol1'], pair['symbol2']
                    pair_name = f"{symbol1}-{symbol2}"
                    
                    if symbol1 not in row.index or symbol2 not in row.index:
                        continue
                    
                    price1, price2 = row[symbol1], row[symbol2]
                    
                    if pd.isna(price1) or pd.isna(price2) or price1 <= 0 or price2 <= 0:
                        continue
                    
                    # Generate signal using pair's trained generator
                    generator = pair['signal_generator']
                    signal_strength, hedge_ratio, volatility, intercept = generator.generate_signal(price1, price2)
                    
                    #   trading logic
                    trade_result = self. _execute_trade_logic(
                        pair_name, signal_strength, price1, price2, 
                        hedge_ratio, intercept, volatility, date
                    )
                    
                    if trade_result:
                        daily_pnl += trade_result['pnl']
                        if trade_result.get('trade'):
                            daily_trades.append(trade_result['trade'])
                    
                    # Count active positions
                    if pair_name in self.positions and self.positions[pair_name]['size'] != 0:
                        active_positions += 1
                        
                except Exception as e:
                    logger.error(f"Error processing pair {pair.get('symbol1', '')}-{pair.get('symbol2', '')} on {date}: {e}")
                    continue
            
            # Update portfolio
            self.portfolio_value += daily_pnl
            daily_return = daily_pnl / (self.portfolio_value - daily_pnl) if (self.portfolio_value - daily_pnl) > 0 else 0.0
            
            results.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'active_positions': active_positions,
                'trades_count': len(daily_trades)
            })
            
            # Progress logging
            if date_idx % 50 == 0:
                logger.info(f"Progress: {date_idx}/{len(test_data)}, Portfolio: ${self.portfolio_value:,.2f}")
        
        return pd.DataFrame(results).set_index('date')
    
    def  _execute_trade_logic(self, pair_name: str, signal_strength: float,
                                   price1: float, price2: float, hedge_ratio: float,
                                   intercept: float, volatility: float, date) -> Optional[Dict]:
        
        result = {'pnl': 0.0, 'trade': None}
        
        # Initialize position if not exists
        if pair_name not in self.positions:
            self.positions[pair_name] = {
                'size': 0.0,
                'entry_price1': 0.0,
                'entry_price2': 0.0,
                'entry_date': None,
                'entry_signal': 0.0,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept
            }
        
        position = self.positions[pair_name]
        current_prices = {'price1': price1, 'price2': price2, 'hedge_ratio': hedge_ratio, 'intercept': intercept}
        
        #   entry logic
        if position['size'] == 0 and abs(signal_strength) >= self.pairs_config.entry_z_score:
            
            # Calculate position size (  approach - equal dollar weighting)
            position_value = self.portfolio_value * 0.05  # 5% per pair
            
            if signal_strength > 0:  # Long spread
                position['size'] = position_value
                signal_type = 'LONG_SPREAD'
            else:  # Short spread
                position['size'] = -position_value
                signal_type = 'SHORT_SPREAD'
            shares1 = int(position_value / price1)
            shares2 = int(shares1 * hedge_ratio)
            # Record entry
            position.update({
                'entry_price1': price1,
                'entry_price2': price2,
                'entry_date': date,
                'entry_signal': signal_strength,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'shares1': shares1,
                'shares2': shares2 
            })
            


            # Calculate detailed transaction costs
            if self.transaction_costs:
                costs = self.calculate_transaction_costs(shares1, shares2, price1, price2)
                transaction_cost = costs['total']
                
                # Track cumulative costs
                for cost_type, cost in costs.items():
                    if cost_type in self.cumulative_costs:
                        self.cumulative_costs[cost_type] += cost
            else:
                transaction_cost = abs(position['size']) * 0.001  # Fallback

            result['pnl'] -= transaction_cost
            
            result['trade'] = {
                'date': date,
                'pair': pair_name,
                'action': 'ENTRY',
                'signal': signal_type,
                'size': position['size'],
                'price1': price1,
                'price2': price2,
                'signal_strength': signal_strength,
                'pnl': -transaction_cost
            }
        
        #   exit logic
        elif position['size'] != 0:
            should_exit = False
            exit_reason = ""
            
            # Mean reversion exit
            if abs(signal_strength) <= self.pairs_config.exit_z_score:
                should_exit = True
                exit_reason = "MEAN_REVERSION"
            
            # Stop loss exit
            elif abs(signal_strength) >= self.pairs_config.stop_loss_z_score:
                should_exit = True
                exit_reason = "STOP_LOSS"
            
            # Time-based exit (  max holding period)
            elif (date - position['entry_date']).days >= 60:  # 60-day max holding
                should_exit = True
                exit_reason = "TIME_LIMIT"
            
            if should_exit:
                # Calculate P&L
                pnl = self._calculate_pnl(position, current_prices)
                
                # Calculate exit transaction costs
                if self.transaction_costs and hasattr(position, 'shares1') and hasattr(position, 'shares2'):
                    costs = self.calculate_transaction_costs(position['shares1'], position['shares2'], price1, price2)
                    transaction_cost = costs['total']
                    
                    # Track cumulative costs
                    for cost_type, cost in costs.items():
                        if cost_type in self.cumulative_costs:
                            self.cumulative_costs[cost_type] += cost
                else:
                    transaction_cost = abs(position['size']) * 0.001  # Fallback

                net_pnl = pnl - transaction_cost
                
                result['pnl'] = net_pnl
                result['trade'] = {
                    'date': date,
                    'pair': pair_name,
                    'action': 'EXIT',
                    'signal': exit_reason,
                    'size': position['size'],
                    'price1': price1,
                    'price2': price2,
                    'signal_strength': signal_strength,
                    'pnl': net_pnl,
                    'hold_days': (date - position['entry_date']).days
                }
                
                # Reset position
                position.update({
                    'size': 0.0,
                    'entry_price1': 0.0,
                    'entry_price2': 0.0,
                    'entry_date': None,
                    'entry_signal': 0.0,    
                })
                
                # Store trade
                self.trades.append(result['trade'])
        
        return result
    
    def _calculate_pnl(self, position: Dict, current_prices: Dict) -> float:
        
        if position['size'] == 0:
            return 0.0
        
        # Entry spread
        entry_spread = (position['entry_price1'] - 
                       position['hedge_ratio'] * position['entry_price2'] - 
                       position['intercept'])
        
        # Current spread
        current_spread = (current_prices['price1'] - 
                         current_prices['hedge_ratio'] * current_prices['price2'] - 
                         current_prices['intercept'])
        
        # P&L based on spread change
        spread_change = current_spread - entry_spread
        
        # Position size determines direction
        pnl = spread_change * abs(position['size']) / position['entry_price1'] * np.sign(position['size'])
        
        return pnl
    
    def _calculate_performance_metrics(self, portfolio_results: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics following   methodology"""
        
        if portfolio_results.empty:
            return {}
        
        # Basic metrics
        total_return = (portfolio_results['portfolio_value'].iloc[-1] - self.risk_config.initial_capital) / self.risk_config.initial_capital
        daily_returns = portfolio_results['daily_return'].dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / (volatility) if volatility > 0 else 0
        
        # Drawdown analysis
        running_max = portfolio_results['portfolio_value'].expanding().max()
        drawdown = (portfolio_results['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        trades_df = pd.DataFrame([t for t in self.trades if t['action'] == 'EXIT'])
        
        if not trades_df.empty:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            avg_hold_days = trades_df['hold_days'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_hold_days = 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return / (len(portfolio_results) / 252),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_hold_days': avg_hold_days,
            'final_portfolio_value': portfolio_results['portfolio_value'].iloc[-1]
        }
