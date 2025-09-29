import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

class RiskManager:

    """Comprehensive risk management for statistical arbitrage."""

    def __init__(self, config):
        self.config = config
        self.position_tracker = {}
        self.portfolio_value = config.initial_capital
        self.daily_pnl = 0

    def reset(self):
        self.position_tracker.clear()
    def calculate_position_size(self, signal_strength: float, volatility: float, portfolio_value: float) -> float:
        if portfolio_value <= 0 or volatility <= 0:
            return 0.0
            
        # Base position size (20% of portfolio maximum)
        base_size = self.config.max_position_size * portfolio_value
        
        # Scale by signal strength (capped at 100%)
        signal_adjusted_size = base_size * min(abs(signal_strength), 1.0)
        
        # Higher volatility -> smaller multiplier -> smaller position
        volatility_multiplier = max(0.1, min(1.0, 0.05 / volatility))
        vol_adjusted_size = signal_adjusted_size * volatility_multiplier
        
        # absolute position size limits
        max_position_limit = portfolio_value * self.config.max_position_size
        final_size = min(vol_adjusted_size, max_position_limit)
        
        return final_size
    
    def check_risk_limits(self) -> bool:
        total_exposure = sum(abs(p['size']) * p['entry_price1'] for p in self.position_tracker.values())
        if total_exposure >= self.config.max_portfolio_exposure * self.portfolio_value:
            return False
        leverage = total_exposure / self.portfolio_value
        if leverage >= self.config.max_leverage:
            return False
        return True

    def calculate_transaction_costs(self, position_size: float) -> float:

        """Calculate transaction costs including slippage."""
        commission = abs(position_size) * self.config.transaction_cost
        slippage = abs(position_size) * self.config.slippage

        return commission + slippage

    def update_position(self, pair_name: str, signal: str, position_size: float, prices: Dict, current_volatility: float = None) -> Dict:

        """Update a position based on a trading signal."""
        if pair_name not in self.position_tracker:

            self.position_tracker[pair_name] = {
                'size': 0.0, 
                'entry_price1': 0.0, 
                'entry_price2': 0.0, 
                'entry_time': None, 
                'unrealized_pnl': 0.0,
                'max_favorable_pnl': 0.0,  # Track best profit for trailing stop
                'dynamic_stop_loss': None,  # Dynamic stop loss level
                'trailing_stop_distance': None,  # Trailing stop distance
                'entry_volatility': 0.0  # Volatility at entry
            }

        position = self.position_tracker[pair_name]
        exit_reason = None
        pnl = 0.0

        if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
            abs_position_size = abs(position_size)
            if signal == 'LONG_SPREAD':
                position['size'] = abs_position_size
            elif signal == 'SHORT_SPREAD':
                position['size'] = -abs_position_size
            position['entry_price1'] = prices['price1']
            position['entry_price2'] = prices['price2']
            position['entry_time'] = pd.Timestamp.now()
            position['entry_volatility'] = current_volatility or 0.02  # Default 2% if not provided
            
            # Set initial dynamic stop loss based on volatility
            entry_spread = position['entry_price1'] - position['entry_price2']
            volatility_multiplier = max(
                getattr(self.config, 'min_volatility_multiplier', 2.0), 
                min(getattr(self.config, 'max_volatility_multiplier', 5.0), position['entry_volatility'] * 100)
            )
            position['dynamic_stop_loss'] = abs(entry_spread) * volatility_multiplier * 0.01
            
            # Set trailing stop distance
            trailing_threshold = getattr(self.config, 'trailing_stop_threshold', 0.01)
            position['trailing_stop_distance'] = max(
                abs(entry_spread) * trailing_threshold,
                position['entry_volatility'] * 0.5
            )
            
            position['max_favorable_pnl'] = 0.0

        # Update trailing stop and check exit conditions
        if position['size'] != 0:
            self._update_trailing_stop(position, prices)

        if signal == 'CLOSE_POSITION':
            pnl = self.calculate_pnl(position, prices)
            exit_reason = "CLOSE_POSITION"
            # Reset position
            position['size'] = 0.0
            position['max_favorable_pnl'] = 0.0
            position['dynamic_stop_loss'] = None
            position['trailing_stop_distance'] = None
        elif self.stop_loss_hit(position, prices):
            pnl = self.calculate_pnl(position, prices)
            exit_reason = "STOP_LOSS"
            position['size'] = 0.0
            position['max_favorable_pnl'] = 0.0
            position['dynamic_stop_loss'] = None
            position['trailing_stop_distance'] = None
        return {'realized_pnl': pnl, 'exit_reason': exit_reason}


    def _update_trailing_stop(self, position: Dict, prices: Dict):
        """Update trailing stop based on current P&L."""
        
        if position['size'] == 0:
            return
            
        current_pnl = self.calculate_pnl(position, prices)
        # Update max favorable P&L
        if current_pnl > position['max_favorable_pnl']:
            position['max_favorable_pnl'] = current_pnl
            
            # Update trailing stop: move stop loss closer when in profit
            profit_lock_ratio = getattr(self.config, 'profit_lock_ratio', 0.5)
            if position['max_favorable_pnl'] > position['trailing_stop_distance']:
                # Reduce stop loss distance as profits increase
                reduction_factor = min(profit_lock_ratio, position['max_favorable_pnl'] / (position['trailing_stop_distance'] * 4))
                new_stop_distance = position['dynamic_stop_loss'] * (1 - reduction_factor)
                position['dynamic_stop_loss'] = max(new_stop_distance, position['trailing_stop_distance'])

    def stop_loss_hit(self, position: Dict, prices: Dict) -> bool:
        """Check if a stop loss has been triggered - now with dynamic and trailing stops."""
        if position['size'] == 0:
            return False
        current_pnl = self.calculate_pnl(position, prices)
        
        # Check if current loss exceeds dynamic stop loss
        if current_pnl < -abs(position['dynamic_stop_loss']):
            return True
            
        # Check trailing stop: if we're below max favorable P&L minus trailing distance
        if (position['max_favorable_pnl'] > position['trailing_stop_distance'] and 
            current_pnl < position['max_favorable_pnl'] - position['trailing_stop_distance']):
            return True
            
        # Original fixed stop loss as final backstop
        entry_spread = position['entry_price1'] - position['entry_price2']
        current_spread = prices['price1'] - prices['price2']
        
        if abs(entry_spread) > 0:
            drawdown = abs(current_spread - entry_spread) / abs(entry_spread)
            if drawdown > self.config.stop_loss_pct:
                return True

        return False

    def calculate_pnl(self, position: Dict, current_prices: Dict) -> float:
        if position['size'] == 0:
            return 0.0
        entry_spread = position['entry_price1'] - position['entry_price2']
        current_spread = current_prices['price1'] - current_prices['price2']
        spread_change = current_spread - entry_spread

        return spread_change * position['size']
