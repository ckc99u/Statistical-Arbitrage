"""
Risk Management System with position sizing, stop losses, and portfolio limits
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

class RiskManager:
    """Comprehensive risk management for statistical arbitrage"""
    
    def __init__(self, config):
        self.config = config
        self.position_tracker = {}
        self.portfolio_value = config.initial_capital
        self.daily_pnl = []
        
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                              portfolio_value: float) -> float:
        """Calculate position size based on signal strength and risk parameters"""
        # Base position size as percentage of portfolio
        base_size = self.config.max_position_size * portfolio_value
        
        # Adjust for signal strength
        adjusted_size = base_size * min(abs(signal_strength), 1.0)
        
        # Adjust for volatility (Kelly criterion-inspired)
        if volatility > 0:
            vol_adjusted_size = adjusted_size / (1 + volatility)
        else:
            vol_adjusted_size = adjusted_size
        
        return vol_adjusted_size
    
    def check_risk_limits(self, pair_name: str, position_size: float) -> bool:
        """Check if position passes risk limits"""
        # Portfolio exposure limit
        total_exposure = sum(abs(pos['size']) for pos in self.position_tracker.values())
        if total_exposure + abs(position_size) > self.config.max_portfolio_exposure * self.portfolio_value:
            return False
        
        # Maximum leverage check
        leverage = (total_exposure + abs(position_size)) / self.portfolio_value
        if leverage > self.config.max_leverage:
            return False
        
        return True
    
    def calculate_transaction_costs(self, position_size: float) -> float:
        """Calculate transaction costs including slippage"""
        commission = abs(position_size) * self.config.transaction_cost
        slippage = abs(position_size) * self.config.slippage
        return commission + slippage
    
    def update_position(self, pair_name: str, signal: str, position_size: float, 
                       prices: Dict) -> Dict:
        """Update position tracking"""
        if pair_name not in self.position_tracker:
            self.position_tracker[pair_name] = {
                'size': 0.0,
                'entry_price1': 0.0,
                'entry_price2': 0.0,
                'entry_time': None,
                'unrealized_pnl': 0.0
            }
        
        position = self.position_tracker[pair_name]
        
        if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
            # New position
            position['size'] = position_size if signal == 'LONG_SPREAD' else -position_size
            position['entry_price1'] = prices['price1']
            position['entry_price2'] = prices['price2']
            position['entry_time'] = pd.Timestamp.now()
        
        elif signal == 'CLOSE_POSITION':
            # Close position
            realized_pnl = self._calculate_realized_pnl(position, prices)
            position['size'] = 0.0
            return {'realized_pnl': realized_pnl}
        
        return {'realized_pnl': 0.0}
    
    def _calculate_realized_pnl(self, position: Dict, current_prices: Dict) -> float:
        """Calculate realized P&L when closing position"""
        if position['size'] == 0:
            return 0.0
        
        # Calculate P&L based on spread movement
        entry_spread = position['entry_price1'] - position['entry_price2']
        current_spread = current_prices['price1'] - current_prices['price2']
        spread_change = current_spread - entry_spread
        
        # P&L depends on position direction
        if position['size'] > 0:  # Long spread
            pnl = spread_change * abs(position['size'])
        else:  # Short spread
            pnl = -spread_change * abs(position['size'])
        
        return pnl
