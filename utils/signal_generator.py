"""
Signal Generation with Moving Average Z-Scores
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)

class SignalGenerator:
    
    def __init__(self, config):
        self.config = config
        self.current_pair = None
        self.spread_history = pd.Series(dtype=float)
        self.fitted = False
        
        self.lookback_window = getattr(config, 'lookback_window', 30)
        self.entry_z_score = getattr(config, 'entry_z_score', 2.0)
        self.exit_z_score = getattr(config, 'exit_z_score', 0.5)
        self.stop_loss_z_score = getattr(config, 'stop_loss_z_score', 3.5)
        
        # Modern enhancements
        self.use_bollinger_bands = True
        self.bollinger_window = getattr(config, 'bollinger_window', 20)
        self.bollinger_std = getattr(config, 'bollinger_std', 2.0)
        
    def fit(self, price1: pd.Series, price2: pd.Series, pair_stats: dict = None) -> bool:
        """Fit the signal generator using   methodology"""
        try:
            if len(price1) < self.lookback_window or len(price2) < self.lookback_window:
                logger.warning("Insufficient data for fitting")
                return False
            
            # Store pair information
            self.current_pair = pair_stats or {}
            
            # Calculate hedge ratio and spread (using OLS)
            if pair_stats and 'hedge_ratio' in pair_stats and 'intercept' in pair_stats:
                self.hedge_ratio = pair_stats['hedge_ratio']
                self.intercept = pair_stats['intercept']
            else:
                # Fallback to simple OLS
                X = price2.values.reshape(-1, 1)
                y = price1.values
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(X, y)
                self.hedge_ratio = reg.coef_[0]
                self.intercept = reg.intercept_
            
            # Calculate historical spread
            spread = price1 - (self.intercept + self.hedge_ratio * price2)
            self.spread_history = spread.copy()
            
            # Calculate spread statistics
            self.spread_mean = spread.mean()
            self.spread_std = spread.std()
            
            self.fitted = True
            logger.info(f"Fitted signal generator with hedge ratio: {self.hedge_ratio:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting signal generator: {e}")
            return False
    
    def generate_signal(self, price1: float, price2: float) -> Tuple[float, float, float, float]:
        if not self.fitted:
            return 0.0, self.hedge_ratio if hasattr(self, 'hedge_ratio') else 1.0, 0.01, 0.0
        
        try:
            # Calculate current spread
            current_spread = price1 - (self.intercept + self.hedge_ratio * price2)
            
            # Update spread history
            self.spread_history = pd.concat([
                self.spread_history, 
                pd.Series([current_spread])
            ], ignore_index=True)
            
            # Keep only recent history
            if len(self.spread_history) > 252:
                self.spread_history = self.spread_history.tail(252)
            
            z_score = self. _calculate_z_score(current_spread)
            
            # Enhanced signal with Bollinger Bands
            bollinger_signal = self._calculate_bollinger_signal(current_spread)
            
            signal_strength = 0.7 * z_score + 0.3 * bollinger_signal
            
            # Calculate current volatility
            recent_spreads = self.spread_history.tail(30)
            volatility = recent_spreads.std() if len(recent_spreads) > 5 else 0.01
            
            return signal_strength, self.hedge_ratio, volatility, self.intercept
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0.0, getattr(self, 'hedge_ratio', 1.0), 0.01, 0.0
    
    def  _calculate_z_score(self, current_spread: float) -> float:
        
        if len(self.spread_history) < self.lookback_window:
            return 0.0
        recent_spreads = self.spread_history.tail(self.lookback_window)
        
        rolling_mean = recent_spreads.mean()
        rolling_std = recent_spreads.std()
        
        if rolling_std == 0 or np.isnan(rolling_std):
            return 0.0
        
        z_score = (current_spread - rolling_mean) / rolling_std
        
        # Invert z-score for trading signal (negative z-score means spread is below mean -> go long spread)
        return -z_score
    
    def _calculate_bollinger_signal(self, current_spread: float) -> float:
        """Calculate Bollinger Band based signal"""
        
        if len(self.spread_history) < self.bollinger_window:
            return 0.0
        
        recent_spreads = self.spread_history.tail(self.bollinger_window)
        
        bb_mean = recent_spreads.mean()
        bb_std = recent_spreads.std()
        
        if bb_std == 0:
            return 0.0
        
        # Bollinger Bands
        upper_band = bb_mean + (self.bollinger_std * bb_std)
        lower_band = bb_mean - (self.bollinger_std * bb_std)
        
        # Signal generation
        if current_spread > upper_band:
            # Spread above upper band -> short spread
            return -2.0
        elif current_spread < lower_band:
            # Spread below lower band -> long spread  
            return 2.0
        else:
            # Inside bands -> neutral
            return 0.0
    
    def get_signal_strength_interpretation(self, signal_strength: float) -> str:
        abs_signal = abs(signal_strength)
        
        if abs_signal >= self.stop_loss_z_score:
            return "STOP_LOSS"
        elif abs_signal >= self.entry_z_score:
            return "STRONG_ENTRY"  
        elif abs_signal >= self.entry_z_score * 0.75:
            return "MODERATE_ENTRY"
        elif abs_signal <= self.exit_z_score:
            return "EXIT"
        else:
            return "HOLD"
    
    def calculate_position_target(self, signal_strength: float, portfolio_value: float) -> float:
        
        abs_signal = abs(signal_strength)
        
        # Base position size
        base_size = portfolio_value * 0.1  # 10% base allocation
        
        # Scale by signal strength
        signal_multiplier = min(abs_signal / self.entry_z_score, 2.0)  # Cap at 2x
        
        target_size = base_size * signal_multiplier
        
        return target_size if signal_strength > 0 else -target_size
