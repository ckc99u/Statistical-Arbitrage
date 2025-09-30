"""
Configuration Management for Statistical Arbitrage System with   Strategy Integration.
"""

import os
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

@dataclass
class DataConfig:
    universe: List[str] = field(default_factory=lambda: [
        # Financial Services
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'CVS',
        # Energy  
        'XOM', 'CVX', 'COP', 'EOG', 'SLB',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'XLF'
    ])
    start_date: str = "2018-01-01"
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    datasource: str = 'yahoo'
    frequency: str = '1d'
    fill_method: str = 'forward'
    min_history_days: int = 504  # 2 years minimum for 
    cache_data: bool = True
    cache_directory: str = ".datacache"

@dataclass
class PairsConfig:
    #   strategy parameters
    correlation_window: int = 252  # 1 year lookback
    min_correlation: float = 0.65  # Higher threshold for better pairs
    max_correlation: float = 0.98
    cointegration_threshold: float = 0.05  # More stringent
    min_spread_std: float = 0.01
    max_spread_std: float = 0.15
    min_observations: int = 252
    exclude_pairs: List[str] = field(default_factory=list)
    
    # Moving average parameters for z-score calculation
    lookback_window: int = 30  # Rolling window for spread statistics
    entry_z_score: float = 2.0  # Entry threshold
    exit_z_score: float = 0.5   # Exit threshold
    stop_loss_z_score: float = 3.5  # Stop loss threshold

@dataclass  
class ModelConfig:
    # Enhanced model parameters combining your LSTM-Kalman with   approach
    use_kalman_hedge: bool = True
    use_lstm_prediction: bool = True
    
    # Kalman filter parameters
    process_var: float = 0.01
    obs_var: float = 0.1
    delta: float = 1e-3
    hedge_obs_var: float = 0.01
    
    # LSTM parameters
    lookback: int = 60
    units: int = 32
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 16
    
    # Signal generation
    z_weight: float = 0.7  # Weight for z-score vs LSTM signal
    
    #  -inspired parameters
    ma_short: int = 10   # Short moving average
    ma_long: int = 30    # Long moving average
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclass
class TransactionCostConfig:
    """Simplified Interactive Brokers transaction cost parameters"""
    
    # Commission (per share, IB approximate)
    commission_per_share: float = 0.005  # $0.005 per share (conservative estimate)
    min_commission: float = 1.0  # $1 minimum per order
    
    # Slippage (as percentage of trade value)
    slippage_bps: float = 5.0  # 5 basis points (0.05%) slippage


@dataclass
class RiskConfig:
    max_position_size: float = 0.15  # Conservative sizing
    max_portfolio_exposure: float = 0.50  # Allow more pairs
    max_leverage: float = 2.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    
    max_holding_period: int = 60  # Force exit after 60 days
    position_concentration: float = 0.25  # Max % in single pair
    sector_concentration: float = 0.40    # Max % in single sector
    
    # Stop loss parameters
    stop_loss_pct: float = 0.03  # 3% fixed stop loss
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    profit_target_pct: float = 0.05   # 5% profit target
    initial_capital: float = 100000.0
@dataclass
class BacktestConfig:
    train_start_date: str = "2018-01-01"
    train_end_date: str = "2021-12-31"
    test_start_date: str = "2022-01-01"
    test_end_date: str = datetime.now().strftime('%Y-%m-%d')
    initial_capital: float = 100000.0
    
    entry_threshold: float = 2.0    # Z-score entry
    exit_threshold: float = 0.5     # Z-score exit
    rebalance_frequency: str = 'monthly'  # Portfolio rebalancing

@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    tranc:TransactionCostConfig = field(default=TransactionCostConfig)
    log_level: str = "INFO"
    random_seed: int = 42
