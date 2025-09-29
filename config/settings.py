# settings.py

"""

Configuration Management for Statistical Arbitrage System.

"""

import os

from dataclasses import dataclass, field

from typing import Dict, List, Optional

import json

from pathlib import Path

from datetime import datetime

@dataclass

class DataConfig:

    universe: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'GLD', 'SLV', 'USO', 'TLT',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        'JPM', 'GS', 'BAC', 'WFC',
        'XLE', 'XOM', 'CVX'
    ])

    start_date: str = "2015-01-01"
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    datasource: str = 'yahoo'
    frequency: str = '1d'
    fill_method: str = 'forward'
    min_history_days: int = 252
    cache_data: bool = True
    cache_directory: str = ".datacache"

@dataclass

class PairsConfig:
    min_correlation: float = 0.6
    max_correlation: float = 0.98
    cointegration_threshold: float = 0.05
    min_spread_std: float = 0.005
    max_spread_std: float = 0.25
    min_observations: int = 100
    exclude_pairs: List[str] = field(default_factory=list)

@dataclass

class ModelConfig:

    # Parameters for the LSTM-Kalman model
    process_var: float = 0.01
    obs_var: float = 0.1
    delta: float = 1e-3
    hedge_obs_var: float = 0.01
    lookback: int = 30
    units: int = 32
    dropout: float = 0.2
    z_thresh: float = 2.0
    alpha: float = 0.6

@dataclass

class RiskConfig:
    initial_capital: float = 50000.0
    max_position_size: float = 0.2
    max_portfolio_exposure: float = 0.30
    max_leverage: float = 2.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    stop_loss_pct: float = 0.02  # Fixed stop loss as final backstop
    # New dynamic and trailing stop parameters
    min_volatility_multiplier: float = 2.0  # Minimum volatility multiplier for dynamic stops
    max_volatility_multiplier: float = 5.0  # Maximum volatility multiplier for dynamic stops
    trailing_stop_threshold: float = 0.01   # Minimum profit % before trailing stop activates
    profit_lock_ratio: float = 0.5         # Ratio of profits to lock in with trailing stop

@dataclass

class BacktestConfig:
    test_start_date: str = "2022-01-01"
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5

@dataclass

class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log_level: str = "INFO"
    random_seed: int = 42
