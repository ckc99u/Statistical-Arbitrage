"""
Configuration Management for Statistical Arbitrage System
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path

@dataclass
class DataConfig:
    universe: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ'])
    start_date: str = '2019-01-01'
    end_date: str = '2024-01-01'
    data_source: str = 'yahoo'
    frequency: str = '1d'
    fill_method: str = 'forward'
    min_history_days: int = 252
    cache_data: bool = True
    cache_directory: str = './data/cache'

@dataclass
class PairsConfig:
    min_correlation: float = 0.7
    max_correlation: float = 0.95
    cointegration_threshold: float = 0.05
    min_spread_std: float = 0.01
    max_spread_std: float = 0.20
    min_observations: int = 100
    exclude_pairs: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    initial_state_variance: float = 1.0
    observation_variance: float = 0.1
    process_variance: float = 0.01
    lookback_period: int = 60
    lstm_units: int = 64
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = -3.0

@dataclass
class RiskConfig:
    initial_capital: float = 1000000.0
    max_position_size: float = 0.05
    max_portfolio_exposure: float = 0.30
    max_leverage: float = 2.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005

@dataclass
class BacktestConfig:
    train_test_split: float = 0.5
    rebalance_frequency: str = 'daily'
    benchmark: str = 'SPY'

@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log_level: str = 'INFO'
    random_seed: int = 42
