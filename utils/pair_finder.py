"""
Pairs Selection for Statistical Arbitrage
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from itertools import combinations
import logging
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class PairsFinder:
    def __init__(self, config):
        self.config = config
        self.pairs_cache = {}

    def find_pairs(self, data: pd.DataFrame) -> List[Dict]:
        returns = data.pct_change().dropna()
        valid_pairs = []
        total_combinations = len(list(combinations(data.columns, 2)))
        
        for i, (symbol1, symbol2) in enumerate(combinations(data.columns, 2)):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{total_combinations} pairs tested")
            
            pair_stats = self._analyze_pair(data, returns, symbol1, symbol2)
            if pair_stats and self._is_valid_pair(pair_stats):
                valid_pairs.append(pair_stats)
        
        valid_pairs.sort(key=lambda x: x['quality_score'], reverse=True)
        return valid_pairs

    def _analyze_pair(self, data: pd.DataFrame, returns: pd.DataFrame, symbol1: str, symbol2: str) -> Optional[Dict]:
        price1 = data[symbol1].dropna()
        price2 = data[symbol2].dropna()
        
        common_index = price1.index.intersection(price2.index)
        if len(common_index) < self.config.min_observations:
            return None
        
        price1 = price1.loc[common_index]
        price2 = price2.loc[common_index]
        
        ret1 = returns[symbol1].loc[common_index[1:]]
        ret2 = returns[symbol2].loc[common_index[1:]]
        correlation = ret1.corr(ret2)

        if abs(correlation) < self.config.min_correlation:
            return None
        log_price1 = np.log(price1)
        log_price2 = np.log(price2) 
        coint_stat, p_value, critical_values = coint(log_price1, log_price2)
        is_cointegrated = p_value < self.config.cointegration_threshold
        if not is_cointegrated:
            return None
        
        X = price2.values.reshape(-1, 1)
        y = price1.values
        reg = LinearRegression().fit(X, y)
        hedge_ratio = reg.coef_[0]
        
        raw_spread = price1 - hedge_ratio * price2
        normalized_spread = raw_spread / price1  # or use geometric mean of both prices
        spread_std = normalized_spread.std()
        if spread_std < self.config.min_spread_std or spread_std > self.config.max_spread_std:
            return None
        half_life = self._calculate_half_life(raw_spread)
        sharpe_ratio = self._calculate_spread_sharpe(raw_spread)
        volatility_ratio = ret1.std() / ret2.std() if ret2.std() > 0 else 0
        quality_score = self._calculate_quality_score(abs(correlation), p_value, spread_std, half_life, sharpe_ratio)
        return {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'pair_name': f"{symbol1}-{symbol2}",
            'correlation': correlation,
            'cointegration_pvalue': p_value,
            'hedge_ratio': hedge_ratio,
            'spread_mean': float(raw_spread.mean()),
            'spread_std': spread_std,
            'half_life': half_life,
            'sharpe_ratio': sharpe_ratio,
            'volatility_ratio': volatility_ratio,
            'quality_score': quality_score,
            'observations': len(common_index)
        }

    def _is_valid_pair(self, pair_stats: Dict) -> bool:
        checks = [
            abs(pair_stats['correlation']) >= self.config.min_correlation,
            abs(pair_stats['correlation']) <= self.config.max_correlation,
            pair_stats['cointegration_pvalue'] < self.config.cointegration_threshold,
            pair_stats['spread_std'] >= self.config.min_spread_std,
            pair_stats['spread_std'] <= self.config.max_spread_std,
            pair_stats['observations'] >= self.config.min_observations,
            pair_stats['half_life'] > 0,
            pair_stats['half_life'] < 100,
        ]
        
        if self.config.exclude_pairs:
            pair_name = pair_stats['pair_name']
            if any(excluded in pair_name for excluded in self.config.exclude_pairs):
                return False
        
        return all(checks)

    def _calculate_half_life(self, spread: pd.Series) -> float:
        spread_lag = spread.shift(1).dropna()
        spread_current = spread[1:len(spread_lag)+1]
        
        X = spread_lag.values.reshape(-1, 1)
        y = spread_current.values
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        
        if beta >= 1 or beta <= 0:
            return np.inf
        
        half_life = -np.log(2) / (beta - 1)
        return half_life

    def _calculate_spread_sharpe(self, spread: pd.Series) -> float:
        spread_returns = spread.pct_change().dropna()
        if len(spread_returns) < 10 or spread_returns.std() == 0:
            return 0.0
        
        mean_spread = spread.mean()
        signals = np.where(spread > mean_spread, -1, 1)
        strategy_returns = signals[1:] * spread_returns.values
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        return sharpe

    def _calculate_quality_score(self, correlation: float, p_value: float, spread_std: float, 
                                half_life: float, sharpe_ratio: float) -> float:
        corr_score = min(correlation, 1.0)
        coint_score = max(0, 1 - p_value / 0.05)
        
        if half_life == np.inf:
            hl_score = 0
        else:
            optimal_hl = 20
            hl_score = max(0, 1 - abs(half_life - optimal_hl) / optimal_hl)
        
        sharpe_score = max(0, min(1, (sharpe_ratio + 1) / 2))
        
        quality_score = (0.4 * coint_score + 0.3 * corr_score + 0.2 * hl_score + 0.1 * sharpe_score)
        return quality_score
