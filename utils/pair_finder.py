"""
Fixed to match existing system interface
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from itertools import combinations
import logging
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)

class PairsFinder:
    
    def __init__(self, config, ticker):
        self.config = config
        
        self.financial_tickers = ticker
        self.cointegration_pvalue_threshold = 0.02
        
    def find_pairs(self, data: pd.DataFrame) -> List[Dict]:
        
        available_tickers = [ticker for ticker in self.financial_tickers 
                           if ticker in data.columns]
        
        if len(available_tickers) < 5:
            logger.warning("Limited financial tickers available, using all symbols")
            available_tickers = list(data.columns)
            
        analysis_data = data[available_tickers].copy()
        cointegrated_pairs = self._find_cointegrated_pairs(analysis_data)
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        return cointegrated_pairs
    
    def _find_cointegrated_pairs(self, data: pd.DataFrame) -> List[Dict]:
        pairs = []
        keys = data.columns
        k = len(keys)
        
        # Test all combinations
        for i in range(k):
            for j in range(i+1, k):
                try:
                    series_1 = data[keys[i]].dropna()
                    series_2 = data[keys[j]].dropna()
                    
                    common_index = series_1.index.intersection(series_2.index)
                    if len(common_index) < 252:  # At least 1 year
                        continue
                        
                    series_1 = series_1.loc[common_index]
                    series_2 = series_2.loc[common_index]
                    
                    # Cointegration test (  core logic)
                    coint_result = coint(series_1, series_2)
                    pvalue = coint_result[1]
                    
                    #   threshold
                    if pvalue < self.cointegration_pvalue_threshold:
                        
                        # Calculate additional statistics needed by your system
                        pair_stats = self._calculate_pair_statistics(
                            series_1, series_2, keys[i], keys[j], pvalue
                        )
                        
                        if pair_stats:
                            pairs.append(pair_stats)
                            
                except Exception as e:
                    logger.warning(f"Error testing {keys[i]}-{keys[j]}: {e}")
                    continue
        
        # Sort by p-value (  approach)
        pairs.sort(key=lambda x: x['cointegration_pvalue'])
        return pairs
    
    def _calculate_pair_statistics(self, series_1: pd.Series, series_2: pd.Series,
                                 symbol1: str, symbol2: str, pvalue: float) -> Optional[Dict]:
        """Calculate comprehensive statistics matching your system's expected interface"""
        
        try:
            # Basic correlation
            returns1 = series_1.pct_change().dropna()
            returns2 = series_2.pct_change().dropna()
            
            # Align returns
            common_returns_index = returns1.index.intersection(returns2.index)
            returns1 = returns1.loc[common_returns_index]
            returns2 = returns2.loc[common_returns_index]
            
            correlation = returns1.corr(returns2)
            
            # OLS regression for hedge ratio (  approach)
            X = add_constant(series_2.values)
            model = OLS(series_1.values, X).fit()
            intercept = model.params[0]
            hedge_ratio = model.params[1]
            
            # Calculate spread
            spread = series_1 - (intercept + hedge_ratio * series_2)
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Calculate volatilities
            vol1 = returns1.std() * np.sqrt(252)
            vol2 = returns2.std() * np.sqrt(252)
            
            # Return dictionary matching your system's expected keys
            return {
                # Core identifiers (required by your system)
                'symbol1': symbol1,
                'symbol2': symbol2,
                'pair_name': f"{symbol1}-{symbol2}",  # This was missing!
                
                # Statistical measures
                'correlation': correlation,
                'cointegration_pvalue': pvalue,
                'intercept': intercept,
                'hedge_ratio': hedge_ratio,
                
                # Spread characteristics
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                
                # Risk measures
                'vol1': vol1,
                'vol2': vol2,
                
                # Optional sector info
                'sector': 'Financial' if symbol1 in self.financial_tickers and symbol2 in self.financial_tickers else 'Mixed'
            }
            
        except Exception as e:
            logger.warning(f"Error calculating statistics for {symbol1}-{symbol2}: {e}")
            return None
