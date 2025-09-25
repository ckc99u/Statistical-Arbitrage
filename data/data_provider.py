"""
Data Provider for Statistical Arbitrage System - FIXED VERSION
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProvider:
    """Enhanced data provider with proper error handling and updated pandas methods"""
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}

    def get_data(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get market data with caching and error handling"""
        try:
            symbols = symbols or self.config.universe
            start_date = start_date or self.config.start_date
            end_date = end_date or self.config.end_date
            
            cache_key = f"{'-'.join(sorted(symbols))}_{start_date}_{end_date}_{self.config.frequency}"
            
            # Check memory cache
            if self.config.cache_data and cache_key in self.data_cache:
                logger.info(f"Loading data from memory cache: {len(symbols)} symbols")
                return self.data_cache[cache_key].copy()
            
            # Check file cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if self.config.cache_data and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        self.data_cache[cache_key] = data
                        logger.info(f"Loading data from file cache: {len(symbols)} symbols")
                        return data.copy()
                except Exception as e:
                    logger.warning(f"Failed to load cache file: {e}")
            
            # Fetch fresh data
            logger.info(f"Fetching fresh data: {len(symbols)} symbols from {start_date} to {end_date}")
            data = self._fetch_yahoo_data(symbols, start_date, end_date)
            
            if data.empty:
                logger.error("No data fetched")
                return pd.DataFrame()
                
            data = self._process_data(data)
            
            # Cache the data
            if self.config.cache_data:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    self.data_cache[cache_key] = data
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")
            
            logger.info(f"Successfully loaded data: {data.shape}")
            return data.copy()
            
        except Exception as e:
            logger.error(f"Error in get_data: {e}")
            return pd.DataFrame()

    def _fetch_yahoo_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with error handling"""
        try:
            # Download data with error handling
            data = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                interval=self.config.frequency,
                group_by='ticker',
                progress=False,  # Suppress progress bar
                threads=True    # Use threading for faster downloads
            )
            
            if data.empty:
                logger.warning("No data returned from yfinance")
                return pd.DataFrame()
            
            # Handle single vs multiple symbols
            if len(symbols) == 1:
                # Single symbol case
                if isinstance(data.columns, pd.MultiIndex):
                    close_prices = {symbols[0]: data['Close'] if 'Close' in data.columns else data['Adj Close']}
                else:
                    close_prices = {symbols[0]: data['Close'] if 'Close' in data.columns else data['Adj Close']}
            else:
                # Multiple symbols case
                close_prices = {}
                for symbol in symbols:
                    try:
                        if symbol in data.columns.levels[0]:  # Check if symbol exists in data
                            symbol_data = data[symbol]
                            if 'Close' in symbol_data.columns:
                                close_prices[symbol] = symbol_data['Close']
                            elif 'Adj Close' in symbol_data.columns:
                                close_prices[symbol] = symbol_data['Adj Close']
                            else:
                                logger.warning(f"No Close or Adj Close data for {symbol}")
                        else:
                            logger.warning(f"No data found for symbol: {symbol}")
                    except Exception as e:
                        logger.warning(f"Error processing symbol {symbol}: {e}")
                        continue
            
            if not close_prices:
                logger.error("No valid price data extracted")
                return pd.DataFrame()
                
            df = pd.DataFrame(close_prices)
            df.index.name = 'Date'
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {e}")
            return pd.DataFrame()

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with updated pandas methods"""
        try:
            if data.empty:
                return data
            
            # Remove columns that are all NaN
            data = data.dropna(axis=1, how='all')
            
            if data.empty:
                return data
            
            # Handle filling missing values with updated pandas methods
            if self.config.fill_method == 'forward':
                data = data.ffill()  # Updated method
            elif self.config.fill_method == 'backward':
                data = data.bfill()  # Updated method
            elif self.config.fill_method == 'interpolate':
                data = data.interpolate(method='linear')
            
            # Remove rows with too many missing values
            min_valid = max(1, int(len(data.columns) * 0.8))
            data = data.dropna(axis=0, thresh=min_valid)
            
            # Handle zero or negative prices
            if (data <= 0).any().any():
                logger.warning("Found zero or negative prices, forward filling")
                data = data.mask(data <= 0).ffill()  # Updated method
            
            # Remove extreme outliers (returns > 50%)
            daily_returns = data.pct_change()
            outlier_mask = (abs(daily_returns) > 0.5)
            
            if outlier_mask.any().any():
                logger.warning("Found extreme outliers, cleaning data")
                data = data.mask(outlier_mask).ffill()  # Updated method
            
            # Final cleanup
            data = data.dropna(how='all')
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def get_returns(self, data: pd.DataFrame = None, period: str = 'daily') -> pd.DataFrame:
        """Calculate returns for given period"""
        if data is None:
            data = self.get_data()
        if period == 'daily':
            returns = data.pct_change().dropna()
        elif period == 'weekly':
            returns = data.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            returns = data.resample('M').last().pct_change().dropna()
        else:
            raise ValueError(f"Unsupported period: {period}")
        return returns
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        if data.empty:
            logger.error("Data is empty")
            return False
            
        if len(data) < self.config.min_history_days:
            logger.warning(f"Insufficient data: {len(data)} < {self.config.min_history_days}")
            return False
            
        # Check for sufficient non-null data
        null_pct = data.isnull().sum() / len(data)
        high_null_cols = null_pct[null_pct > 0.5].index.tolist()
        
        if high_null_cols:
            logger.warning(f"High null percentage in columns: {high_null_cols}")
            
        return True