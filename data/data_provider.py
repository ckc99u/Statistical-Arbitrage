"""
Data Provider for Statistical Arbitrage System
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProvider:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}

    def get_data(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        symbols = symbols or self.config.universe
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date}_{end_date}_{self.config.frequency}"
        
        if self.config.cache_data and cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if self.config.cache_data and cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.data_cache[cache_key] = data
                return data.copy()
        
        data = self._fetch_yahoo_data(symbols, start_date, end_date)
        data = self._process_data(data)
        
        if self.config.cache_data:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.data_cache[cache_key] = data
        
        return data.copy()

    def _fetch_yahoo_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(symbols, start=start_date, end=end_date, interval=self.config.frequency, group_by='ticker')
        
        if len(symbols) == 1:
            data.columns = pd.MultiIndex.from_product([symbols, data.columns])
        
        close_prices = {}
        for symbol in symbols:
            if len(symbols) == 1:
                close_prices[symbol] = data[symbol]['Close'] if 'Close' in data[symbol].columns else data[symbol]['Adj Close']
            else:
                close_prices[symbol] = data[symbol]['Close'] if 'Close' in data[symbol].columns else data[symbol]['Adj Close']
        
        df = pd.DataFrame(close_prices)
        df.index.name = 'Date'
        return df

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.dropna(axis=1, how='all')
        
        if self.config.fill_method == 'forward':
            data = data.fillna(method='ffill')
        elif self.config.fill_method == 'backward':
            data = data.fillna(method='bfill')
        elif self.config.fill_method == 'interpolate':
            data = data.interpolate(method='linear')
        
        min_valid = len(data.columns) * 0.8
        data = data.dropna(axis=0, thresh=min_valid)
        
        if (data <= 0).any().any():
            data = data.mask(data <= 0).fillna(method='ffill')
        
        daily_returns = data.pct_change()
        outlier_mask = (abs(daily_returns) > 0.5)
        if outlier_mask.any().any():
            data = data.mask(outlier_mask).fillna(method='ffill')
        
        return data

    def get_returns(self, data: pd.DataFrame = None, period: str = 'daily') -> pd.DataFrame:
        if data is None:
            data = self.get_data()
        
        if period == 'daily':
            returns = data.pct_change().dropna()
        elif period == 'weekly':
            returns = data.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            returns = data.resample('M').last().pct_change().dropna()
        
        return returns
