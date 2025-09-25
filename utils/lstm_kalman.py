"""
Spread-Focused LSTM-Kalman for Statistical Arbitrage (DeepPairs Style)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

class SpreadKalmanFilter:
    """Kalman Filter for spread state estimation"""
    def __init__(self, process_variance=0.01, observation_variance=0.1):
        self.Q = process_variance  # Process noise
        self.R = observation_variance  # Observation noise
        self.P = 1.0  # State covariance
        self.spread_mean = 0.0  # Spread mean state
        self.spread_var = 1.0  # Spread variance state
    def update(self, observed_spread):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.spread_mean += K * (observed_spread - self.spread_mean)
        self.P = (1 - K) * self.P
        return self.spread_mean

class SpreadLSTM:
    """LSTM for spread residual and pattern modeling"""
    def __init__(self, lookback_period=60, lstm_units=32):
        self.lookback_period = lookback_period
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    def build_model(self):
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.lookback_period, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
    def _prepare_sequences(self, scaled_data):
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_period:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)
    def train(self, spread_series):
        if len(spread_series) < self.lookback_period + 50:
            return False
        # Fit scaler on training data only (no lookahead)
        scaled_spreads = self.scaler.fit_transform(spread_series.values.reshape(-1, 1))
        self.is_fitted = True
        X, y = self._prepare_sequences(scaled_spreads)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        if self.model is None:
            self.build_model()
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        return True
    def predict_spread(self, recent_spreads):
        if len(recent_spreads) < self.lookback_period or not self.is_fitted or self.model is None:
            return 0.0
        scaled_input = self.scaler.transform(recent_spreads[-self.lookback_period:].values.reshape(-1, 1))
        X = scaled_input.reshape((1, self.lookback_period, 1))
        prediction = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(prediction)[0, 0]

class DeepPairsSignalGenerator:
    """Spread-focused LSTM-Kalman signal generator with correct per-pair state"""
    def __init__(self, config):
        self.config = config
        self.kalman_filters = {}      # pair_name -> SpreadKalmanFilter
        self.lstm_predictors = {}     # pair_name -> SpreadLSTM
        self.spread_histories = {}    # pair_name -> list of float
        self.z_scores = {}            # pair_name -> list of float
        self.half_lives = {}          # pair_name -> float
    def _get_or_create_state(self, pair_name):
        if pair_name not in self.kalman_filters:
            self.kalman_filters[pair_name] = SpreadKalmanFilter()
            self.lstm_predictors[pair_name] = SpreadLSTM(self.config.lookback_period)
            self.spread_histories[pair_name] = []
            self.z_scores[pair_name] = []
            self.half_lives[pair_name] = np.inf
        return (self.kalman_filters[pair_name],
                self.lstm_predictors[pair_name],
                self.spread_histories[pair_name],
                self.z_scores[pair_name],
                self.half_lives)
    def reset_pair_state(self, pair_name):
        if pair_name in self.kalman_filters:
            del self.kalman_filters[pair_name]
            del self.lstm_predictors[pair_name]
            del self.spread_histories[pair_name]
            del self.z_scores[pair_name]
            del self.half_lives[pair_name]
    def calculate_static_hedge_ratio(self, price1_series, price2_series):
        from sklearn.linear_model import LinearRegression
        X = price2_series.values.reshape(-1, 1)
        y = price1_series.values
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0]
    def calculate_spread(self, price1, price2, hedge_ratio):
        return price1 - hedge_ratio * price2
    def update_spread_statistics(self, pair_name, spread):
        kalman = self.kalman_filters[pair_name]
        history = self.spread_histories[pair_name]
        kalman_mean = kalman.update(spread)
        history.append(spread)
        if len(history) >= 20:
            recent_spreads = history[-20:]
            spread_std = np.std(recent_spreads)
            z_score = (spread - kalman_mean) / spread_std if spread_std > 0 else 0.0
            self.z_scores[pair_name].append(z_score)
            half_life = self._calculate_half_life(pd.Series(recent_spreads))
            self.half_lives[pair_name] = half_life
        else:
            z_score = 0.0
            spread_std = 1.0
        return z_score, kalman_mean, spread_std
    def _calculate_half_life(self, spread_series):
        if len(spread_series) < 10:
            return np.inf
        spread_lag = spread_series.shift(1).dropna()
        spread_curr = spread_series[1:len(spread_lag)+1]
        from sklearn.linear_model import LinearRegression
        X = spread_lag.values.reshape(-1, 1)
        y = spread_curr.values
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        if beta >= 1 or beta <= 0:
            return np.inf
        # Use correct OU half-life formula
        return -np.log(2) / (beta - 1)
    def generate_signals(self, pair_name, price1, price2, hedge_ratio):
        kalman, lstm, history, z_scores, half_lives = self._get_or_create_state(pair_name)
        current_spread = self.calculate_spread(price1, price2, hedge_ratio)
        z_score, kalman_mean, spread_std = self.update_spread_statistics(pair_name, current_spread)
        lstm_signal = 0.0
        if len(history) >= lstm.lookback_period:
            lstm_prediction = lstm.predict_spread(pd.Series(history))
            lstm_signal = np.tanh((lstm_prediction - current_spread) / spread_std) if spread_std > 0 else 0.0
        combined_signal = self._generate_combined_signal(z_score, lstm_signal)
        return {
            'signal': combined_signal,
            'z_score': z_score,
            'half_life': half_lives[pair_name],
            'spread': current_spread,
            'spread_mean': kalman_mean,
            'lstm_signal': lstm_signal,
            'hedge_ratio': hedge_ratio
        }
    def _generate_combined_signal(self, z_score, lstm_signal):
        entry_threshold = self.config.entry_threshold
        exit_threshold = self.config.exit_threshold
        z_signal = 0
        if z_score > entry_threshold:
            z_signal = -1
        elif z_score < -entry_threshold:
            z_signal = 1
        elif abs(z_score) < exit_threshold:
            z_signal = 0
        combined_strength = 0.7 * z_signal + 0.3 * lstm_signal
        if combined_strength > 0.5:
            return 'LONG_SPREAD'
        elif combined_strength < -0.5:
            return 'SHORT_SPREAD'
        elif abs(combined_strength) < 0.2:
            return 'CLOSE_POSITION'
        else:
            return 'HOLD'
    def train_models(self, historical_data, symbol1, symbol2):
        pair_name = f"{symbol1}-{symbol2}"
        self.reset_pair_state(pair_name)
        self._get_or_create_state(pair_name)
        price1 = historical_data[symbol1]
        price2 = historical_data[symbol2]
        hedge_ratio = self.calculate_static_hedge_ratio(price1, price2)
        spreads = price1 - hedge_ratio * price2
        lstm_predictor = self.lstm_predictors[pair_name]
        return lstm_predictor.train(spreads)
