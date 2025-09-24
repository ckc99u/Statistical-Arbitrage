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
        self.spread_var = 1.0   # Spread variance state
        
    def update(self, observed_spread):
        """Update spread mean and variance estimates"""
        # Prediction step
        self.P += self.Q
        
        # Update step
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
        
    def build_model(self):
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.lookback_period, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='tanh')  # Output spread prediction
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        
    def prepare_spread_sequences(self, spread_series):
        """Prepare sequences for spread modeling"""
        scaled_spreads = self.scaler.fit_transform(spread_series.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_spreads)):
            X.append(scaled_spreads[i-self.lookback_period:i, 0])
            y.append(scaled_spreads[i, 0])
            
        return np.array(X), np.array(y)
    
    def train(self, spread_series):
        if len(spread_series) < self.lookback_period + 50:
            return False
            
        X, y = self.prepare_spread_sequences(spread_series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if self.model is None:
            self.build_model()
            
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        return True
    
    def predict_spread(self, recent_spreads):
        """Predict next spread value"""
        if len(recent_spreads) < self.lookback_period or self.model is None:
            return 0.0
            
        scaled_input = self.scaler.transform(recent_spreads[-self.lookback_period:].values.reshape(-1, 1))
        X = scaled_input.reshape((1, self.lookback_period, 1))
        
        prediction = self.model.predict(X, verbose=0)[0, 0]
        return self.scaler.inverse_transform([[prediction]])[0, 0]

class DeepPairsSignalGenerator:
    """Spread-focused LSTM-Kalman signal generator matching PDF requirements"""
    
    def __init__(self, config):
        self.config = config
        self.kalman_filter = SpreadKalmanFilter()
        self.lstm_predictor = SpreadLSTM(config.lookback_period)
        
        # Signal tracking
        self.spread_history = []
        self.z_scores = []
        self.half_life = np.inf
        
    def calculate_static_hedge_ratio(self, price1_series, price2_series):
        """Calculate initial hedge ratio using cointegration"""
        from sklearn.linear_model import LinearRegression
        X = price2_series.values.reshape(-1, 1)
        y = price1_series.values
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0]
    
    def calculate_spread(self, price1, price2, hedge_ratio):
        """Calculate spread: S(t) = P1(t) - β * P2(t)"""
        return price1 - hedge_ratio * price2
    
    def update_spread_statistics(self, spread):
        """Update spread mean via Kalman filter"""
        kalman_mean = self.kalman_filter.update(spread)
        self.spread_history.append(spread)
        
        # Calculate rolling statistics for Z-score (PDF requirement)
        if len(self.spread_history) >= 20:
            recent_spreads = self.spread_history[-20:]
            spread_std = np.std(recent_spreads)
            
            if spread_std > 0:
                z_score = (spread - kalman_mean) / spread_std
            else:
                z_score = 0.0
                
            self.z_scores.append(z_score)
            
            # Calculate half-life (PDF requirement)
            self.half_life = self._calculate_half_life(pd.Series(recent_spreads))
            
            return z_score, kalman_mean, spread_std
        
        return 0.0, kalman_mean, 1.0
    
    def _calculate_half_life(self, spread_series):
        """Calculate mean reversion half-life (PDF requirement)"""
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
            
        return -np.log(2) / np.log(beta)
    
    def generate_signals(self, price1, price2, hedge_ratio):
        """Generate trading signals based on Z-score and LSTM prediction (PDF requirements)"""
        
        # Calculate spread
        current_spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Update spread statistics
        z_score, kalman_mean, spread_std = self.update_spread_statistics(current_spread)
        
        # LSTM prediction for spread behavior
        lstm_signal = 0.0
        if len(self.spread_history) >= self.lstm_predictor.lookback_period:
            lstm_prediction = self.lstm_predictor.predict_spread(pd.Series(self.spread_history))
            # LSTM signal: positive if predicting spread increase, negative if decrease
            lstm_signal = np.tanh((lstm_prediction - current_spread) / spread_std) if spread_std > 0 else 0.0
        
        # Combined signal: Z-score (mean reversion) + LSTM (pattern recognition)
        combined_signal = self._generate_combined_signal(z_score, lstm_signal)
        
        return {
            'signal': combined_signal,
            'z_score': z_score,  # PDF requirement
            'half_life': self.half_life,  # PDF requirement
            'spread': current_spread,
            'spread_mean': kalman_mean,
            'lstm_signal': lstm_signal,
            'hedge_ratio': hedge_ratio
        }
    
    def _generate_combined_signal(self, z_score, lstm_signal):
        """Generate trading signals combining Z-score and LSTM (PDF requirements)"""
        entry_threshold = self.config.entry_threshold
        exit_threshold = self.config.exit_threshold
        
        # Z-score based signals (PDF requirement)
        z_signal = 0
        if z_score > entry_threshold:
            z_signal = -1  # Short spread (sell asset1, buy asset2)
        elif z_score < -entry_threshold:
            z_signal = 1   # Long spread (buy asset1, sell asset2)
        elif abs(z_score) < exit_threshold:
            z_signal = 0   # Close position
        
        # Weight Z-score and LSTM signals
        combined_strength = 0.7 * z_signal + 0.3 * lstm_signal
        
        # Generate final signal
        if combined_strength > 0.5:
            return 'LONG_SPREAD'
        elif combined_strength < -0.5:
            return 'SHORT_SPREAD'
        elif abs(combined_strength) < 0.2:
            return 'CLOSE_POSITION'
        else:
            return 'HOLD'
    
    def train_models(self, historical_data, symbol1, symbol2):
        """Train LSTM on historical spread data"""
        price1 = historical_data[symbol1]
        price2 = historical_data[symbol2]
        
        # Calculate hedge ratio
        hedge_ratio = self.calculate_static_hedge_ratio(price1, price2)
        
        # Calculate historical spreads
        spreads = price1 - hedge_ratio * price2
        
        # Train LSTM on spread patterns
        return self.lstm_predictor.train(spreads)
