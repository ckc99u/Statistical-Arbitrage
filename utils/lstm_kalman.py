import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DynamicHedgeKalman:
    """
    Corrected Kalman filter for time-varying hedge ratio estimation.
    This version implements the standard vector Kalman filter equations correctly.
    """
    def __init__(self, delta=1e-4, R=0.01):
        self.delta = delta
        self.R = R
        self.P = np.eye(2) * 1e3
        self.beta = np.zeros((2, 1))

    def get_params(self):
        """Return intercept (alpha) and hedge ratio (beta) as floats."""
        alpha = float(self.beta[0, 0])
        beta = float(self.beta[1, 0])
        return alpha, beta

    def update(self, y_obs, x_obs):
        y = y_obs.values if isinstance(y_obs, pd.Series) else np.array(y_obs)
        x = x_obs.values if isinstance(x_obs, pd.Series) else np.array(x_obs)
        for i in range(len(y)):
            self.P += self.delta * np.eye(2)
            H = np.array([[1.0, x[i]]])
            S = H @ self.P @ H.T + self.R
            if S <= 0:
                continue
            K = self.P @ H.T / S
            e = y[i] - H @ self.beta
            self.beta += K * e
            self.P = (np.eye(2) - K @ H) @ self.P
        return float(self.beta[1, 0])  # beta

class SpreadKalmanFilter:
    """Kalman Filter for spread state estimation. No changes needed."""
    def __init__(self, process_variance=0.01, observation_variance=0.1):
        self.Q = process_variance
        self.R = observation_variance
        self.P = 1.0
        self.spread_mean = 0.0

    def update(self, observed_spread):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.spread_mean += K * (observed_spread - self.spread_mean)
        self.P *= (1 - K)
        return self.spread_mean

class SpreadLSTM:
    """LSTM model for spread prediction. No changes needed."""
    def __init__(self, lookback=30, units=32, dropout_rate=0.2):
        self.lookback = lookback
        inputs = Input(shape=(lookback, 1))
        x = LSTM(units, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(units // 2)(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='tanh')(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        self.predict_fn = tf.function(self.model)
        self.scaler = None # Scaler is now initialized in train()

    def prepare_data(self, spread_series):
        scaler = StandardScaler()
        data = scaler.fit_transform(spread_series.values.reshape(-1, 1))
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y), scaler

    def train(self, spread_series, epochs=50, batch_size=32):
        if len(spread_series) < self.lookback * 2:
            logger.warning("Not enough data to train LSTM.")
            return False
        X, y, scaler = self.prepare_data(spread_series)
        if X.shape[0] == 0:
            logger.warning("LSTM training data is empty after processing.")
            return False
        self.scaler = scaler # Save the scaler for this specific model
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)], verbose=0)
        return True

    def predict(self, recent_spread):
        if self.scaler is None or len(recent_spread) < self.lookback:
            return recent_spread.iloc[-1] if not recent_spread.empty else 0.0
        data = self.scaler.transform(np.array(recent_spread[-self.lookback:]).reshape(-1, 1))
        X = np.expand_dims(data, axis=0)
        pred = self.predict_fn(X)
        return float(self.scaler.inverse_transform(pred.numpy().reshape(-1, 1)))

class DeepPairsSignalGenerator:
    """Combined LSTM-Kalman signal generation with dynamic hedge ratio."""
    def __init__(self, model_config):
        self.kalman_spread = SpreadKalmanFilter(model_config.process_var, model_config.obs_var)
        self.kalman_hedge = DynamicHedgeKalman(delta=model_config.delta, R=model_config.hedge_obs_var)
        self.lstm = SpreadLSTM(model_config.lookback, model_config.units, model_config.dropout)
        self.z_threshold = model_config.z_thresh
        self.model_config = model_config
        self.current_hedge_ratio = 1.0
        self.current_intercept = 0.0
        self._spread_history = None # To maintain state during backtest
        self.current_intercept = 0.0

    def fit(self, price1, price2):
        # re-init LSTM as before
        self.lstm = SpreadLSTM(self.model_config.lookback, self.model_config.units, self.model_config.dropout)
        _ = self.kalman_hedge.update(price1, price2)
        alpha, beta = self.kalman_hedge.get_params()
        self.current_intercept, self.current_hedge_ratio = alpha, beta

        spread = price1 - (alpha + beta * price2)  # use hedged spread
        self._spread_history = spread.copy() if isinstance(spread, pd.Series) else pd.Series(spread)
        epochs = getattr(self.model_config, 'epochs', 1)
        batch_size = getattr(self.model_config, 'batch_size', 32)
        return self.lstm.train(self._spread_history, epochs=epochs, batch_size=batch_size)

    def generate_signal(self, price1, price2):
        # Update hedge ratio
        _ = self.kalman_hedge.update(pd.Series([price1]), pd.Series([price2]))
        alpha, beta = self.kalman_hedge.get_params()
        self.current_intercept, self.current_hedge_ratio = alpha, beta
        spread_val = price1 - (self.current_intercept + self.current_hedge_ratio * price2)
        if self._spread_history is None:
            self._spread_history = pd.Series([spread_val])
        else:
            # Append new spread value to history
            self._spread_history = pd.concat([self._spread_history, pd.Series([spread_val])], ignore_index=True)
            max_hist = 252 + self.lstm.lookback
            if len(self._spread_history) > max_hist:
                self._spread_history = self._spread_history[-max_hist:]

        mu = self.kalman_spread.update(spread_val)
        sigma = float(self._spread_history[-252:].std() if len(self._spread_history) >= 30 else 1.0)
        z_score = (spread_val - mu) / sigma if sigma > 0 else 0.0
        
        lstm_pred = self.lstm.predict(self._spread_history)
        lstm_signal = (lstm_pred - spread_val) / sigma if sigma > 0 else 0.0
        signal_strength = self.model_config.z_weight * (-z_score) + (1 - self.model_config.z_weight) * lstm_signal
        # Return volatility for the risk manager
        volatility = sigma
        print(lstm_pred, z_score)
        
        return z_score, beta, volatility, alpha