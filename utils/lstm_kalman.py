# FILE: lstm_kalman.py
# DESCRIPTION: Refactored module for pair trading signal generation.
# Includes hyperparameter tuning, regularization, and dynamic state-space model selection.

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import logging
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ENHANCED STATE-SPACE MODELING WITH KALMAN FILTER ---

class EnhancedKalmanFilter:
    """
    Dynamically selects the best state-space model (e.g., Local Level, AR(1))
    based on AIC and applies the Kalman Filter for spread state estimation.
    """
    def __init__(self):
        self.model_fit = None
        self.spread_mean = 0.0
        self.spread_var = 1.0
        self.best_model_type = "None"

    def select_and_fit_model(self, spread_series: pd.Series):
        """
        Fits multiple state-space models and selects the best one based on AIC.

        Args:
            spread_series (pd.Series): The historical series of the spread.
        """
        if len(spread_series) < 20:
            logging.warning("Not enough data to select a state-space model. Need > 20, have %d.", len(spread_series))
            return False

        models = {
            # Model 1: A simple random walk
            "local_level": sm.tsa.UnobservedComponents(spread_series, 'local level'),
            # Model 2: An autoregressive model of order 1
            "ar1": sm.tsa.SARIMAX(spread_series, order=(1, 0, 0))
        }

        best_aic = np.inf
        best_model_fit = None
        best_model_name = "None"

        for name, model in models.items():
            try:
                fit = model.fit(disp=False)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_model_fit = fit
                    best_model_name = name
            except Exception as e:
                logging.error("Failed to fit model %s: %s", name, e)

        if best_model_fit:
            self.model_fit = best_model_fit
            self.best_model_type = best_model_name
            logging.info("Selected state-space model: %s with AIC: %.2f", self.best_model_type, best_aic)
            return True
        return False

    def update(self, observed_spread: float) -> tuple[float, float]:
        """
        Updates the state (mean and variance) using the fitted Kalman filter.
        If no model is fitted, performs a simple moving average update.
        """
        if self.model_fit:
            # Predict the next state based on the fitted model
            # Note: For real-time use, we would append and refit or use the forecast method
            # This is a simplified update for demonstration
            forecast = self.model_fit.get_forecast(steps=1)
            self.spread_mean = forecast.predicted_mean.iloc[0]
            self.spread_var = forecast.var.iloc[0]
        else:
            # Fallback to a simple exponential moving average if model fitting fails
            self.spread_mean = 0.95 * self.spread_mean + 0.05 * observed_spread
            self.spread_var = 1.0  # Default variance

        return self.spread_mean, self.spread_var

# --- ROBUST LSTM MODEL WITH REGULARIZATION AND TUNING ---

class RobustSpreadLSTM:
    """
    An LSTM model for spread modeling featuring hyperparameter tuning,
    L2 regularization, and Early Stopping to prevent overfitting.
    """
    def __init__(self, lookback_period=60, lstm_units=50, dropout_rate=0.2, l2_reg=0.01):
        self.lookback_period = lookback_period
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_model(self):
        """Builds the LSTM model with L2 regularization and Dropout."""
        inputs = Input(shape=(self.lookback_period, 1))
        x = LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg)
        )(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(
            self.lstm_units,
            return_sequences=False,
            kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def _prepare_sequences(self, data: np.ndarray):
        """Prepares sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback_period, len(data)):
            X.append(data[i-self.lookback_period:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train(self, spread_series: pd.Series, validation_split=0.2, patience=10):
        """
        Trains the LSTM model with early stopping.

        Args:
            spread_series (pd.Series): The historical series of the spread.
            validation_split (float): Fraction of data to use for validation.
            patience (int): Epochs to wait for improvement before stopping.
        """
        if len(spread_series) < self.lookback_period + 20:
            logging.warning("Not enough data to train LSTM. Need > %d, have %d.", self.lookback_period + 20, len(spread_series))
            return False

        scaled_spreads = self.scaler.fit_transform(spread_series.values.reshape(-1, 1))
        X, y = self._prepare_sequences(scaled_spreads)
        
        if X.shape[0] == 0:
            logging.error("Failed to create sequences for LSTM training.")
            return False
        
        X = X.reshape(X.shape[0], X.shape[1], 1)

        if self.model is None:
            self.build_model()
        monitor_metric = 'val_loss' if validation_split > 0 else 'loss'
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True
        )

        self.model.fit(
            X, y,
            epochs=5,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        self.is_fitted = True
        return True

    def predict(self, recent_spreads: pd.Series) -> float:
        """Predicts the next spread value."""
        if not self.is_fitted or len(recent_spreads) < self.lookback_period:
            return 0.0

        input_data = recent_spreads.values[-self.lookback_period:].reshape(-1, 1)
        scaled_input = self.scaler.transform(input_data)
        input_tensor = scaled_input.reshape(1, self.lookback_period, 1)
        
        prediction = self.model.predict(input_tensor, verbose=0)
        inversed_prediction = self.scaler.inverse_transform(prediction)
        
        return float(inversed_prediction[0, 0])


# --- MAIN SIGNAL GENERATOR WITH HYPERPARAMETER TUNING ---

class DeepPairsSignalGenerator:
    """
    Orchestrates the Kalman Filter and LSTM to generate trading signals,
    now with integrated hyperparameter tuning.
    """
    def __init__(self, config):
        self.config = config
        self.kalman_filters = {}
        self.lstm_predictors = {}
        self.spread_histories = {}
        self.best_params = {}

    def _get_or_create_state(self, pair_name):
        """Initializes state for a new pair."""
        if pair_name not in self.kalman_filters:
            self.kalman_filters[pair_name] = EnhancedKalmanFilter()
            
            # Use best params if available, otherwise defaults
            params = self.best_params.get(pair_name, {})
            self.lstm_predictors[pair_name] = RobustSpreadLSTM(
                lookback_period=params.get('lookback_period', self.config.lookback_period),
                lstm_units=params.get('lstm_units', self.config.lstm_units),
                dropout_rate=params.get('dropout_rate', self.config.dropout_rate),
                l2_reg=params.get('l2_reg', self.config.l2_reg)
            )
            self.spread_histories[pair_name] = []
        return self.kalman_filters[pair_name], self.lstm_predictors[pair_name], self.spread_histories[pair_name]

    def tune_and_train_models(self, historical_data: pd.DataFrame, symbol1: str, symbol2: str):
        """
        Performs hyperparameter tuning using walk-forward validation and then
        trains the final models on the full historical dataset.
        """
        pair_name = f"{symbol1}-{symbol2}"
        
        # 1. Calculate spread
        hedge_ratio = self._calculate_static_hedge_ratio(historical_data[symbol1], historical_data[symbol2])
        spreads = historical_data[symbol1] - hedge_ratio * historical_data[symbol2]
        
        # 2. Hyperparameter Tuning for LSTM
        logging.info("Starting hyperparameter tuning for pair: %s", pair_name)
        self._tune_lstm_hyperparameters(spreads, pair_name)
        
        # 3. Train final models with best params on full history
        logging.info("Training final models for pair: %s with best parameters.", pair_name)
        kalman, lstm, _ = self._get_or_create_state(pair_name)
        
        # Train Kalman Filter model selector
        kalman.select_and_fit_model(spreads)
        
        # Train LSTM
        lstm.train(spreads)
        
        return True, hedge_ratio

    def _tune_lstm_hyperparameters(self, spread_series: pd.Series, pair_name: str):
        """
        Uses TimeSeriesSplit to find the best LSTM hyperparameters.
        """
        param_grid = {
            'lookback_period': [30, 60],
            'lstm_units': [32, 64],
            'dropout_rate': [0.2, 0.3],
            'l2_reg': [0.01, 0.001]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = np.inf
        best_params = {}

        # Generate all combinations of parameters
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in param_combinations:
            scores = []
            for train_index, test_index in tscv.split(spread_series):
                train_spread = spread_series.iloc[train_index]
                test_spread = spread_series.iloc[test_index]

                if len(train_spread) < params['lookback_period'] + 20:
                    continue

                # Create and train a temporary LSTM model
                temp_lstm = RobustSpreadLSTM(**params)
                temp_lstm.train(train_spread, validation_split=0, patience=5) # No validation set needed here

                # Evaluate on the test fold
                if temp_lstm.is_fitted:
                    predictions = []
                    history = train_spread.copy()
                    for i in range(len(test_spread)):
                        pred = temp_lstm.predict(history)
                        predictions.append(pred)
                        # Append the actual value to history for the next prediction
                        history = pd.concat([history, pd.Series([test_spread.iloc[i]])], ignore_index=True)
                    
                    mse = np.mean((np.array(predictions) - test_spread.values)**2)
                    scores.append(mse)

            if scores:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
        
        if best_params:
            self.best_params[pair_name] = best_params
            logging.info("Best parameters for %s: %s (Score: %.4f)", pair_name, best_params, best_score)
        else:
            logging.warning("Hyperparameter tuning failed for %s. Using default parameters.", pair_name)


    def generate_signal(self, pair_name: str, price1: float, price2: float, hedge_ratio: float) -> dict:
        """Generates a trading signal for the given pair and prices."""
        kalman, lstm, history = self._get_or_create_state(pair_name)
        
        current_spread = price1 - hedge_ratio * price2
        history.append(current_spread)
        
        # 1. Get filtered spread from Kalman Filter
        kalman_mean, kalman_var = kalman.update(current_spread)
        kalman_std = np.sqrt(kalman_var)
        
        # 2. Calculate Z-Score
        z_score = (current_spread - kalman_mean) / kalman_std if kalman_std > 0 else 0.0
        
        # 3. Get prediction from LSTM
        lstm_signal = 0.0
        if lstm.is_fitted:
            lstm_prediction = lstm.predict(pd.Series(history))
            if kalman_std > 0:
                # Signal is based on expected deviation from the mean
                lstm_signal = np.tanh((lstm_prediction - current_spread) / kalman_std)

        # 4. Combine signals
        signal = self._generate_combined_signal(z_score, lstm_signal)
        
        return {
            'signal': signal,
            'z_score': z_score,
            'spread': current_spread,
            'kalman_mean': kalman_mean,
            'lstm_signal': lstm_signal,
        }

    def _generate_combined_signal(self, z_score, lstm_signal):
        """Combines Z-Score and LSTM signals into a final trading decision."""
        entry_thresh = self.config.entry_threshold
        exit_thresh = self.config.exit_threshold

        # Z-Score based signal
        z_signal = 0
        if z_score > entry_thresh: z_signal = -1  # Short the spread
        elif z_score < -entry_thresh: z_signal = 1  # Long the spread
        elif abs(z_score) < exit_thresh: z_signal = 2 # Close position
        
        # Weighted combination
        # Giving more weight to the primary Z-score signal
        combined_strength = 0.7 * z_signal + 0.3 * lstm_signal
        
        if z_signal == 2:
            return "CLOSE"
        if combined_strength > 0.5:
            return "LONG"
        elif combined_strength < -0.5:
            return "SHORT"
        else:
            return "HOLD"
            
    def _calculate_static_hedge_ratio(self, price1_series, price2_series):
        """Calculates hedge ratio using linear regression."""
        model = sm.OLS(price1_series, sm.add_constant(price2_series)).fit()
        return model.params[1]
