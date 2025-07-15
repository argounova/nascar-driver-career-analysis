"""
NASCAR LSTM Career Trajectory Prediction Model

This module implements LSTM neural networks to predict NASCAR driver career trajectories
including future performance metrics, career peak timing, and total career achievements.
Uses time series sequences of driver seasons to learn performance patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings

# TensorFlow imports with compatibility handling
try:
    import tensorflow as tf
    
    # Try modern TensorFlow 2.x imports first
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, callbacks, optimizers
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        TF_VERSION = "2.x"
    except ImportError:
        # Fallback to older import style if needed
        import keras
        from keras import layers, callbacks, optimizers
        from keras.models import Sequential, Model
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        TF_VERSION = "1.x"
        
except ImportError:
    raise ImportError(
        "TensorFlow is not installed. Please install it with:\n"
        "pip install tensorflow\n"
        "or\n"
        "conda install tensorflow"
    )

# ML imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
from config import get_config, get_data_paths


class NASCARLSTMPredictor:
    """
    LSTM neural network for predicting NASCAR driver career trajectories.
    
    Predicts multiple targets:
    - Next season performance metrics
    - Career peak timing and magnitude
    - Long-term career achievements
    - Performance trend directions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the LSTM predictor.
        
        Args:
            config (Optional[Dict]): Configuration dictionary
        """
        self.config = config if config is not None else get_config()
        self.lstm_config = self.config['models']['lstm']
        self.paths = get_data_paths(self.config)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log TensorFlow version info
        self.logger.info(f"TensorFlow version: {tf.__version__}")
        self.logger.info(f"Using import style: {TF_VERSION}")
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
        # Model components
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.history = None
        
        # Data storage
        self.sequences = None
        self.targets = None
        self.driver_names = None
        self.feature_names = None
        
        # Training data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Prediction targets from config
        self.prediction_targets = self.lstm_config.get('prediction_targets', [
            'next_season_avg_finish',
            'next_season_win_rate',
            'career_peak_score',
            'seasons_to_peak'
        ])
    
    def _configure_tensorflow(self) -> None:
        """Configure TensorFlow settings for optimal performance."""
        try:
            # GPU configuration
            if self.config['runtime']['use_gpu']:
                if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'list_physical_devices'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            self.logger.info(f"Configured {len(gpus)} GPU(s) for TensorFlow")
                        except RuntimeError as e:
                            self.logger.warning(f"GPU configuration failed: {e}")
                    else:
                        self.logger.info("No GPUs found, using CPU")
                else:
                    self.logger.info("GPU configuration not available in this TensorFlow version")
            
            # Set random seeds for reproducibility
            if self.config['runtime']['deterministic']:
                if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
                    tf.random.set_seed(self.config['runtime']['random_seed'])
                np.random.seed(self.config['runtime']['random_seed'])
                
        except Exception as e:
            self.logger.warning(f"TensorFlow configuration warning: {e}")
    
    def load_sequences(self, sequences: Optional[np.ndarray] = None, 
                      targets: Optional[np.ndarray] = None,
                      driver_names: Optional[List[str]] = None) -> None:
        """
        Load LSTM sequence data.
        
        Args:
            sequences (Optional[np.ndarray]): Input sequences
            targets (Optional[np.ndarray]): Target values
            driver_names (Optional[List[str]]): Driver names for each sequence
        """
        if sequences is not None:
            self.sequences = sequences
            self.targets = targets
            self.driver_names = driver_names
        else:
            # Try to load from saved file
            sequences_path = Path(self.paths['processed_data']) / 'lstm_sequences.npz'
            if sequences_path.exists():
                data = np.load(sequences_path, allow_pickle=True)
                self.sequences = data['sequences']
                self.targets = data['targets']
                self.driver_names = data['driver_names'].tolist()
                self.logger.info(f"Loaded {len(self.sequences)} sequences from file")
            else:
                raise ValueError("No sequence data provided or found")
        
        self.logger.info(f"Loaded {len(self.sequences)} sequences with shape {self.sequences.shape}")
    
    def prepare_prediction_targets(self, engineered_features: pd.DataFrame) -> np.ndarray:
        """
        Prepare specific prediction targets from engineered features.
        
        Args:
            engineered_features (pd.DataFrame): Full engineered feature set
            
        Returns:
            np.ndarray: Prepared target matrix
        """
        self.logger.info("Preparing prediction targets...")
        
        # Create targets for each driver sequence
        target_data = []
        
        for i, driver_name in enumerate(self.driver_names):
            driver_data = engineered_features[
                engineered_features['Driver'] == driver_name
            ].sort_values('Season')
            
            if len(driver_data) == 0:
                continue
            
            # Get the season corresponding to this sequence
            # Sequences are created with sequence_length seasons, so target is next season
            sequence_end_idx = self.lstm_config['sequence_length']
            
            if len(driver_data) > sequence_end_idx:
                target_season_data = driver_data.iloc[sequence_end_idx]
                
                # Create target vector
                targets = []
                
                # Next season metrics
                targets.extend([
                    target_season_data.get('avg_finish', 20.0),  # Default to mid-pack
                    target_season_data.get('win_rate', 0.0),
                    target_season_data.get('top_5_rate', 0.1),
                    target_season_data.get('top_10_rate', 0.2)
                ])
                
                # Career progression metrics
                targets.extend([
                    target_season_data.get('performance_score', 0.5),
                    target_season_data.get('seasons_to_peak', 5.0),
                    target_season_data.get('career_progress', 0.5),
                    target_season_data.get('peak_timing', 0.5)
                ])
                
                target_data.append(targets)
            else:
                # If we don't have next season data, use current season as proxy
                current_data = driver_data.iloc[-1]
                targets = [
                    current_data.get('avg_finish', 20.0),
                    current_data.get('win_rate', 0.0),
                    current_data.get('top_5_rate', 0.1),
                    current_data.get('top_10_rate', 0.2),
                    current_data.get('performance_score', 0.5),
                    current_data.get('seasons_to_peak', 5.0),
                    current_data.get('career_progress', 0.5),
                    current_data.get('peak_timing', 0.5)
                ]
                target_data.append(targets)
        
        target_matrix = np.array(target_data)
        
        # Define target names for interpretation
        self.target_names = [
            'next_avg_finish', 'next_win_rate', 'next_top5_rate', 'next_top10_rate',
            'performance_score', 'seasons_to_peak', 'career_progress', 'peak_timing'
        ]
        
        self.logger.info(f"Prepared {target_matrix.shape[1]} prediction targets")
        return target_matrix
    
    def preprocess_data(self, sequences: Optional[np.ndarray] = None,
                       targets: Optional[np.ndarray] = None) -> None:
        """
        Preprocess sequences and targets for training.
        
        Args:
            sequences (Optional[np.ndarray]): Input sequences
            targets (Optional[np.ndarray]): Target values
        """
        if sequences is None:
            sequences = self.sequences
        if targets is None:
            targets = self.targets
        
        self.logger.info("Preprocessing data for LSTM training...")
        
        # Handle missing values in sequences
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        # Reshape sequences for scaling (flatten to 2D, scale, then reshape back)
        original_shape = sequences.shape
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])
        
        self.feature_scaler = StandardScaler()
        sequences_scaled = self.feature_scaler.fit_transform(sequences_2d)
        sequences = sequences_scaled.reshape(original_shape)
        
        # Scale targets
        self.target_scaler = StandardScaler()
        targets = self.target_scaler.fit_transform(targets)
        
        # Split data
        test_size = 0.2
        val_size = 0.2  # 20% of remaining data after test split
        
        # First split: train+val vs test
        X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
            sequences, targets, 
            test_size=test_size, 
            random_state=self.config['runtime']['random_seed'],
            shuffle=True
        )
        
        # Second split: train vs val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_size,
            random_state=self.config['runtime']['random_seed'],
            shuffle=True
        )
        
        self.logger.info(f"Data split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
    
    def build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.
        
        Returns:
            keras.Model: Compiled LSTM model
        """
        self.logger.info("Building LSTM model architecture...")
        
        # Model parameters
        sequence_length = self.X_train.shape[1]
        n_features = self.X_train.shape[2]
        n_targets = self.y_train.shape[1]
        
        hidden_units = self.lstm_config['hidden_units']
        dropout_rate = self.lstm_config['dropout_rate']
        recurrent_dropout = self.lstm_config['recurrent_dropout']
        
        # Build model
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            
            # First LSTM layer
            LSTM(
                hidden_units[0], 
                return_sequences=True if len(hidden_units) > 1 else False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                name='lstm_1'
            ),
            
            # Batch normalization
            BatchNormalization(name='batch_norm_1'),
        ])
        
        # Additional LSTM layers if specified
        for i, units in enumerate(hidden_units[1:], 2):
            model.add(LSTM(
                units,
                return_sequences=False,  # Last LSTM layer should not return sequences
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                name=f'lstm_{i}'
            ))
            model.add(BatchNormalization(name=f'batch_norm_{i}'))
        
        # Dense layers for final prediction
        model.add(Dense(
            hidden_units[-1] // 2, 
            activation='relu',
            name='dense_1'
        ))
        model.add(Dropout(dropout_rate, name='dropout_final'))
        
        # Output layer
        model.add(Dense(n_targets, activation='linear', name='output'))
        
        # Compile model
        if TF_VERSION == "2.x":
            optimizer = optimizers.Adam(learning_rate=self.lstm_config['learning_rate'])
        else:
            optimizer = optimizers.Adam(lr=self.lstm_config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss=self.lstm_config['loss_function'],
            metrics=self.lstm_config['metrics']
        )
        
        self.model = model
        
        # Print model summary
        self.logger.info("Model architecture:")
        try:
            model.summary(print_fn=lambda x: self.logger.info(x))
        except:
            self.logger.info("Model summary not available")
        
        return model
    
    def train_model(self) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Returns:
            keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting LSTM model training...")
        
        # Callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.lstm_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Model checkpoint
        checkpoint_path = Path(self.paths['models']) / 'lstm_best_model.h5'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks_list.append(ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.logger.info("Training completed!")
        return self.history
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Inverse transform predictions and true values
        y_pred_orig = self.target_scaler.inverse_transform(y_pred)
        y_test_orig = self.target_scaler.inverse_transform(self.y_test)
        
        # Calculate metrics for each target
        evaluation_results = {}
        
        for i, target_name in enumerate(self.target_names):
            y_true = y_test_orig[:, i]
            y_pred_target = y_pred_orig[:, i]
            
            mse = mean_squared_error(y_true, y_pred_target)
            mae = mean_absolute_error(y_true, y_pred_target)
            r2 = r2_score(y_true, y_pred_target)
            
            evaluation_results[target_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_test_orig, y_pred_orig)
        overall_mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        evaluation_results['overall'] = {
            'mse': overall_mse,
            'mae': overall_mae,
            'rmse': np.sqrt(overall_mse)
        }
        
        # Log results
        self.logger.info("Evaluation Results:")
        for target, metrics in evaluation_results.items():
            if target != 'overall':
                self.logger.info(f"  {target}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
        
        self.logger.info(f"  Overall: MAE={evaluation_results['overall']['mae']:.3f}")
        
        return evaluation_results
    
    def predict_career_trajectory(self, driver_sequence: np.ndarray) -> Dict:
        """
        Predict career trajectory for a single driver sequence.
        
        Args:
            driver_sequence (np.ndarray): Driver's performance sequence
            
        Returns:
            Dict: Predicted career metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure sequence has correct shape
        if driver_sequence.ndim == 2:
            driver_sequence = driver_sequence.reshape(1, *driver_sequence.shape)
        
        # Scale input
        sequence_2d = driver_sequence.reshape(-1, driver_sequence.shape[-1])
        sequence_scaled = self.feature_scaler.transform(sequence_2d)
        sequence_scaled = sequence_scaled.reshape(driver_sequence.shape)
        
        # Make prediction
        prediction = self.model.predict(sequence_scaled, verbose=0)
        
        # Inverse transform prediction
        prediction_orig = self.target_scaler.inverse_transform(prediction)
        
        # Format results
        results = {}
        for i, target_name in enumerate(self.target_names):
            results[target_name] = prediction_orig[0, i]
        
        return results
    
    def save_model(self) -> None:
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        model_path = Path(self.paths['models']) / 'lstm_career_predictor.h5'
        self.model.save(str(model_path))
        
        # Save preprocessing components
        import joblib
        preprocessor_path = Path(self.paths['models']) / 'lstm_preprocessors.pkl'
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'target_names': self.target_names,
            'sequence_length': self.lstm_config['sequence_length']
        }, preprocessor_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Preprocessors saved to {preprocessor_path}")


def check_tensorflow_installation() -> bool:
    """
    Check if TensorFlow is properly installed and working.
    
    Returns:
        bool: True if TensorFlow is working, False otherwise
    """
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
        
        # Test basic functionality
        test_tensor = tf.constant([1, 2, 3, 4])
        print(f"TensorFlow test: {test_tensor}")
        
        return True
        
    except Exception as e:
        print(f"TensorFlow error: {e}")
        return False


if __name__ == "__main__":
    print("🧠 NASCAR LSTM Model - TensorFlow Compatibility Check")
    print("=" * 60)
    
    # Check TensorFlow installation
    if check_tensorflow_installation():
        print("✅ TensorFlow is working correctly!")
        
        # Try initializing the predictor
        try:
            predictor = NASCARLSTMPredictor()
            print("✅ LSTM predictor initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing predictor: {e}")
    else:
        print("❌ TensorFlow installation issues detected")
        print("\nTo install TensorFlow:")
        print("pip install tensorflow")
        print("or")
        print("conda install tensorflow")