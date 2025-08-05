"""
NASCAR Finishing Position Predictor Model

This model uses linear regression to predict a driver's finishing position
based on their historical performance, starting position tendencies, and recent form.

Features used:
- Average finishing position (last 5, 10, 20 races)
- Average starting position (last 5, 10, 20 races)
- Finishing position improvement trend (start vs finish)
- Recent consistency (coefficient of variation)
- Track type performance adjustments

Model outputs:
- Predicted finishing position
- Confidence interval
- Performance compared to starting position
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class FinishPositionPredictor:
    """
    Predicts NASCAR finishing positions using linear regression
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor
        
        Args:
            random_state: Random state for reproducibility
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'avg_finish_5',
            'avg_finish_10', 
            'avg_finish_20',
            'avg_start_5',
            'avg_start_10',
            'avg_start_20',
            'position_improvement_trend',
            'consistency_score',
            'track_performance_factor',
            'recent_form_score'
        ]
        self.random_state = random_state
        self.training_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, race_history: pd.DataFrame, track_type: str = 'intermediate') -> np.ndarray:
        """
        Prepare features for a driver's historical data
        
        Args:
            race_history: DataFrame of race results for the driver
            track_type: Type of track ('superspeedway', 'short', 'intermediate', 'road')
            
        Returns:
            Feature vector as numpy array
        """
        if race_history.empty:
            raise ValueError("Race history is required")
        
        # Sort by most recent first (assuming Season/Race columns exist)
        if 'Season' in race_history.columns and 'Race' in race_history.columns:
            race_history = race_history.sort_values(['Season', 'Race'], ascending=False)
        
        # Calculate rolling averages
        last_5 = race_history.head(5)
        last_10 = race_history.head(10)
        last_20 = race_history.head(20)
        
        avg_finish_5 = self._safe_mean(last_5, 'Finish')
        avg_finish_10 = self._safe_mean(last_10, 'Finish')
        avg_finish_20 = self._safe_mean(last_20, 'Finish')
        
        avg_start_5 = self._safe_mean(last_5, 'Start')
        avg_start_10 = self._safe_mean(last_10, 'Start')
        avg_start_20 = self._safe_mean(last_20, 'Start')
        
        # Position improvement trend (negative means they typically gain positions)
        position_improvements = []
        for _, race in last_10.iterrows():
            finish = race.get('Finish', 40)
            start = race.get('Start', 40)
            if pd.notna(finish) and pd.notna(start):
                position_improvements.append(finish - start)
        
        improvement_trend = np.mean(position_improvements) if position_improvements else 0
        
        # Consistency score (coefficient of variation)
        finish_positions = last_10['Finish'].dropna().values
        consistency_score = self._calculate_consistency(finish_positions)
        
        # Track performance factor
        track_performance_factor = self._get_track_performance_factor(race_history, track_type)
        
        # Recent form score (weighted average of last 5 races)
        recent_form_score = self._calculate_recent_form(last_5)
        
        features = np.array([
            avg_finish_5,
            avg_finish_10,
            avg_finish_20,
            avg_start_5,
            avg_start_10,
            avg_start_20,
            improvement_trend,
            consistency_score,
            track_performance_factor,
            recent_form_score
        ])
        
        return features
    
    def _safe_mean(self, df: pd.DataFrame, column: str, default: float = 20.0) -> float:
        """Calculate mean with handling for missing values"""
        if column not in df.columns or df[column].empty:
            return default
        
        values = df[column].dropna()
        return values.mean() if not values.empty else default
    
    def _calculate_consistency(self, values: np.ndarray) -> float:
        """Calculate consistency score (coefficient of variation)"""
        if len(values) < 2:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        return std_val / mean_val if mean_val > 0 else 1.0
    
    def _get_track_performance_factor(self, race_history: pd.DataFrame, track_type: str) -> float:
        """Get performance factor for specific track type"""
        if 'Track' not in race_history.columns or 'Length' not in race_history.columns:
            return 1.0
        
        # Filter races by track type
        if track_type.lower() == 'superspeedway':
            similar_tracks = race_history[
                (race_history['Length'] >= 2.0) | 
                (race_history['Track'].str.contains('Daytona|Talladega', case=False, na=False))
            ]
        elif track_type.lower() == 'short':
            similar_tracks = race_history[race_history['Length'] < 1.0]
        elif track_type.lower() == 'road':
            similar_tracks = race_history[
                race_history['Track'].str.contains('Road|Glen|Sonoma|COTA', case=False, na=False)
            ]
        else:  # intermediate
            similar_tracks = race_history[
                (race_history['Length'] >= 1.0) & (race_history['Length'] < 2.0)
            ]
        
        if similar_tracks.empty:
            return 1.0
        
        track_avg = self._safe_mean(similar_tracks, 'Finish')
        overall_avg = self._safe_mean(race_history, 'Finish')
        
        return track_avg / overall_avg if overall_avg > 0 else 1.0
    
    def _calculate_recent_form(self, recent_races: pd.DataFrame) -> float:
        """Calculate recent form score with exponential weighting"""
        if recent_races.empty:
            return 20.0
        
        weighted_sum = 0
        total_weight = 0
        
        for i, (_, race) in enumerate(recent_races.iterrows()):
            weight = 0.8 ** i  # Exponential decay
            finish = race.get('Finish', 40)
            if pd.notna(finish):
                weighted_sum += finish * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 20.0
    
    def train(self, data: pd.DataFrame, target_column: str = 'next_race_finish', 
              track_type_column: str = 'track_type') -> Dict:
        """
        Train the model with historical data
        
        Args:
            data: DataFrame with driver histories and target finish positions
            target_column: Column name containing the target finish positions
            track_type_column: Column name containing track type information
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Training finishing position predictor...")
        
        if data.empty:
            raise ValueError("Training data is required")
        
        features_list = []
        targets = []
        
        for idx, row in data.iterrows():
            try:
                # Assume the race history is stored in a column or can be reconstructed
                # This would need to be adapted based on your actual data structure
                driver_history = self._extract_driver_history(row)
                track_type = row.get(track_type_column, 'intermediate')
                target_finish = row.get(target_column)
                
                if driver_history is not None and pd.notna(target_finish):
                    feature_vector = self.prepare_features(driver_history, track_type)
                    features_list.append(feature_vector)
                    targets.append(target_finish)
                    
            except Exception as e:
                self.logger.warning(f"Skipping row {idx}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No valid training examples found")
        
        X = np.array(features_list)
        y = np.array(targets)
        
        self.logger.info(f"Training with {len(X)} examples")
        
        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        self.training_metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_importance': dict(zip(self.feature_names, self.model.coef_))
        }
        
        self.logger.info(f"Training complete - Val MAE: {self.training_metrics['val_mae']:.2f}, "
                        f"Val R²: {self.training_metrics['val_r2']:.3f}")
        
        return self.training_metrics
    
    def _extract_driver_history(self, row) -> Optional[pd.DataFrame]:
        """
        Extract driver history from a row - this would need to be customized
        based on your actual data structure
        """
        # This is a placeholder - you'd implement this based on how your data is structured
        # For example, if you have a 'driver_history' column with serialized data:
        # return pd.DataFrame(row['driver_history'])
        
        # Or if you need to query from the main dataset:
        # return self.main_dataset[self.main_dataset['Driver'] == row['Driver']]
        
        return None
    
    def predict(self, race_history: pd.DataFrame, track_type: str = 'intermediate', 
                starting_position: Optional[int] = None) -> Dict:
        """
        Make prediction for a driver
        
        Args:
            race_history: DataFrame of driver's race history
            track_type: Type of track for next race
            starting_position: Starting position for next race (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.prepare_features(race_history, track_type)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence interval based on historical variance
        recent_finishes = race_history.head(10)['Finish'].dropna().values
        if len(recent_finishes) > 1:
            std_dev = np.std(recent_finishes)
            confidence_interval = {
                'lower': max(1, prediction - 1.96 * std_dev),
                'upper': min(40, prediction + 1.96 * std_dev)
            }
        else:
            confidence_interval = {'lower': max(1, prediction - 5), 'upper': min(40, prediction + 5)}
        
        # Position improvement prediction
        avg_start = self._safe_mean(race_history.head(10), 'Start')
        position_improvement = avg_start - prediction
        
        # Confidence score based on model performance and data quality
        confidence = max(0, min(1, self.training_metrics.get('val_r2', 0.5)))
        
        result = {
            'predicted_finish': max(1, min(40, round(prediction))),
            'predicted_finish_raw': prediction,
            'confidence_interval': confidence_interval,
            'position_improvement': round(position_improvement),
            'confidence': confidence,
            'features': dict(zip(self.feature_names, features))
        }
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_names, self.model.coef_))
    
    def save(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FinishPositionPredictor':
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data['training_metrics']
        predictor.is_trained = model_data['is_trained']
        
        return predictor
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'name': 'Finishing Position Predictor',
            'type': 'Linear Regression',
            'features': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'description': 'Predicts NASCAR finishing positions based on historical performance and track characteristics'
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the model
    predictor = FinishPositionPredictor()
    
    # Create sample data for testing
    sample_history = pd.DataFrame({
        'Season': [2024] * 10,
        'Race': range(1, 11),
        'Finish': [12, 8, 15, 5, 22, 9, 18, 3, 11, 7],
        'Start': [15, 12, 20, 8, 25, 14, 22, 6, 16, 10],
        'Track': ['Charlotte'] * 10,
        'Length': [1.5] * 10
    })
    
    print("Sample prediction:")
    print("Features:", predictor.prepare_features(sample_history, 'intermediate'))
    
    print("\nModel info:")
    print(predictor.get_model_info())