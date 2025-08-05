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
    
    def train(self, cup_series_df: pd.DataFrame, min_history_races: int = 10, 
              seasons_to_predict: List[int] = None) -> Dict:
        """
        Train the model with NASCAR Cup Series data
        
        Args:
            cup_series_df: DataFrame with NASCAR Cup Series race data
            min_history_races: Minimum races needed to make a prediction
            seasons_to_predict: List of seasons to use for training targets (default: 2010-2024)
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Training finishing position predictor with NASCAR data...")
        
        if cup_series_df.empty:
            raise ValueError("NASCAR data is required")
        
        # Store reference to full dataset for history extraction
        self.full_dataset = cup_series_df.copy()
        
        # Clean up the data
        self.full_dataset['Start'] = pd.to_numeric(self.full_dataset['Start'], errors='coerce')
        self.full_dataset['Finish'] = pd.to_numeric(self.full_dataset['Finish'], errors='coerce')
        self.full_dataset['Length'] = pd.to_numeric(self.full_dataset['Length'], errors='coerce')
        
        # Default seasons for training targets
        if seasons_to_predict is None:
            seasons_to_predict = list(range(2010, 2025))
        
        self.logger.info(f"Creating training examples from seasons {min(seasons_to_predict)}-{max(seasons_to_predict)}")
        
        features_list = []
        targets = []
        training_examples = []
        
        # Group by driver to create training examples
        for driver_name in self.full_dataset['Driver'].unique():
            if pd.isna(driver_name) or driver_name == "NA":
                continue
                
            driver_races = self.full_dataset[
                self.full_dataset['Driver'] == driver_name
            ].sort_values(['Season', 'Race'])
            
            # Skip drivers with insufficient data
            if len(driver_races) < min_history_races + 5:
                continue
            
            # Create training examples: use race N-k through N-1 to predict race N
            for i in range(min_history_races, len(driver_races)):
                target_race = driver_races.iloc[i]
                
                # Only use races from our target seasons
                if target_race['Season'] not in seasons_to_predict:
                    continue
                
                # Skip if target finish is missing
                if pd.isna(target_race['Finish']):
                    continue
                
                # Get history (races 0 through i-1)
                history = driver_races.iloc[:i]
                
                # Skip if insufficient history
                if len(history) < min_history_races:
                    continue
                
                try:
                    # Determine track type for target race
                    track_type = self._classify_track_type(
                        target_race['Track'], target_race['Length']
                    )
                    
                    # Prepare features
                    feature_vector = self.prepare_features(history, track_type)
                    
                    features_list.append(feature_vector)
                    targets.append(target_race['Finish'])
                    
                    training_examples.append({
                        'driver': driver_name,
                        'target_season': target_race['Season'],
                        'target_race': target_race['Race'],
                        'history_races': len(history),
                        'track_type': track_type
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Skipping training example for {driver_name}: {str(e)}")
                    continue
        
        if not features_list:
            raise ValueError("No valid training examples could be created")
        
        X = np.array(features_list)
        y = np.array(targets)
        
        self.logger.info(f"Created {len(X)} training examples from {len(set(ex['driver'] for ex in training_examples))} drivers")
        
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
            'feature_importance': dict(zip(self.feature_names, self.model.coef_)),
            'drivers_count': len(set(ex['driver'] for ex in training_examples)),
            'seasons_used': sorted(seasons_to_predict)
        }
        
        self.logger.info(f"Training complete - Val MAE: {self.training_metrics['val_mae']:.2f}, "
                        f"Val R²: {self.training_metrics['val_r2']:.3f}")
        
        return self.training_metrics
    
    def _extract_driver_history(self, row) -> Optional[pd.DataFrame]:
        """
        Extract driver history from a row using the NASCAR CSV data structure
        """
        # In training, we assume the full dataset is available as self.full_dataset
        if not hasattr(self, 'full_dataset') or self.full_dataset is None:
            return None
        
        driver_name = row.get('driver_name') or row.get('Driver')
        cutoff_season = row.get('cutoff_season', 2024)
        cutoff_race = row.get('cutoff_race', 999)  # Use high number if not specified
        
        if not driver_name:
            return None
        
        # Get all races for this driver up to the cutoff point
        driver_races = self.full_dataset[
            (self.full_dataset['Driver'] == driver_name) &
            (
                (self.full_dataset['Season'] < cutoff_season) |
                (
                    (self.full_dataset['Season'] == cutoff_season) &
                    (self.full_dataset['Race'] < cutoff_race)
                )
            )
        ].copy()
        
        # Sort chronologically
        driver_races = driver_races.sort_values(['Season', 'Race'])
        
        # Clean up the data - handle missing values
        driver_races['Start'] = pd.to_numeric(driver_races['Start'], errors='coerce')
        driver_races['Finish'] = pd.to_numeric(driver_races['Finish'], errors='coerce')
        
        # Filter out races with missing finish positions (can't use for training)
        driver_races = driver_races.dropna(subset=['Finish'])
        
        return driver_races if len(driver_races) >= 5 else None
    
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
    
    def _classify_track_type(self, track_name: str, track_length: float) -> str:
        """
        Classify track type based on name and length using NASCAR data
        """
        if pd.isna(track_name):
            track_name = ""
        if pd.isna(track_length):
            track_length = 1.5
            
        track_name = str(track_name).lower()
        
        # Road courses - check name first
        road_keywords = ['road', 'glen', 'sonoma', 'cota', 'roval', 'mexico city']
        if any(keyword in track_name for keyword in road_keywords):
            return 'road'
        
        # Superspeedways - 2.0+ miles or specific tracks
        superspeedway_tracks = ['daytona', 'talladega']
        if track_length >= 2.0 or any(name in track_name for name in superspeedway_tracks):
            return 'superspeedway'
        
        # Short tracks - under 1.0 mile
        if track_length < 1.0:
            return 'short'
        
        # Everything else is intermediate (1.0 - 2.0 miles)
        return 'intermediate'
    
    def predict_for_driver(self, cup_series_df: pd.DataFrame, driver_name: str, 
                          next_track_name: str = None, next_track_length: float = 1.5) -> Dict:
        """
        Make a prediction for a specific driver using their NASCAR history
        
        Args:
            cup_series_df: Full NASCAR Cup Series DataFrame
            driver_name: Name of driver to predict for
            next_track_name: Name of next track (for track type classification)
            next_track_length: Length of next track in miles
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get driver's race history
        driver_races = cup_series_df[
            cup_series_df['Driver'] == driver_name
        ].copy()
        
        if driver_races.empty:
            raise ValueError(f"No data found for driver: {driver_name}")
        
        # Clean and sort the data
        driver_races['Start'] = pd.to_numeric(driver_races['Start'], errors='coerce')
        driver_races['Finish'] = pd.to_numeric(driver_races['Finish'], errors='coerce')
        driver_races = driver_races.dropna(subset=['Finish'])
        driver_races = driver_races.sort_values(['Season', 'Race'])
        
        if len(driver_races) < 5:
            raise ValueError(f"Insufficient race history for {driver_name} ({len(driver_races)} races)")
        
        # Classify track type for next race
        track_type = self._classify_track_type(next_track_name, next_track_length)
        
        # Make prediction using the existing predict method
        return self.predict(driver_races, track_type)
        """Get model information"""
        return {
            'name': 'Finishing Position Predictor',
            'type': 'Linear Regression',
            'features': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'description': 'Predicts NASCAR finishing positions based on historical performance and track characteristics'
        }


# Example usage with NASCAR data
if __name__ == "__main__":
    # Example of how to use the model with your NASCAR CSV data
    
    # Load the NASCAR data
    # df = pd.read_csv('data/raw/cup_series.csv')
    
    # Create and train the predictor
    predictor = FinishPositionPredictor()
    
    # Train the model (this would use the actual data)
    # training_results = predictor.train(df)
    # print("Training results:", training_results)
    
    # Make predictions for specific drivers
    # prediction = predictor.predict_for_driver(
    #     df, 
    #     driver_name="Kyle Larson",
    #     next_track_name="Charlotte Motor Speedway", 
    #     next_track_length=1.5
    # )
    # print("Kyle Larson prediction:", prediction)
    
    # Create sample data for testing the structure
    sample_history = pd.DataFrame({
        'Season': [2024] * 10,
        'Race': range(1, 11),
        'Finish': [12, 8, 15, 5, 22, 9, 18, 3, 11, 7],
        'Start': [15, 12, 20, 8, 25, 14, 22, 6, 16, 10],
        'Track': ['Charlotte Motor Speedway'] * 10,
        'Length': [1.5] * 10,
        'Driver': ['Test Driver'] * 10
    })
    
    print("Sample prediction with test data:")
    try:
        features = predictor.prepare_features(sample_history, 'intermediate')
        print("Features extracted:", dict(zip(predictor.feature_names, features)))
    except Exception as e:
        print("Error:", str(e))
    
    print("\nModel info:")
    print(predictor.get_model_info())