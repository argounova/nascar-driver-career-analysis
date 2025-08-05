"""
NASCAR Win Probability Predictor Model

This model uses logistic regression to predict the probability of a driver
winning their next race based on their historical performance, recent form,
and track-specific factors.

Features used:
- Historical win rate (last 5, 10, 20 races)
- Average finishing position trends
- Track-specific win rates
- Recent performance momentum
- Starting position tendencies
- Equipment quality indicators

Model outputs:
- Win probability (0.0 to 1.0)
- Confidence intervals
- Key contributing factors
- Comparison to field average
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class WinProbabilityPredictor:
    """
    Predicts NASCAR win probabilities using logistic regression
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor
        
        Args:
            random_state: Random state for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'career_win_rate',
            'recent_win_rate_5',
            'recent_win_rate_10', 
            'recent_win_rate_20',
            'avg_finish_5',
            'avg_finish_10',
            'avg_start_5',
            'track_win_rate',
            'track_avg_finish',
            'momentum_score',
            'consistency_score',
            'top5_rate_recent',
            'led_laps_rate',
            'equipment_quality_score'
        ]
        self.random_state = random_state
        self.training_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, race_history: pd.DataFrame, track_type: str = 'intermediate',
                        track_name: str = None) -> np.ndarray:
        """
        Prepare features for a driver's historical data
        
        Args:
            race_history: DataFrame of race results for the driver
            track_type: Type of track ('superspeedway', 'short', 'intermediate', 'road')
            track_name: Specific track name for track-specific analysis
            
        Returns:
            Feature vector as numpy array
        """
        if race_history.empty:
            raise ValueError("Race history is required")
        
        # Sort by most recent first
        if 'Season' in race_history.columns and 'Race' in race_history.columns:
            race_history = race_history.sort_values(['Season', 'Race'], ascending=False)
        
        # Calculate rolling windows
        last_5 = race_history.head(5)
        last_10 = race_history.head(10)
        last_20 = race_history.head(20)
        career = race_history
        
        # Win rates
        career_win_rate = self._safe_mean(career, 'Win')
        recent_win_rate_5 = self._safe_mean(last_5, 'Win')
        recent_win_rate_10 = self._safe_mean(last_10, 'Win')
        recent_win_rate_20 = self._safe_mean(last_20, 'Win')
        
        # Finishing position trends
        avg_finish_5 = self._safe_mean(last_5, 'Finish', 20.0)
        avg_finish_10 = self._safe_mean(last_10, 'Finish', 20.0)
        
        # Starting position trends
        avg_start_5 = self._safe_mean(last_5, 'Start', 20.0)
        
        # Track-specific performance
        track_win_rate, track_avg_finish = self._get_track_performance(race_history, track_type, track_name)
        
        # Momentum score (recent performance vs career average)
        momentum_score = self._calculate_momentum(race_history)
        
        # Consistency score
        consistency_score = self._calculate_consistency(last_10['Finish'].dropna().values)
        
        # Top-5 rate recent
        top5_rate_recent = self._calculate_top5_rate(last_10)
        
        # Laps led rate (indicator of speed/competitiveness)
        led_laps_rate = self._calculate_led_laps_rate(last_10)
        
        # Equipment quality score
        equipment_quality_score = self._calculate_equipment_quality(race_history)
        
        features = np.array([
            career_win_rate,
            recent_win_rate_5,
            recent_win_rate_10,
            recent_win_rate_20,
            avg_finish_5,
            avg_finish_10,
            avg_start_5,
            track_win_rate,
            track_avg_finish,
            momentum_score,
            consistency_score,
            top5_rate_recent,
            led_laps_rate,
            equipment_quality_score
        ])
        
        return features
    
    def _safe_mean(self, df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        """Calculate mean with handling for missing values"""
        if column not in df.columns or df[column].empty:
            return default
        
        values = pd.to_numeric(df[column], errors='coerce').dropna()
        return values.mean() if not values.empty else default
    
    def _get_track_performance(self, race_history: pd.DataFrame, track_type: str, 
                              track_name: str = None) -> Tuple[float, float]:
        """Get track-specific win rate and average finish"""
        if track_name:
            # Specific track performance
            track_races = race_history[
                race_history['Track'].str.contains(track_name, case=False, na=False)
            ]
        else:
            # Track type performance
            track_races = self._filter_by_track_type(race_history, track_type)
        
        if track_races.empty:
            return 0.0, 20.0
        
        win_rate = self._safe_mean(track_races, 'Win')
        avg_finish = self._safe_mean(track_races, 'Finish', 20.0)
        
        return win_rate, avg_finish
    
    def _filter_by_track_type(self, race_history: pd.DataFrame, track_type: str) -> pd.DataFrame:
        """Filter races by track type"""
        if 'Track' not in race_history.columns or 'Length' not in race_history.columns:
            return pd.DataFrame()
        
        if track_type.lower() == 'superspeedway':
            return race_history[
                (race_history['Length'] >= 2.0) | 
                (race_history['Track'].str.contains('Daytona|Talladega', case=False, na=False))
            ]
        elif track_type.lower() == 'short':
            return race_history[race_history['Length'] < 1.0]
        elif track_type.lower() == 'road':
            return race_history[
                race_history['Track'].str.contains('Road|Glen|Sonoma|COTA|Roval', case=False, na=False)
            ]
        else:  # intermediate
            return race_history[
                (race_history['Length'] >= 1.0) & (race_history['Length'] < 2.0)
            ]
    
    def _calculate_momentum(self, race_history: pd.DataFrame) -> float:
        """Calculate performance momentum (recent vs career)"""
        if len(race_history) < 10:
            return 0.0
        
        recent_avg = self._safe_mean(race_history.head(5), 'Finish', 20.0)
        career_avg = self._safe_mean(race_history, 'Finish', 20.0)
        
        # Momentum is positive when recent performance is better (lower finish)
        # Normalize to -1 to 1 scale
        if career_avg > 0:
            momentum = (career_avg - recent_avg) / career_avg
            return max(-1.0, min(1.0, momentum))
        return 0.0
    
    def _calculate_consistency(self, finish_positions: np.ndarray) -> float:
        """Calculate consistency score (inverse of coefficient of variation)"""
        if len(finish_positions) < 2:
            return 0.5
        
        mean_finish = np.mean(finish_positions)
        std_finish = np.std(finish_positions)
        
        if mean_finish > 0:
            cv = std_finish / mean_finish
            # Convert to 0-1 scale where 1 is most consistent
            consistency = max(0.0, min(1.0, 1.0 - (cv / 2.0)))
            return consistency
        return 0.5
    
    def _calculate_top5_rate(self, races: pd.DataFrame) -> float:
        """Calculate top-5 finish rate"""
        if 'Finish' not in races.columns or races.empty:
            return 0.0
        
        finishes = pd.to_numeric(races['Finish'], errors='coerce').dropna()
        if finishes.empty:
            return 0.0
        
        top5_count = (finishes <= 5).sum()
        return top5_count / len(finishes)
    
    def _calculate_led_laps_rate(self, races: pd.DataFrame) -> float:
        """Calculate rate of races where driver led laps"""
        if 'Led' not in races.columns or races.empty:
            return 0.0
        
        led_laps = pd.to_numeric(races['Led'], errors='coerce').fillna(0)
        races_led = (led_laps > 0).sum()
        return races_led / len(races) if len(races) > 0 else 0.0
    
    def _calculate_equipment_quality(self, race_history: pd.DataFrame) -> float:
        """
        Estimate equipment quality based on team/manufacturer consistency
        and average performance vs field
        """
        if race_history.empty:
            return 0.5
        
        # Use average finish as a proxy for equipment quality
        # Better equipment should lead to consistently better finishes
        avg_finish = self._safe_mean(race_history, 'Finish', 20.0)
        
        # Normalize to 0-1 scale (1 = best equipment, 0 = worst)
        # Assume field average is around position 20
        equipment_score = max(0.0, min(1.0, (40 - avg_finish) / 40))
        return equipment_score
    
    def train(self, cup_series_df: pd.DataFrame, min_history_races: int = 15,
              seasons_to_predict: List[int] = None) -> Dict:
        """
        Train the model with NASCAR Cup Series data
        
        Args:
            cup_series_df: DataFrame with NASCAR Cup Series race data
            min_history_races: Minimum races needed to make a prediction
            seasons_to_predict: List of seasons to use for training targets
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Training win probability predictor...")
        
        if cup_series_df.empty:
            raise ValueError("NASCAR data is required")
        
        # Store reference to full dataset
        self.full_dataset = cup_series_df.copy()
        
        # Clean up the data
        self.full_dataset['Start'] = pd.to_numeric(self.full_dataset['Start'], errors='coerce')
        self.full_dataset['Finish'] = pd.to_numeric(self.full_dataset['Finish'], errors='coerce')
        self.full_dataset['Win'] = pd.to_numeric(self.full_dataset['Win'], errors='coerce').fillna(0)
        self.full_dataset['Led'] = pd.to_numeric(self.full_dataset['Led'], errors='coerce').fillna(0)
        
        # Default seasons for training
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
            
            # Create training examples
            for i in range(min_history_races, len(driver_races)):
                target_race = driver_races.iloc[i]
                
                # Only use races from our target seasons
                if target_race['Season'] not in seasons_to_predict:
                    continue
                
                # Skip if target win status is missing
                if pd.isna(target_race['Win']):
                    continue
                
                # Get history
                history = driver_races.iloc[:i]
                
                if len(history) < min_history_races:
                    continue
                
                try:
                    # Determine track type
                    track_type = self._classify_track_type(
                        target_race['Track'], target_race['Length']
                    )
                    
                    # Prepare features
                    feature_vector = self.prepare_features(
                        history, track_type, target_race['Track']
                    )
                    
                    features_list.append(feature_vector)
                    targets.append(int(target_race['Win']))  # Ensure binary
                    
                    training_examples.append({
                        'driver': driver_name,
                        'target_season': target_race['Season'],
                        'target_race': target_race['Race'],
                        'won': bool(target_race['Win'])
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Skipping training example for {driver_name}: {str(e)}")
                    continue
        
        if not features_list:
            raise ValueError("No valid training examples could be created")
        
        X = np.array(features_list)
        y = np.array(targets)
        
        # Check class balance
        win_rate = y.mean()
        self.logger.info(f"Created {len(X)} training examples with {win_rate:.1%} win rate")
        
        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        train_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        val_pred = self.model.predict(X_val_scaled)
        val_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'val_precision': precision_score(y_val, val_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'val_recall': recall_score(y_val, val_pred, zero_division=0),
            'train_f1': f1_score(y_train, train_pred, zero_division=0),
            'val_f1': f1_score(y_val, val_pred, zero_division=0),
            'train_auc': roc_auc_score(y_train, train_pred_proba),
            'val_auc': roc_auc_score(y_val, val_pred_proba),
            'train_log_loss': log_loss(y_train, train_pred_proba),
            'val_log_loss': log_loss(y_val, val_pred_proba),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'win_rate': win_rate,
            'feature_importance': dict(zip(self.feature_names, self.model.coef_[0])),
            'drivers_count': len(set(ex['driver'] for ex in training_examples))
        }
        
        self.logger.info(f"Training complete - Val Accuracy: {self.training_metrics['val_accuracy']:.3f}, "
                        f"Val AUC: {self.training_metrics['val_auc']:.3f}")
        
        return self.training_metrics
    
    def _classify_track_type(self, track_name: str, track_length: float) -> str:
        """Classify track type based on name and length"""
        if pd.isna(track_name):
            track_name = ""
        if pd.isna(track_length):
            track_length = 1.5
            
        track_name = str(track_name).lower()
        
        # Road courses
        road_keywords = ['road', 'glen', 'sonoma', 'cota', 'roval', 'mexico']
        if any(keyword in track_name for keyword in road_keywords):
            return 'road'
        
        # Superspeedways
        superspeedway_tracks = ['daytona', 'talladega']
        if track_length >= 2.0 or any(name in track_name for name in superspeedway_tracks):
            return 'superspeedway'
        
        # Short tracks
        if track_length < 1.0:
            return 'short'
        
        return 'intermediate'
    
    def predict_for_driver(self, cup_series_df: pd.DataFrame, driver_name: str, 
                          next_track_name: str = None, next_track_length: float = 1.5) -> Dict:
        """
        Predict win probability for a specific driver
        
        Args:
            cup_series_df: Full NASCAR Cup Series DataFrame
            driver_name: Name of driver to predict for
            next_track_name: Name of next track
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
        driver_races = self._clean_driver_data(driver_races)
        
        if len(driver_races) < 5:
            raise ValueError(f"Insufficient race history for {driver_name}")
        
        # Classify track type
        track_type = self._classify_track_type(next_track_name, next_track_length)
        
        # Prepare features
        features = self.prepare_features(driver_races, track_type, next_track_name)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        win_probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Calculate field-adjusted probability (assuming ~3% base win rate in 36-car field)
        field_average = 1.0 / 36  # ~2.8%
        relative_probability = win_probability / field_average
        
        # Get feature contributions
        feature_contributions = dict(zip(self.feature_names, features))
        
        return {
            'win_probability': win_probability,
            'win_probability_percent': win_probability * 100,
            'relative_to_field': relative_probability,
            'confidence': min(1.0, max(0.0, self.training_metrics.get('val_auc', 0.5))),
            'track_type': track_type,
            'features': feature_contributions,
            'interpretation': self._interpret_probability(win_probability)
        }
    
    def _clean_driver_data(self, driver_races: pd.DataFrame) -> pd.DataFrame:
        """Clean driver race data"""
        driver_races['Start'] = pd.to_numeric(driver_races['Start'], errors='coerce')
        driver_races['Finish'] = pd.to_numeric(driver_races['Finish'], errors='coerce')
        driver_races['Win'] = pd.to_numeric(driver_races['Win'], errors='coerce').fillna(0)
        driver_races['Led'] = pd.to_numeric(driver_races['Led'], errors='coerce').fillna(0)
        
        # Remove races with missing finish data
        driver_races = driver_races.dropna(subset=['Finish'])
        driver_races = driver_races.sort_values(['Season', 'Race'])
        
        return driver_races
    
    def _interpret_probability(self, probability: float) -> str:
        """Provide human-readable interpretation of win probability"""
        if probability < 0.01:
            return "Very Low"
        elif probability < 0.03:
            return "Low"
        elif probability < 0.07:
            return "Below Average"
        elif probability < 0.15:
            return "Average"
        elif probability < 0.25:
            return "Above Average"
        elif probability < 0.40:
            return "High"
        else:
            return "Very High"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_names, self.model.coef_[0]))
    
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
    def load(cls, filepath: str) -> 'WinProbabilityPredictor':
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
            'name': 'Win Probability Predictor',
            'type': 'Logistic Regression',
            'features': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'description': 'Predicts NASCAR win probabilities based on historical performance and track factors'
        }


# Example usage
if __name__ == "__main__":
    predictor = WinProbabilityPredictor()
    
    # Create sample data for testing
    sample_history = pd.DataFrame({
        'Season': [2024] * 20,
        'Race': range(1, 21),
        'Finish': [12, 8, 15, 5, 22, 9, 18, 3, 11, 7, 14, 6, 20, 2, 16, 4, 13, 1, 19, 10],
        'Start': [15, 12, 20, 8, 25, 14, 22, 6, 16, 10, 18, 9, 24, 5, 19, 7, 17, 3, 23, 11],
        'Win': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Led': [0, 0, 0, 5, 0, 0, 0, 15, 0, 0, 0, 10, 0, 25, 0, 8, 0, 120, 0, 0],
        'Track': ['Charlotte Motor Speedway'] * 20,
        'Length': [1.5] * 20,
        'Driver': ['Test Driver'] * 20
    })
    
    print("Sample win probability features:")
    try:
        features = predictor.prepare_features(sample_history, 'intermediate', 'Charlotte Motor Speedway')
        print("Features extracted:", dict(zip(predictor.feature_names, features)))
    except Exception as e:
        print("Error:", str(e))
    
    print("\nModel info:")
    print(predictor.get_model_info())