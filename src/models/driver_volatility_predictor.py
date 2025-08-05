"""
NASCAR Driver Volatility Predictor Model

This model predicts the variability/volatility in a driver's performance using
a combination of regression and time series analysis. It forecasts how consistent
or inconsistent a driver's finishing positions will be in upcoming races.

Volatility is measured as the standard deviation of finishing positions over
a rolling window, providing insights into driver reliability and risk.

Features used:
- Historical finish position variance patterns
- Track-specific volatility patterns
- Equipment change indicators
- Season progression effects
- Weather sensitivity proxies
- Competitive field strength indicators

Model outputs:
- Expected volatility (standard deviation of finishes)
- Volatility percentile vs field
- Risk assessment categories
- Consistency confidence intervals
- Performance bands (typical range of finishes)

Perfect for histogram visualization showing distribution of predicted finish positions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

class DriverVolatilityPredictor:
    """
    Predicts NASCAR driver performance volatility using random forest regression
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor
        
        Args:
            random_state: Random state for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'historical_volatility',
            'recent_volatility_5',
            'recent_volatility_10',
            'volatility_trend',
            'track_volatility',
            'season_progress',
            'avg_field_strength',
            'equipment_stability_score',
            'weather_sensitivity',
            'starting_position_variance',
            'performance_momentum',
            'consistency_decay_rate',
            'competitive_pressure_index',
            'mechanical_failure_rate'
        ]
        self.random_state = random_state
        self.training_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, race_history: pd.DataFrame, track_type: str = 'intermediate',
                        season_progress: float = 0.5, field_strength: float = 1.0) -> np.ndarray:
        """
        Prepare features for volatility prediction
        
        Args:
            race_history: DataFrame of race results for the driver
            track_type: Type of track for next race
            season_progress: Progress through season (0.0 to 1.0)
            field_strength: Relative strength of competitive field
            
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
        
        # Historical volatility (career standard deviation)
        historical_volatility = self._calculate_volatility(career, 'Finish')
        
        # Recent volatility patterns
        recent_volatility_5 = self._calculate_volatility(last_5, 'Finish')
        recent_volatility_10 = self._calculate_volatility(last_10, 'Finish')
        
        # Volatility trend (recent vs historical)
        volatility_trend = self._calculate_volatility_trend(race_history)
        
        # Track-specific volatility
        track_volatility = self._get_track_volatility(race_history, track_type)
        
        # Season progression effect (volatility often increases late in season)
        # Passed as parameter
        
        # Average field strength (affects variance in results)
        # Passed as parameter
        
        # Equipment stability score
        equipment_stability_score = self._calculate_equipment_stability(race_history)
        
        # Weather sensitivity (proxy using track surface/type performance variance)
        weather_sensitivity = self._calculate_weather_sensitivity(race_history)
        
        # Starting position variance
        starting_position_variance = self._calculate_volatility(career, 'Start')
        
        # Performance momentum
        performance_momentum = self._calculate_performance_momentum(race_history)
        
        # Consistency decay rate
        consistency_decay_rate = self._calculate_consistency_decay(race_history)
        
        # Competitive pressure index
        competitive_pressure_index = self._calculate_competitive_pressure(race_history)
        
        # Mechanical failure rate
        mechanical_failure_rate = self._calculate_failure_rate(race_history)
        
        features = np.array([
            historical_volatility,
            recent_volatility_5,
            recent_volatility_10,
            volatility_trend,
            track_volatility,
            season_progress,
            field_strength,
            equipment_stability_score,
            weather_sensitivity,
            starting_position_variance,
            performance_momentum,
            consistency_decay_rate,
            competitive_pressure_index,
            mechanical_failure_rate
        ])
        
        return features
    
    def _calculate_volatility(self, races: pd.DataFrame, column: str) -> float:
        """Calculate standard deviation of specified column"""
        if column not in races.columns or races.empty:
            return 5.0  # Default moderate volatility
        
        values = pd.to_numeric(races[column], errors='coerce').dropna()
        if len(values) < 2:
            return 5.0
        
        return float(np.std(values))
    
    def _calculate_volatility_trend(self, race_history: pd.DataFrame) -> float:
        """Calculate trend in volatility over time"""
        if len(race_history) < 20:
            return 0.0
        
        # Split into early and recent periods
        mid_point = len(race_history) // 2
        early_races = race_history.iloc[mid_point:]  # Earlier races (bottom half)
        recent_races = race_history.iloc[:mid_point]  # Recent races (top half)
        
        early_volatility = self._calculate_volatility(early_races, 'Finish')
        recent_volatility = self._calculate_volatility(recent_races, 'Finish')
        
        # Positive trend means increasing volatility
        if early_volatility > 0:
            return (recent_volatility - early_volatility) / early_volatility
        return 0.0
    
    def _get_track_volatility(self, race_history: pd.DataFrame, track_type: str) -> float:
        """Get volatility for specific track type"""
        track_races = self._filter_by_track_type(race_history, track_type)
        return self._calculate_volatility(track_races, 'Finish')
    
    def _filter_by_track_type(self, race_history: pd.DataFrame, track_type: str) -> pd.DataFrame:
        """Filter races by track type"""
        if 'Track' not in race_history.columns or 'Length' not in race_history.columns:
            return race_history.head(10)  # Return sample if no track data
        
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
    
    def _calculate_equipment_stability(self, race_history: pd.DataFrame) -> float:
        """
        Calculate equipment stability score based on team/manufacturer changes
        """
        if 'Team' not in race_history.columns or race_history.empty:
            return 0.8  # Default high stability
        
        # Count unique teams (proxy for equipment changes)
        unique_teams = race_history['Team'].nunique()
        total_seasons = race_history['Season'].nunique() if 'Season' in race_history.columns else 1
        
        # Normalize team changes per season
        team_changes_per_season = unique_teams / max(total_seasons, 1)
        
        # Convert to stability score (lower changes = higher stability)
        stability = max(0.0, min(1.0, 1.0 - (team_changes_per_season - 1.0) * 0.3))
        return stability
    
    def _calculate_weather_sensitivity(self, race_history: pd.DataFrame) -> float:
        """
        Calculate weather sensitivity using surface type performance variance
        """
        if 'Surface' not in race_history.columns or race_history.empty:
            return 0.5  # Default moderate sensitivity
        
        # Compare performance on different surfaces
        surfaces = race_history['Surface'].unique()
        if len(surfaces) < 2:
            return 0.3  # Low sensitivity if only one surface type
        
        surface_volatilities = []
        for surface in surfaces:
            surface_races = race_history[race_history['Surface'] == surface]
            if len(surface_races) >= 3:
                volatility = self._calculate_volatility(surface_races, 'Finish')
                surface_volatilities.append(volatility)
        
        if len(surface_volatilities) < 2:
            return 0.3
        
        # Weather sensitivity is the variance in volatility across surfaces
        sensitivity = np.std(surface_volatilities) / 10.0  # Normalize
        return min(1.0, max(0.0, sensitivity))
    
    def _calculate_performance_momentum(self, race_history: pd.DataFrame) -> float:
        """Calculate performance momentum using recent trend"""
        if len(race_history) < 10:
            return 0.0
        
        recent_finishes = pd.to_numeric(race_history.head(10)['Finish'], errors='coerce').dropna()
        if len(recent_finishes) < 5:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_finishes))
        slope, _, r_value, _, _ = stats.linregress(x, recent_finishes)
        
        # Positive slope means worsening performance (higher finish positions)
        # Negative slope means improving performance
        momentum = -slope * (r_value ** 2)  # Weight by correlation strength
        return float(momentum)
    
    def _calculate_consistency_decay(self, race_history: pd.DataFrame) -> float:
        """Calculate rate at which consistency changes over career"""
        if len(race_history) < 20:
            return 0.0
        
        # Split career into quarters and calculate volatility for each
        n = len(race_history)
        quarter_size = n // 4
        
        volatilities = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else n
            quarter_races = race_history.iloc[n - end_idx:n - start_idx]  # Reverse order for chronological
            
            if len(quarter_races) >= 5:
                vol = self._calculate_volatility(quarter_races, 'Finish')
                volatilities.append(vol)
        
        if len(volatilities) < 3:
            return 0.0
        
        # Calculate trend in volatility over career quarters
        x = np.arange(len(volatilities))
        slope, _, _, _, _ = stats.linregress(x, volatilities)
        return float(slope)
    
    def _calculate_competitive_pressure(self, race_history: pd.DataFrame) -> float:
        """Calculate competitive pressure index based on field position distribution"""
        if race_history.empty:
            return 0.5
        
        finishes = pd.to_numeric(race_history['Finish'], errors='coerce').dropna()
        if len(finishes) < 5:
            return 0.5
        
        # Competitive pressure increases when driver frequently runs in middle of pack
        # where small mistakes have big position consequences
        mean_finish = np.mean(finishes)
        
        # Pressure is highest around positions 10-25
        if 10 <= mean_finish <= 25:
            pressure = 1.0 - abs(mean_finish - 17.5) / 7.5
        else:
            pressure = max(0.0, 1.0 - abs(mean_finish - 17.5) / 17.5)
        
        return min(1.0, max(0.0, pressure))
    
    def _calculate_failure_rate(self, race_history: pd.DataFrame) -> float:
        """Calculate mechanical failure rate from status field"""
        if 'Status' not in race_history.columns or race_history.empty:
            return 0.05  # Default low failure rate
        
        # Count races with mechanical issues
        failure_keywords = ['engine', 'transmission', 'gear', 'motor', 'mechanical', 'vibration']
        total_races = len(race_history)
        
        failures = 0
        for status in race_history['Status'].fillna(''):
            status_lower = str(status).lower()
            if any(keyword in status_lower for keyword in failure_keywords):
                failures += 1
        
        return failures / max(total_races, 1)
    
    def train(self, cup_series_df: pd.DataFrame, min_history_races: int = 20,
              seasons_to_predict: List[int] = None, volatility_window: int = 5) -> Dict:
        """
        Train the volatility prediction model
        
        Args:
            cup_series_df: DataFrame with NASCAR Cup Series race data
            min_history_races: Minimum races needed to calculate target volatility
            seasons_to_predict: List of seasons to use for training
            volatility_window: Number of races to look ahead for target volatility
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Training driver volatility predictor...")
        
        if cup_series_df.empty:
            raise ValueError("NASCAR data is required")
        
        # Store reference to full dataset
        self.full_dataset = cup_series_df.copy()
        
        # Clean up the data
        self.full_dataset['Finish'] = pd.to_numeric(self.full_dataset['Finish'], errors='coerce')
        self.full_dataset['Start'] = pd.to_numeric(self.full_dataset['Start'], errors='coerce')
        
        # Default seasons for training
        if seasons_to_predict is None:
            seasons_to_predict = list(range(2005, 2025))
        
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
            if len(driver_races) < min_history_races + volatility_window + 5:
                continue
            
            # Create training examples
            for i in range(min_history_races, len(driver_races) - volatility_window):
                target_race_idx = i + volatility_window
                
                if target_race_idx >= len(driver_races):
                    continue
                
                current_race = driver_races.iloc[i]
                
                # Only use races from target seasons
                if current_race['Season'] not in seasons_to_predict:
                    continue
                
                # Get history up to current point
                history = driver_races.iloc[:i]
                
                # Get future races for target volatility calculation
                future_races = driver_races.iloc[i:i + volatility_window]
                
                if len(future_races) < volatility_window:
                    continue
                
                try:
                    # Calculate target volatility (what we want to predict)
                    target_volatility = self._calculate_volatility(future_races, 'Finish')
                    
                    # Skip if target is invalid
                    if np.isnan(target_volatility) or target_volatility == 0:
                        continue
                    
                    # Determine track type for current race
                    track_type = self._classify_track_type(
                        current_race['Track'], current_race['Length']
                    )
                    
                    # Calculate season progress
                    season_progress = current_race['Race'] / 36.0  # Approximate races per season
                    season_progress = min(1.0, max(0.0, season_progress))
                    
                    # Field strength (simplified to 1.0 for now)
                    field_strength = 1.0
                    
                    # Prepare features
                    feature_vector = self.prepare_features(
                        history, track_type, season_progress, field_strength
                    )
                    
                    features_list.append(feature_vector)
                    targets.append(target_volatility)
                    
                    training_examples.append({
                        'driver': driver_name,
                        'season': current_race['Season'],
                        'race': current_race['Race'],
                        'target_volatility': target_volatility
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
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'drivers_count': len(set(ex['driver'] for ex in training_examples)),
            'mean_target_volatility': np.mean(y)
        }
        
        self.logger.info(f"Training complete - Val MAE: {self.training_metrics['val_mae']:.2f}, "
                        f"Val R²: {self.training_metrics['val_r2']:.3f}")
        
        return self.training_metrics
    
    def _classify_track_type(self, track_name: str, track_length: float) -> str:
        """Classify track type based on name and length"""
        if pd.isna(track_name):
            track_name = ""
        if pd.isna(track_length):
            track_length = 1.5
            
        track_name = str(track_name).lower()
        
        road_keywords = ['road', 'glen', 'sonoma', 'cota', 'roval', 'mexico']
        if any(keyword in track_name for keyword in road_keywords):
            return 'road'
        
        superspeedway_tracks = ['daytona', 'talladega']
        if track_length >= 2.0 or any(name in track_name for name in superspeedway_tracks):
            return 'superspeedway'
        
        if track_length < 1.0:
            return 'short'
        
        return 'intermediate'
    
    def predict_for_driver(self, cup_series_df: pd.DataFrame, driver_name: str, 
                          next_track_name: str = None, next_track_length: float = 1.5,
                          season_progress: float = 0.5) -> Dict:
        """
        Predict volatility for a specific driver
        
        Args:
            cup_series_df: Full NASCAR Cup Series DataFrame
            driver_name: Name of driver to predict for
            next_track_name: Name of next track
            next_track_length: Length of next track in miles
            season_progress: Progress through season (0.0 to 1.0)
            
        Returns:
            Dictionary with prediction results including histogram data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get driver race history
        driver_races = cup_series_df[
            cup_series_df['Driver'] == driver_name
        ].copy()
        
        if driver_races.empty:
            raise ValueError(f"No data found for driver: {driver_name}")
        
        # Clean and sort data
        driver_races = self._clean_driver_data(driver_races)
        
        if len(driver_races) < 10:
            raise ValueError(f"Insufficient race history for {driver_name}")
        
        # Classify track type
        track_type = self._classify_track_type(next_track_name, next_track_length)
        
        # Prepare features
        features = self.prepare_features(driver_races, track_type, season_progress, 1.0)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        predicted_volatility = self.model.predict(features_scaled)[0]
        
        # Calculate additional insights
        recent_avg_finish = self._safe_mean(driver_races.head(10), 'Finish', 20.0)
        
        # Generate performance bands for histogram visualization
        performance_bands = self._generate_performance_bands(recent_avg_finish, predicted_volatility)
        
        # Calculate percentile vs field
        field_percentile = self._calculate_field_percentile(predicted_volatility)
        
        return {
            'predicted_volatility': predicted_volatility,
            'volatility_category': self._categorize_volatility(predicted_volatility),
            'field_percentile': field_percentile,
            'recent_avg_finish': recent_avg_finish,
            'performance_bands': performance_bands,
            'histogram_data': self._create_histogram_data(recent_avg_finish, predicted_volatility),
            'risk_assessment': self._assess_risk(predicted_volatility, recent_avg_finish),
            'confidence': min(1.0, max(0.0, self.training_metrics.get('val_r2', 0.5))),
            'track_type': track_type
        }
    
    def _safe_mean(self, df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        """Calculate mean with handling for missing values"""
        if column not in df.columns or df.empty:
            return default
        
        values = pd.to_numeric(df[column], errors='coerce').dropna()
        return values.mean() if not values.empty else default
    
    def _clean_driver_data(self, driver_races: pd.DataFrame) -> pd.DataFrame:
        """Clean driver race data"""
        driver_races['Finish'] = pd.to_numeric(driver_races['Finish'], errors='coerce')
        driver_races['Start'] = pd.to_numeric(driver_races['Start'], errors='coerce')
        
        # Remove races with missing finish data
        driver_races = driver_races.dropna(subset=['Finish'])
        driver_races = driver_races.sort_values(['Season', 'Race'], ascending=False)
        
        return driver_races
    
    def _generate_performance_bands(self, avg_finish: float, volatility: float) -> Dict:
        """Generate performance bands for visualization"""
        return {
            'best_case': max(1, avg_finish - 2 * volatility),
            'likely_best': max(1, avg_finish - volatility),
            'expected': avg_finish,
            'likely_worst': min(40, avg_finish + volatility),
            'worst_case': min(40, avg_finish + 2 * volatility)
        }
    
    def _calculate_field_percentile(self, volatility: float) -> float:
        """Calculate where this volatility ranks vs typical field"""
        # Based on typical NASCAR volatility distribution
        # Lower volatility = higher percentile (more consistent)
        if volatility < 3.0:
            return 0.95
        elif volatility < 5.0:
            return 0.80
        elif volatility < 7.0:
            return 0.60
        elif volatility < 10.0:
            return 0.40
        elif volatility < 12.0:
            return 0.20
        else:
            return 0.05
    
    def _create_histogram_data(self, avg_finish: float, volatility: float) -> Dict:
        """Create data for histogram visualization of predicted finish distribution"""
        # Generate normal distribution around average finish with predicted volatility
        finish_positions = np.arange(1, 41)
        probabilities = stats.norm.pdf(finish_positions, avg_finish, volatility)
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Create bins for histogram
        bins = []
        for i in range(0, 40, 5):
            bin_start = i + 1
            bin_end = min(i + 5, 40)
            bin_prob = np.sum(probabilities[i:i+5])
            bins.append({
                'range': f"{bin_start}-{bin_end}",
                'probability': bin_prob,
                'positions': list(range(bin_start, bin_end + 1))
            })
        
        return {
            'bins': bins,
            'raw_probabilities': probabilities.tolist(),
            'positions': finish_positions.tolist()
        }
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level"""
        if volatility < 3.0:
            return "Very Consistent"
        elif volatility < 5.0:
            return "Consistent"
        elif volatility < 7.0:
            return "Moderate"
        elif volatility < 10.0:
            return "Volatile"
        else:
            return "Very Volatile"
    
    def _assess_risk(self, volatility: float, avg_finish: float) -> Dict:
        """Assess risk level and provide insights"""
        if volatility < 5.0 and avg_finish < 15:
            risk_level = "Low"
            description = "Consistent top-15 performer with predictable results"
        elif volatility < 7.0 and avg_finish < 20:
            risk_level = "Moderate"  
            description = "Solid performer with occasional variance"
        elif volatility > 10.0:
            risk_level = "High"
            description = "Unpredictable results with high variance"
        else:
            risk_level = "Moderate"
            description = "Average volatility for field position"
        
        return {
            'level': risk_level,
            'description': description,
            'boom_bust_potential': volatility > 8.0 and avg_finish < 25
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
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
    def load(cls, filepath: str) -> 'DriverVolatilityPredictor':
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
            'name': 'Driver Volatility Predictor',
            'type': 'Random Forest Regression',
            'features': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'description': 'Predicts NASCAR driver performance volatility and consistency patterns'
        }


# Example usage
if __name__ == "__main__":
    predictor = DriverVolatilityPredictor()
    
    # Create sample data with varying consistency
    sample_history = pd.DataFrame({
        'Season': [2024] * 25,
        'Race': range(1, 26),
        'Finish': [12, 8, 25, 5, 22, 9, 18, 30, 11, 7, 14, 6, 35, 2, 16, 4, 28, 1, 19, 10, 15, 8, 24, 3, 20],
        'Start': [15, 12, 20, 8, 25, 14, 22, 28, 16, 10, 18, 9, 30, 5, 19, 7, 26, 3, 23, 11, 17, 12, 22, 6, 21],
        'Track': ['Charlotte Motor Speedway'] * 25,
        'Length': [1.5] * 25,
        'Surface': ['asphalt'] * 25,
        'Driver': ['Test Driver'] * 25,
        'Team': ['Test Team'] * 25,
        'Status': ['running'] * 25
    })
    
    print("Sample volatility prediction features:")
    try:
        features = predictor.prepare_features(sample_history, 'intermediate', 0.5, 1.0)
        print("Features extracted:", dict(zip(predictor.feature_names, features)))
        print(f"Historical volatility: {predictor._calculate_volatility(sample_history, 'Finish'):.2f}")
    except Exception as e:
        print("Error:", str(e))
    
    print("\nModel info:")
    print(predictor.get_model_info())