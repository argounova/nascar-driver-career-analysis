"""
NASCAR Feature Engineering Module

This module creates advanced features for machine learning models including:
- Rolling window calculations (3, 5, 10 season averages)
- Career phase identification (Rookie, Prime, Veteran)
- Performance trends and improvement rates
- Consistency metrics and volatility measures
- Peak performance detection and timing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings

# Statistical imports
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

# Import our modules
from config import get_config, get_data_paths


class NASCARFeatureEngineer:
    """
    Creates advanced features for NASCAR driver performance analysis.
    
    Generates features for both clustering analysis and LSTM time series prediction:
    - Rolling statistics and moving averages
    - Career phase identification
    - Performance trend analysis
    - Consistency and volatility metrics
    - Peak performance detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config (Optional[Dict]): Configuration dictionary
        """
        self.config = config if config is not None else get_config()
        self.feature_config = self.config['features']
        self.paths = get_data_paths(self.config)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Feature storage
        self.driver_seasons = None
        self.engineered_features = None
        self.lstm_sequences = None
        
        # Configuration shortcuts
        self.rolling_windows = self.feature_config['rolling_windows']
        self.career_phases = self.feature_config['career_phases']
    
    def load_driver_seasons(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load driver season data for feature engineering.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Loaded driver season data
        """
        if driver_seasons is not None:
            self.driver_seasons = driver_seasons.copy()
        else:
            # Try to load from saved processed data
            processed_path = Path(self.paths['processed_data']) / 'driver_season_summaries.parquet'
            if processed_path.exists():
                self.driver_seasons = pd.read_parquet(processed_path)
                self.logger.info(f"Loaded {len(self.driver_seasons)} driver-seasons from file")
            else:
                raise ValueError("No driver season data provided or found")
        
        # Ensure proper sorting
        self.driver_seasons = self.driver_seasons.sort_values(['Driver', 'Season']).reset_index(drop=True)
        
        return self.driver_seasons
    
    def create_rolling_features(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create rolling window features for performance metrics.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Data with rolling features added
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info("Creating rolling window features...")
        
        # Core metrics to calculate rolling statistics for
        core_metrics = [
            'avg_finish', 'wins', 'top_5s', 'top_10s', 'win_rate', 
            'top_5_rate', 'top_10_rate', 'dnf_rate', 'avg_rating'
        ]
        
        # Initialize result dataframe
        rolling_data = driver_seasons.copy()
        
        # Calculate rolling features for each driver
        for driver in rolling_data['Driver'].unique():
            driver_mask = rolling_data['Driver'] == driver
            driver_data = rolling_data[driver_mask].copy()
            
            # Skip drivers with insufficient data
            if len(driver_data) < 2:
                continue
            
            # Calculate rolling statistics for each window size
            for window_name, window_size in self.rolling_windows.items():
                if len(driver_data) >= window_size:
                    for metric in core_metrics:
                        if metric in driver_data.columns:
                            # Rolling mean
                            rolling_mean = driver_data[metric].rolling(
                                window=window_size, 
                                min_periods=1
                            ).mean()
                            rolling_data.loc[driver_mask, f'{metric}_{window_name}_avg'] = rolling_mean
                            
                            # Rolling standard deviation (consistency)
                            rolling_std = driver_data[metric].rolling(
                                window=window_size, 
                                min_periods=2
                            ).std()
                            rolling_data.loc[driver_mask, f'{metric}_{window_name}_std'] = rolling_std
                            
                            # Rolling trend (linear slope)
                            rolling_trend = self._calculate_rolling_trend(
                                driver_data[metric], 
                                window_size
                            )
                            rolling_data.loc[driver_mask, f'{metric}_{window_name}_trend'] = rolling_trend
        
        self.logger.info(f"Added rolling features for {len(self.rolling_windows)} window sizes")
        return rolling_data
    
    def _calculate_rolling_trend(self, series: pd.Series, window_size: int) -> pd.Series:
        """
        Calculate rolling linear trend (slope) for a time series.
        
        Args:
            series (pd.Series): Time series data
            window_size (int): Rolling window size
            
        Returns:
            pd.Series: Rolling trend values
        """
        def calc_slope(window_data):
            if len(window_data) < 2:
                return np.nan
            x = np.arange(len(window_data))
            try:
                slope, _, _, _, _ = stats.linregress(x, window_data)
                return slope
            except:
                return np.nan
        
        return series.rolling(window=window_size, min_periods=2).apply(calc_slope, raw=False)
    
    def identify_career_phases(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Identify career phases for each driver-season.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Data with career phase information
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info("Identifying career phases...")
        
        phase_data = driver_seasons.copy()
        
        # Calculate career information for each driver
        for driver in phase_data['Driver'].unique():
            driver_mask = phase_data['Driver'] == driver
            driver_data = phase_data[driver_mask].copy()
            
            # Sort by season to ensure proper ordering
            driver_data = driver_data.sort_values('Season')
            driver_indices = driver_data.index
            
            # Calculate season number in career (1-based)
            season_numbers = np.arange(1, len(driver_data) + 1)
            
            # Assign career phases
            career_phases = []
            for season_num in season_numbers:
                if season_num <= self.career_phases['rookie_years']:
                    career_phases.append('Rookie')
                elif season_num <= self.career_phases['prime_start'] + self.career_phases['prime_duration']:
                    career_phases.append('Prime')
                else:
                    career_phases.append('Veteran')
            
            # Update dataframe
            phase_data.loc[driver_indices, 'career_season_number'] = season_numbers
            phase_data.loc[driver_indices, 'career_phase'] = career_phases
            phase_data.loc[driver_indices, 'total_career_seasons'] = len(driver_data)
            
            # Calculate career progress (0-1 scale)
            career_progress = (season_numbers - 1) / (len(driver_data) - 1) if len(driver_data) > 1 else [0.5]
            phase_data.loc[driver_indices, 'career_progress'] = career_progress
        
        self.logger.info("Career phases identified")
        return phase_data
    
    def calculate_performance_trends(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate performance trends and improvement rates.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Data with trend features
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info("Calculating performance trends...")
        
        trend_data = driver_seasons.copy()
        
        # Metrics to analyze trends for
        trend_metrics = ['avg_finish', 'win_rate', 'top_5_rate', 'top_10_rate', 'avg_rating']
        
        for driver in trend_data['Driver'].unique():
            driver_mask = trend_data['Driver'] == driver
            driver_data = trend_data[driver_mask].copy().sort_values('Season')
            driver_indices = driver_data.index
            
            if len(driver_data) < 3:  # Need at least 3 seasons for meaningful trends
                continue
            
            # Calculate overall career trends
            seasons = np.arange(len(driver_data))
            
            for metric in trend_metrics:
                if metric in driver_data.columns:
                    values = driver_data[metric].values
                    
                    # Overall linear trend
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(seasons, values)
                        
                        # Store trend information
                        trend_data.loc[driver_indices, f'{metric}_career_slope'] = slope
                        trend_data.loc[driver_indices, f'{metric}_career_r2'] = r_value ** 2
                        trend_data.loc[driver_indices, f'{metric}_trend_significance'] = p_value
                        
                        # Calculate improvement direction (for avg_finish, negative slope is improvement)
                        if metric == 'avg_finish':
                            improvement_direction = -slope  # Negative slope means better finish
                        else:
                            improvement_direction = slope   # Positive slope means improvement
                        
                        trend_data.loc[driver_indices, f'{metric}_improvement_rate'] = improvement_direction
                        
                    except:
                        # Handle cases where regression fails
                        trend_data.loc[driver_indices, f'{metric}_career_slope'] = 0
                        trend_data.loc[driver_indices, f'{metric}_career_r2'] = 0
                        trend_data.loc[driver_indices, f'{metric}_trend_significance'] = 1.0
                        trend_data.loc[driver_indices, f'{metric}_improvement_rate'] = 0
            
            # Calculate year-over-year changes
            for i, idx in enumerate(driver_indices[1:], 1):
                prev_idx = driver_indices[i-1]
                
                for metric in trend_metrics:
                    if metric in driver_data.columns:
                        current_val = trend_data.loc[idx, metric]
                        prev_val = trend_data.loc[prev_idx, metric]
                        
                        if pd.notna(current_val) and pd.notna(prev_val):
                            # Calculate year-over-year change
                            if metric == 'avg_finish':
                                # For avg_finish, lower is better, so negative change is improvement
                                yoy_change = prev_val - current_val
                            else:
                                # For rates and ratings, higher is better
                                yoy_change = current_val - prev_val
                            
                            trend_data.loc[idx, f'{metric}_yoy_change'] = yoy_change
                            
                            # Calculate percentage change (handle division by zero)
                            if prev_val != 0:
                                pct_change = (current_val - prev_val) / abs(prev_val)
                                trend_data.loc[idx, f'{metric}_yoy_pct_change'] = pct_change
        
        self.logger.info("Performance trends calculated")
        return trend_data
    
    def detect_peak_performance(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Detect peak performance periods for each driver.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Data with peak detection features
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info("Detecting peak performance periods...")
        
        peak_data = driver_seasons.copy()
        
        for driver in peak_data['Driver'].unique():
            driver_mask = peak_data['Driver'] == driver
            driver_data = peak_data[driver_mask].copy().sort_values('Season')
            driver_indices = driver_data.index
            
            if len(driver_data) < 3:
                continue
            
            # Calculate composite performance score
            # Normalize metrics to 0-1 scale for comparison
            metrics_for_peak = ['win_rate', 'top_5_rate', 'avg_rating']
            available_metrics = [m for m in metrics_for_peak if m in driver_data.columns and driver_data[m].notna().any()]
            
            if not available_metrics:
                continue
            
            # Create composite score
            composite_score = np.zeros(len(driver_data))
            
            for metric in available_metrics:
                values = driver_data[metric].values
                if metric == 'avg_finish':
                    # For avg_finish, lower is better, so invert
                    normalized = 1 - ((values - values.min()) / (values.max() - values.min() + 1e-8))
                else:
                    # For rates and ratings, higher is better
                    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
                
                composite_score += normalized
            
            composite_score /= len(available_metrics)
            
            # Find peak seasons
            peak_threshold = np.percentile(composite_score, 75)  # Top 25% of seasons
            peak_seasons = composite_score >= peak_threshold
            
            # Identify the absolute peak season
            peak_season_idx = np.argmax(composite_score)
            
            # Calculate peak-related features
            for i, idx in enumerate(driver_indices):
                peak_data.loc[idx, 'performance_score'] = composite_score[i]
                peak_data.loc[idx, 'is_peak_season'] = peak_seasons[i]
                peak_data.loc[idx, 'is_career_peak'] = (i == peak_season_idx)
                
                # Seasons to/from peak
                seasons_to_peak = peak_season_idx - i
                peak_data.loc[idx, 'seasons_to_peak'] = seasons_to_peak if seasons_to_peak >= 0 else 0
                peak_data.loc[idx, 'seasons_from_peak'] = max(0, i - peak_season_idx)
                
                # Peak timing (0 = early career, 1 = late career)
                peak_timing = peak_season_idx / (len(driver_data) - 1) if len(driver_data) > 1 else 0.5
                peak_data.loc[idx, 'peak_timing'] = peak_timing
            
            # Calculate peak duration (consecutive peak seasons)
            peak_durations = self._calculate_peak_durations(peak_seasons)
            for i, idx in enumerate(driver_indices):
                peak_data.loc[idx, 'peak_duration'] = peak_durations[i]
        
        self.logger.info("Peak performance detection completed")
        return peak_data
    
    def _calculate_peak_durations(self, peak_seasons: np.ndarray) -> List[int]:
        """
        Calculate duration of peak performance periods.
        
        Args:
            peak_seasons (np.ndarray): Boolean array of peak seasons
            
        Returns:
            List[int]: Peak duration for each season
        """
        durations = []
        current_duration = 0
        
        for is_peak in peak_seasons:
            if is_peak:
                current_duration += 1
            else:
                current_duration = 0
            durations.append(current_duration)
        
        return durations
    
    def calculate_consistency_metrics(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate consistency and volatility metrics.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Data with consistency features
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info("Calculating consistency metrics...")
        
        consistency_data = driver_seasons.copy()
        
        # Metrics to analyze for consistency
        consistency_metrics = ['avg_finish', 'win_rate', 'top_5_rate', 'avg_rating']
        
        for driver in consistency_data['Driver'].unique():
            driver_mask = consistency_data['Driver'] == driver
            driver_data = consistency_data[driver_mask].copy().sort_values('Season')
            driver_indices = driver_data.index
            
            if len(driver_data) < 3:
                continue
            
            for metric in consistency_metrics:
                if metric in driver_data.columns:
                    values = driver_data[metric].dropna()
                    
                    if len(values) < 2:
                        continue
                    
                    # Standard deviation (volatility)
                    std_dev = values.std()
                    
                    # Coefficient of variation (normalized volatility)
                    mean_val = values.mean()
                    cv = std_dev / abs(mean_val) if mean_val != 0 else np.inf
                    
                    # Consistency score (inverse of CV, capped at reasonable values)
                    consistency_score = 1 / (1 + cv) if cv != np.inf else 0
                    
                    # Range (max - min)
                    value_range = values.max() - values.min()
                    
                    # Interquartile range
                    q75, q25 = np.percentile(values, [75, 25])
                    iqr = q75 - q25
                    
                    # Store consistency metrics for all seasons of this driver
                    consistency_data.loc[driver_indices, f'{metric}_volatility'] = std_dev
                    consistency_data.loc[driver_indices, f'{metric}_cv'] = cv
                    consistency_data.loc[driver_indices, f'{metric}_consistency_score'] = consistency_score
                    consistency_data.loc[driver_indices, f'{metric}_range'] = value_range
                    consistency_data.loc[driver_indices, f'{metric}_iqr'] = iqr
        
        self.logger.info("Consistency metrics calculated")
        return consistency_data
    
    def create_lag_features(self, driver_seasons: Optional[pd.DataFrame] = None, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            lags (List[int]): Number of seasons to lag
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        self.logger.info(f"Creating lag features for {lags} seasons...")
        
        lag_data = driver_seasons.copy()
        
        # Metrics to create lags for
        lag_metrics = ['avg_finish', 'wins', 'win_rate', 'top_5_rate', 'top_10_rate', 'avg_rating']
        
        for driver in lag_data['Driver'].unique():
            driver_mask = lag_data['Driver'] == driver
            driver_data = lag_data[driver_mask].copy().sort_values('Season')
            driver_indices = driver_data.index
            
            for lag in lags:
                for metric in lag_metrics:
                    if metric in driver_data.columns:
                        lagged_values = driver_data[metric].shift(lag)
                        lag_data.loc[driver_indices, f'{metric}_lag_{lag}'] = lagged_values.values
        
        self.logger.info("Lag features created")
        return lag_data
    
    def create_lstm_sequences(self, driver_seasons: Optional[pd.DataFrame] = None, 
                            sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for LSTM training.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            sequence_length (Optional[int]): Length of sequences. Uses config default if None.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, driver names
        """
        if driver_seasons is None:
            driver_seasons = self.driver_seasons
        
        if sequence_length is None:
            sequence_length = self.config['models']['lstm']['sequence_length']
        
        self.logger.info(f"Creating LSTM sequences with length {sequence_length}...")
        
        # Features to use for LSTM
        feature_columns = [
            'avg_finish', 'win_rate', 'top_5_rate', 'top_10_rate', 'dnf_rate',
            'avg_rating', 'career_season_number', 'career_progress'
        ]
        
        # Add rolling features if they exist
        for window in self.rolling_windows.keys():
            for metric in ['avg_finish', 'win_rate', 'top_5_rate']:
                col = f'{metric}_{window}_avg'
                if col in driver_seasons.columns:
                    feature_columns.append(col)
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in driver_seasons.columns]
        
        sequences = []
        targets = []
        driver_names = []
        
        for driver in driver_seasons['Driver'].unique():
            driver_data = driver_seasons[driver_seasons['Driver'] == driver].sort_values('Season')
            
            if len(driver_data) < sequence_length + 1:
                continue
            
            # Get feature values
            features = driver_data[available_features].values
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                sequence = features[i:i + sequence_length]
                target = features[i + sequence_length]  # Next season's features
                
                sequences.append(sequence)
                targets.append(target)
                driver_names.append(driver)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        self.lstm_sequences = (sequences, targets, driver_names)
        
        self.logger.info(f"Created {len(sequences)} LSTM sequences for {len(set(driver_names))} drivers")
        
        return sequences, targets, driver_names
    
    def engineer_all_features(self, driver_seasons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all engineered features in one pipeline.
        
        Args:
            driver_seasons (Optional[pd.DataFrame]): Driver season data
            
        Returns:
            pd.DataFrame: Fully engineered feature set
        """
        if driver_seasons is None:
            if self.driver_seasons is None:
                raise ValueError("No driver season data available")
            driver_seasons = self.driver_seasons
        
        self.logger.info("🔧 Starting comprehensive feature engineering pipeline...")
        
        # Step 1: Load and prepare data
        data = self.load_driver_seasons(driver_seasons)
        
        # Step 2: Create rolling features
        data = self.create_rolling_features(data)
        
        # Step 3: Identify career phases
        data = self.identify_career_phases(data)
        
        # Step 4: Calculate performance trends
        data = self.calculate_performance_trends(data)
        
        # Step 5: Detect peak performance
        data = self.detect_peak_performance(data)
        
        # Step 6: Calculate consistency metrics
        data = self.calculate_consistency_metrics(data)
        
        # Step 7: Create lag features
        data = self.create_lag_features(data)
        
        self.engineered_features = data
        
        # Feature summary
        original_features = len(driver_seasons.columns)
        new_features = len(data.columns)
        added_features = new_features - original_features
        
        self.logger.info(f"✅ Feature engineering complete!")
        self.logger.info(f"   Original features: {original_features}")
        self.logger.info(f"   Engineered features: {new_features}")
        self.logger.info(f"   Added features: {added_features}")
        
        return data
    
    def save_engineered_features(self) -> None:
        """Save engineered features to file."""
        if self.engineered_features is None:
            raise ValueError("No engineered features to save")
        
        # Save main feature set
        features_path = Path(self.paths['processed_data']) / 'engineered_features.parquet'
        self.engineered_features.to_parquet(features_path, index=False)
        
        # Save LSTM sequences if available
        if self.lstm_sequences is not None:
            sequences, targets, driver_names = self.lstm_sequences
            
            sequences_path = Path(self.paths['processed_data']) / 'lstm_sequences.npz'
            np.savez_compressed(
                sequences_path,
                sequences=sequences,
                targets=targets,
                driver_names=driver_names
            )
            
            self.logger.info(f"LSTM sequences saved to {sequences_path}")
        
        self.logger.info(f"Engineered features saved to {features_path}")
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of engineered features.
        
        Returns:
            Dict: Feature summary statistics
        """
        if self.engineered_features is None:
            return {"error": "No engineered features available"}
        
        data = self.engineered_features
        
        # Categorize features
        feature_categories = {
            'original': [],
            'rolling': [],
            'trend': [],
            'career_phase': [],
            'peak': [],
            'consistency': [],
            'lag': []
        }
        
        for col in data.columns:
            if any(window in col for window in self.rolling_windows.keys()):
                feature_categories['rolling'].append(col)
            elif any(trend_word in col for trend_word in ['slope', 'trend', 'improvement', 'yoy']):
                feature_categories['trend'].append(col)
            elif any(phase_word in col for phase_word in ['career_', 'phase']):
                feature_categories['career_phase'].append(col)
            elif any(peak_word in col for peak_word in ['peak', 'performance_score']):
                feature_categories['peak'].append(col)
            elif any(cons_word in col for cons_word in ['consistency', 'volatility', 'cv', 'iqr']):
                feature_categories['consistency'].append(col)
            elif 'lag_' in col:
                feature_categories['lag'].append(col)
            else:
                feature_categories['original'].append(col)
        
        summary = {
            'total_features': len(data.columns),
            'total_driver_seasons': len(data),
            'unique_drivers': data['Driver'].nunique(),
            'season_range': f"{data['Season'].min()}-{data['Season'].max()}",
            'feature_categories': {k: len(v) for k, v in feature_categories.items()},
            'feature_breakdown': feature_categories
        }
        
        if self.lstm_sequences is not None:
            sequences, targets, driver_names = self.lstm_sequences
            summary['lstm_sequences'] = {
                'total_sequences': len(sequences),
                'sequence_length': sequences.shape[1],
                'feature_count': sequences.shape[2],
                'unique_drivers': len(set(driver_names))
            }
        
        return summary


def create_nascar_features(config_path: Optional[str] = None, 
                          save_results: bool = True) -> NASCARFeatureEngineer:
    """
    Convenience function to create all NASCAR features.
    
    Args:
        config_path (Optional[str]): Path to config file
        save_results (bool): Whether to save results to files
        
    Returns:
        NASCARFeatureEngineer: Fitted feature engineer with results
    """
    # Load data
    from data.data_loader import load_nascar_data
    
    print("🏁 Loading NASCAR data...")
    data_loader = load_nascar_data()
    
    # Initialize feature engineer
    print("🔧 Initializing feature engineering...")
    engineer = NASCARFeatureEngineer()
    
    # Create all features
    print("⚙️  Engineering features...")
    engineered_data = engineer.engineer_all_features(data_loader.driver_seasons)
    
    # Create LSTM sequences
    print("🧠 Creating LSTM sequences...")
    sequences, targets, driver_names = engineer.create_lstm_sequences(engineered_data)
    
    # Show summary
    summary = engineer.get_feature_summary()
    print("\n📊 Feature Engineering Summary:")
    print("=" * 50)
    print(f"Total Features: {summary['total_features']}")
    print(f"Driver-Seasons: {summary['total_driver_seasons']}")
    print(f"Unique Drivers: {summary['unique_drivers']}")
    print(f"Season Range: {summary['season_range']}")
    
    print("\n📋 Feature Categories:")
    for category, count in summary['feature_categories'].items():
        if count > 0:
            print(f"  {category.replace('_', ' ').title()}: {count} features")
    
    if 'lstm_sequences' in summary:
        print(f"\n🧠 LSTM Sequences:")
        print(f"  Total Sequences: {summary['lstm_sequences']['total_sequences']}")
        print(f"  Sequence Length: {summary['lstm_sequences']['sequence_length']} seasons")
        print(f"  Features per Sequence: {summary['lstm_sequences']['feature_count']}")
        print(f"  Drivers with Sequences: {summary['lstm_sequences']['unique_drivers']}")
    
    # Save results
    if save_results:
        print("\n💾 Saving engineered features...")
        engineer.save_engineered_features()
    
    print("✅ Feature engineering complete!")
    return engineer


if __name__ == "__main__":
    # Example usage
    engineer = create_nascar_features()
    
    # Show example of engineered features for a driver
    if engineer.engineered_features is not None:
        print("\n🔍 Example: Kyle Larson's Career Features")
        print("=" * 50)
        
        larson_data = engineer.engineered_features[
            engineer.engineered_features['Driver'] == 'Kyle Larson'
        ].sort_values('Season')
        
        if not larson_data.empty:
            # Show key features for recent seasons
            recent_seasons = larson_data.tail(3)
            key_features = [
                'Season', 'avg_finish', 'win_rate', 'top_5_rate',
                'career_phase', 'career_progress', 'peak_timing',
                'avg_finish_short_term_avg', 'performance_score'
            ]
            
            available_features = [f for f in key_features if f in recent_seasons.columns]
            print(recent_seasons[available_features].to_string(index=False))
        else:
            print("Kyle Larson not found in dataset")