"""
NASCAR Data Loader Module

This module handles loading and initial processing of NASCAR Cup Series data
from the nascaR.data package, providing clean interfaces for data access
and basic validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings

# Import our config utilities
from config import get_config, get_data_paths


class NASCARDataLoader:
    """
    Handles loading and initial processing of NASCAR Cup Series data.
    
    This class provides methods to load data from the nascaR.data package,
    apply basic filtering, and prepare data for analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the NASCAR data loader.
        
        Args:
            config (Optional[Dict]): Configuration dictionary. If None, loads from config.yaml
        """
        self.config = config if config is not None else get_config()
        self.data_config = self.config['data']
        self.paths = get_data_paths(self.config)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize data storage
        self.raw_data = None
        self.filtered_data = None
        self.driver_seasons = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw NASCAR Cup Series data.
        
        Note: This assumes you have the nascaR.data package installed and loaded in R,
        or you have exported the data to CSV. For now, this is a placeholder that
        would need to be adapted based on your specific data source.
        
        Returns:
            pd.DataFrame: Raw NASCAR Cup Series data
            
        Raises:
            FileNotFoundError: If data source cannot be found
            ValueError: If data format is invalid
        """
        try:
            # Option 1: If you have CSV file exported from R
            csv_path = Path(self.paths['raw_data']) / 'cup_series.csv'
            if csv_path.exists():
                self.logger.info(f"Loading data from CSV: {csv_path}")
                self.raw_data = pd.read_csv(csv_path)
                
            # Option 2: If you have parquet file (faster for large datasets)
            elif (Path(self.paths['raw_data']) / 'cup_series.parquet').exists():
                parquet_path = Path(self.paths['raw_data']) / 'cup_series.parquet'
                self.logger.info(f"Loading data from Parquet: {parquet_path}")
                self.raw_data = pd.read_parquet(parquet_path)
                
            # Option 3: Placeholder for R integration (would need rpy2 or similar)
            else:
                self.logger.warning("No data file found. You'll need to export from R first.")
                # For development, create sample data structure
                self.raw_data = self._create_sample_data()
                
            self.logger.info(f"Loaded {len(self.raw_data)} records from {self.raw_data['Season'].min()} to {self.raw_data['Season'].max()}")
            return self.raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample NASCAR data for development/testing purposes.
        
        Returns:
            pd.DataFrame: Sample NASCAR data with realistic structure
        """
        self.logger.info("Creating sample data for development")
        
        # Create sample data that matches NASCAR structure
        np.random.seed(42)
        
        seasons = range(2000, 2026)
        drivers = ['Kyle Larson', 'Denny Hamlin', 'Chase Elliott', 'William Byron', 
                  'Ryan Blaney', 'Christopher Bell', 'Tyler Reddick', 'Joey Logano',
                  'Kevin Harvick', 'Martin Truex Jr.', 'Kyle Busch', 'Brad Keselowski']
        
        sample_data = []
        
        for season in seasons:
            for race_num in range(1, 37):  # 36 races per season
                for i, driver in enumerate(drivers):
                    # Simulate race results with some realism
                    base_skill = np.random.normal(15, 5)  # Base average finish
                    season_factor = np.random.normal(0, 2)
                    race_luck = np.random.normal(0, 8)
                    
                    finish = max(1, min(40, int(base_skill + season_factor + race_luck)))
                    
                    sample_data.append({
                        'Season': season,
                        'Race': race_num,
                        'Driver': driver,
                        'Finish': finish,
                        'Start': max(1, min(40, finish + np.random.randint(-10, 11))),
                        'Car': 5 + i,  # Car numbers
                        'Make': np.random.choice(['Chevrolet', 'Ford', 'Toyota']),
                        'Pts': max(1, 50 - finish + (5 if finish == 1 else 0)),
                        'Laps': np.random.randint(180, 501),
                        'Led': np.random.randint(0, 100) if finish <= 5 else 0,
                        'Status': 'running' if np.random.random() > 0.1 else 'mechanical',
                        'Rating': max(0, min(150, 100 + np.random.normal(0, 20))),
                        'Win': 1 if finish == 1 else 0
                    })
        
        return pd.DataFrame(sample_data)
    
    def apply_data_filtering(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply basic data filtering based on configuration.
        
        Args:
            data (Optional[pd.DataFrame]): Data to filter. Uses self.raw_data if None.
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_raw_data() first.")
            data = self.raw_data.copy()
        
        original_rows = len(data)
        filtering_config = self.data_config['filtering']
        
        # Filter by date range
        date_range = self.data_config['date_range']
        data = data[
            (data['Season'] >= date_range['start_year']) & 
            (data['Season'] <= date_range['end_year'])
        ]
        self.logger.info(f"Date filtering: {original_rows} -> {len(data)} records")
        
        # Remove records with missing critical data
        critical_columns = ['Season', 'Driver', 'Finish']
        before_missing = len(data)
        data = data.dropna(subset=critical_columns)
        self.logger.info(f"Missing data removal: {before_missing} -> {len(data)} records")
        
        # Convert data types
        data = self._standardize_data_types(data)
        
        self.filtered_data = data
        return data
    
    def _standardize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types for consistency.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with standardized types
        """
        # Ensure numeric columns are properly typed
        numeric_columns = ['Season', 'Race', 'Finish', 'Start', 'Car', 'Pts', 'Laps', 'Led', 'Rating', 'Win']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Ensure string columns are properly typed
        string_columns = ['Driver', 'Make', 'Status']
        for col in string_columns:
            if col in data.columns:
                data[col] = data[col].astype('string')
        
        return data
    
    def create_driver_season_summary(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create season-level summary statistics for each driver.
        
        Args:
            data (Optional[pd.DataFrame]): Race-level data. Uses filtered_data if None.
            
        Returns:
            pd.DataFrame: Driver season summaries
        """
        if data is None:
            if self.filtered_data is None:
                raise ValueError("No filtered data available. Call apply_data_filtering() first.")
            data = self.filtered_data
        
        # Group by driver and season
        season_stats = data.groupby(['Season', 'Driver']).agg({
            'Finish': ['count', 'mean', 'std', lambda x: (x <= 5).sum(), lambda x: (x <= 10).sum()],
            'Win': 'sum',
            'Pts': 'sum',
            'Led': 'sum',
            'Rating': 'mean',
            'Status': lambda x: (x != 'running').sum()  # DNF count
        }).round(2)
        
        # Flatten column names
        season_stats.columns = [
            'races_run', 'avg_finish', 'finish_std', 'top_5s', 'top_10s',
            'wins', 'total_points', 'laps_led', 'avg_rating', 'dnfs'
        ]
        
        # Calculate derived metrics
        season_stats['top_5_rate'] = (season_stats['top_5s'] / season_stats['races_run']).round(3)
        season_stats['top_10_rate'] = (season_stats['top_10s'] / season_stats['races_run']).round(3)
        season_stats['dnf_rate'] = (season_stats['dnfs'] / season_stats['races_run']).round(3)
        season_stats['win_rate'] = (season_stats['wins'] / season_stats['races_run']).round(3)
        
        # Reset index to make Season and Driver regular columns
        season_stats = season_stats.reset_index()
        
        # Filter drivers with minimum races
        min_races = self.data_config['filtering']['min_races_per_season']
        season_stats = season_stats[season_stats['races_run'] >= min_races]
        
        self.driver_seasons = season_stats
        self.logger.info(f"Created season summaries for {len(season_stats)} driver-seasons")
        
        return season_stats
    
    def update_data_from_r(self) -> bool:
        """
        Run the R script to update NASCAR data from nascaR.data package.
        
        Returns:
            bool: True if update successful, False otherwise
        """
        import subprocess
        import json
        from pathlib import Path
        
        # Path to the R script
        r_script_path = Path(__file__).parent.parent.parent / "scripts" / "update_data.R"
        
        if not r_script_path.exists():
            self.logger.error(f"R script not found at {r_script_path}")
            return False
        
        self.logger.info("Running R script to update NASCAR data...")
        
        try:
            # Run the R script
            result = subprocess.run(
                ['Rscript', str(r_script_path)],
                cwd=Path(__file__).parent.parent.parent,  # Run from project root
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log R script output
            if result.stdout:
                self.logger.info(f"R script output: {result.stdout}")
            
            if result.stderr:
                self.logger.warning(f"R script warnings: {result.stderr}")
            
            if result.returncode == 0:
                self.logger.info("R script completed successfully")
                
                # Try to read metadata
                metadata_path = Path(self.paths['raw_data']) / 'data_metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.logger.info(f"Updated data: {metadata['total_records']} records, "
                                   f"seasons {metadata['seasons_covered']}")
                
                return True
            else:
                self.logger.error(f"R script failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("R script timed out after 5 minutes")
            return False
        except FileNotFoundError:
            self.logger.error("Rscript not found. Make sure R is installed and in your PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error running R script: {e}")
            return False
    
    def check_data_freshness(self) -> Dict:
        """
        Check if the data is up-to-date and read metadata if available.
        
        Returns:
            Dict: Information about data freshness and metadata
        """
        freshness_info = {}
        
        # Check for metadata file first
        metadata_path = Path(self.paths['raw_data']) / 'data_metadata.json'
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                freshness_info['metadata'] = metadata
                
                # Parse update time
                import datetime
                update_time = datetime.datetime.fromisoformat(
                    metadata['update_time'].replace('Z', '+00:00').split('+')[0]
                )
                days_old = (datetime.datetime.now() - update_time).days
                
                freshness_info.update({
                    'last_updated': metadata['update_time'],
                    'days_old': days_old,
                    'is_recent': days_old <= 7,
                    'status': 'Fresh' if days_old <= 7 else 'Stale',
                    'total_records': metadata['total_records'],
                    'seasons_covered': metadata['seasons_covered']
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading metadata: {e}")
        
        # Fallback to file modification time checking
        if 'status' not in freshness_info:
            csv_path = Path(self.paths['raw_data']) / 'cup_series.csv'
            parquet_path = Path(self.paths['raw_data']) / 'cup_series.parquet'
            
            data_file = None
            if parquet_path.exists():
                data_file = parquet_path
            elif csv_path.exists():
                data_file = csv_path
                
            if data_file:
                import datetime
                mod_time = datetime.datetime.fromtimestamp(data_file.stat().st_mtime)
                days_old = (datetime.datetime.now() - mod_time).days
                
                freshness_info.update({
                    'data_file': str(data_file),
                    'last_modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'days_old': days_old,
                    'is_recent': days_old <= 7,
                    'status': 'Fresh' if days_old <= 7 else 'Stale'
                })
            else:
                freshness_info = {
                    'status': 'No data file found',
                    'recommendation': 'Run update_data_from_r() to fetch latest data'
                }
        
        return freshness_info
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics about the loaded data.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {}
        
        if self.raw_data is not None:
            summary['raw_data'] = {
                'total_records': len(self.raw_data),
                'seasons': f"{self.raw_data['Season'].min()} - {self.raw_data['Season'].max()}",
                'unique_drivers': self.raw_data['Driver'].nunique(),
                'total_races': self.raw_data['Race'].nunique()
            }
        
        if self.filtered_data is not None:
            summary['filtered_data'] = {
                'total_records': len(self.filtered_data),
                'seasons': f"{self.filtered_data['Season'].min()} - {self.filtered_data['Season'].max()}",
                'unique_drivers': self.filtered_data['Driver'].nunique()
            }
        
        if self.driver_seasons is not None:
            summary['driver_seasons'] = {
                'total_driver_seasons': len(self.driver_seasons),
                'unique_drivers': self.driver_seasons['Driver'].nunique(),
                'avg_races_per_season': self.driver_seasons['races_run'].mean().round(1)
            }
        
        return summary
    
    def save_processed_data(self) -> None:
        """
        Save processed data to files for later use.
        """
        if self.filtered_data is not None:
            filtered_path = Path(self.paths['processed_data']) / 'filtered_race_data.parquet'
            self.filtered_data.to_parquet(filtered_path, index=False)
            self.logger.info(f"Saved filtered data to {filtered_path}")
        
        if self.driver_seasons is not None:
            seasons_path = Path(self.paths['processed_data']) / 'driver_season_summaries.parquet'
            self.driver_seasons.to_parquet(seasons_path, index=False)
            self.logger.info(f"Saved driver seasons to {seasons_path}")


def load_nascar_data(config_path: Optional[str] = None) -> NASCARDataLoader:
    """
    Convenience function to load NASCAR data with default settings.
    
    Args:
        config_path (Optional[str]): Path to config file
        
    Returns:
        NASCARDataLoader: Configured data loader with data loaded
    """
    loader = NASCARDataLoader()
    loader.load_raw_data()
    loader.apply_data_filtering()
    loader.create_driver_season_summary()
    
    return loader


if __name__ == "__main__":
    # Example usage
    print("Loading NASCAR data...")
    
    loader = load_nascar_data()
    summary = loader.get_data_summary()
    
    print("\nData Summary:")
    for section, stats in summary.items():
        print(f"\n{section.replace('_', ' ').title()}:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Save processed data
    loader.save_processed_data()
    print("\nProcessed data saved successfully!")