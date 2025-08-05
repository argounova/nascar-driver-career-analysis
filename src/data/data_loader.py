"""
NASCAR Data Loader Module

This module handles loading and initial processing of NASCAR Cup Series data,
providing clean interfaces for data access and basic validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
import json
import datetime

# Import our config utilities
from config import get_config, get_data_paths


class NASCARDataLoader:
    """
    Handles loading and initial processing of NASCAR Cup Series data.
    
    This class provides methods to load data from Parquet/CSV files,
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
        
        # Make sure we have the df property for backward compatibility
        self.df = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw NASCAR Cup Series data.
        
        Prioritizes Parquet format for performance, falls back to CSV.
        
        Returns:
            pd.DataFrame: Raw NASCAR Cup Series data
            
        Raises:
            FileNotFoundError: If no data files are found
            ValueError: If data format is invalid
        """
        try:
            # Priority 1: Parquet file (faster for large datasets)
            parquet_path = Path(self.paths['raw_data']) / 'cup_series.parquet'
            if parquet_path.exists():
                self.logger.info(f"Loading data from Parquet: {parquet_path}")
                self.raw_data = pd.read_parquet(parquet_path)
                
            # Priority 2: CSV file (backup/fallback)
            elif (Path(self.paths['raw_data']) / 'cup_series.csv').exists():
                csv_path = Path(self.paths['raw_data']) / 'cup_series.csv'
                self.logger.info(f"Loading data from CSV: {csv_path}")
                self.raw_data = pd.read_csv(csv_path)
                self.logger.info("💡 Consider converting to Parquet format for faster loading")
                
            else:
                raise FileNotFoundError(
                    f"No NASCAR data found. Expected files:\n"
                    f"  Primary: {parquet_path}\n"
                    f"  Fallback: {Path(self.paths['raw_data']) / 'cup_series.csv'}\n"
                    f"Run 'Rscript scripts/update_data.R' to download data."
                )
                
            # Set df property for backward compatibility
            self.df = self.raw_data
                
            self.logger.info(f"Loaded {len(self.raw_data)} records from {self.raw_data['Season'].min()} to {self.raw_data['Season'].max()}")
            return self.raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            raise
    
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
        
        original_count = len(data)
        
        # Apply date filtering if specified in config
        if 'min_season' in self.data_config:
            data = data[data['Season'] >= self.data_config['min_season']]
        
        if 'max_season' in self.data_config:
            data = data[data['Season'] <= self.data_config['max_season']]
        
        date_filtered_count = len(data)
        self.logger.info(f"Date filtering: {original_count} -> {date_filtered_count} records")
        
        # Remove rows with missing critical data
        critical_columns = ['Driver', 'Season', 'Race', 'Finish']
        data = data.dropna(subset=critical_columns)
        
        final_count = len(data)
        self.logger.info(f"Missing data removal: {date_filtered_count} -> {final_count} records")
        
        self.filtered_data = data
        return data
    
    def create_driver_season_summary(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create season-level summaries for each driver.
        
        Args:
            data (Optional[pd.DataFrame]): Data to summarize. Uses filtered_data if None.
            
        Returns:
            pd.DataFrame: Driver season summaries
        """
        if data is None:
            if self.filtered_data is None:
                self.apply_data_filtering()
            data = self.filtered_data
            
        # Group by driver and season
        season_stats = []
        
        for (driver, season), group in data.groupby(['Driver', 'Season']):
            # Basic race statistics
            stats = {
                'Driver': driver,
                'Season': season,
                'races_run': len(group),
                'wins': (group['Finish'] == 1).sum(),
                'top_5s': (group['Finish'] <= 5).sum(),  # Match clustering expectation
                'top_10s': (group['Finish'] <= 10).sum(),  # Match clustering expectation
                'avg_finish': group['Finish'].mean(),
                'avg_start': group['Start'].mean() if 'Start' in group.columns else np.nan,
                'total_points': group['Pts'].sum() if 'Pts' in group.columns else np.nan,
                'laps_led': group['Led'].sum() if 'Led' in group.columns else 0,
                'dnfs': (group['Status'] != 'running').sum() if 'Status' in group.columns else 0,  # Match clustering expectation
                'avg_rating': group['Rating'].mean() if 'Rating' in group.columns else np.nan  # Match clustering expectation
            }
            
            # Calculate rates
            if stats['races_run'] > 0:
                stats['win_rate'] = stats['wins'] / stats['races_run']
                stats['top_5_rate'] = stats['top_5s'] / stats['races_run']
                stats['top_10_rate'] = stats['top_10s'] / stats['races_run']
                stats['dnf_rate'] = stats['dnfs'] / stats['races_run']
            else:
                stats.update({'win_rate': 0, 'top_5_rate': 0, 'top_10_rate': 0, 'dnf_rate': 0})
            
            season_stats.append(stats)
        
        self.driver_seasons = pd.DataFrame(season_stats)
        self.logger.info(f"Created season summaries for {len(self.driver_seasons)} driver-seasons")
        
        return self.driver_seasons
    
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
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                freshness_info['metadata'] = metadata
                
                # Parse update time
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
            parquet_path = Path(self.paths['raw_data']) / 'cup_series.parquet'
            csv_path = Path(self.paths['raw_data']) / 'cup_series.csv'
            
            data_file = None
            if parquet_path.exists():
                data_file = parquet_path
            elif csv_path.exists():
                data_file = csv_path
                
            if data_file:
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
                    'recommendation': 'Run Rscript scripts/update_data.R to fetch latest data'
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
                'season_range': f"{self.raw_data['Season'].min()} - {self.raw_data['Season'].max()}",
                'unique_drivers': self.raw_data['Driver'].nunique(),
                'total_races': len(self.raw_data.groupby(['Season', 'Race']))
            }
        
        if self.filtered_data is not None:
            summary['filtered_data'] = {
                'total_records': len(self.filtered_data),
                'season_range': f"{self.filtered_data['Season'].min()} - {self.filtered_data['Season'].max()}",
                'unique_drivers': self.filtered_data['Driver'].nunique()
            }
        
        if self.driver_seasons is not None:
            summary['aggregated'] = {
                'driver_seasons': len(self.driver_seasons),
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