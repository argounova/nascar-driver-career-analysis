#!/usr/bin/env python3
"""
NASCAR Data Update and Load Script with Track Data Cleaning

This script handles updating NASCAR data from the R package and loading it for analysis.
It checks data freshness and updates only when needed.

ENHANCED: Now includes comprehensive track data cleaning to fix NA values and 
properly classify track types based on your specifications:
- Surface types: dirt, paved, road
- Track classification: Uses length and keywords to properly identify road courses
- Fixes missing/NA track names with proper road course identification
"""

import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Add project root and src to Python path for proper imports
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'

# Add both project root and src to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from data.data_loader import NASCARDataLoader


def clean_track_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize track data according to specifications:
    
    Rules:
    1. Surface types: dirt, paved, road only
    2. Longest oval is 2.66 miles - anything over is road course
    3. Tracks 1.5-2.66 miles: use keywords to determine if road course
    4. Fix NA/missing track names for known road courses
    
    Args:
        df: Raw NASCAR DataFrame
        
    Returns:
        DataFrame with cleaned track data
    """
    print("\n🧹 Cleaning track data...")
    
    df_clean = df.copy()
    
    # Convert length to numeric, handling any string values
    df_clean['Length'] = pd.to_numeric(df_clean['Length'], errors='coerce')
    
    # Track cleaning statistics
    na_tracks_fixed = 0
    surface_corrections = 0
    track_type_corrections = 0
    
    # Keywords that indicate road courses (case-insensitive)
    road_keywords = [
        'road course', 'road', 'roval', 'autodromo', 'mexico', 'grand prix', 
        'street', 'circuit', 'sonoma', 'glen', 'watkins', 'cota',
        'infineon', 'sears point', 'montreal', 'chicago street'
    ]
    
    # Process each row to clean track data
    for idx, row in df_clean.iterrows():
        track_name = str(row.get('Track', '')).lower().strip()
        race_name = str(row.get('Name', '')).lower().strip()
        length = row.get('Length', 1.5)
        surface = str(row.get('Surface', '')).lower().strip()
        
        # Handle missing length
        if pd.isna(length) or length <= 0:
            length = 1.5  # Default reasonable length
            df_clean.at[idx, 'Length'] = length
        
        # Fix NA track names - look for road course indicators
        if track_name in ['na', 'nan', ''] or pd.isna(row.get('Track')):
            # Check race name for clues
            if any(keyword in race_name for keyword in ['mexico', 'autodromo', 'grand prix', 'road']):
                if 'mexico' in race_name:
                    df_clean.at[idx, 'Track'] = 'Autodromo Hermanos Rodriguez'
                    na_tracks_fixed += 1
                elif 'montreal' in race_name:
                    df_clean.at[idx, 'Track'] = 'Circuit Gilles Villeneuve'
                    na_tracks_fixed += 1
                elif 'chicago' in race_name and 'street' in race_name:
                    df_clean.at[idx, 'Track'] = 'Chicago Street Course'
                    na_tracks_fixed += 1
            
            # Update track_name for further processing
            track_name = str(df_clean.at[idx, 'Track']).lower().strip()
        
        # Determine if this is a road course based on length and keywords
        is_road_course = False
        
        # Rule 1: Anything over 2.66 miles is a road course (longest oval)
        if length > 2.66:
            is_road_course = True
        
        # Rule 2: For tracks 1.5-2.66 miles, check for road course keywords
        elif length >= 1.5:
            # Check both track name and race name for road course keywords
            combined_text = f"{track_name} {race_name}".lower()
            if any(keyword in combined_text for keyword in road_keywords):
                is_road_course = True
        
        # Rule 3: Some road courses might be shorter (like street courses)
        elif any(keyword in f"{track_name} {race_name}".lower() for keyword in 
                ['street', 'circuit', 'road course', 'roval']):
            is_road_course = True
        
        # Clean surface data
        original_surface = surface
        
        if is_road_course:
            # Road courses should have "road" surface
            if surface != 'road':
                df_clean.at[idx, 'Surface'] = 'road'
                surface_corrections += 1
        else:
            # Non-road courses: standardize to dirt or paved
            if surface in ['dirt', 'clay', 'sand']:
                df_clean.at[idx, 'Surface'] = 'dirt'
                if original_surface != 'dirt':
                    surface_corrections += 1
            else:
                # Default to paved for ovals/standard tracks
                df_clean.at[idx, 'Surface'] = 'paved'
                if original_surface not in ['paved', 'asphalt', 'concrete']:
                    surface_corrections += 1
    
    # Create standardized track type column for easy filtering
    df_clean['track_type_cleaned'] = df_clean.apply(_classify_cleaned_track_type, axis=1)
    
    # Report cleaning results
    print(f"✅ Track data cleaning completed:")
    print(f"   📝 Fixed {na_tracks_fixed} NA/missing track names")
    print(f"   🏁 Corrected {surface_corrections} surface classifications")
    
    # Show track type distribution
    track_type_counts = df_clean['track_type_cleaned'].value_counts()
    print(f"   📊 Track type distribution:")
    for track_type, count in track_type_counts.items():
        print(f"      {track_type}: {count} races")
    
    # Show surface distribution
    surface_counts = df_clean['Surface'].value_counts()
    print(f"   🏗️  Surface distribution:")
    for surface, count in surface_counts.items():
        print(f"      {surface}: {count} races")
    
    return df_clean


def _classify_cleaned_track_type(row) -> str:
    """
    Classify track type based on cleaned data
    
    Args:
        row: DataFrame row with cleaned track data
        
    Returns:
        str: Track type (road, superspeedway, short, intermediate)
    """
    length = row.get('Length', 1.5)
    surface = str(row.get('Surface', '')).lower()
    track_name = str(row.get('Track', '')).lower()
    
    # Road courses - identified by surface type or specific characteristics
    if surface == 'road' or length > 2.66:
        return 'road'
    
    # For paved/dirt ovals, classify by length
    if length >= 2.0:
        return 'superspeedway'
    elif length < 1.0:
        return 'short'
    else:
        return 'intermediate'


def run_r_update_script() -> bool:
    """
    Run the R data update script.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Path to the R script
    r_script_path = Path(__file__).parent / "update_data.R"
    
    if not r_script_path.exists():
        print(f"❌ R script not found at {r_script_path}")
        print("Please ensure the update_data.R script exists.")
        return False
    
    print("🔄 Running R data update script...")
    try:
        result = subprocess.run(
            ['Rscript', str(r_script_path)],
            cwd=Path(__file__).parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Show R script output
        if result.stdout:
            print("R Script Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        if result.stderr:
            print("R Script Warnings:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"  ⚠️  {line}")
        
        if result.returncode == 0:
            print("✅ R script completed successfully")
            return True
        else:
            print(f"❌ R script failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ R script timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print("❌ Rscript not found. Please ensure R is installed and in your PATH")
        return False
    except Exception as e:
        print(f"❌ Error running R script: {e}")
        return False


def apply_post_load_cleaning():
    """
    Apply track data cleaning to the loaded data files.
    This runs after R script but before the data loader processes it.
    """
    # Look for both parquet and CSV files - data loader prioritizes parquet
    parquet_path = project_root / 'data' / 'raw' / 'cup_series.parquet'
    csv_path = project_root / 'data' / 'raw' / 'cup_series.csv'
    
    data_file = None
    file_format = None
    
    if parquet_path.exists():
        data_file = parquet_path
        file_format = 'parquet'
        print(f"📂 Found Parquet file: {data_file}")
    elif csv_path.exists():
        data_file = csv_path
        file_format = 'csv'
        print(f"📂 Found CSV file: {data_file}")
    else:
        print("⚠️ No raw data file found to clean")
        return False
    
    print(f"📊 Loading {file_format.upper()} data from {data_file}")
    
    try:
        # Load the raw data
        if file_format == 'parquet':
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)
        
        print(f"📊 Loaded {len(df):,} records")
        
        # Create backup of original data
        backup_suffix = f"_original_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        if file_format == 'parquet':
            backup_path = project_root / 'data' / 'raw' / f'cup_series{backup_suffix}.parquet'
            if not any(f.name.startswith('cup_series_original_backup') for f in (project_root / 'data' / 'raw').glob('*.parquet')):
                df.to_parquet(backup_path, index=False)
                print(f"📁 Original Parquet backed up to {backup_path}")
        else:
            backup_path = project_root / 'data' / 'raw' / f'cup_series{backup_suffix}.csv'
            if not any(f.name.startswith('cup_series_original_backup') for f in (project_root / 'data' / 'raw').glob('*.csv')):
                df.to_csv(backup_path, index=False)
                print(f"📁 Original CSV backed up to {backup_path}")
        
        # Apply track cleaning
        df_cleaned = clean_track_data(df)
        
        # Replace the original file with cleaned version
        if file_format == 'parquet':
            df_cleaned.to_parquet(parquet_path, index=False)
            print(f"✅ Cleaned data saved to {parquet_path} (Parquet)")
            
            # Also update CSV if it exists
            if csv_path.exists():
                df_cleaned.to_csv(csv_path, index=False)
                print(f"✅ Cleaned data also saved to {csv_path} (CSV)")
        else:
            df_cleaned.to_csv(csv_path, index=False)
            print(f"✅ Cleaned data saved to {csv_path} (CSV)")
        
        # Update last_update.txt with cleaning info
        update_info_path = project_root / 'data' / 'raw' / 'last_update.txt'
        with open(update_info_path, 'a') as f:
            f.write(f"\nTrack data cleaning applied: {pd.Timestamp.now()}\n")
            f.write(f"Records processed: {len(df_cleaned):,}\n")
            f.write(f"Original format: {file_format.upper()}\n")
            f.write(f"Backup created: {backup_path.name}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning track data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Update NASCAR data and perform track cleaning."""
    
    print("🏁 NASCAR Data Update and Analysis with Track Cleaning")
    print("=" * 65)
    
    # Initialize data loader
    loader = NASCARDataLoader()
    
    # Check current data freshness
    print("\n📅 Checking data freshness...")
    freshness = loader.check_data_freshness()
    
    print(f"Status: {freshness.get('status', 'Unknown')}")
    
    if freshness.get('status') == 'No data file found':
        print("❌ No data found. Fetching fresh data...")
        update_needed = True
    elif freshness.get('last_updated'):
        days_old = freshness.get('days_old', 999)
        print(f"Data age: {days_old} days old")
        if not freshness.get('is_recent', False):
            print(f"⚠️  Data needs updating (older than 7 days)")
            update_needed = True
        else:
            print(f"✅ Data is fresh")
            # Ask if user wants to clean anyway
            clean_anyway = input("Apply track cleaning to existing data? (y/n): ").lower().strip()
            update_needed = clean_anyway == 'y'
    else:
        # Fallback to file modification time
        if freshness.get('last_modified'):
            days_old = freshness.get('days_old', 999)
            print(f"File age: {days_old} days old")
            update_needed = days_old > 7
        else:
            print("⚠️  Cannot determine data age, updating to be safe")
            update_needed = True
    
    # Update data if needed
    if update_needed:
        print("\n🔄 Updating NASCAR data...")
        success = run_r_update_script()
        
        if not success:
            print("❌ Data update failed. Checking for existing data...")
            # Try to continue with existing data
            try:
                loader.load_raw_data()
                print("✅ Found existing data, will proceed with cleaning")
            except Exception as e:
                print(f"❌ Cannot proceed without data: {e}")
                return
        else:
            print("✅ Data update completed successfully!")
        
        # Apply track data cleaning
        print("\n🧽 Applying track data cleaning...")
        cleaning_success = apply_post_load_cleaning()
        
        if cleaning_success:
            print("✅ Track data cleaning completed!")
        else:
            print("⚠️ Track cleaning failed, but proceeding with original data")
    
    # Load and validate the final data
    print("\n📊 Loading final processed data...")
    try:
        loader.load_raw_data()
        loader.apply_data_filtering()
        
        summary = loader.get_data_summary()
        print(f"✅ Final data loaded successfully")
        print(f"   📊 {summary['raw_data']['total_records']:,} total records")
        print(f"   👨‍💼 {summary['raw_data']['unique_drivers']} unique drivers") 
        print(f"   📅 Seasons: {summary['raw_data']['season_range']}")
        
        # Show track type distribution if available
        if hasattr(loader, 'raw_data') and 'track_type_cleaned' in loader.raw_data.columns:
            track_types = loader.raw_data['track_type_cleaned'].value_counts()
            print(f"   🏁 Track types:")
            for track_type, count in track_types.items():
                print(f"      {track_type}: {count:,} races")
        
        print("\n🎯 Data is ready for analysis!")
        print("   Models can now use cleaned track classifications")
        print("   Shane van Gisbergen's road course data should be complete")
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")


if __name__ == "__main__":
    main()