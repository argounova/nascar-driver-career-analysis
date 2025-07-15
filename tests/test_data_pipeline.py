#!/usr/bin/env python3
"""
Real NASCAR Data Pipeline Test Script

This script tests the complete data pipeline with actual nascaR.data:
1. R script integration and data updates
2. Data loading and validation
3. Data quality assessment
4. Processing pipeline verification

Run this from the project root directory.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import json
from datetime import datetime

# Add project root and src to Python path
project_root = Path(__file__).parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print("🏁 Real NASCAR Data Pipeline Test")
print("=" * 50)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from data.data_loader import NASCARDataLoader
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_r_environment():
    """Test R environment and nascaR.data package availability."""
    print("\n🔧 Testing R Environment")
    print("-" * 30)
    
    try:
        # Test R installation
        result = subprocess.run(['R', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            r_version = result.stdout.split('\n')[0]
            print(f"✅ R installed: {r_version}")
        else:
            print("❌ R not found or not working")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ R not available")
        print("Install R from: https://www.r-project.org/")
        return False
    
    # Test nascaR.data package
    r_script = '''
    if (!require("nascaR.data", quietly = TRUE)) {
        cat("PACKAGE_NOT_INSTALLED\\n")
    } else {
        cat("PACKAGE_AVAILABLE\\n")
        cat("Version:", as.character(packageVersion("nascaR.data")), "\\n")
        
        # Try to load data
        tryCatch({
            data("cup_series", package = "nascaR.data")
            cat("Data loaded successfully\\n")
            cat("Records:", nrow(cup_series), "\\n")
            cat("Columns:", ncol(cup_series), "\\n")
            cat("Years:", min(cup_series$Season, na.rm=TRUE), "-", max(cup_series$Season, na.rm=TRUE), "\\n")
        }, error = function(e) {
            cat("DATA_LOAD_ERROR:", e$message, "\\n")
        })
    }
    '''
    
    try:
        result = subprocess.run(['R', '--slave', '-e', r_script], 
                              capture_output=True, text=True, timeout=30)
        
        output = result.stdout
        print("R package test output:")
        for line in output.split('\n'):
            if line.strip():
                if 'PACKAGE_NOT_INSTALLED' in line:
                    print("❌ nascaR.data package not installed")
                    print("Install with: install.packages('nascaR.data')")
                    return False
                elif 'DATA_LOAD_ERROR' in line:
                    print(f"❌ Data loading error: {line}")
                    return False
                else:
                    print(f"   {line}")
        
        if 'PACKAGE_AVAILABLE' in output and 'Data loaded successfully' in output:
            print("✅ nascaR.data package working correctly")
            return True
        else:
            print("⚠️ Package tests incomplete")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ R script timed out")
        return False
    except Exception as e:
        print(f"❌ R test failed: {e}")
        return False

def test_data_update_script():
    """Test the R data update script."""
    print("\n📊 Testing Data Update Script")
    print("-" * 35)
    
    # Check if update script exists
    update_script = project_root / 'scripts' / 'update_data.R'
    if not update_script.exists():
        print("⚠️ update_data.R script not found")
        print("Creating basic update script...")
        
        # Create the script directory if needed
        scripts_dir = project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Create basic R update script
        r_script_content = '''
# NASCAR Data Update Script
# Updates data from nascaR.data package

library(nascaR.data)
library(arrow)

cat("Loading nascaR.data package...\\n")

# Load Cup Series data
data("cup_series", package = "nascaR.data")

cat("Loaded", nrow(cup_series), "records\\n")
cat("Seasons:", min(cup_series$Season, na.rm=TRUE), "-", max(cup_series$Season, na.rm=TRUE), "\\n")

# Create data directory if it doesn't exist
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("data/raw")) dir.create("data/raw")

# Save as parquet for fast loading in Python
arrow::write_parquet(cup_series, "data/raw/cup_series.parquet")

# Also save as CSV as backup
write.csv(cup_series, "data/raw/cup_series.csv", row.names = FALSE)

# Create metadata
metadata <- list(
    update_time = Sys.time(),
    total_records = nrow(cup_series),
    seasons_covered = paste(min(cup_series$Season, na.rm=TRUE), 
                           max(cup_series$Season, na.rm=TRUE), sep="-"),
    columns = ncol(cup_series),
    column_names = colnames(cup_series)
)

# Save metadata as JSON
cat(jsonlite::toJSON(metadata, pretty=TRUE), file="data/raw/data_metadata.json")

cat("Data saved successfully!\\n")
cat("Files created:\\n")
cat("- data/raw/cup_series.parquet\\n")
cat("- data/raw/cup_series.csv\\n") 
cat("- data/raw/data_metadata.json\\n")
'''
        
        with open(update_script, 'w') as f:
            f.write(r_script_content)
        print(f"✅ Created update script at {update_script}")
    
    # Run the update script
    print("Running data update script...")
    try:
        result = subprocess.run(['Rscript', str(update_script)], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Data update script completed")
            print("Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            # Check if files were created
            parquet_file = project_root / 'data' / 'raw' / 'cup_series.parquet'
            csv_file = project_root / 'data' / 'raw' / 'cup_series.csv'
            metadata_file = project_root / 'data' / 'raw' / 'data_metadata.json'
            
            files_created = []
            if parquet_file.exists():
                files_created.append(f"✅ {parquet_file}")
            if csv_file.exists():
                files_created.append(f"✅ {csv_file}")
            if metadata_file.exists():
                files_created.append(f"✅ {metadata_file}")
            
            if files_created:
                print("Files created:")
                for file_info in files_created:
                    print(f"   {file_info}")
                return True
            else:
                print("❌ No output files found")
                return False
        else:
            print("❌ Data update script failed")
            print("Error output:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data update script timed out (2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        return False

def test_data_loading():
    """Test loading the updated NASCAR data."""
    print("\n📂 Testing Data Loading")
    print("-" * 25)
    
    try:
        # Initialize data loader
        loader = NASCARDataLoader()
        
        # Check data freshness
        freshness = loader.check_data_freshness()
        print(f"Data status: {freshness.get('status', 'Unknown')}")
        
        if 'last_updated' in freshness:
            print(f"Last updated: {freshness['last_updated']}")
        if 'total_records' in freshness:
            print(f"Total records: {freshness['total_records']:,}")
        if 'seasons_covered' in freshness:
            print(f"Seasons: {freshness['seasons_covered']}")
        
        # Load raw data
        print("\nLoading raw data...")
        raw_data = loader.load_raw_data()
        print(f"✅ Loaded {len(raw_data):,} raw records")
        
        # Display basic info
        print(f"   Columns: {len(raw_data.columns)}")
        print(f"   Seasons: {raw_data['Season'].min()} - {raw_data['Season'].max()}")
        print(f"   Drivers: {raw_data['Driver'].nunique():,}")
        print(f"   Races: {raw_data.groupby('Season')['Race'].max().sum():,} total")
        
        # Apply filtering
        print("\nApplying data filtering...")
        filtered_data = loader.apply_data_filtering()
        print(f"✅ Filtered to {len(filtered_data):,} records")
        
        # Create season summaries
        print("\nCreating driver season summaries...")
        driver_seasons = loader.create_driver_season_summary()
        print(f"✅ Created {len(driver_seasons):,} driver-season summaries")
        print(f"   Unique drivers: {driver_seasons['Driver'].nunique():,}")
        print(f"   Season range: {driver_seasons['Season'].min()} - {driver_seasons['Season'].max()}")
        
        # Show some example data
        print("\n📊 Sample Driver Season Data:")
        if 'Kyle Larson' in driver_seasons['Driver'].values:
            larson_data = driver_seasons[driver_seasons['Driver'] == 'Kyle Larson'].sort_values('Season')
            print("Kyle Larson recent seasons:")
            if len(larson_data) > 0:
                for _, row in larson_data.tail(3).iterrows():
                    print(f"   {row['Season']}: {row['wins']} wins, {row['avg_finish']:.1f} avg finish, {row['races_run']} races")
        
        # Get summary statistics
        summary = loader.get_data_summary()
        print("\n📈 Data Pipeline Summary:")
        for section, stats in summary.items():
            print(f"  {section.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"    {key.replace('_', ' ').title()}: {value}")
        
        return True, loader
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_data_quality(loader):
    """Test data quality and completeness."""
    print("\n🔍 Testing Data Quality")
    print("-" * 25)
    
    try:
        if loader is None or loader.driver_seasons is None:
            print("❌ No data available for quality testing")
            return False
        
        data = loader.driver_seasons
        
        # Check for missing values
        print("Missing value analysis:")
        missing_cols = []
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            if missing_count > 0:
                missing_cols.append((col, missing_count, missing_pct))
                print(f"   {col}: {missing_count} ({missing_pct:.1f}%)")
        
        if not missing_cols:
            print("   ✅ No missing values found")
        
        # Check data ranges
        print("\nData range validation:")
        
        # Wins should be >= 0
        invalid_wins = data[data['wins'] < 0]
        if len(invalid_wins) > 0:
            print(f"   ❌ {len(invalid_wins)} records with negative wins")
        else:
            print("   ✅ Win counts valid")
        
        # Average finish should be between 1-43 (reasonable NASCAR field size)
        invalid_finish = data[(data['avg_finish'] < 1) | (data['avg_finish'] > 43)]
        if len(invalid_finish) > 0:
            print(f"   ❌ {len(invalid_finish)} records with invalid avg_finish")
        else:
            print("   ✅ Average finish values valid")
        
        # Rates should be between 0-1
        rate_columns = ['top_5_rate', 'top_10_rate', 'win_rate', 'dnf_rate']
        for col in rate_columns:
            if col in data.columns:
                invalid_rates = data[(data[col] < 0) | (data[col] > 1)]
                if len(invalid_rates) > 0:
                    print(f"   ❌ {len(invalid_rates)} records with invalid {col}")
                else:
                    print(f"   ✅ {col} values valid")
        
        # Check for reasonable driver career lengths
        driver_careers = data.groupby('Driver')['Season'].count()
        print(f"\nCareer length analysis:")
        print(f"   Average seasons per driver: {driver_careers.mean():.1f}")
        print(f"   Longest career: {driver_careers.max()} seasons")
        print(f"   Shortest career: {driver_careers.min()} seasons")
        
        # Check for modern era data
        modern_data = data[data['Season'] >= 2000]
        print(f"\nModern era (2000+): {len(modern_data):,} driver-seasons")
        
        if len(modern_data) > 1000:
            print("   ✅ Sufficient modern era data for analysis")
        else:
            print("   ⚠️ Limited modern era data")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality testing failed: {e}")
        return False

# Main test execution
def main():
    """Run all data pipeline tests."""
    
    print("Starting Real NASCAR Data Pipeline Tests...")
    
    # Test 1: R environment
    r_ok = test_r_environment()
    
    # Test 2: Data update script (only if R works)
    if r_ok:
        update_ok = test_data_update_script()
    else:
        update_ok = False
        print("⚠️ Skipping data update due to R environment issues")
    
    # Test 3: Data loading (try even if update failed, might have existing data)
    loading_ok, loader = test_data_loading()
    
    # Test 4: Data quality (only if loading worked)
    if loading_ok:
        quality_ok = test_data_quality(loader)
    else:
        quality_ok = False
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 Data Pipeline Test Summary")
    print(f"{'='*50}")
    print(f"R Environment: {'✅ OK' if r_ok else '❌ Issues'}")
    print(f"Data Update: {'✅ OK' if update_ok else '❌ Issues'}")
    print(f"Data Loading: {'✅ OK' if loading_ok else '❌ Issues'}")
    print(f"Data Quality: {'✅ OK' if quality_ok else '❌ Issues'}")
    
    if all([r_ok, update_ok, loading_ok, quality_ok]):
        print("\n🎉 All pipeline tests passed!")
        print("✅ Real NASCAR data is ready for model training")
        print("\nNext steps:")
        print("1. Test feature engineering with real data")
        print("2. Test clustering with actual driver careers")
        print("3. Train LSTM with real performance sequences")
    else:
        print("\n⚠️ Some pipeline tests failed")
        print("Check the errors above and fix issues before proceeding")
        
        if not r_ok:
            print("\n🔧 R Environment Issues:")
            print("- Install R from: https://www.r-project.org/")
            print("- Install nascaR.data: install.packages('nascaR.data')")
        
        if r_ok and not update_ok:
            print("\n🔧 Data Update Issues:")
            print("- Check R package dependencies")
            print("- Verify network connection for data download")
    
    return all([r_ok, update_ok, loading_ok, quality_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)