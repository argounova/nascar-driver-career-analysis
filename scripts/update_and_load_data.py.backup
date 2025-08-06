#!/usr/bin/env python3
"""
NASCAR Data Update and Load Script

This script handles updating NASCAR data from the R package and loading it for analysis.
It checks data freshness and updates only when needed.
"""

import sys
import subprocess
from pathlib import Path

# Add project root and src to Python path for proper imports
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'

# Add both project root and src to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from data.data_loader import NASCARDataLoader


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


def main():
    """Update NASCAR data and perform basic analysis."""
    
    print("🏁 NASCAR Data Update and Analysis")
    print("=" * 50)
    
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
            update_needed = False
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
            print("❌ Data update failed. Proceeding with existing data if available...")
            # Try to continue with existing data
            try:
                loader.load_raw_data()
            except Exception as e:
                print(f"❌ Cannot proceed without data: {e}")
                return
        else:
            print("✅ Data update completed successfully!")
    
    # Load and process data
    print("\n📊 Loading and processing data...")
    try:
        loader.load_raw_data()
        loader.apply_data_filtering()
        loader.create_driver_season_summary()
        
        # Display summary
        summary = loader.get_data_summary()
        
        print("\n📈 Data Summary:")
        print("-" * 30)
        
        if 'raw_data' in summary:
            raw = summary['raw_data']
            print(f"Raw Data: {raw['total_records']:,} records")
            print(f"Seasons: {raw['season_range']}")
            print(f"Drivers: {raw['unique_drivers']:,}")
        
        if 'aggregated' in summary:
            agg = summary['aggregated']
            print(f"Driver Seasons: {agg['driver_seasons']:,}")
            print(f"Avg Races/Season: {agg['avg_races_per_season']}")
        
        # Save processed data
        print("\n💾 Saving processed data...")
        loader.save_processed_data()
        print("✅ Processed data saved to data/processed/")
        
        # Show some sample data
        if loader.driver_seasons is not None:
            print("\n🏆 Sample Recent Driver Seasons:")
            recent_data = (loader.driver_seasons
                          .query('Season >= 2023')
                          .sort_values(['Season', 'wins'], ascending=[False, False])
                          .head(10))
            
            if len(recent_data) > 0:
                print(recent_data[['Season', 'Driver', 'wins', 'avg_finish', 'top_5_rate']].to_string(index=False))
            else:
                print("No recent data available (2023+)")
        
        print("\n🎯 Data ready for machine learning analysis!")
        print("Next steps:")
        print("  - Run: python scripts/train_models.py")
        print("  - Or explore data in Jupyter notebooks")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()