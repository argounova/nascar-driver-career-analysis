#!/usr/bin/env python3
"""
Example script showing how to update NASCAR data and load it for analysis.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import NASCARDataLoader


def main():
    """Update NASCAR data and perform basic analysis."""
    
    print("🏁 NASCAR Data Update and Analysis")
    print("=" * 50)
    
    # Initialize data loader
    loader = NASCARDataLoader()
    
    # Check current data freshness
    print("\n📅 Checking data freshness...")
    freshness = loader.check_data_freshness()
    
    if freshness.get('status') == 'No data file found':
        print("❌ No data found. Fetching fresh data from R...")
        update_needed = True
    elif not freshness.get('is_recent', False):
        print(f"⚠️  Data is {freshness['days_old']} days old. Updating...")
        update_needed = True
    else:
        print(f"✅ Data is fresh ({freshness['days_old']} days old)")
        update_needed = False
    
    # Update data if needed
    if update_needed:
        print("\n🔄 Updating data from nascaR.data package...")
        success = loader.update_data_from_r()
        
        if success:
            print("✅ Data update completed successfully!")
        else:
            print("❌ Data update failed. Check that R is installed and nascaR.data package is available.")
            return
    
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
            print(f"Seasons: {raw['seasons']}")
            print(f"Drivers: {raw['unique_drivers']:,}")
        
        if 'driver_seasons' in summary:
            seasons = summary['driver_seasons']
            print(f"Driver Seasons: {seasons['total_driver_seasons']:,}")
            print(f"Avg Races/Season: {seasons['avg_races_per_season']}")
        
        # Save processed data
        print("\n💾 Saving processed data...")
        loader.save_processed_data()
        print("✅ Data ready for machine learning analysis!")
        
        # Show some sample data
        if loader.driver_seasons is not None:
            print("\n🏆 Sample Recent Driver Seasons:")
            recent_data = (loader.driver_seasons
                          .query('Season >= 2023')
                          .sort_values(['Season', 'wins'], ascending=[False, False])
                          .head(10))
            
            print(recent_data[['Season', 'Driver', 'wins', 'avg_finish', 'top_5_rate']].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return
    
    print("\n🎯 Ready for driver career analysis!")


if __name__ == "__main__":
    main()