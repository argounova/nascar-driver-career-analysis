#!/usr/bin/env python3
"""
Test script for the Feature Engineering module.
Run this from the project root directory.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root and src to Python path
project_root = Path(__file__).parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print("🔧 Testing Feature Engineering Module")
print("=" * 50)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from data.feature_engineering import NASCARFeatureEngineer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test configuration
try:
    print("\n⚙️  Testing configuration...")
    config = get_config()
    engineer = NASCARFeatureEngineer(config)
    print("✅ Feature engineer initialized")
    print(f"   Rolling windows: {engineer.rolling_windows}")
    print(f"   Career phases: {engineer.career_phases}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Create sample NASCAR data for testing
def create_sample_driver_data():
    """Create realistic sample NASCAR driver season data."""
    print("\n📊 Creating sample NASCAR data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Sample drivers with different career patterns
    drivers = [
        'Kyle Larson',      # Dominant performer
        'Chase Elliott',    # Consistent contender  
        'Denny Hamlin',     # Veteran
        'William Byron',    # Rising star
        'Austin Dillon'     # Mid-pack
    ]
    
    data = []
    
    for driver in drivers:
        # Each driver gets 8-12 seasons
        career_length = np.random.randint(8, 13)
        start_season = 2025 - career_length + 1
        
        for season_num in range(career_length):
            season = start_season + season_num
            
            # Create performance that varies by driver and career stage
            if driver == 'Kyle Larson':
                # Dominant performer - improving over time
                base_finish = 8 + season_num * -0.5  # Gets better
                win_rate = 0.15 + season_num * 0.02
            elif driver == 'Chase Elliott':
                # Consistent performer
                base_finish = 12 + np.random.normal(0, 2)
                win_rate = 0.08 + np.random.normal(0, 0.02)
            elif driver == 'Denny Hamlin':
                # Veteran - peak then decline
                career_peak = career_length // 2
                distance_from_peak = abs(season_num - career_peak)
                base_finish = 10 + distance_from_peak * 1.5
                win_rate = 0.12 - distance_from_peak * 0.015
            elif driver == 'William Byron':
                # Rising star - late bloomer
                base_finish = 25 - season_num * 2  # Rapid improvement
                win_rate = season_num * 0.02
            else:  # Austin Dillon
                # Mid-pack with some variation
                base_finish = 18 + np.random.normal(0, 3)
                win_rate = 0.03 + np.random.normal(0, 0.01)
            
            # Ensure realistic bounds
            avg_finish = max(5, min(30, base_finish + np.random.normal(0, 2)))
            win_rate = max(0, min(0.3, win_rate))
            
            # Calculate derived metrics
            races_run = 36  # Full season
            wins = int(races_run * win_rate)
            top_5_rate = min(0.8, win_rate * 3 + 0.1)
            top_10_rate = min(0.9, top_5_rate * 1.5)
            dnf_rate = max(0.02, min(0.15, 0.1 + np.random.normal(0, 0.03)))
            avg_rating = max(50, min(150, 100 - (avg_finish - 15) * 2 + np.random.normal(0, 10)))
            
            data.append({
                'Driver': driver,
                'Season': season,
                'races_run': races_run,
                'avg_finish': round(avg_finish, 1),
                'wins': wins,
                'win_rate': round(win_rate, 3),
                'top_5s': int(races_run * top_5_rate),
                'top_5_rate': round(top_5_rate, 3),
                'top_10s': int(races_run * top_10_rate), 
                'top_10_rate': round(top_10_rate, 3),
                'total_points': int(1000 + (40 - avg_finish) * 20 + wins * 50),
                'laps_led': int(wins * 50 + np.random.randint(0, 100)),
                'dnfs': int(races_run * dnf_rate),
                'dnf_rate': round(dnf_rate, 3),
                'avg_rating': round(avg_rating, 1)
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Created sample data: {len(df)} driver-seasons")
    print(f"   Drivers: {df['Driver'].nunique()}")
    print(f"   Seasons: {df['Season'].min()}-{df['Season'].max()}")
    
    return df

# Test feature engineering pipeline
def test_feature_engineering_pipeline():
    """Test the complete feature engineering pipeline."""
    
    # Create sample data
    sample_data = create_sample_driver_data()
    
    print("\n🔧 Testing Feature Engineering Pipeline")
    print("-" * 40)
    
    try:
        # Initialize engineer
        engineer = NASCARFeatureEngineer()
        
        # Test 1: Load data
        print("1. Loading driver season data...")
        engineer.load_driver_seasons(sample_data)
        print(f"   ✅ Loaded {len(engineer.driver_seasons)} seasons")
        
        # Test 2: Rolling features
        print("2. Creating rolling features...")
        rolling_data = engineer.create_rolling_features()
        original_cols = len(sample_data.columns)
        rolling_cols = len(rolling_data.columns)
        print(f"   ✅ Added {rolling_cols - original_cols} rolling features")
        
        # Test 3: Career phases
        print("3. Identifying career phases...")
        phase_data = engineer.identify_career_phases(rolling_data)
        phase_cols = len(phase_data.columns)
        print(f"   ✅ Added {phase_cols - rolling_cols} career phase features")
        
        # Show example of career phases
        sample_driver = phase_data[phase_data['Driver'] == 'Kyle Larson'].sort_values('Season')
        if not sample_driver.empty:
            print(f"   Example - Kyle Larson career phases:")
            for _, row in sample_driver.iterrows():
                print(f"     {row['Season']}: {row.get('career_phase', 'Unknown')} (Season {row.get('career_season_number', '?')})")
        
        # Test 4: Performance trends
        print("4. Calculating performance trends...")
        trend_data = engineer.calculate_performance_trends(phase_data)
        trend_cols = len(trend_data.columns)
        print(f"   ✅ Added {trend_cols - phase_cols} trend features")
        
        # Test 5: Peak detection
        print("5. Detecting peak performance...")
        peak_data = engineer.detect_peak_performance(trend_data)
        peak_cols = len(peak_data.columns)
        print(f"   ✅ Added {peak_cols - trend_cols} peak detection features")
        
        # Test 6: Consistency metrics
        print("6. Calculating consistency metrics...")
        consistency_data = engineer.calculate_consistency_metrics(peak_data)
        consistency_cols = len(consistency_data.columns)
        print(f"   ✅ Added {consistency_cols - peak_cols} consistency features")
        
        # Test 7: Lag features
        print("7. Creating lag features...")
        lag_data = engineer.create_lag_features(consistency_data)
        lag_cols = len(lag_data.columns)
        print(f"   ✅ Added {lag_cols - consistency_cols} lag features")
        
        # Test 8: LSTM sequences
        print("8. Creating LSTM sequences...")
        sequences, targets, driver_names = engineer.create_lstm_sequences(lag_data)
        print(f"   ✅ Created {len(sequences)} LSTM sequences")
        print(f"   ✅ Sequence shape: {sequences.shape}")
        print(f"   ✅ Drivers with sequences: {len(set(driver_names))}")
        
        # Test 9: Complete pipeline
        print("9. Testing complete pipeline...")
        engineered_features = engineer.engineer_all_features(sample_data)
        final_cols = len(engineered_features.columns)
        print(f"   ✅ Complete pipeline: {original_cols} → {final_cols} features")
        
        # Show feature summary
        summary = engineer.get_feature_summary()
        print(f"\n📊 Feature Engineering Summary:")
        print(f"   Total Features: {summary['total_features']}")
        print(f"   Driver-Seasons: {summary['total_driver_seasons']}")
        print(f"   Feature Categories:")
        for category, count in summary['feature_categories'].items():
            if count > 0:
                print(f"     {category.replace('_', ' ').title()}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main test execution
def main():
    """Run all feature engineering tests."""
    
    print("Starting Feature Engineering Tests...")
    
    # Test the pipeline
    success = test_feature_engineering_pipeline()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 All Feature Engineering Tests Passed!")
        print("✅ Rolling windows working")
        print("✅ Career phases identified") 
        print("✅ Performance trends calculated")
        print("✅ Peak detection working")
        print("✅ Consistency metrics computed")
        print("✅ LSTM sequences created")
        print("\nNext steps:")
        print("1. Test clustering module")
        print("2. Test complete pipeline with real data")
        print("3. Run end-to-end training")
    else:
        print("❌ Feature Engineering Tests Failed!")
        print("Check the error messages above")

if __name__ == "__main__":
    main()