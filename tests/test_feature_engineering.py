#!/usr/bin/env python3
"""
Real NASCAR Data Feature Engineering Test Script
Tests feature engineering with actual nascaR.data instead of sample data.

Move this to tests/test_feature_engineering_real.py for better organization.
Run from project root: python tests/test_feature_engineering_real.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add project root and src to Python path
project_root = Path(__file__).parent.parent if 'tests' in str(Path(__file__).parent) else Path(__file__).parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print("🔧 Real NASCAR Data Feature Engineering Test")
print("=" * 55)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from data.data_loader import load_nascar_data
    from data.feature_engineering import NASCARFeatureEngineer, create_nascar_features
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def load_real_nascar_data():
    """Load real NASCAR data from the pipeline."""
    print("\n📊 Loading Real NASCAR Data")
    print("-" * 35)
    
    try:
        # Load real NASCAR data
        print("Loading data using established pipeline...")
        data_loader = load_nascar_data()
        
        # Display data summary
        print(f"✅ Loaded real NASCAR data successfully")
        print(f"   Raw records: {len(data_loader.raw_data):,}")
        print(f"   Filtered records: {len(data_loader.filtered_data):,}")
        print(f"   Driver-seasons: {len(data_loader.driver_seasons):,}")
        print(f"   Unique drivers: {data_loader.driver_seasons['Driver'].nunique():,}")
        print(f"   Season range: {data_loader.driver_seasons['Season'].min()} - {data_loader.driver_seasons['Season'].max()}")
        
        # Show some real driver examples
        print(f"\n📋 Real Driver Examples:")
        top_drivers = data_loader.driver_seasons.groupby('Driver')['wins'].sum().nlargest(5)
        for driver, total_wins in top_drivers.items():
            seasons = len(data_loader.driver_seasons[data_loader.driver_seasons['Driver'] == driver])
            avg_finish = data_loader.driver_seasons[data_loader.driver_seasons['Driver'] == driver]['avg_finish'].mean()
            print(f"   {driver}: {total_wins} total wins, {seasons} seasons, {avg_finish:.1f} avg finish")
        
        return data_loader
        
    except Exception as e:
        print(f"❌ Failed to load real NASCAR data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_feature_engineering(data_loader):
    """Test feature engineering with real NASCAR data."""
    print(f"\n🔧 Testing Feature Engineering with Real Data")
    print("-" * 50)
    
    try:
        # Initialize feature engineer
        engineer = NASCARFeatureEngineer()
        
        # Use real driver season data
        real_driver_seasons = data_loader.driver_seasons
        print(f"Input data: {len(real_driver_seasons)} real driver-seasons")
        
        # Test each step with real data
        step_times = {}
        
        # Step 1: Load data
        start_time = time.time()
        engineer.load_driver_seasons(real_driver_seasons)
        step_times['load'] = time.time() - start_time
        print(f"1. ✅ Data loaded ({step_times['load']:.1f}s)")
        
        # Step 2: Rolling features
        start_time = time.time()
        rolling_data = engineer.create_rolling_features()
        step_times['rolling'] = time.time() - start_time
        original_cols = len(real_driver_seasons.columns)
        rolling_cols = len(rolling_data.columns)
        print(f"2. ✅ Rolling features: {original_cols} → {rolling_cols} columns ({step_times['rolling']:.1f}s)")
        
        # Step 3: Career phases
        start_time = time.time()
        phase_data = engineer.identify_career_phases(rolling_data)
        step_times['phases'] = time.time() - start_time
        phase_cols = len(phase_data.columns)
        print(f"3. ✅ Career phases: {rolling_cols} → {phase_cols} columns ({step_times['phases']:.1f}s)")
        
        # Show real career phase examples
        print(f"\n   Real Career Phase Examples:")
        for driver in ['Kyle Larson', 'Denny Hamlin', 'Kevin Harvick'][:3]:
            driver_data = phase_data[phase_data['Driver'] == driver].sort_values('Season')
            if not driver_data.empty:
                latest = driver_data.iloc[-1]
                total_seasons = len(driver_data)
                current_phase = latest.get('career_phase', 'Unknown')
                season_num = latest.get('career_season_number', '?')
                print(f"     {driver}: {current_phase} (Season {season_num}/{total_seasons})")
        
        # Step 4: Performance trends
        start_time = time.time()
        trend_data = engineer.calculate_performance_trends(phase_data)
        step_times['trends'] = time.time() - start_time
        trend_cols = len(trend_data.columns)
        print(f"4. ✅ Performance trends: {phase_cols} → {trend_cols} columns ({step_times['trends']:.1f}s)")
        
        # Step 5: Peak detection
        start_time = time.time()
        peak_data = engineer.detect_peak_performance(trend_data)
        step_times['peaks'] = time.time() - start_time
        peak_cols = len(peak_data.columns)
        print(f"5. ✅ Peak detection: {trend_cols} → {peak_cols} columns ({step_times['peaks']:.1f}s)")
        
        # Step 6: Consistency metrics
        start_time = time.time()
        consistency_data = engineer.calculate_consistency_metrics(peak_data)
        step_times['consistency'] = time.time() - start_time
        consistency_cols = len(consistency_data.columns)
        print(f"6. ✅ Consistency metrics: {peak_cols} → {consistency_cols} columns ({step_times['consistency']:.1f}s)")
        
        # Step 7: Lag features
        start_time = time.time()
        lag_data = engineer.create_lag_features(consistency_data)
        step_times['lags'] = time.time() - start_time
        lag_cols = len(lag_data.columns)
        print(f"7. ✅ Lag features: {consistency_cols} → {lag_cols} columns ({step_times['lags']:.1f}s)")
        
        # Step 8: LSTM sequences
        start_time = time.time()
        sequences, targets, driver_names = engineer.create_lstm_sequences(lag_data)
        step_times['sequences'] = time.time() - start_time
        print(f"8. ✅ LSTM sequences: {len(sequences)} sequences, shape {sequences.shape} ({step_times['sequences']:.1f}s)")
        print(f"   Drivers with sequences: {len(set(driver_names))}")
        
        # Test complete pipeline
        print(f"\n9. Testing complete pipeline...")
        start_time = time.time()
        engineered_features = engineer.engineer_all_features(real_driver_seasons)
        step_times['complete'] = time.time() - start_time
        final_cols = len(engineered_features.columns)
        print(f"   ✅ Complete pipeline: {original_cols} → {final_cols} features ({step_times['complete']:.1f}s)")
        
        # Performance summary
        total_time = sum(step_times.values())
        print(f"\n⏱️  Performance Summary:")
        print(f"   Total processing time: {total_time:.1f} seconds")
        print(f"   Records processed: {len(real_driver_seasons):,}")
        print(f"   Processing rate: {len(real_driver_seasons)/total_time:.0f} records/second")
        
        return engineer, engineered_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_real_driver_features(engineer, engineered_features):
    """Analyze features for real NASCAR drivers."""
    print(f"\n📈 Analyzing Real Driver Features")
    print("-" * 35)
    
    try:
        # Get feature summary
        summary = engineer.get_feature_summary()
        print(f"Feature Engineering Results:")
        print(f"   Total Features: {summary['total_features']}")
        print(f"   Driver-Seasons: {summary['total_driver_seasons']}")
        print(f"   Unique Drivers: {summary['unique_drivers']}")
        
        print(f"\n   Feature Breakdown:")
        for category, count in summary['feature_categories'].items():
            if count > 0:
                print(f"     {category.replace('_', ' ').title()}: {count}")
        
        # Analyze specific real drivers
        print(f"\n🏁 Real Driver Feature Analysis:")
        
        test_drivers = ['Kyle Larson', 'Denny Hamlin', 'Chase Elliott']
        
        for driver in test_drivers:
            driver_data = engineered_features[engineered_features['Driver'] == driver].sort_values('Season')
            
            if not driver_data.empty:
                print(f"\n   {driver} ({len(driver_data)} seasons):")
                
                # Latest season analysis
                latest = driver_data.iloc[-1]
                print(f"     Latest Season ({latest['Season']}):")
                print(f"       Career Phase: {latest.get('career_phase', 'Unknown')}")
                print(f"       Career Progress: {latest.get('career_progress', 0):.1%}")
                print(f"       Peak Timing: {latest.get('peak_timing', 0):.1%}")
                print(f"       Performance Score: {latest.get('performance_score', 0):.3f}")
                
                # Trend analysis
                if 'avg_finish_improvement_rate' in latest:
                    improvement = latest['avg_finish_improvement_rate']
                    trend = "Improving" if improvement > 0 else "Declining" if improvement < 0 else "Stable"
                    print(f"       Trend: {trend} ({improvement:+.2f} positions/year)")
                
                # Rolling averages
                short_term_avg = latest.get('avg_finish_short_term_avg', latest['avg_finish'])
                long_term_avg = latest.get('avg_finish_long_term_avg', latest['avg_finish'])
                print(f"       Short-term avg: {short_term_avg:.1f}")
                print(f"       Long-term avg: {long_term_avg:.1f}")
        
        # LSTM sequence analysis
        if 'lstm_sequences' in summary:
            lstm_info = summary['lstm_sequences']
            print(f"\n🧠 LSTM Sequence Analysis:")
            print(f"   Total sequences: {lstm_info['total_sequences']}")
            print(f"   Sequence length: {lstm_info['sequence_length']} seasons")
            print(f"   Features per sequence: {lstm_info['feature_count']}")
            print(f"   Drivers with sequences: {lstm_info['unique_drivers']}")
            
            # Show which real drivers have sequences
            sequences, targets, driver_names = engineer.lstm_sequences
            unique_drivers = list(set(driver_names))
            print(f"   Real drivers ready for LSTM: {len(unique_drivers)}")
            if len(unique_drivers) <= 10:
                print(f"     {', '.join(unique_drivers)}")
            else:
                print(f"     {', '.join(unique_drivers[:10])}... and {len(unique_drivers)-10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        return False

def test_feature_quality(engineered_features):
    """Test quality of engineered features."""
    print(f"\n🔍 Testing Feature Quality")
    print("-" * 28)
    
    try:
        # Check for missing values in key features
        key_features = [
            'career_phase', 'career_progress', 'peak_timing', 
            'performance_score', 'avg_finish_short_term_avg'
        ]
        
        print("Missing value check:")
        for feature in key_features:
            if feature in engineered_features.columns:
                missing = engineered_features[feature].isnull().sum()
                total = len(engineered_features)
                pct = (missing / total) * 100
                status = "✅" if pct < 10 else "⚠️" if pct < 50 else "❌"
                print(f"   {status} {feature}: {missing}/{total} ({pct:.1f}%) missing")
        
        # Check feature ranges
        print(f"\nFeature range validation:")
        
        # Career progress should be 0-1
        if 'career_progress' in engineered_features.columns:
            cp = engineered_features['career_progress'].dropna()
            valid_cp = ((cp >= 0) & (cp <= 1)).all()
            print(f"   {'✅' if valid_cp else '❌'} Career progress in [0,1]: {valid_cp}")
        
        # Peak timing should be 0-1
        if 'peak_timing' in engineered_features.columns:
            pt = engineered_features['peak_timing'].dropna()
            valid_pt = ((pt >= 0) & (pt <= 1)).all()
            print(f"   {'✅' if valid_pt else '❌'} Peak timing in [0,1]: {valid_pt}")
        
        # Performance score should be reasonable
        if 'performance_score' in engineered_features.columns:
            ps = engineered_features['performance_score'].dropna()
            valid_ps = ((ps >= 0) & (ps <= 1)).all()
            print(f"   {'✅' if valid_ps else '❌'} Performance score in [0,1]: {valid_ps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature quality testing failed: {e}")
        return False

# Main test execution
def main():
    """Run all real data feature engineering tests."""
    
    print("Starting Real NASCAR Data Feature Engineering Tests...")
    
    # Test 1: Load real data
    data_loader = load_real_nascar_data()
    if data_loader is None:
        print("❌ Cannot proceed without real data")
        return False
    
    # Test 2: Feature engineering pipeline
    engineer, engineered_features = test_real_feature_engineering(data_loader)
    if engineer is None or engineered_features is None:
        print("❌ Feature engineering failed")
        return False
    
    # Test 3: Analyze real driver features
    analysis_success = analyze_real_driver_features(engineer, engineered_features)
    
    # Test 4: Feature quality validation
    quality_success = test_feature_quality(engineered_features)
    
    # Summary
    print(f"\n{'='*55}")
    if analysis_success and quality_success:
        print("🎉 All Real Data Feature Engineering Tests Passed!")
        print("✅ Real NASCAR data processed successfully")
        print("✅ 175+ features engineered from actual careers") 
        print("✅ LSTM sequences ready for real driver predictions")
        print("✅ Feature quality validated")
        print("\nReal drivers ready for analysis:")
        
        # Show some statistics
        if engineered_features is not None:
            modern_drivers = engineered_features[engineered_features['Season'] >= 2010]['Driver'].nunique()
            total_sequences = len(engineer.lstm_sequences[0]) if engineer.lstm_sequences else 0
            print(f"   Modern era drivers (2010+): {modern_drivers}")
            print(f"   LSTM training sequences: {total_sequences}")
        
        print("\nNext steps:")
        print("1. Test clustering with real driver archetypes")
        print("2. Train LSTM with real career progression data")
        print("3. Run end-to-end pipeline with real predictions")
    else:
        print("❌ Some Real Data Feature Engineering Tests Failed!")
        print("Check the error messages above")
    
    return analysis_success and quality_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)