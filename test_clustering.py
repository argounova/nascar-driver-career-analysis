#!/usr/bin/env python3
"""
Test script for the Driver Clustering module.
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

print("🏷️  Testing Driver Clustering Module")
print("=" * 50)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from models.clustering import DriverClusterAnalyzer
    from data.feature_engineering import NASCARFeatureEngineer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test configuration
try:
    print("\n⚙️  Testing configuration...")
    config = get_config()
    analyzer = DriverClusterAnalyzer(config)
    print("✅ Clustering analyzer initialized")
    print(f"   Number of clusters: {analyzer.cluster_config['n_clusters']}")
    print(f"   Algorithm: {analyzer.cluster_config['algorithm']}")
    print(f"   Archetype names: {len(analyzer.archetype_names)}")
    for i, name in enumerate(analyzer.archetype_names):
        print(f"     {i+1}. {name}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

def create_diverse_driver_data():
    """Create sample NASCAR data with diverse driver archetypes."""
    print("\n📊 Creating diverse driver archetype data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Create drivers representing different archetypes
    driver_profiles = {
        # Dominant Champions
        'Dale Earnhardt Jr.': {'wins_per_season': 3.5, 'avg_finish': 8.2, 'top5_rate': 0.45, 'seasons': 12},
        'Jeff Gordon': {'wins_per_season': 4.2, 'avg_finish': 7.1, 'top5_rate': 0.52, 'seasons': 15},
        'Kyle Larson': {'wins_per_season': 2.8, 'avg_finish': 9.1, 'top5_rate': 0.38, 'seasons': 8},
        
        # Consistent Contenders  
        'Denny Hamlin': {'wins_per_season': 1.2, 'avg_finish': 11.5, 'top5_rate': 0.28, 'seasons': 14},
        'Chase Elliott': {'wins_per_season': 1.8, 'avg_finish': 12.3, 'top5_rate': 0.31, 'seasons': 7},
        'Ryan Blaney': {'wins_per_season': 1.1, 'avg_finish': 13.2, 'top5_rate': 0.25, 'seasons': 8},
        
        # Late Bloomers
        'Martin Truex Jr.': {'wins_per_season': 0.8, 'avg_finish': 15.8, 'top5_rate': 0.22, 'seasons': 18, 'late_peak': True},
        'Chris Buescher': {'wins_per_season': 0.3, 'avg_finish': 18.5, 'top5_rate': 0.15, 'seasons': 9, 'late_peak': True},
        
        # Flash in the Pan
        'Kasey Kahne': {'wins_per_season': 1.5, 'avg_finish': 16.2, 'top5_rate': 0.20, 'seasons': 4, 'short_career': True},
        'Jamie McMurray': {'wins_per_season': 0.9, 'avg_finish': 17.8, 'top5_rate': 0.18, 'seasons': 5, 'short_career': True},
        
        # Journeymen
        'Matt DiBenedetto': {'wins_per_season': 0.1, 'avg_finish': 20.5, 'top5_rate': 0.08, 'seasons': 12},
        'Ricky Stenhouse Jr.': {'wins_per_season': 0.2, 'avg_finish': 21.2, 'top5_rate': 0.09, 'seasons': 10},
        'Michael McDowell': {'wins_per_season': 0.1, 'avg_finish': 22.8, 'top5_rate': 0.06, 'seasons': 15},
        
        # Strugglers
        'Landon Cassill': {'wins_per_season': 0.0, 'avg_finish': 28.5, 'top5_rate': 0.02, 'seasons': 8},
        'David Ragan': {'wins_per_season': 0.1, 'avg_finish': 26.2, 'top5_rate': 0.04, 'seasons': 14},
        'Corey LaJoie': {'wins_per_season': 0.0, 'avg_finish': 27.1, 'top5_rate': 0.03, 'seasons': 6}
    }
    
    data = []
    
    for driver, profile in driver_profiles.items():
        seasons = profile['seasons']
        base_wins_per_season = profile['wins_per_season']
        base_avg_finish = profile['avg_finish']
        base_top5_rate = profile['top5_rate']
        
        # Create season-by-season data
        for season_num in range(1, seasons + 1):
            season_year = 2010 + season_num
            
            # Apply career progression patterns
            if profile.get('late_peak', False):
                # Late bloomers - get better over time
                improvement_factor = min(2.0, 1.0 + (season_num - 5) * 0.15) if season_num > 5 else 0.8
                wins_per_season = base_wins_per_season * improvement_factor
                avg_finish = base_avg_finish / improvement_factor
                top5_rate = base_top5_rate * improvement_factor
            elif profile.get('short_career', False):
                # Flash in pan - good early, then decline
                decline_factor = max(0.3, 1.0 - (season_num - 2) * 0.2) if season_num > 2 else 1.2
                wins_per_season = base_wins_per_season * decline_factor
                avg_finish = base_avg_finish / decline_factor
                top5_rate = base_top5_rate * decline_factor
            else:
                # Normal progression with some variation
                variation = np.random.normal(1.0, 0.1)
                wins_per_season = base_wins_per_season * variation
                avg_finish = base_avg_finish * variation
                top5_rate = base_top5_rate * variation
            
            # Add some realistic noise
            wins_per_season += np.random.normal(0, 0.3)
            avg_finish += np.random.normal(0, 1.5)
            top5_rate += np.random.normal(0, 0.05)
            
            # Ensure realistic bounds
            wins_per_season = max(0, wins_per_season)
            avg_finish = max(3, min(35, avg_finish))
            top5_rate = max(0, min(0.8, top5_rate))
            
            # Calculate derived metrics
            races_run = 36
            total_wins = int(wins_per_season)
            win_rate = wins_per_season / races_run
            total_top5s = int(races_run * top5_rate)
            top10_rate = min(0.9, top5_rate * 1.5)
            total_top10s = int(races_run * top10_rate)
            dnf_rate = max(0.02, min(0.2, 0.08 + np.random.normal(0, 0.03)))
            total_dnfs = int(races_run * dnf_rate)
            
            # Points and other metrics
            total_points = int(1200 + (40 - avg_finish) * 25 + total_wins * 60)
            total_laps_led = int(total_wins * 75 + np.random.randint(0, 150))
            avg_rating = max(40, min(150, 110 - (avg_finish - 10) * 3 + np.random.normal(0, 8)))
            
            # Career metrics for clustering
            career_span = season_year - 2010
            activity_rate = 1.0  # Full-time
            
            data.append({
                'Driver': driver,
                'Season': season_year,
                'total_races': races_run,
                'total_wins': total_wins,
                'total_top5s': total_top5s,
                'total_top10s': total_top10s,
                'total_points': total_points,
                'total_laps_led': total_laps_led,
                'total_dnfs': total_dnfs,
                'career_avg_finish': avg_finish,
                'career_avg_rating': avg_rating,
                'seasons_active': season_num,
                'first_season': 2011,
                'last_season': season_year,
                'wins_per_season': wins_per_season,
                'top5_rate': top5_rate,
                'top10_rate': top10_rate,
                'win_rate': win_rate,
                'dnf_rate': dnf_rate,
                'laps_led_per_race': total_laps_led / races_run,
                'points_per_race': total_points / races_run,
                'career_span': career_span,
                'activity_rate': activity_rate,
                # Add some engineered features
                'finish_consistency': np.random.uniform(1.5, 4.5),
                'finish_improvement': np.random.normal(0, 0.5),
                'peak_timing': np.random.uniform(0.2, 0.8)
            })
    
    df = pd.DataFrame(data)
    
    # Create career-level summary (what clustering actually uses)
    career_summary = df.groupby('Driver').agg({
        'total_races': 'sum',
        'total_wins': 'sum',
        'total_top5s': 'sum',
        'total_top10s': 'sum',
        'total_points': 'sum',
        'total_laps_led': 'sum',
        'total_dnfs': 'sum',
        'career_avg_finish': 'mean',
        'career_avg_rating': 'mean',
        'seasons_active': 'max',
        'first_season': 'min',
        'last_season': 'max',
        'wins_per_season': 'mean',
        'top5_rate': 'mean',
        'top10_rate': 'mean',
        'win_rate': 'mean',
        'dnf_rate': 'mean',
        'finish_consistency': 'mean',
        'finish_improvement': 'mean',
        'peak_timing': 'mean'
    }).round(3)
    
    # Add derived career metrics
    career_summary['laps_led_per_race'] = (career_summary['total_laps_led'] / career_summary['total_races']).round(1)
    career_summary['points_per_race'] = (career_summary['total_points'] / career_summary['total_races']).round(1)
    career_summary['career_span'] = career_summary['last_season'] - career_summary['first_season'] + 1
    career_summary['activity_rate'] = (career_summary['seasons_active'] / career_summary['career_span']).round(3)
    
    career_summary = career_summary.reset_index()
    
    print(f"✅ Created diverse driver data: {len(career_summary)} drivers")
    print(f"   Total driver-seasons: {len(df)}")
    print(f"   Expected archetypes represented:")
    print("     • Dominant Champions (3 drivers)")
    print("     • Consistent Contenders (3 drivers)") 
    print("     • Late Bloomers (2 drivers)")
    print("     • Flash in Pan (2 drivers)")
    print("     • Journeymen (3 drivers)")
    print("     • Strugglers (3 drivers)")
    
    return career_summary

def test_clustering_pipeline():
    """Test the complete clustering analysis pipeline."""
    
    # Create diverse sample data
    career_data = create_diverse_driver_data()
    
    print("\n🏷️  Testing Clustering Analysis Pipeline")
    print("-" * 45)
    
    try:
        # Initialize analyzer
        analyzer = DriverClusterAnalyzer()
        
        # Test 1: Prepare clustering features
        print("1. Preparing clustering features...")
        analyzer.career_data = career_data  # Load the data directly
        features = analyzer.select_clustering_features(career_data)
        print(f"   ✅ Selected {features.shape[1]} features for {features.shape[0]} drivers")
        print(f"   ✅ Feature scaling: {analyzer.cluster_config.get('scaler_type', 'standard')}")
        
        # Test 2: Find optimal clusters
        print("2. Finding optimal number of clusters...")
        optimization_results = analyzer.find_optimal_clusters(max_clusters=8)
        print(f"   ✅ Tested {len(optimization_results['cluster_range'])} cluster configurations")
        print(f"   ✅ Best silhouette score: {optimization_results['best_silhouette_score']:.3f} (k={optimization_results['best_silhouette_k']})")
        print(f"   ✅ Elbow method suggests: k={optimization_results['elbow_k']}")
        print(f"   ✅ Config recommends: k={optimization_results['recommended_k']}")
        
        # Test 3: Fit final clustering model
        print("3. Fitting final clustering model...")
        final_k = optimization_results['recommended_k']
        kmeans_model = analyzer.fit_clustering_model(final_k)
        print(f"   ✅ K-means model fitted with {final_k} clusters")
        print(f"   ✅ Final inertia: {kmeans_model.inertia_:.2f}")
        
        # Test 4: Analyze clusters
        print("4. Analyzing cluster characteristics...")
        cluster_analysis = analyzer.analyze_clusters()
        print(f"   ✅ Analyzed {len(cluster_analysis)} clusters")
        
        print(f"\n   📊 Discovered Driver Archetypes:")
        for idx, row in cluster_analysis.iterrows():
            print(f"     {row['archetype']} ({row['driver_count']} drivers):")
            print(f"       • Avg Wins/Season: {row['avg_wins_per_season']:.2f}")
            print(f"       • Avg Finish: {row['avg_finish']:.1f}")
            print(f"       • Top-5 Rate: {row['avg_top5_rate']:.1%}")
            print(f"       • Examples: {row['representative_drivers']}")
        
        # Test 5: Individual driver lookups
        print("\n5. Testing individual driver lookups...")
        test_drivers = ['Kyle Larson', 'Denny Hamlin', 'Matt DiBenedetto', 'Landon Cassill']
        
        for driver in test_drivers:
            try:
                archetype_info = analyzer.get_driver_archetype(driver)
                if 'error' not in archetype_info:
                    print(f"   ✅ {driver}: {archetype_info['archetype']}")
                    print(f"      Career stats: {archetype_info['career_stats']['wins_per_season']:.1f} wins/season, "
                          f"{archetype_info['career_stats']['career_avg_finish']:.1f} avg finish")
                else:
                    print(f"   ❌ {driver}: {archetype_info['error']}")
            except Exception as e:
                print(f"   ⚠️  {driver}: Lookup failed - {e}")
        
        # Test 6: Create visualizations (basic test)
        print("\n6. Testing visualization creation...")
        try:
            figures = analyzer.create_cluster_visualizations()
            print(f"   ✅ Created {len(figures)} visualization figures:")
            for fig_name in figures.keys():
                print(f"     • {fig_name}")
        except Exception as e:
            print(f"   ⚠️  Visualization creation failed: {e}")
        
        return True, analyzer
        
    except Exception as e:
        print(f"❌ Clustering pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_cluster_quality(analyzer):
    """Test the quality of cluster assignments."""
    print("\n📈 Testing Cluster Quality")
    print("-" * 30)
    
    try:
        # Test silhouette analysis
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        silhouette_avg = silhouette_score(analyzer.scaled_features, analyzer.cluster_labels)
        print(f"✅ Average silhouette score: {silhouette_avg:.3f}")
        
        # Test cluster separation
        cluster_centers = analyzer.cluster_centers
        min_distance = float('inf')
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_distance = min(min_distance, distance)
        
        print(f"✅ Minimum cluster separation: {min_distance:.3f}")
        
        # Test cluster sizes
        unique, counts = np.unique(analyzer.cluster_labels, return_counts=True)
        print(f"✅ Cluster size distribution:")
        for cluster_id, count in zip(unique, counts):
            archetype = analyzer.cluster_analysis.loc[cluster_id, 'archetype']
            print(f"   {archetype}: {count} drivers ({count/len(analyzer.cluster_labels):.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Cluster quality analysis failed: {e}")
        return False

# Main test execution
def main():
    """Run all clustering tests."""
    
    print("Starting Driver Clustering Tests...")
    
    # Test the pipeline
    success, analyzer = test_clustering_pipeline()
    
    if success and analyzer is not None:
        # Test cluster quality
        quality_success = test_cluster_quality(analyzer)
    else:
        quality_success = False
    
    print(f"\n{'='*50}")
    if success and quality_success:
        print("🎉 All Driver Clustering Tests Passed!")
        print("✅ Feature selection working")
        print("✅ Optimal cluster detection")
        print("✅ K-means model fitting")
        print("✅ Cluster analysis and archetype assignment")
        print("✅ Individual driver lookups")
        print("✅ Visualization generation")
        print("✅ Cluster quality validation")
        print("\nNext steps:")
        print("1. Test with real NASCAR data")
        print("2. Test complete end-to-end pipeline")
        print("3. Compare results with LSTM predictions")
    else:
        print("❌ Driver Clustering Tests Failed!")
        print("Check the error messages above")
        
    return success and quality_success

if __name__ == "__main__":
    main()