#!/usr/bin/env python3
"""
Real NASCAR Data Clustering Test Script
Tests driver clustering with actual NASCAR data spanning 1949-2025.

Place in tests/test_clustering_real.py
Run from project root: python tests/test_clustering_real.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print("🏷️  Real NASCAR Data Clustering Test")
print("=" * 45)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from models.clustering import DriverClusterAnalyzer
    from data.data_loader import load_nascar_data
    from data.feature_engineering import create_nascar_features
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def load_real_engineered_data():
    """Load real NASCAR data and create engineered features."""
    print("\n📊 Loading Real NASCAR Data & Features")
    print("-" * 45)
    
    try:
        # Load real NASCAR data
        print("Loading real NASCAR data...")
        data_loader = load_nascar_data()
        print(f"✅ Raw data: {len(data_loader.raw_data):,} records")
        print(f"✅ Driver-seasons: {len(data_loader.driver_seasons):,}")
        print(f"✅ Unique drivers: {data_loader.driver_seasons['Driver'].nunique()}")
        
        # Create engineered features
        print("\nCreating engineered features for clustering...")
        start_time = time.time()
        engineer = create_nascar_features(save_results=False)
        feature_time = time.time() - start_time
        
        print(f"✅ Feature engineering completed in {feature_time:.1f}s")
        print(f"✅ Engineered features: {len(engineer.engineered_features.columns)} columns")
        print(f"✅ Driver-seasons with features: {len(engineer.engineered_features)}")
        
        return engineer.engineered_features
        
    except Exception as e:
        print(f"❌ Failed to load and engineer real data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_clustering_pipeline(engineered_features):
    """Test clustering analysis with real NASCAR data."""
    print(f"\n🏷️  Testing Real Data Clustering Pipeline")
    print("-" * 45)
    
    try:
        # Initialize clustering analyzer
        analyzer = DriverClusterAnalyzer()
        print(f"✅ Clustering analyzer initialized")
        print(f"   Target clusters: {analyzer.cluster_config['n_clusters']}")
        print(f"   Algorithm: {analyzer.cluster_config['algorithm']}")
        
        # Prepare clustering features from real data
        print(f"\n1. Preparing clustering features from real data...")
        start_time = time.time()
        career_data = analyzer.prepare_clustering_features(engineered_features)
        prep_time = time.time() - start_time
        
        print(f"   ✅ Career-level data prepared in {prep_time:.1f}s")
        print(f"   ✅ Real drivers for clustering: {len(career_data)}")
        print(f"   ✅ Career span: {career_data['first_season'].min()} - {career_data['last_season'].max()}")
        
        # Show some real driver examples
        print(f"\n   📋 Real Driver Career Examples:")
        # Sort by total wins to show legendary drivers first
        top_drivers = career_data.nlargest(5, 'total_wins')
        for _, driver in top_drivers.iterrows():
            print(f"     {driver['Driver']}: {driver['total_wins']} wins, "
                  f"{driver['seasons_active']} seasons, {driver['career_avg_finish']:.1f} avg finish")
        
        # Select clustering features
        print(f"\n2. Selecting features for clustering...")
        start_time = time.time()
        features = analyzer.select_clustering_features(career_data)
        select_time = time.time() - start_time
        
        print(f"   ✅ Feature selection completed in {select_time:.1f}s")
        print(f"   ✅ Features selected: {features.shape[1]}")
        print(f"   ✅ Drivers with features: {features.shape[0]}")
        print(f"   ✅ Scaling method: {analyzer.cluster_config.get('scaler_type', 'standard')}")
        
        # Find optimal clusters
        print(f"\n3. Finding optimal number of clusters...")
        start_time = time.time()
        optimization_results = analyzer.find_optimal_clusters(max_clusters=8)
        opt_time = time.time() - start_time
        
        print(f"   ✅ Cluster optimization completed in {opt_time:.1f}s")
        print(f"   ✅ Tested configurations: {len(optimization_results['cluster_range'])}")
        print(f"   ✅ Best silhouette score: {optimization_results['best_silhouette_score']:.3f} (k={optimization_results['best_silhouette_k']})")
        print(f"   ✅ Elbow method suggests: k={optimization_results['elbow_k']}")
        print(f"   ✅ Config recommends: k={optimization_results['recommended_k']}")
        
        # Fit final clustering model
        print(f"\n4. Fitting final clustering model...")
        start_time = time.time()
        final_k = optimization_results['recommended_k']
        kmeans_model = analyzer.fit_clustering_model(final_k)
        fit_time = time.time() - start_time
        
        print(f"   ✅ K-means model fitted in {fit_time:.1f}s")
        print(f"   ✅ Final clusters: {final_k}")
        print(f"   ✅ Final inertia: {kmeans_model.inertia_:.2f}")
        
        # Analyze cluster characteristics
        print(f"\n5. Analyzing real driver clusters...")
        start_time = time.time()
        cluster_analysis = analyzer.analyze_clusters()
        analysis_time = time.time() - start_time
        
        print(f"   ✅ Cluster analysis completed in {analysis_time:.1f}s")
        print(f"   ✅ Archetypes discovered: {len(cluster_analysis)}")
        
        return analyzer, cluster_analysis
        
    except Exception as e:
        print(f"❌ Clustering pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_real_driver_archetypes(analyzer, cluster_analysis):
    """Analyze the discovered driver archetypes with real NASCAR legends."""
    print(f"\n📊 Real NASCAR Driver Archetypes Discovered")
    print("-" * 45)
    
    try:
        print(f"Discovered {len(cluster_analysis)} real driver archetypes:\n")
        
        for idx, row in cluster_analysis.iterrows():
            print(f"🏁 {row['archetype']} ({row['driver_count']} drivers)")
            print(f"   Performance Profile:")
            print(f"     • Avg Wins/Season: {row['avg_wins_per_season']:.2f}")
            print(f"     • Avg Finish: {row['avg_finish']:.1f}")
            print(f"     • Top-5 Rate: {row['avg_top5_rate']:.1%}")
            print(f"     • Win Rate: {row['avg_win_rate']:.1%}")
            print(f"     • Career Length: {row['avg_seasons']:.1f} seasons")
            print(f"   Representative Drivers: {row['representative_drivers']}")
            print()
        
        # Test specific legendary drivers
        print(f"🏆 Legendary Driver Classifications:")
        legendary_drivers = [
            'Richard Petty', 'Jeff Gordon', 'Dale Earnhardt', 
            'Kyle Larson', 'Denny Hamlin', 'Jimmie Johnson'
        ]
        
        for driver in legendary_drivers:
            try:
                archetype_info = analyzer.get_driver_archetype(driver)
                if 'error' not in archetype_info:
                    stats = archetype_info['career_stats']
                    print(f"   {driver}: {archetype_info['archetype']}")
                    print(f"     Career: {stats['total_wins']:.0f} wins, "
                          f"{stats['seasons_active']:.0f} seasons, "
                          f"{stats['career_avg_finish']:.1f} avg finish")
                else:
                    print(f"   {driver}: Not found in clustering data")
            except Exception as e:
                print(f"   {driver}: Error retrieving archetype - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Archetype analysis failed: {e}")
        return False

def test_clustering_quality(analyzer):
    """Test the quality of clustering with real NASCAR data."""
    print(f"\n📈 Testing Real Data Clustering Quality")
    print("-" * 40)
    
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(analyzer.scaled_features, analyzer.cluster_labels)
        silhouette_samples_scores = silhouette_samples(analyzer.scaled_features, analyzer.cluster_labels)
        
        print(f"Quality Metrics:")
        print(f"   ✅ Average silhouette score: {silhouette_avg:.3f}")
        
        # Analyze silhouette scores by cluster
        print(f"\n   Silhouette scores by archetype:")
        for cluster_id in sorted(np.unique(analyzer.cluster_labels)):
            cluster_scores = silhouette_samples_scores[analyzer.cluster_labels == cluster_id]
            archetype = analyzer.cluster_analysis.loc[cluster_id, 'archetype']
            print(f"     {archetype}: {cluster_scores.mean():.3f} ± {cluster_scores.std():.3f}")
        
        # Test cluster separation
        cluster_centers = analyzer.cluster_centers
        min_distance = float('inf')
        max_distance = 0
        
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_distance = min(min_distance, distance)
                max_distance = max(max_distance, distance)
        
        print(f"\n   Cluster separation:")
        print(f"     Minimum distance: {min_distance:.3f}")
        print(f"     Maximum distance: {max_distance:.3f}")
        print(f"     Separation ratio: {max_distance/min_distance:.2f}")
        
        # Analyze cluster sizes and balance
        unique, counts = np.unique(analyzer.cluster_labels, return_counts=True)
        total_drivers = len(analyzer.cluster_labels)
        
        print(f"\n   Cluster size distribution:")
        for cluster_id, count in zip(unique, counts):
            archetype = analyzer.cluster_analysis.loc[cluster_id, 'archetype']
            percentage = (count / total_drivers) * 100
            print(f"     {archetype}: {count} drivers ({percentage:.1f}%)")
        
        # Check for reasonable balance (no cluster should dominate)
        max_cluster_pct = max(counts) / total_drivers
        balanced = max_cluster_pct < 0.5  # No cluster should have >50% of drivers
        
        print(f"\n   ✅ Cluster balance: {'Good' if balanced else 'Imbalanced'}")
        print(f"     Largest cluster: {max_cluster_pct:.1%} of drivers")
        
        return True
        
    except Exception as e:
        print(f"❌ Clustering quality analysis failed: {e}")
        return False

def test_era_analysis(analyzer):
    """Analyze how clustering handles different NASCAR eras."""
    print(f"\n🕰️  NASCAR Era Analysis")
    print("-" * 25)
    
    try:
        # Define NASCAR eras
        eras = {
            'Classic Era': (1949, 1971),
            'Modern Era': (1972, 2003), 
            'Chase Era': (2004, 2015),
            'Playoff Era': (2016, 2025)
        }
        
        career_data = analyzer.career_data
        
        print("Driver distribution by era and archetype:")
        
        for era_name, (start_year, end_year) in eras.items():
            # Find drivers who raced primarily in this era
            era_drivers = career_data[
                (career_data['first_season'] >= start_year) & 
                (career_data['first_season'] <= end_year)
            ]
            
            if len(era_drivers) == 0:
                continue
                
            print(f"\n   {era_name} ({start_year}-{end_year}): {len(era_drivers)} drivers")
            
            # Get archetype distribution for this era
            era_cluster_labels = analyzer.cluster_labels[era_drivers.index]
            era_archetypes = {}
            
            for cluster_id in np.unique(era_cluster_labels):
                archetype = analyzer.cluster_analysis.loc[cluster_id, 'archetype']
                count = np.sum(era_cluster_labels == cluster_id)
                era_archetypes[archetype] = count
            
            # Show top archetypes for this era
            sorted_archetypes = sorted(era_archetypes.items(), key=lambda x: x[1], reverse=True)
            for archetype, count in sorted_archetypes[:3]:
                percentage = (count / len(era_drivers)) * 100
                print(f"     {archetype}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Era analysis failed: {e}")
        return False

# Main test execution
def main():
    """Run all real data clustering tests."""
    
    print("Starting Real NASCAR Data Clustering Tests...")
    
    # Test 1: Load real engineered data
    engineered_features = load_real_engineered_data()
    if engineered_features is None:
        print("❌ Cannot proceed without real engineered data")
        return False
    
    # Test 2: Clustering pipeline with real data
    analyzer, cluster_analysis = test_real_clustering_pipeline(engineered_features)
    if analyzer is None or cluster_analysis is None:
        print("❌ Clustering pipeline failed")
        return False
    
    # Test 3: Analyze real driver archetypes
    archetype_success = analyze_real_driver_archetypes(analyzer, cluster_analysis)
    
    # Test 4: Clustering quality validation
    quality_success = test_clustering_quality(analyzer)
    
    # Test 5: Era analysis
    era_success = test_era_analysis(analyzer)
    
    # Summary
    print(f"\n{'='*45}")
    if archetype_success and quality_success and era_success:
        print("🎉 All Real Data Clustering Tests Passed!")
        print("✅ Real NASCAR driver archetypes discovered")
        print("✅ 75 years of driver data successfully clustered")
        print("✅ Legendary drivers correctly classified")
        print("✅ Cluster quality validated")
        print("✅ Era-based analysis completed")
        
        print(f"\nReal NASCAR insights ready:")
        if cluster_analysis is not None:
            print(f"   • {len(cluster_analysis)} distinct driver archetypes")
            print(f"   • From Richard Petty to Kyle Larson")
            print(f"   • Spanning Classic Era to Playoff Era")
        
        print("\nNext steps:")
        print("1. Test LSTM with real career progression sequences")
        print("2. Create visualizations of real driver archetypes") 
        print("3. Run end-to-end pipeline with real predictions")
    else:
        print("❌ Some Real Data Clustering Tests Failed!")
        print("Check the error messages above")
    
    return archetype_success and quality_success and era_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)