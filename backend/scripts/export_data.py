#!/usr/bin/env python3
"""
Data Export Script for NASCAR FastAPI Backend

This script exports data from the existing NASCAR analysis models
into JSON files that the FastAPI backend can serve efficiently.

Run this script whenever update the API data needs to be updated.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

# Add existing src directory to path
# Script is in backend/scripts/, so go up 2 levels to get to project root
project_root = Path(__file__).parent.parent.parent  # Go up 2 levels from backend/scripts/
src_dir = project_root / 'src'

# Add both project root and src to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Print debug info to see what's happening
print(f"Project root: {project_root}")
print(f"Src directory: {src_dir}")
print(f"Src exists: {src_dir.exists()}")

# Import existing modules
from data.data_loader import load_nascar_data
from data.feature_engineering import create_nascar_features
from models.clustering import run_clustering_analysis
from models.lstm_model import NASCARLSTMPredictor

print("✅ All imports successful")


def create_driver_slug(driver_name: str) -> str:
    """Convert driver name to URL-safe slug."""
    # Convert to lowercase, replace spaces with hyphens, remove special chars
    slug = re.sub(r'[^\w\s-]', '', driver_name.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def export_drivers_data() -> Dict[str, Any]:
    """Export core driver data for the API."""
    print("🏁 Loading NASCAR data...")
    data_loader = load_nascar_data()
    
    # Get driver career summaries
    driver_seasons = data_loader.driver_seasons
    
    # Create driver profiles
    drivers = []
    driver_careers = driver_seasons.groupby('Driver').agg({
        'Season': ['min', 'max', 'count'],
        'wins': 'sum',
        'avg_finish': 'mean',
        'top_5_rate': 'mean',
        'top_10_rate': 'mean',
        'win_rate': 'mean',
        'races_run': 'sum'
    }).round(3)
    
    # Flatten column names
    driver_careers.columns = [
        'first_season', 'last_season', 'total_seasons',
        'total_wins', 'career_avg_finish', 'career_top5_rate',
        'career_top10_rate', 'career_win_rate', 'total_races'
    ]
    
    for driver_name, stats in driver_careers.iterrows():
        # Skip drivers with very short careers for API efficiency
        # if stats['total_seasons'] < 3:
        #     continue
            
        driver_slug = create_driver_slug(driver_name)
        
        # Determine if driver is active
        current_year = datetime.now().year
        is_active = stats['last_season'] >= (current_year - 1)
        
        # Get recent performance (last season)
        recent_seasons = driver_seasons[
            (driver_seasons['Driver'] == driver_name) & 
            (driver_seasons['Season'] >= stats['last_season'])
        ]
        
        recent_avg_finish = recent_seasons['avg_finish'].mean() if not recent_seasons.empty else stats['career_avg_finish']
        recent_wins = recent_seasons['wins'].sum() if not recent_seasons.empty else 0
        
        driver_data = {
            'id': driver_slug,
            'name': driver_name,
            'is_active': bool(is_active),
            'career_span': f"{int(stats['first_season'])}-{int(stats['last_season'])}",
            'first_season': int(stats['first_season']),
            'last_season': int(stats['last_season']),
            'total_seasons': int(stats['total_seasons']),
            'career_stats': {
                'total_wins': int(stats['total_wins']),
                'total_races': int(stats['total_races']),
                'career_avg_finish': float(stats['career_avg_finish']),
                'career_top5_rate': float(stats['career_top5_rate']),
                'career_top10_rate': float(stats['career_top10_rate']),
                'career_win_rate': float(stats['career_win_rate'])
            },
            'recent_performance': {
                'recent_avg_finish': float(recent_avg_finish),
                'recent_wins': int(recent_wins),
                'seasons_analyzed': len(recent_seasons)
            }
        }
        
        drivers.append(driver_data)
    
    # Sort by total wins for better API responses
    drivers.sort(key=lambda x: x['career_stats']['total_wins'], reverse=True)
    
    return {
        'drivers': drivers,
        'total_drivers': len(drivers),
        'export_timestamp': datetime.now().isoformat(),
        'data_source': 'nascaR.data package'
    }


def export_archetypes_data() -> Dict[str, Any]:
    """Export driver archetype clustering data."""
    print("🏷️ Running clustering analysis...")
    
    try:
        # Run your existing clustering analysis
        clustering_analyzer = run_clustering_analysis(save_results=False)
        cluster_analysis = clustering_analyzer.cluster_analysis
        
        # Get driver assignments
        career_data = clustering_analyzer.career_data.copy()
        career_data['cluster'] = clustering_analyzer.cluster_labels
        
        # Map clusters to archetypes
        cluster_archetype_map = dict(zip(
            cluster_analysis.index,
            cluster_analysis['archetype']
        ))
        career_data['archetype'] = career_data['cluster'].map(cluster_archetype_map)
        
        # Create archetype summaries with reassigned names
        cluster_analysis = reassign_archetypes(cluster_analysis)
        
        archetypes = []
        for idx, row in cluster_analysis.iterrows():
            # Get drivers in this archetype (this returns a dataframe)
            archetype_drivers_df = career_data[career_data['cluster'] == idx]
            
            # Get top drivers by total wins
            top_drivers = archetype_drivers_df.nlargest(5, 'total_wins')['Driver'].tolist()
            
            # Convert dataframe to list of driver objects
            archetype_drivers_list = []
            for _, driver_row in archetype_drivers_df.iterrows():
                # Determine active status
                current_year = datetime.now().year
                driver_last_season = driver_row.get('last_season', 0)
                driver_is_active = driver_last_season >= (current_year - 1)

                driver_obj = {
                    'id': create_driver_slug(driver_row['Driver']),
                    'name': driver_row['Driver'],
                    'total_wins': int(driver_row['total_wins']),
                    'career_avg_finish': float(driver_row['career_avg_finish']),
                    'total_seasons': int(driver_row['seasons_active']),
                    'total_races': int(driver_row['total_races']),
                    'career_top5_rate': float(driver_row['top5_rate']),
                    'career_win_rate': float(driver_row['win_rate']),
                    'is_active': bool(driver_is_active),
                    'cluster_id': int(driver_row['cluster']),
                    'archetype': driver_row['archetype']
                }
                archetype_drivers_list.append(driver_obj)

            # Sort by total wins (highest first)
            archetype_drivers_list.sort(key=lambda x: x['total_wins'], reverse=True)

            archetype_data = {
                'id': create_driver_slug(row['archetype']),
                'name': row['archetype'],
                'color': row['color'],
                'driver_count': int(row['driver_count']),
                'characteristics': {
                    'avg_wins_per_season': float(row['avg_wins_per_season']),
                    'avg_finish': float(row['avg_finish']),
                    'avg_top5_rate': float(row['avg_top5_rate']),
                    'avg_win_rate': float(row['avg_win_rate']),
                    'avg_seasons': float(row['avg_seasons'])
                },
                'representative_drivers': row['representative_drivers'],
                'top_drivers': top_drivers,
                'archetype_drivers': archetype_drivers_list,
                'description': get_archetype_description(row['archetype'])
            }
            
            archetypes.append(archetype_data)
        
        # Create driver-to-archetype mapping
        driver_archetypes = {}
        for _, driver_row in career_data.iterrows():
            driver_slug = create_driver_slug(driver_row['Driver'])
            archetype_slug = create_driver_slug(driver_row['archetype'])
            
            driver_archetypes[driver_slug] = {
                'archetype_id': archetype_slug,
                'archetype_name': driver_row['archetype'],
                'cluster_id': int(driver_row['cluster'])
            }
        
        return {
            'archetypes': archetypes,
            'driver_archetypes': driver_archetypes,
            'clustering_info': {
                'total_drivers_clustered': len(career_data),
                'silhouette_score': getattr(clustering_analyzer, 'silhouette_scores', {}).get(len(archetypes), 'N/A'),
                'algorithm': 'K-Means',
                'features_used': len(clustering_analyzer.features.columns) if clustering_analyzer.features is not None else 'N/A'
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in clustering analysis: {e}")
        return {
            'archetypes': [],
            'driver_archetypes': {},
            'error': str(e),
            'export_timestamp': datetime.now().isoformat()
        }


def reassign_archetypes(cluster_analysis: pd.DataFrame) -> pd.DataFrame:
    """Reassign archetype names to ensure 6 unique, meaningful categories."""
    
    # Sort by performance metrics to assign better names
    cluster_analysis_sorted = cluster_analysis.sort_values(['avg_wins_per_season', 'avg_top5_rate'], ascending=False)
    
    # Define 6 distinct archetype names (one for each cluster)
    archetype_names = [
        "Elite Champions",        # Best overall performers
        "Race Winners",          # Regular win contenders  
        "Consistent Contenders", # Top-10 regulars
        "Veteran Journeymen",    # Long careers, moderate success
        "Mid-Pack Drivers",      # Middle of the field
        "Backfield Runners"      # Back of the pack
    ]
    
    # Assign names in performance order (best to worst)
    new_archetypes = []
    
    for idx, (cluster_idx, row) in enumerate(cluster_analysis_sorted.iterrows()):
        # Simply assign the names in order of performance
        if idx < len(archetype_names):
            archetype = archetype_names[idx]
        else:
            # Fallback if somehow we have more than 6 clusters
            archetype = f"Group {idx + 1}"
        
        new_archetypes.append(archetype)
        
        # Debug info
        wins_per_season = row['avg_wins_per_season']
        avg_finish = row['avg_finish']
        top5_rate = row['avg_top5_rate']
        print(f"Cluster {cluster_idx}: {archetype} - {wins_per_season:.2f} wins/season, {avg_finish:.1f} avg finish, {top5_rate:.1%} top-5")
    
    # Update the original dataframe with new names
    name_mapping = dict(zip(cluster_analysis_sorted.index, new_archetypes))
    cluster_analysis_copy = cluster_analysis.copy()
    cluster_analysis_copy['archetype'] = cluster_analysis_copy.index.map(name_mapping)
    
    return cluster_analysis_copy


def get_archetype_description(archetype_name: str) -> str:
    """Get human-readable description for archetype."""
    descriptions = {
        'Elite Champions': 'The absolute best drivers in NASCAR history. Multiple wins per season and consistent championship contenders.',
        'Race Winners': 'Drivers who regularly win races and compete for championships. Solid performers with proven winning ability.',
        'Consistent Contenders': 'Reliable drivers who regularly finish in the top-10 with occasional wins. Solid performers you can count on.',
        'Veteran Journeymen': 'Long-career drivers with moderate success. The experienced backbone of NASCAR competition.',
        'Mid-Pack Drivers': 'Drivers who typically finish in the middle of the field. Solid contributors to the sport.',
        'Backfield Runners': 'Drivers who consistently ran in the back of the field throughout their careers.',
        'Late Bloomers': 'Drivers who showed significant improvement over their careers, often reaching peak performance later.',
        'Flash in the Pan': 'Drivers with short but impressive peak performance periods before declining or retiring.',
        'Strugglers': 'Drivers who consistently ran in the back of the field throughout their careers.'
    }
    return descriptions.get(archetype_name, 'A distinct group of NASCAR drivers with similar career patterns.')


def export_predictions_data() -> Dict[str, Any]:
    """Export LSTM prediction data (simplified for now)."""
    print("🧠 Preparing prediction data...")
    
    # For now, create a simple structure
    # Expand this once the LSTM model is properly integrated
    return {
        'predictions_available': False,
        'message': 'LSTM predictions will be integrated in next phase',
        'export_timestamp': datetime.now().isoformat()
    }


def main():
    """Main export function."""
    print("🚀 Starting NASCAR data export for FastAPI...")
    
    # Create backend data directory
    # Script is in backend/scripts/, so parent is backend/
    backend_dir = Path(__file__).parent.parent  # This gets to backend/
    data_dir = backend_dir / 'app' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🗂️  Backend directory: {backend_dir}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Data directory exists: {data_dir.exists()}")
    
    # Export each data type
    try:
        # 1. Export drivers data
        print("\n📊 Exporting drivers data...")
        drivers_data = export_drivers_data()
        with open(data_dir / 'drivers.json', 'w') as f:
            json.dump(drivers_data, f, indent=2, default=str)
        print(f"✅ Exported {drivers_data['total_drivers']} drivers")
        
        # 2. Export archetypes data
        print("\n🏷️ Exporting archetypes data...")
        archetypes_data = export_archetypes_data()
        with open(data_dir / 'archetypes.json', 'w') as f:
            json.dump(archetypes_data, f, indent=2, default=str)
        print(f"✅ Exported {len(archetypes_data['archetypes'])} archetypes")
        
        # 3. Export predictions data
        print("\n🔮 Exporting predictions data...")
        predictions_data = export_predictions_data()
        with open(data_dir / 'predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2, default=str)
        print("✅ Predictions data structure created")
        
        # Create summary
        summary = {
            'export_completed': True,
            'export_timestamp': datetime.now().isoformat(),
            'files_created': [
                'drivers.json',
                'archetypes.json', 
                'predictions.json'
            ],
            'stats': {
                'total_drivers': drivers_data['total_drivers'],
                'total_archetypes': len(archetypes_data['archetypes'])
            }
        }
        
        with open(data_dir / 'export_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 Data exported to: {data_dir}")
        print(f"📈 {summary['stats']['total_drivers']} drivers, {summary['stats']['total_archetypes']} archetypes")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()