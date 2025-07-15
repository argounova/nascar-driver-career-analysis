"""
NASCAR Driver Clustering Module

This module performs K-means clustering analysis to identify distinct driver archetypes
based on career performance metrics. Implements the clustering configuration from
config.yaml to discover patterns in driver careers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Import our modules
from config import get_config, get_data_paths
from data.data_loader import NASCARDataLoader


class DriverClusterAnalyzer:
    """
    Analyzes NASCAR driver career patterns using K-means clustering.
    
    Identifies distinct driver archetypes based on performance metrics:
    - Dominant Champions
    - Consistent Contenders  
    - Late Bloomers
    - Flash in the Pan
    - Journeymen
    - Strugglers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the clustering analyzer.
        
        Args:
            config (Optional[Dict]): Configuration dictionary
        """
        self.config = config if config is not None else get_config()
        self.cluster_config = self.config['models']['clustering']
        self.paths = get_data_paths(self.config)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scaler = None
        self.kmeans = None
        self.pca = None
        
        # Data storage
        self.career_data = None
        self.features = None
        self.scaled_features = None
        self.cluster_labels = None
        self.cluster_centers = None
        
        # Analysis results
        self.silhouette_scores = {}
        self.elbow_scores = {}
        self.archetype_names = self.config['archetypes']['names']
    
    def prepare_clustering_features(self, driver_seasons: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare career-level features for clustering analysis.
        
        Args:
            driver_seasons (pd.DataFrame): Driver season summaries
            
        Returns:
            pd.DataFrame: Career-level features for each driver
        """
        self.logger.info("Preparing clustering features from driver seasons")
        
        # Create career-level aggregations
        career_metrics = driver_seasons.groupby('Driver').agg({
            'races_run': 'sum',
            'wins': 'sum', 
            'top_5s': 'sum',
            'top_10s': 'sum',
            'total_points': 'sum',
            'laps_led': 'sum',
            'dnfs': 'sum',
            'avg_finish': 'mean',
            'avg_rating': 'mean',
            'Season': ['count', 'min', 'max']  # Seasons active, first, last
        }).round(3)
        
        # Flatten column names
        career_metrics.columns = [
            'total_races', 'total_wins', 'total_top5s', 'total_top10s',
            'total_points', 'total_laps_led', 'total_dnfs', 'career_avg_finish',
            'career_avg_rating', 'seasons_active', 'first_season', 'last_season'
        ]
        
        # Calculate derived features
        career_metrics['wins_per_season'] = (career_metrics['total_wins'] / career_metrics['seasons_active']).round(3)
        career_metrics['top5_rate'] = (career_metrics['total_top5s'] / career_metrics['total_races']).round(3)
        career_metrics['top10_rate'] = (career_metrics['total_top10s'] / career_metrics['total_races']).round(3)
        career_metrics['win_rate'] = (career_metrics['total_wins'] / career_metrics['total_races']).round(3)
        career_metrics['dnf_rate'] = (career_metrics['total_dnfs'] / career_metrics['total_races']).round(3)
        career_metrics['laps_led_per_race'] = (career_metrics['total_laps_led'] / career_metrics['total_races']).round(1)
        career_metrics['points_per_race'] = (career_metrics['total_points'] / career_metrics['total_races']).round(1)
        
        # Career span and consistency metrics
        career_metrics['career_span'] = career_metrics['last_season'] - career_metrics['first_season'] + 1
        career_metrics['activity_rate'] = (career_metrics['seasons_active'] / career_metrics['career_span']).round(3)
        
        # Calculate consistency (need season-by-season data)
        consistency_metrics = driver_seasons.groupby('Driver').agg({
            'avg_finish': 'std',
            'wins': 'std',
            'top_5_rate': 'std'
        }).round(3)
        
        consistency_metrics.columns = ['finish_consistency', 'wins_consistency', 'top5_consistency']
        career_metrics = career_metrics.join(consistency_metrics)
        
        # Calculate improvement trends
        improvement_metrics = self._calculate_improvement_trends(driver_seasons)
        career_metrics = career_metrics.join(improvement_metrics)
        
        # Filter drivers with minimum career requirements
        min_seasons = self.config['data']['filtering']['min_seasons_for_career']
        career_metrics = career_metrics[career_metrics['seasons_active'] >= min_seasons]
        
        # Reset index to make Driver a column
        career_metrics = career_metrics.reset_index()
        
        self.career_data = career_metrics
        self.logger.info(f"Prepared features for {len(career_metrics)} drivers")
        
        return career_metrics
    
    def _calculate_improvement_trends(self, driver_seasons: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate career improvement trends for each driver.
        
        Args:
            driver_seasons (pd.DataFrame): Season-by-season data
            
        Returns:
            pd.DataFrame: Improvement metrics by driver
        """
        improvement_data = []
        
        for driver in driver_seasons['Driver'].unique():
            driver_data = driver_seasons[driver_seasons['Driver'] == driver].sort_values('Season')
            
            if len(driver_data) >= 3:  # Need at least 3 seasons
                # Calculate linear trends in key metrics
                seasons = np.arange(len(driver_data))
                
                # Average finish improvement (negative = better)
                finish_trend = np.polyfit(seasons, driver_data['avg_finish'], 1)[0]
                
                # Win rate improvement
                if driver_data['win_rate'].sum() > 0:
                    win_trend = np.polyfit(seasons, driver_data['win_rate'], 1)[0]
                else:
                    win_trend = 0
                
                # Top-5 rate improvement  
                top5_trend = np.polyfit(seasons, driver_data['top_5_rate'], 1)[0]
                
                # Peak performance identification
                peak_season_idx = driver_data['avg_finish'].idxmin()
                peak_season_num = driver_data.loc[peak_season_idx, 'Season']
                seasons_to_peak = len(driver_data[driver_data['Season'] <= peak_season_num])
                peak_timing = seasons_to_peak / len(driver_data)  # 0-1 scale
                
                improvement_data.append({
                    'Driver': driver,
                    'finish_improvement': -finish_trend,  # Flip sign so positive = improvement
                    'win_rate_improvement': win_trend,
                    'top5_improvement': top5_trend,
                    'peak_timing': peak_timing,  # 0 = early career, 1 = late career
                    'seasons_to_peak': seasons_to_peak
                })
        
        improvement_df = pd.DataFrame(improvement_data).set_index('Driver')
        return improvement_df
    
    def select_clustering_features(self, career_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Select and scale features for clustering.
        
        Args:
            career_data (Optional[pd.DataFrame]): Career data. Uses self.career_data if None.
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        if career_data is None:
            career_data = self.career_data
        
        # Define core clustering features based on config
        feature_columns = [
            'wins_per_season',
            'career_avg_finish', 
            'top5_rate',
            'top10_rate',
            'win_rate',
            'dnf_rate',
            'finish_consistency',
            'finish_improvement',
            'peak_timing',
            'seasons_active',
            'career_avg_rating'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in career_data.columns]
        if len(available_features) < len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")
        
        # Extract features
        features = career_data[available_features].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Scale features
        scaler_type = self.cluster_config.get('scaler_type', 'standard')
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        scaled_features = self.scaler.fit_transform(features)
        
        self.features = features
        self.scaled_features = scaled_features
        
        self.logger.info(f"Selected {len(available_features)} features for clustering")
        return scaled_features
    
    def find_optimal_clusters(self, max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            Dict: Analysis results with optimal cluster recommendation
        """
        self.logger.info("Finding optimal number of clusters...")
        
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            # Fit K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.cluster_config['random_state'],
                init=self.cluster_config['init'],
                n_init=self.cluster_config['n_init'],
                max_iter=self.cluster_config['max_iter']
            )
            
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.scaled_features, cluster_labels)
            silhouette_scores.append(sil_score)
            
            self.logger.info(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil_score:.3f}")
        
        # Store results
        self.elbow_scores = dict(zip(cluster_range, inertias))
        self.silhouette_scores = dict(zip(cluster_range, silhouette_scores))
        
        # Find optimal clusters
        # Method 1: Best silhouette score
        best_silhouette_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        
        # Method 2: Elbow method (find the "knee")
        elbow_k = self._find_elbow_point(list(cluster_range), inertias)
        
        results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores_list': silhouette_scores,
            'silhouette_scores_dict': self.silhouette_scores,
            'best_silhouette_k': best_silhouette_k,
            'best_silhouette_score': self.silhouette_scores[best_silhouette_k],
            'elbow_k': elbow_k,
            'recommended_k': self.cluster_config['n_clusters']  # From config
        }
        
        self.logger.info(f"Optimal clusters - Silhouette: {best_silhouette_k}, Elbow: {elbow_k}, Config: {results['recommended_k']}")
        
        return results
    
    def _find_elbow_point(self, x_values: List, y_values: List) -> int:
        """
        Find the elbow point using the kneedle algorithm approximation.
        
        Args:
            x_values (List): X coordinates (number of clusters)
            y_values (List): Y coordinates (inertias)
            
        Returns:
            int: Optimal number of clusters at elbow point
        """
        # Simple elbow detection: find point with maximum distance from line
        if len(x_values) < 3:
            return x_values[0]
        
        # Normalize values
        x_norm = np.array(x_values)
        y_norm = np.array(y_values)
        
        # Line from first to last point
        first_point = np.array([x_norm[0], y_norm[0]])
        last_point = np.array([x_norm[-1], y_norm[-1]])
        
        max_distance = 0
        elbow_idx = 0
        
        for i in range(1, len(x_norm) - 1):
            point = np.array([x_norm[i], y_norm[i]])
            
            # Calculate distance from point to line
            distance = np.abs(np.cross(last_point - first_point, first_point - point)) / np.linalg.norm(last_point - first_point)
            
            if distance > max_distance:
                max_distance = distance
                elbow_idx = i
        
        return x_values[elbow_idx]
    
    def fit_clustering_model(self, n_clusters: Optional[int] = None) -> KMeans:
        """
        Fit the final clustering model.
        
        Args:
            n_clusters (Optional[int]): Number of clusters. Uses config default if None.
            
        Returns:
            KMeans: Fitted clustering model
        """
        if n_clusters is None:
            n_clusters = self.cluster_config['n_clusters']
        
        self.logger.info(f"Fitting K-means with {n_clusters} clusters")
        
        # Initialize K-means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.cluster_config['random_state'],
            init=self.cluster_config['init'],
            n_init=self.cluster_config['n_init'],
            max_iter=self.cluster_config['max_iter']
        )
        
        # Fit model
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate final metrics
        final_silhouette = silhouette_score(self.scaled_features, self.cluster_labels)
        
        self.logger.info(f"Final model - Inertia: {self.kmeans.inertia_:.2f}, Silhouette: {final_silhouette:.3f}")
        
        return self.kmeans
    
    def analyze_clusters(self) -> pd.DataFrame:
        """
        Analyze cluster characteristics and assign archetype names.
        
        Returns:
            pd.DataFrame: Cluster analysis with driver assignments
        """
        if self.kmeans is None:
            raise ValueError("Must fit clustering model first")
        
        # Add cluster labels to career data
        cluster_data = self.career_data.copy()
        cluster_data['cluster'] = self.cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = cluster_data.groupby('cluster').agg({
            'wins_per_season': 'mean',
            'career_avg_finish': 'mean',
            'top5_rate': 'mean',
            'win_rate': 'mean',
            'seasons_active': 'mean',
            'finish_improvement': 'mean',
            'peak_timing': 'mean',
            'Driver': 'count'
        }).round(3)
        
        cluster_stats.columns = [
            'avg_wins_per_season', 'avg_finish', 'avg_top5_rate', 
            'avg_win_rate', 'avg_seasons', 'avg_improvement',
            'avg_peak_timing', 'driver_count'
        ]
        
        # Assign archetype names based on characteristics
        archetype_assignments = self._assign_archetypes(cluster_stats)
        cluster_stats['archetype'] = archetype_assignments
        
        # Add color mapping
        colors = self.config['visualization']['colors']['archetypes']
        cluster_stats['color'] = [colors[i % len(colors)] for i in range(len(cluster_stats))]
        
        # Get representative drivers for each cluster
        representative_drivers = self._get_representative_drivers(cluster_data)
        cluster_stats['representative_drivers'] = representative_drivers
        
        self.cluster_analysis = cluster_stats
        return cluster_stats
    
    def _assign_archetypes(self, cluster_stats: pd.DataFrame) -> List[str]:
        """
        Assign archetype names to clusters based on characteristics.
        
        Args:
            cluster_stats (pd.DataFrame): Cluster statistics
            
        Returns:
            List[str]: Archetype names for each cluster
        """
        archetypes = []
        
        for idx, row in cluster_stats.iterrows():
            # Dominant Champions: High wins, good finish, high top-5 rate
            if row['avg_wins_per_season'] >= 1.0 and row['avg_top5_rate'] >= 0.3:
                archetypes.append("Dominant Champions")
            
            # Consistent Contenders: Good finish, high top-5, moderate wins
            elif row['avg_finish'] <= 15 and row['avg_top5_rate'] >= 0.2:
                archetypes.append("Consistent Contenders")
            
            # Late Bloomers: Positive improvement, late peak timing
            elif row['avg_improvement'] > 0 and row['avg_peak_timing'] > 0.6:
                archetypes.append("Late Bloomers")
            
            # Flash in the Pan: Short careers but decent performance
            elif row['avg_seasons'] <= 5 and row['avg_finish'] <= 20:
                archetypes.append("Flash in the Pan")
            
            # Strugglers: Poor performance across metrics
            elif row['avg_finish'] >= 25 and row['avg_top5_rate'] <= 0.05:
                archetypes.append("Strugglers")
            
            # Journeymen: Long careers, moderate performance
            else:
                archetypes.append("Journeymen")
        
        return archetypes
    
    def _get_representative_drivers(self, cluster_data: pd.DataFrame) -> List[str]:
        """
        Get representative drivers for each cluster.
        
        Args:
            cluster_data (pd.DataFrame): Data with cluster assignments
            
        Returns:
            List[str]: Representative driver names for each cluster
        """
        representatives = []
        
        for cluster_id in sorted(cluster_data['cluster'].unique()):
            cluster_drivers = cluster_data[cluster_data['cluster'] == cluster_id]
            
            # Sort by a combination of wins and consistency
            cluster_drivers['rep_score'] = (
                cluster_drivers['total_wins'] * 0.4 +
                cluster_drivers['seasons_active'] * 0.3 +
                (100 - cluster_drivers['career_avg_finish']) * 0.3
            )
            
            top_drivers = cluster_drivers.nlargest(3, 'rep_score')['Driver'].tolist()
            representatives.append(", ".join(top_drivers))
        
        return representatives
    
    def create_cluster_visualizations(self) -> Dict[str, go.Figure]:
        """
        Create comprehensive cluster visualization plots.
        
        Returns:
            Dict[str, go.Figure]: Dictionary of Plotly figures
        """
        if self.cluster_analysis is None:
            raise ValueError("Must analyze clusters first")
        
        figures = {}
        
        # 1. Elbow Plot
        figures['elbow'] = self._create_elbow_plot()
        
        # 2. Silhouette Analysis
        figures['silhouette'] = self._create_silhouette_plot()
        
        # 3. 2D Cluster Scatter (PCA)
        figures['scatter_2d'] = self._create_2d_scatter()
        
        # 4. 3D Cluster Scatter
        figures['scatter_3d'] = self._create_3d_scatter()
        
        # 5. Cluster Characteristics Radar
        figures['radar'] = self._create_cluster_radar()
        
        # 6. Driver Distribution by Archetype
        figures['distribution'] = self._create_distribution_plot()
        
        return figures
    
    def _create_elbow_plot(self) -> go.Figure:
        """Create elbow method visualization."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(self.elbow_scores.keys()),
            y=list(self.elbow_scores.values()),
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Elbow Method for Optimal Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Inertia (Within-cluster sum of squares)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _create_silhouette_plot(self) -> go.Figure:
        """Create silhouette analysis visualization."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(self.silhouette_scores.keys()),
            y=list(self.silhouette_scores.values()),
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Add recommended k-value
        recommended_k = self.cluster_config['n_clusters']
        if recommended_k in self.silhouette_scores:
            fig.add_vline(
                x=recommended_k,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Config K={recommended_k}"
            )
        
        fig.update_layout(
            title='Silhouette Analysis for Optimal Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _create_2d_scatter(self) -> go.Figure:
        """Create 2D PCA scatter plot of clusters."""
        # Perform PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(self.scaled_features)
        
        # Create DataFrame for plotting
        plot_data = self.career_data.copy()
        plot_data['PC1'] = pca_features[:, 0]
        plot_data['PC2'] = pca_features[:, 1]
        plot_data['cluster'] = self.cluster_labels
        
        # Map clusters to archetypes
        cluster_archetype_map = dict(zip(
            self.cluster_analysis.index,
            self.cluster_analysis['archetype']
        ))
        plot_data['archetype'] = plot_data['cluster'].map(cluster_archetype_map)
        
        fig = px.scatter(
            plot_data,
            x='PC1',
            y='PC2',
            color='archetype',
            hover_data=['Driver', 'total_wins', 'career_avg_finish', 'seasons_active'],
            title='NASCAR Driver Archetypes (PCA Visualization)',
            color_discrete_sequence=self.config['visualization']['colors']['archetypes']
        )
        
        fig.update_traces(marker_size=8, marker_opacity=0.7)
        fig.update_layout(template='plotly_white', height=600)
        
        return fig
    
    def _create_3d_scatter(self) -> go.Figure:
        """Create 3D scatter plot using key performance metrics."""
        plot_data = self.career_data.copy()
        plot_data['cluster'] = self.cluster_labels
        
        # Map clusters to archetypes
        cluster_archetype_map = dict(zip(
            self.cluster_analysis.index,
            self.cluster_analysis['archetype']
        ))
        plot_data['archetype'] = plot_data['cluster'].map(cluster_archetype_map)
        
        fig = px.scatter_3d(
            plot_data,
            x='wins_per_season',
            y='career_avg_finish',
            z='top5_rate',
            color='archetype',
            hover_data=['Driver', 'total_wins', 'seasons_active'],
            title='NASCAR Driver Archetypes (3D Performance Space)',
            color_discrete_sequence=self.config['visualization']['colors']['archetypes']
        )
        
        fig.update_traces(marker_size=5, marker_opacity=0.8)
        fig.update_layout(height=700)
        
        return fig
    
    def _create_cluster_radar(self) -> go.Figure:
        """Create radar chart showing cluster characteristics."""
        # Normalize metrics for radar chart (0-1 scale)
        metrics = ['avg_wins_per_season', 'avg_top5_rate', 'avg_seasons', 'avg_improvement']
        radar_data = self.cluster_analysis[metrics].copy()
        
        # Normalize each metric to 0-1 scale
        for col in metrics:
            radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
        
        fig = go.Figure()
        
        colors = self.config['visualization']['colors']['archetypes']
        
        for i, (idx, row) in enumerate(radar_data.iterrows()):
            archetype = self.cluster_analysis.loc[idx, 'archetype']
            
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=archetype,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Driver Archetype Characteristics",
            height=600
        )
        
        return fig
    
    def _create_distribution_plot(self) -> go.Figure:
        """Create driver count distribution by archetype."""
        archetype_counts = self.cluster_analysis['driver_count'].values
        archetype_names = self.cluster_analysis['archetype'].values
        colors = self.config['visualization']['colors']['archetypes']
        
        fig = go.Figure(data=[
            go.Bar(
                x=archetype_names,
                y=archetype_counts,
                marker_color=colors[:len(archetype_names)],
                text=archetype_counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Driver Distribution by Archetype',
            xaxis_title='Driver Archetype',
            yaxis_title='Number of Drivers',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def get_driver_archetype(self, driver_name: str) -> Dict:
        """
        Get archetype information for a specific driver.
        
        Args:
            driver_name (str): Name of the driver
            
        Returns:
            Dict: Driver archetype information
        """
        if self.career_data is None:
            raise ValueError("No career data available")
        
        driver_data = self.career_data[self.career_data['Driver'] == driver_name]
        
        if driver_data.empty:
            return {'error': f"Driver '{driver_name}' not found"}
        
        driver_idx = driver_data.index[0]
        cluster_id = self.cluster_labels[driver_idx]
        archetype = self.cluster_analysis.loc[cluster_id, 'archetype']
        
        return {
            'driver': driver_name,
            'archetype': archetype,
            'cluster_id': cluster_id,
            'career_stats': driver_data.iloc[0].to_dict(),
            'cluster_characteristics': self.cluster_analysis.loc[cluster_id].to_dict()
        }
    
    def save_clustering_results(self) -> None:
        """Save clustering results to files."""
        if self.kmeans is None:
            raise ValueError("No clustering model to save")
        
        # Save model
        import joblib
        model_path = Path(self.paths['models']) / 'driver_clustering_model.pkl'
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_names': self.features.columns.tolist(),
            'cluster_analysis': self.cluster_analysis
        }, model_path)
        
        # Save driver assignments
        driver_assignments = self.career_data.copy()
        driver_assignments['cluster'] = self.cluster_labels
        cluster_archetype_map = dict(zip(
            self.cluster_analysis.index,
            self.cluster_analysis['archetype']
        ))
        driver_assignments['archetype'] = driver_assignments['cluster'].map(cluster_archetype_map)
        
        assignments_path = Path(self.paths['processed_data']) / 'driver_archetypes.parquet'
        driver_assignments.to_parquet(assignments_path, index=False)
        
        # Save cluster analysis
        analysis_path = Path(self.paths['processed_data']) / 'cluster_analysis.parquet'
        self.cluster_analysis.to_parquet(analysis_path, index=False)
        
        self.logger.info(f"Clustering results saved to {model_path}")
        self.logger.info(f"Driver assignments saved to {assignments_path}")
        self.logger.info(f"Cluster analysis saved to {analysis_path}")


def run_clustering_analysis(config_path: Optional[str] = None, 
                          n_clusters: Optional[int] = None,
                          save_results: bool = True) -> DriverClusterAnalyzer:
    """
    Convenience function to run complete clustering analysis.
    
    Args:
        config_path (Optional[str]): Path to config file
        n_clusters (Optional[int]): Number of clusters to use
        save_results (bool): Whether to save results to files
        
    Returns:
        DriverClusterAnalyzer: Fitted analyzer with results
    """
    # Load data
    from data.data_loader import load_nascar_data
    
    print("🏁 Loading NASCAR data...")
    data_loader = load_nascar_data()
    
    # Initialize clustering analyzer
    print("🔬 Initializing clustering analysis...")
    analyzer = DriverClusterAnalyzer()
    
    # Prepare features
    print("⚙️  Preparing clustering features...")
    analyzer.prepare_clustering_features(data_loader.driver_seasons)
    analyzer.select_clustering_features()
    
    # Find optimal clusters
    print("📊 Finding optimal number of clusters...")
    optimization_results = analyzer.find_optimal_clusters()
    
    print(f"   Best silhouette score: {optimization_results['best_silhouette_score']:.3f} (k={optimization_results['best_silhouette_k']})")
    print(f"   Elbow method suggests: k={optimization_results['elbow_k']}")
    print(f"   Config recommends: k={optimization_results['recommended_k']}")
    
    # Fit final model
    final_k = n_clusters if n_clusters is not None else optimization_results['recommended_k']
    print(f"🎯 Fitting final model with {final_k} clusters...")
    analyzer.fit_clustering_model(final_k)
    
    # Analyze clusters
    print("🏷️  Analyzing cluster characteristics...")
    cluster_analysis = analyzer.analyze_clusters()
    
    print("\n📋 Driver Archetypes Discovered:")
    print("=" * 60)
    for idx, row in cluster_analysis.iterrows():
        print(f"{row['archetype']}: {row['driver_count']} drivers")
        print(f"   Avg Wins/Season: {row['avg_wins_per_season']:.2f}")
        print(f"   Avg Finish: {row['avg_finish']:.1f}")
        print(f"   Top-5 Rate: {row['avg_top5_rate']:.1%}")
        print(f"   Representatives: {row['representative_drivers']}")
        print()
    
    # Save results
    if save_results:
        print("💾 Saving clustering results...")
        analyzer.save_clustering_results()
    
    print("✅ Clustering analysis complete!")
    return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = run_clustering_analysis()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    figures = analyzer.create_cluster_visualizations()
    
    # Show example driver archetype lookup
    print("\n🔍 Example Driver Lookups:")
    test_drivers = ['Kyle Larson', 'Kevin Harvick', 'Denny Hamlin']
    for driver in test_drivers:
        result = analyzer.get_driver_archetype(driver)
        if 'error' not in result:
            print(f"{driver}: {result['archetype']}")