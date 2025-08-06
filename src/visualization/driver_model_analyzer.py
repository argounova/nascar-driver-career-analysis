#!/usr/bin/env python3
"""
NASCAR Driver Model Visualization Tool

This standalone script loads the trained NASCAR prediction models and creates
interactive visualizations for any driver. Simply modify the DRIVER_NAME variable
below to analyze different drivers.

Requirements:
- Run from project root directory
- Ensure models have been trained (run scripts/train_models.py first)
- Install: pip install plotly pandas numpy scikit-learn

Usage:
1. Modify the DRIVER_NAME variable below
2. Run: python src/visualization/driver_model_analyzer.py
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

# Add project paths
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import your existing modules
try:
    from models.finish_position_predictor import FinishPositionPredictor
    from models.driver_volatility_predictor import DriverVolatilityPredictor
    from models.win_probability_predictor import WinProbabilityPredictor
    from data.data_loader import load_nascar_data
    print("✅ Successfully imported NASCAR analysis modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# =============================================================================
# CONFIGURATION - MODIFY THIS TO ANALYZE DIFFERENT DRIVERS
# =============================================================================
DRIVER_NAME = "Shane van Gisbergen"  # Change this to any driver name
NEXT_TRACK_NAME = "Watkins Glen International"  # Optional: specify next track
NEXT_TRACK_LENGTH = 1.5  # Track length in miles

# Optional: Analyze multiple drivers at once
COMPARE_DRIVERS = [
  
]

# =============================================================================


class NASCARModelVisualizer:
    """
    Creates interactive visualizations from NASCAR prediction models
    """
    
    def __init__(self):
        """Initialize the visualizer and load data/models"""
        print("🏁 Loading NASCAR data...")
        self.data_loader = load_nascar_data()
        self.cup_series_df = self.data_loader.df
        
        print("🤖 Loading trained models...")
        self.models = self._load_models()
        
        print("✅ Initialization complete!")
    
    def _load_models(self) -> Dict[str, Any]:
        """Load all trained models"""
        models = {}
        
        try:
            # Load finish position predictor
            models['finish_position'] = FinishPositionPredictor.load(
                'data/models/finish_position_predictor.pkl'
            )
            print("  ✅ Finish Position Predictor loaded")
        except FileNotFoundError:
            print("  ❌ Finish Position Predictor not found")
            models['finish_position'] = None
        
        try:
            # Load volatility predictor  
            models['volatility'] = DriverVolatilityPredictor.load(
                'data/models/driver_volatility_predictor.pkl'
            )
            print("  ✅ Driver Volatility Predictor loaded")
        except FileNotFoundError:
            print("  ❌ Driver Volatility Predictor not found")
            models['volatility'] = None
        
        try:
            # Load win probability predictor
            models['win_probability'] = WinProbabilityPredictor.load(
                'data/models/win_probability_predictor.pkl'
            )
            print("  ✅ Win Probability Predictor loaded")
        except FileNotFoundError:
            print("  ❌ Win Probability Predictor not found")
            models['win_probability'] = None
        
        return models
    
    def get_driver_data(self, driver_name: str) -> pd.DataFrame:
        """Get race history for a specific driver"""
        driver_data = self.cup_series_df[
            self.cup_series_df['Driver'] == driver_name
        ].copy()
        
        if driver_data.empty:
            available_drivers = sorted(self.cup_series_df['Driver'].unique())
            print(f"❌ No data found for '{driver_name}'")
            print(f"Available drivers include: {available_drivers[:10]}...")
            return pd.DataFrame()
        
        return driver_data.sort_values(['Season', 'Race'])
    
    def analyze_single_driver(self, driver_name: str, 
                            next_track: str = None, 
                            track_length: float = 1.5) -> Dict[str, Any]:
        """Analyze a single driver using all available models"""
        print(f"\n🔍 Analyzing {driver_name}...")
        
        driver_data = self.get_driver_data(driver_name)
        if driver_data.empty:
            return {}
        
        results = {
            'driver_name': driver_name,
            'total_races': len(driver_data),
            'seasons': f"{driver_data['Season'].min()}-{driver_data['Season'].max()}",
            'career_stats': self._get_career_stats(driver_data),
            'predictions': {}
        }
        
        # Finish position prediction
        if self.models['finish_position'] is not None:
            try:
                finish_pred = self.models['finish_position'].predict_for_driver(
                    self.cup_series_df, driver_name, next_track, track_length
                )
                results['predictions']['finish_position'] = finish_pred
                print(f"  📊 Predicted finish: {finish_pred['predicted_finish']}")
            except Exception as e:
                print(f"  ❌ Finish position prediction failed: {e}")
        
        # Volatility prediction
        if self.models['volatility'] is not None:
            try:
                volatility_pred = self.models['volatility'].predict_for_driver(
                    self.cup_series_df, driver_name, next_track, track_length
                )
                results['predictions']['volatility'] = volatility_pred
                print(f"  📈 Predicted volatility: {volatility_pred['predicted_volatility']:.2f}")
            except Exception as e:
                print(f"  ❌ Volatility prediction failed: {e}")
        
        # Win probability prediction
        if self.models['win_probability'] is not None:
            try:
                win_pred = self.models['win_probability'].predict_for_driver(
                    self.cup_series_df, driver_name, next_track, track_length
                )
                results['predictions']['win_probability'] = win_pred
                print(f"  🏆 Win probability: {win_pred['win_probability']:.1%}")
            except Exception as e:
                print(f"  ❌ Win probability prediction failed: {e}")
        
        return results
    
    def _get_career_stats(self, driver_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic career statistics"""
        return {
            'avg_finish': driver_data['Finish'].mean(),
            'avg_start': driver_data['Start'].mean(),
            'wins': (driver_data['Finish'] == 1).sum(),
            'top_5s': (driver_data['Finish'] <= 5).sum(),
            'top_10s': (driver_data['Finish'] <= 10).sum(),
            'win_rate': (driver_data['Finish'] == 1).mean() * 100,
            'top_5_rate': (driver_data['Finish'] <= 5).mean() * 100,
            'top_10_rate': (driver_data['Finish'] <= 10).mean() * 100,
        }
    
    def create_driver_dashboard(self, driver_name: str, 
                              next_track: str = None, 
                              track_length: float = 1.5) -> go.Figure:
        """Create comprehensive dashboard for a single driver"""
        analysis = self.analyze_single_driver(driver_name, next_track, track_length)
        
        if not analysis:
            return go.Figure().add_annotation(
                text=f"No data available for {driver_name}",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Recent Finish Positions', 
                'Model Predictions',
                'Career Performance Distribution',
                'Start vs Finish Correlation',
                'Feature Importance (Finish Model)',
                'Performance Bands'
            ],
            specs=[[{"secondary_y": False}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "box"}]]
        )
        
        driver_data = self.get_driver_data(driver_name)
        recent_data = driver_data.tail(20)  # Last 20 races
        
        # 1. Recent finish positions trend
        fig.add_trace(
            go.Scatter(
                x=list(range(len(recent_data))),
                y=recent_data['Finish'],
                mode='lines+markers',
                name='Finish Position',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 2. Model predictions bar chart
        predictions_data = []
        predictions_names = []
        
        if 'finish_position' in analysis['predictions']:
            predictions_data.append(analysis['predictions']['finish_position']['predicted_finish'])
            predictions_names.append('Predicted Finish')
        
        if 'volatility' in analysis['predictions']:
            predictions_data.append(analysis['predictions']['volatility']['predicted_volatility'])
            predictions_names.append('Volatility (StdDev)')
        
        if 'win_probability' in analysis['predictions']:
            predictions_data.append(analysis['predictions']['win_probability']['win_probability'] * 100)
            predictions_names.append('Win Prob (%)')
        
        if predictions_data:
            fig.add_trace(
                go.Bar(
                    x=predictions_names,
                    y=predictions_data,
                    name='Predictions',
                    marker_color=['lightblue', 'orange', 'green'][:len(predictions_data)]
                ),
                row=1, col=2
            )
        
        # 3. Career performance histogram
        fig.add_trace(
            go.Histogram(
                x=driver_data['Finish'],
                nbinsx=20,
                name='Finish Distribution',
                marker_color='lightcoral'
            ),
            row=1, col=3
        )
        
        # 4. Start vs Finish scatter
        fig.add_trace(
            go.Scatter(
                x=driver_data['Start'],
                y=driver_data['Finish'],
                mode='markers',
                name='Start vs Finish',
                marker=dict(color='purple', size=4, opacity=0.6)
            ),
            row=2, col=1
        )
        
        # 5. Feature importance (if available)
        if (self.models['finish_position'] is not None and 
            'finish_position' in analysis['predictions']):
            features = analysis['predictions']['finish_position'].get('features', {})
            if features:
                fig.add_trace(
                    go.Bar(
                        x=list(features.keys()),
                        y=list(features.values()),
                        name='Feature Values',
                        marker_color='lightgreen'
                    ),
                    row=2, col=2
                )
        
        # 6. Performance box plot by season
        seasons_data = []
        season_labels = []
        for season in sorted(driver_data['Season'].unique())[-5:]:  # Last 5 seasons
            season_data = driver_data[driver_data['Season'] == season]['Finish']
            seasons_data.extend(season_data.tolist())
            season_labels.extend([str(season)] * len(season_data))
        
        fig.add_trace(
            go.Box(
                y=seasons_data,
                x=season_labels,
                name='Season Performance',
                marker_color='lightblue'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"NASCAR Driver Analysis Dashboard: {driver_name}",
            title_x=0.5,
            showlegend=False
        )
        
        # Invert y-axis for finish positions (1st place at top)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=3)
        
        return fig
    
    def create_driver_comparison(self, driver_names: List[str]) -> go.Figure:
        """Create comparison visualization for multiple drivers"""
        print(f"\n🔄 Comparing {len(driver_names)} drivers...")
        
        comparison_data = []
        for driver in driver_names:
            analysis = self.analyze_single_driver(driver)
            if analysis:
                stats = analysis['career_stats']
                pred_data = analysis['predictions']
                
                row = {
                    'Driver': driver,
                    'Avg Finish': stats['avg_finish'],
                    'Win Rate (%)': stats['win_rate'],
                    'Top 5 Rate (%)': stats['top_5_rate'],
                    'Total Races': analysis['total_races']
                }
                
                # Add predictions if available
                if 'finish_position' in pred_data:
                    row['Predicted Finish'] = pred_data['finish_position']['predicted_finish']
                
                if 'win_probability' in pred_data:
                    row['Win Probability (%)'] = pred_data['win_probability']['win_probability'] * 100
                
                comparison_data.append(row)
        
        if not comparison_data:
            return go.Figure().add_annotation(
                text="No valid driver data found for comparison",
                x=0.5, y=0.5, showarrow=False
            )
        
        df_comp = pd.DataFrame(comparison_data)
        
        # Create radar chart comparison
        fig = go.Figure()
        
        categories = ['Avg Finish', 'Win Rate (%)', 'Top 5 Rate (%)']
        
        for _, row in df_comp.iterrows():
            values = [
                40 - row['Avg Finish'],  # Invert so higher is better
                row['Win Rate (%)'],
                row['Top 5 Rate (%)']
            ]
            values += [values[0]]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Driver']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 40]
                )),
            title="Driver Performance Comparison",
            title_x=0.5
        )
        
        return fig
    
    def create_volatility_distribution(self, driver_name: str) -> go.Figure:
        """Create finish position distribution based on volatility prediction"""
        if self.models['volatility'] is None:
            return go.Figure().add_annotation(
                text="Volatility model not available",
                x=0.5, y=0.5, showarrow=False
            )
        
        analysis = self.analyze_single_driver(driver_name)
        if not analysis or 'volatility' not in analysis['predictions']:
            return go.Figure().add_annotation(
                text=f"No volatility prediction available for {driver_name}",
                x=0.5, y=0.5, showarrow=False
            )
        
        volatility_data = analysis['predictions']['volatility']
        predicted_finish = analysis['predictions'].get('finish_position', {}).get('predicted_finish', 15)
        predicted_volatility = volatility_data['predicted_volatility']
        
        # Generate distribution based on prediction
        positions = np.arange(1, 41)  # Positions 1-40
        
        # Create normal distribution centered on predicted finish
        probabilities = np.exp(-0.5 * ((positions - predicted_finish) / predicted_volatility) ** 2)
        probabilities = probabilities / probabilities.sum()  # Normalize
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=positions,
            y=probabilities,
            name='Finish Position Probability',
            marker_color='lightblue',
            hovertemplate='Position %{x}<br>Probability: %{y:.1%}<extra></extra>'
        ))
        
        # Add vertical line at predicted finish
        fig.add_vline(
            x=predicted_finish,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Predicted: P{predicted_finish}"
        )
        
        fig.update_layout(
            title=f"Predicted Finish Distribution: {driver_name}",
            title_x=0.5,
            xaxis_title="Finish Position",
            yaxis_title="Probability",
            xaxis=dict(range=[0.5, 40.5]),
            height=500
        )
        
        fig.update_yaxes(tickformat='.1%')
        
        return fig


def main():
    """Main execution function"""
    print("🏁 NASCAR Driver Model Visualization Tool")
    print("=" * 50)
    
    # Initialize visualizer
    try:
        viz = NASCARModelVisualizer()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Single driver analysis
    print(f"\n📊 Creating dashboard for {DRIVER_NAME}...")
    dashboard = viz.create_driver_dashboard(
        DRIVER_NAME, 
        NEXT_TRACK_NAME, 
        NEXT_TRACK_LENGTH
    )
    dashboard.show()
    
    # Driver comparison
    if len(COMPARE_DRIVERS) > 1:
        print(f"\n⚖️ Creating comparison chart...")
        comparison = viz.create_driver_comparison(COMPARE_DRIVERS)
        comparison.show()
    
    # Volatility distribution
    print(f"\n📈 Creating volatility distribution for {DRIVER_NAME}...")
    volatility_dist = viz.create_volatility_distribution(DRIVER_NAME)
    volatility_dist.show()
    
    print("\n✅ Visualization complete! Check your browser for interactive charts.")
    
    # Optional: Save figures as HTML
    save_option = input("\nSave visualizations to HTML files? (y/n): ").lower().strip()
    if save_option == 'y':
        output_dir = Path('outputs/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        dashboard.write_html(output_dir / f'dashboard_{DRIVER_NAME.replace(" ", "_")}_{timestamp}.html')
        if len(COMPARE_DRIVERS) > 1:
            comparison.write_html(output_dir / f'comparison_{timestamp}.html')
        volatility_dist.write_html(output_dir / f'volatility_{DRIVER_NAME.replace(" ", "_")}_{timestamp}.html')
        
        print(f"📁 Files saved to {output_dir}")


if __name__ == "__main__":
    main()