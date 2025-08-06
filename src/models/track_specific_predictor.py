#!/usr/bin/env python3
"""
Track-Type-Only NASCAR Driver Performance Predictor

This model makes predictions based SOLELY on historical performance at the same track type:
- Road course prediction → only uses road course history
- Superspeedway prediction → only uses superspeedway history
- Short track prediction → only uses short track history
- Intermediate prediction → only uses intermediate track history

Perfect for specialists like Shane van Gisbergen (road courses) or Michael McDowell (superspeedways).
When track type is selected, the model automatically filters to only that track type's historical data.
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import existing modules
try:
    from data.data_loader import load_nascar_data
    print("✅ Successfully imported NASCAR data loader")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================================================
DRIVER_NAME = "Shane van Gisbergen"
NEXT_TRACK_NAME = "Watkins Glen International"  # The track type will be auto-detected

# Alternative examples to test different scenarios:
# DRIVER_NAME = "Michael McDowell"
# NEXT_TRACK_NAME = "Daytona International Speedway"

# DRIVER_NAME = "Chase Elliott" 
# NEXT_TRACK_NAME = "Charlotte Motor Speedway"

# DRIVER_NAME = "Kyle Larson"
# NEXT_TRACK_NAME = "Bristol Motor Speedway"
# =============================================================================


class TrackTypeOnlyPredictor:
    """
    NASCAR finishing position predictor using ONLY same track type historical data
    """
    
    def __init__(self):
        """Initialize predictor and load data"""
        print("🏁 Loading NASCAR data...")
        self.data_loader = load_nascar_data()
        self.cup_series_df = self.data_loader.df
        
        # Clean and prepare data
        self._prepare_data()
        
        # Create track type mapping
        self._create_track_mapping()
        
        print("✅ Track-Type-Only Predictor initialized!")
    
    def _prepare_data(self):
        """Clean and prepare the NASCAR data"""
        # Convert positions to numeric
        self.cup_series_df['Finish'] = pd.to_numeric(self.cup_series_df['Finish'], errors='coerce')
        self.cup_series_df['Start'] = pd.to_numeric(self.cup_series_df['Start'], errors='coerce')
        self.cup_series_df['Length'] = pd.to_numeric(self.cup_series_df['Length'], errors='coerce')
        
        # Remove invalid data
        self.cup_series_df = self.cup_series_df.dropna(subset=['Finish'])
        self.cup_series_df = self.cup_series_df[
            (self.cup_series_df['Finish'] >= 1) & 
            (self.cup_series_df['Finish'] <= 50)  # Reasonable finishing positions
        ]
        
        # Add track type classification
        self.cup_series_df['track_type'] = self.cup_series_df.apply(
            lambda row: self._classify_track_type(row['Track'], row['Length']), axis=1
        )
        
        # Sort by driver and race date for recency weighting
        if 'Season' in self.cup_series_df.columns and 'Race' in self.cup_series_df.columns:
            self.cup_series_df = self.cup_series_df.sort_values(['Driver', 'Season', 'Race'])
    
    def _classify_track_type(self, track_name: str, track_length: float) -> str:
        """Classify track type with enhanced detection"""
        if pd.isna(track_name):
            track_name = ""
        if pd.isna(track_length):
            track_length = 1.5
            
        track_name = str(track_name).lower()
        
        # Enhanced road course detection
        road_keywords = [
            'road', 'glen', 'sonoma', 'cota', 'roval', 'mexico', 
            'watkins', 'infineon', 'sears point', 'circuit', 'road america'
        ]
        if any(keyword in track_name for keyword in road_keywords):
            return 'road'
        
        # Superspeedways - specific tracks and length-based
        superspeedway_tracks = ['daytona', 'talladega']
        if track_length >= 2.0 or any(name in track_name for name in superspeedway_tracks):
            return 'superspeedway'
        
        # Short tracks - under 1.0 mile
        if track_length < 1.0:
            return 'short'
        
        # Intermediate tracks - everything else
        return 'intermediate'
    
    def _create_track_mapping(self):
        """Create mapping of specific tracks to their types"""
        self.track_type_mapping = {}
        
        # Get unique track-type combinations
        track_types = self.cup_series_df.groupby('Track')['track_type'].first()
        
        for track, track_type in track_types.items():
            self.track_type_mapping[track.lower()] = track_type
        
        print(f"📋 Mapped {len(self.track_type_mapping)} tracks to types")
    
    def get_track_type_from_name(self, track_name: str) -> str:
        """Get track type from track name"""
        track_lower = track_name.lower()
        
        # Direct lookup first
        if track_lower in self.track_type_mapping:
            return self.track_type_mapping[track_lower]
        
        # Partial match lookup
        for mapped_track, track_type in self.track_type_mapping.items():
            if any(word in mapped_track for word in track_lower.split()):
                return track_type
        
        # Fallback to classification logic
        # Try to find track length in data
        track_data = self.cup_series_df[
            self.cup_series_df['Track'].str.contains(track_name, case=False, na=False)
        ]
        
        if not track_data.empty:
            avg_length = track_data['Length'].mean()
            return self._classify_track_type(track_name, avg_length)
        
        # Default fallback
        print(f"⚠️ Could not determine track type for '{track_name}', defaulting to intermediate")
        return 'intermediate'
    
    def get_driver_track_type_history(self, driver_name: str, track_type: str) -> pd.DataFrame:
        """Get all races for a driver at specific track type"""
        # First, let's debug by getting ALL driver data
        all_driver_data = self.cup_series_df[
            self.cup_series_df['Driver'] == driver_name
        ].copy()
        
        if not all_driver_data.empty:
            print(f"\n🔍 DEBUGGING {driver_name}:")
            print(f"Total races found: {len(all_driver_data)}")
            print(f"Total wins: {(all_driver_data['Finish'] == 1).sum()}")
            
            # Show track type breakdown
            track_type_counts = all_driver_data['track_type'].value_counts()
            print(f"Track type breakdown:")
            for tt, count in track_type_counts.items():
                wins_at_type = (all_driver_data[all_driver_data['track_type'] == tt]['Finish'] == 1).sum()
                print(f"  {tt}: {count} races, {wins_at_type} wins")
            
            # Show all winning races
            wins = all_driver_data[all_driver_data['Finish'] == 1]
            if not wins.empty:
                print(f"\n🏆 All winning races:")
                for _, win in wins.iterrows():
                    track_name = win.get('Track', 'Unknown')
                    season = win.get('Season', 'Unknown')
                    classified_type = win.get('track_type', 'Unknown')
                    print(f"  {season}: {track_name} (classified as: {classified_type})")
        
        # Now get the filtered data
        driver_data = self.cup_series_df[
            (self.cup_series_df['Driver'] == driver_name) &
            (self.cup_series_df['track_type'] == track_type)
        ].copy()
        
        if not driver_data.empty:
            # Sort by season and race for chronological order (most recent first)
            driver_data = driver_data.sort_values(['Season', 'Race'], ascending=[False, False])
        
        return driver_data
    
    def calculate_track_type_prediction(self, track_type_history: pd.DataFrame, 
                                      use_recency_weighting: bool = True) -> Dict:
        """Calculate prediction based solely on track type history"""
        
        if track_type_history.empty:
            return {
                'predicted_position': 25.0,
                'volatility': 10.0,
                'confidence': 0.0,
                'data_quality': 'no_data',
                'races_used': 0
            }
        
        races_used = len(track_type_history)
        
        if use_recency_weighting and races_used > 1:
            # Apply exponential decay weighting (more recent races weighted higher)
            weights = np.array([0.85 ** i for i in range(races_used)])
            weights = weights / weights.sum()  # Normalize
            
            predicted_position = np.average(track_type_history['Finish'], weights=weights)
            
            # Calculate weighted standard deviation
            weighted_variance = np.average(
                (track_type_history['Finish'] - predicted_position) ** 2, 
                weights=weights
            )
            volatility = np.sqrt(weighted_variance)
        else:
            # Simple average
            predicted_position = track_type_history['Finish'].mean()
            volatility = track_type_history['Finish'].std()
        
        # Handle single race case
        if races_used == 1:
            volatility = 8.0  # Default moderate volatility
        elif pd.isna(volatility):
            volatility = 6.0
        
        # Calculate confidence based on sample size and recency
        if races_used >= 10:
            confidence = 0.9
        elif races_used >= 5:
            confidence = 0.75
        elif races_used >= 3:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Determine data quality
        if races_used >= 8:
            data_quality = 'excellent'
        elif races_used >= 5:
            data_quality = 'good'
        elif races_used >= 3:
            data_quality = 'fair'
        elif races_used >= 1:
            data_quality = 'limited'
        else:
            data_quality = 'no_data'
        
        return {
            'predicted_position': predicted_position,
            'volatility': max(2.0, min(15.0, volatility)),  # Reasonable bounds
            'confidence': confidence,
            'data_quality': data_quality,
            'races_used': races_used
        }
    
    def generate_position_probabilities(self, predicted_position: float, 
                                      volatility: float) -> Dict[int, float]:
        """Generate probability distribution for finishing positions"""
        positions = np.arange(1, 41)  # Positions 1-40
        
        # Use truncated normal distribution
        probs = stats.norm.pdf(positions, loc=predicted_position, scale=volatility)
        
        # Ensure probabilities stay within reasonable bounds (no position > 40)
        # Apply truncation at boundaries
        probs = np.where(positions <= 40, probs, 0)
        probs = np.where(positions >= 1, probs, 0)
        
        # Normalize to sum to 1
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        return {int(pos): float(prob) for pos, prob in zip(positions, probs)}
    
    def predict_driver_performance(self, driver_name: str, track_name: str) -> Dict:
        """
        Make track-type-only prediction for a driver
        
        Args:
            driver_name: Name of driver to analyze
            track_name: Name of track for next race
            
        Returns:
            Dictionary with prediction results
        """
        print(f"🔍 Analyzing {driver_name} at {track_name}...")
        
        # Determine track type
        track_type = self.get_track_type_from_name(track_name)
        print(f"📍 Track type identified: {track_type.upper()}")
        
        # Get track type history
        track_history = self.get_driver_track_type_history(driver_name, track_type)
        
        if track_history.empty:
            return {
                'error': f"No {track_type} track history found for {driver_name}",
                'driver_name': driver_name,
                'track_name': track_name,
                'track_type': track_type
            }
        
        print(f"📊 Found {len(track_history)} {track_type} races for {driver_name}")
        
        # Calculate prediction
        prediction = self.calculate_track_type_prediction(track_history)
        
        # Generate probability distribution
        probabilities = self.generate_position_probabilities(
            prediction['predicted_position'], 
            prediction['volatility']
        )
        
        # Calculate confidence interval
        z_score = 1.96  # 95% confidence
        margin = z_score * prediction['volatility']
        confidence_interval = {
            'lower': max(1, prediction['predicted_position'] - margin),
            'upper': min(40, prediction['predicted_position'] + margin),
            'confidence': 0.95
        }
        
        # Compile comprehensive results
        results = {
            'driver_name': driver_name,
            'track_name': track_name,
            'track_type': track_type,
            'predicted_position': max(1, min(40, round(prediction['predicted_position']))),
            'predicted_position_raw': prediction['predicted_position'],
            'volatility': prediction['volatility'],
            'confidence_interval': confidence_interval,
            'position_probabilities': probabilities,
            'max_probability': max(probabilities.values()) * 100,
            'data_quality': prediction['data_quality'],
            'races_used': prediction['races_used'],
            'model_confidence': prediction['confidence'],
            'track_history': self._summarize_track_history(track_history)
        }
        
        return results
    
    def _summarize_track_history(self, track_history: pd.DataFrame) -> Dict:
        """Summarize track type historical performance"""
        if track_history.empty:
            return {}
        
        return {
            'total_races': len(track_history),
            'avg_finish': track_history['Finish'].mean(),
            'best_finish': track_history['Finish'].min(),
            'worst_finish': track_history['Finish'].max(),
            'wins': (track_history['Finish'] == 1).sum(),
            'top_5s': (track_history['Finish'] <= 5).sum(),
            'top_10s': (track_history['Finish'] <= 10).sum(),
            'win_rate': (track_history['Finish'] == 1).mean(),
            'top_5_rate': (track_history['Finish'] <= 5).mean(),
            'top_10_rate': (track_history['Finish'] <= 10).mean(),
            'recent_5_avg': track_history.head(5)['Finish'].mean() if len(track_history) >= 5 else track_history['Finish'].mean(),
            'std_deviation': track_history['Finish'].std(),
            'seasons_span': f"{track_history['Season'].min()}-{track_history['Season'].max()}" if 'Season' in track_history.columns else 'Unknown'
        }
    
    def create_comprehensive_visualization(self, prediction_results: Dict) -> go.Figure:
        """Create comprehensive visualization of track-type-only prediction"""
        
        if 'error' in prediction_results:
            fig = go.Figure()
            fig.add_annotation(
                text=prediction_results['error'],
                x=0.5, y=0.5, showarrow=False, font_size=16
            )
            fig.update_layout(title="Prediction Error")
            return fig
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Finishing Position Probability Distribution ({prediction_results["track_type"].title()} Tracks Only)',
                f'Historical {prediction_results["track_type"].title()} Track Performance',
                'Data Quality & Confidence Metrics',
                'Performance Summary Statistics'
            ],
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Main probability distribution
        positions = list(prediction_results['position_probabilities'].keys())
        probabilities = [p * 100 for p in prediction_results['position_probabilities'].values()]
        
        # Enhanced color coding
        colors = []
        for pos in positions:
            if pos == 1:
                colors.append('#FFD700')  # Gold for win
            elif pos <= 5:
                colors.append('#32CD32')  # Green for top 5
            elif pos <= 10:
                colors.append('#87CEEB')  # Light blue for top 10
            elif pos <= 15:
                colors.append('#98FB98')  # Light green for decent
            elif pos <= 25:
                colors.append('#FFA500')  # Orange for average
            else:
                colors.append('#FF6B6B')  # Red for poor finish
        
        fig.add_trace(
            go.Bar(
                x=positions,
                y=probabilities,
                name='Position Probability',
                marker_color=colors,
                hovertemplate='Position %{x}<br>Probability: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add predicted position line
        pred_pos = prediction_results['predicted_position']
        fig.add_vline(
            x=pred_pos,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Predicted: P{pred_pos}",
            annotation_position="top",
            row=1, col=1
        )
        
        # Add confidence interval
        ci = prediction_results['confidence_interval']
        fig.add_vrect(
            x0=ci['lower'], x1=ci['upper'],
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below", line_width=0,
            annotation_text=f"95% Confidence",
            annotation_position="top left",
            row=1, col=1
        )
        
        # 2. Historical performance metrics
        history = prediction_results['track_history']
        if history:
            metrics = ['Win Rate (%)', 'Top 5 Rate (%)', 'Top 10 Rate (%)', 'Avg Finish']
            values = [
                history['win_rate'] * 100,
                history['top_5_rate'] * 100, 
                history['top_10_rate'] * 100,
                40 - history['avg_finish']  # Invert for better visual (higher bar = better performance)
            ]
            
            colors_bar = ['gold', 'green', 'lightblue', 'orange']
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name='Historical Performance',
                    marker_color=colors_bar,
                    hovertemplate='%{x}: %{y:.1f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 3. Summary table
        summary_data = [
            ['Driver', prediction_results['driver_name']],
            ['Next Track', prediction_results['track_name']],
            ['Track Type', prediction_results['track_type'].title()],
            ['Predicted Position', f"P{pred_pos}"],
            ['Max Probability', f"{prediction_results['max_probability']:.1f}%"],
            ['Races Used', str(prediction_results['races_used'])],
            ['Data Quality', prediction_results['data_quality'].title()],
            ['Model Confidence', f"{prediction_results['model_confidence']:.1%}"],
            ['Volatility (σ)', f"{prediction_results['volatility']:.1f}"]
        ]
        
        if history:
            summary_data.extend([
                ['Historical Avg', f"P{history['avg_finish']:.1f}"],
                ['Best Finish', f"P{history['best_finish']}"],
                ['Total Wins', f"{history['wins']}"],
                ['Seasons Span', history['seasons_span']]
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='lightgray',
                           align='left',
                           font=dict(size=12)),
                cells=dict(values=[list(zip(*summary_data))[0], list(zip(*summary_data))[1]],
                          fill_color='white',
                          align='left',
                          font=dict(size=11))
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"Track-Type-Only Prediction: {prediction_results['driver_name']} at {prediction_results['track_type'].title()} Tracks",
            title_x=0.5,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Finishing Position", row=1, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
        fig.update_yaxes(title_text="Performance Score", row=2, col=1)
        
        return fig


def main():
    """Main execution function"""
    print("🏁 Track-Type-Only NASCAR Driver Performance Predictor")
    print("=" * 65)
    print(f"Driver: {DRIVER_NAME}")
    print(f"Next Track: {NEXT_TRACK_NAME}")
    print("-" * 65)
    
    # Initialize predictor
    try:
        predictor = TrackTypeOnlyPredictor()
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return
    
    # Make prediction
    try:
        results = predictor.predict_driver_performance(DRIVER_NAME, NEXT_TRACK_NAME)
        
        if 'error' in results:
            print(f"❌ Prediction failed: {results['error']}")
            
            # Show available track types for this driver
            all_driver_data = predictor.cup_series_df[
                predictor.cup_series_df['Driver'] == DRIVER_NAME
            ]
            if not all_driver_data.empty:
                available_types = all_driver_data['track_type'].value_counts()
                print(f"\n📋 Available track types for {DRIVER_NAME}:")
                for track_type, count in available_types.items():
                    print(f"  - {track_type.title()}: {count} races")
            return
        
        # Print comprehensive results
        print(f"\n📊 PREDICTION RESULTS")
        print(f"Track Type: {results['track_type'].upper()}")
        print(f"Predicted Finish: P{results['predicted_position']}")
        print(f"Raw Prediction: P{results['predicted_position_raw']:.1f}")
        print(f"95% Confidence Range: P{results['confidence_interval']['lower']:.0f} - P{results['confidence_interval']['upper']:.0f}")
        print(f"Maximum Probability: {results['max_probability']:.1f}%")
        print(f"Volatility: {results['volatility']:.1f}")
        
        print(f"\n📈 DATA QUALITY")
        print(f"Races Used: {results['races_used']} {results['track_type']} races")
        print(f"Data Quality: {results['data_quality'].title()}")
        print(f"Model Confidence: {results['model_confidence']:.1%}")
        
        if results['track_history']:
            history = results['track_history']
            print(f"\n🏆 {results['track_type'].upper()} TRACK PERFORMANCE")
            print(f"Historical Average: P{history['avg_finish']:.1f}")
            print(f"Best Finish: P{history['best_finish']}")
            print(f"Worst Finish: P{history['worst_finish']}")
            print(f"Wins: {history['wins']} ({history['win_rate']:.1%})")
            print(f"Top 5s: {history['top_5s']} ({history['top_5_rate']:.1%})")
            print(f"Top 10s: {history['top_10s']} ({history['top_10_rate']:.1%})")
            print(f"Recent 5-Race Avg: P{history['recent_5_avg']:.1f}")
            print(f"Seasons: {history['seasons_span']}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Create visualization
    try:
        print(f"\n🎨 Creating visualization...")
        fig = predictor.create_comprehensive_visualization(results)
        
        # Don't automatically show in browser to avoid HTML output spam
        print("📊 Visualization created successfully")
        
        # Only show if user wants to save
        save_option = input("\nSave and show visualization? (y/n): ").lower().strip()
        if save_option == 'y':
            output_dir = Path('outputs/visualizations')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"track_type_only_{DRIVER_NAME.replace(' ', '_')}_{results['track_type']}_{timestamp}.html"
            filepath = output_dir / filename
            
            fig.write_html(filepath)
            print(f"📁 Visualization saved to: {filepath}")
            
            # Ask how to view the file
            view_option = input("\nHow would you like to view it?\n1. Default browser (recommended)\n2. Live Server in VS Code\nChoice (1/2): ").strip()
            
            if view_option == '1' or view_option == '':
                # Open with default browser (bypasses caching issues)
                import webbrowser
                webbrowser.open(f'file://{filepath.absolute()}')
                print("🌐 Opening in default browser...")
            else:
                print(f"📝 Open this file in VS Code Live Server: {filepath}")
                print("💡 If you see old data, try Ctrl+F5 to force refresh!")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


if __name__ == "__main__":
    main()