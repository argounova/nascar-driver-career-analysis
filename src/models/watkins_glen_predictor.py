#!/usr/bin/env python3
"""
NASCAR Watkins Glen International Top 10 Finisher Predictor

This standalone model predicts the top 10 finishers for the upcoming NASCAR race
at Watkins Glen International based on road course performance from 2023-2025.

Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn

Usage: python src/models/watkins_glen_predictor.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class WatkinsGlenPredictor:
    """Predicts top 10 finishers at Watkins Glen International"""
    
    def __init__(self, csv_path='data/raw/cup_series.csv'):
        """Initialize the predictor with data path"""
        self.csv_path = csv_path
        self.model = None
        self.feature_importance = None
        self.predictions = None
        
    def load_and_prepare_data(self):
        """Load and prepare road course data from 2023-2025"""
        print("🏁 Loading NASCAR data...")
        
        # Load data
        self.raw_data = pd.read_csv(self.csv_path)
        print(f"   Total records: {len(self.raw_data):,}")
        
        # Filter for road courses in recent years (2023-2025)
        road_data = self.raw_data[
            (self.raw_data['Surface'] == 'road') & 
            (self.raw_data['Season'] >= 2023)
        ].copy()
        
        print(f"   Road course records (2023-2025): {len(road_data):,}")
        
        # Clean the data
        road_data = road_data.dropna(subset=['Finish', 'Start', 'Driver', 'Laps'])
        
        # Convert data types
        numeric_cols = ['Finish', 'Start', 'Laps', 'Led', 'Pts']
        for col in numeric_cols:
            road_data[col] = pd.to_numeric(road_data[col], errors='coerce')
        
        self.road_data = road_data
        print(f"   Clean records: {len(self.road_data):,}")
        
        return self.road_data
    
    def engineer_features(self):
        """Create features for prediction"""
        print("🔧 Engineering features...")
        
        # Calculate driver road course statistics
        driver_stats = []
        
        for driver in self.road_data['Driver'].unique():
            driver_races = self.road_data[self.road_data['Driver'] == driver]
            
            # Skip drivers with very few races
            if len(driver_races) < 2:
                continue
                
            # Calculate performance metrics
            stats = {
                'Driver': driver,
                'races_run': len(driver_races),
                'avg_finish': driver_races['Finish'].mean(),
                'avg_start': driver_races['Start'].mean(),
                'avg_laps_led': driver_races['Led'].fillna(0).mean(),
                'avg_laps_completed': driver_races['Laps'].mean(),
                'best_finish': driver_races['Finish'].min(),
                'worst_finish': driver_races['Finish'].max(),
                'top_10_rate': (driver_races['Finish'] <= 10).mean(),
                'top_5_rate': (driver_races['Finish'] <= 5).mean(),
                'win_rate': (driver_races['Finish'] == 1).mean(),
                'dnf_rate': (driver_races['Status'] != 'running').mean(),
                'consistency': driver_races['Finish'].std(),
                'recent_form': driver_races[driver_races['Season'] == 2024]['Finish'].mean() if len(driver_races[driver_races['Season'] == 2024]) > 0 else driver_races['Finish'].mean()
            }
            
            # Watkins Glen specific stats
            watkins_races = driver_races[driver_races['Track'] == 'Watkins Glen International']
            if len(watkins_races) > 0:
                stats['watkins_avg_finish'] = watkins_races['Finish'].mean()
                stats['watkins_races'] = len(watkins_races)
                stats['watkins_best_finish'] = watkins_races['Finish'].min()
                stats['watkins_top_10_rate'] = (watkins_races['Finish'] <= 10).mean()
            else:
                stats['watkins_avg_finish'] = stats['avg_finish']  # Use overall road course avg
                stats['watkins_races'] = 0
                stats['watkins_best_finish'] = stats['best_finish']
                stats['watkins_top_10_rate'] = stats['top_10_rate']
            
            driver_stats.append(stats)
        
        self.driver_features = pd.DataFrame(driver_stats)
        
        # Fill NaN values
        self.driver_features['consistency'].fillna(self.driver_features['consistency'].median(), inplace=True)
        
        print(f"   Features created for {len(self.driver_features)} drivers")
        
        return self.driver_features
    
    def train_model(self):
        """Train the prediction model"""
        print("🤖 Training prediction model...")
        
        # Prepare features for training
        feature_cols = [
            'races_run', 'avg_start', 'avg_laps_led', 'avg_laps_completed',
            'best_finish', 'top_10_rate', 'top_5_rate', 'win_rate', 
            'dnf_rate', 'consistency', 'recent_form', 'watkins_races',
            'watkins_best_finish', 'watkins_top_10_rate'
        ]
        
        X = self.driver_features[feature_cols].fillna(0)
        y = self.driver_features['watkins_avg_finish']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"   Training MAE: {train_mae:.2f}")
        print(f"   Test MAE: {test_mae:.2f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def predict_top_10(self):
        """Predict top 10 finishers for Watkins Glen"""
        print("🏆 Predicting top 10 finishers...")
        
        # Prepare features
        feature_cols = [
            'races_run', 'avg_start', 'avg_laps_led', 'avg_laps_completed',
            'best_finish', 'top_10_rate', 'top_5_rate', 'win_rate', 
            'dnf_rate', 'consistency', 'recent_form', 'watkins_races',
            'watkins_best_finish', 'watkins_top_10_rate'
        ]
        
        X = self.driver_features[feature_cols].fillna(0)
        
        # Get predictions
        predicted_finishes = self.model.predict(X)
        
        # Calculate prediction confidence (inverse of standard deviation of tree predictions)
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        prediction_std = np.std(tree_predictions, axis=0)
        confidence = 1 / (1 + prediction_std)  # Higher confidence = lower std
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Driver': self.driver_features['Driver'],
            'Predicted_Finish': predicted_finishes,
            'Confidence': confidence,
            'Road_Course_Avg': self.driver_features['avg_finish'],
            'Watkins_Glen_Avg': self.driver_features['watkins_avg_finish'],
            'Top_10_Rate': self.driver_features['top_10_rate'],
            'Recent_Form': self.driver_features['recent_form']
        })
        
        # Sort by predicted finish and get top 10
        self.predictions = predictions_df.sort_values('Predicted_Finish').head(10).reset_index(drop=True)
        self.predictions['Predicted_Position'] = range(1, 11)
        
        # Convert confidence to probability (normalize)
        total_confidence = self.predictions['Confidence'].sum()
        self.predictions['Probability'] = self.predictions['Confidence'] / total_confidence
        
        return self.predictions
    
    def create_visualization(self):
        """Create visualization of predictions"""
        print("📊 Creating visualization...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NASCAR Watkins Glen International - Top 10 Predictions', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Top 10 Predictions with Probabilities
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 10))
        bars1 = ax1.barh(range(10), self.predictions['Probability'], 
                        color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_yticks(range(10))
        ax1.set_yticklabels([f"{i+1}. {driver}" for i, driver in 
                           enumerate(self.predictions['Driver'])])
        ax1.set_xlabel('Prediction Probability')
        ax1.set_title('Top 10 Predicted Finishers', fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars1, self.predictions['Probability'])):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', ha='left', fontweight='bold')
        
        ax1.invert_yaxis()
        
        # 2. Predicted Finish Position vs Historical Average
        ax2.scatter(self.predictions['Road_Course_Avg'], 
                   self.predictions['Predicted_Finish'],
                   s=self.predictions['Probability']*1000,
                   c=colors, alpha=0.7, edgecolors='black')
        
        # Add diagonal line
        min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
        max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('Historical Road Course Average Finish')
        ax2.set_ylabel('Predicted Finish Position')
        ax2.set_title('Prediction vs Historical Performance', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add driver labels to points
        for i, driver in enumerate(self.predictions['Driver']):
            ax2.annotate(driver.split()[-1], # Last name only
                        (self.predictions['Road_Course_Avg'].iloc[i], 
                         self.predictions['Predicted_Finish'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 3. Feature Importance
        top_features = self.feature_importance.head(8)
        bars3 = ax3.barh(range(len(top_features)), top_features['importance'],
                        color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
        ax3.set_xlabel('Feature Importance')
        ax3.set_title('Most Important Prediction Factors', fontweight='bold', pad=20)
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()
        
        # Add importance values on bars
        for bar, importance in zip(bars3, top_features['importance']):
            ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', va='center', ha='left', fontweight='bold')
        
        # 4. Driver Statistics Summary
        stats_data = self.predictions[['Driver', 'Top_10_Rate', 'Recent_Form', 'Probability']].copy()
        
        # Create a heatmap of normalized statistics
        stats_matrix = stats_data.set_index('Driver')[['Top_10_Rate', 'Recent_Form']].T
        
        # Normalize for better visualization (lower recent_form is better)
        stats_matrix.loc['Recent_Form'] = 1 / (1 + stats_matrix.loc['Recent_Form'])
        
        sns.heatmap(stats_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   ax=ax4, cbar_kws={'label': 'Performance Score (Higher = Better)'})
        ax4.set_title('Driver Performance Heatmap', fontweight='bold', pad=20)
        ax4.set_xlabel('Predicted Top 10 Drivers')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('watkins_glen_predictions.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        plt.show()
        
        return fig
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "="*60)
        print("🏆 WATKINS GLEN INTERNATIONAL - TOP 10 PREDICTIONS")
        print("="*60)
        
        for i, row in self.predictions.iterrows():
            print(f"{row['Predicted_Position']:2d}. {row['Driver']:<25} "
                  f"(Prob: {row['Probability']:5.1%}, "
                  f"Road Avg: {row['Road_Course_Avg']:5.1f}, "
                  f"WG Avg: {row['Watkins_Glen_Avg']:5.1f})")
        
        print("\n" + "-"*60)
        print("📊 MODEL INSIGHTS")
        print("-"*60)
        print(f"Total drivers analyzed: {len(self.driver_features)}")
        print(f"Road course races analyzed: {len(self.road_data)}")
        print(f"Top prediction factors:")
        
        for i, row in self.feature_importance.head(5).iterrows():
            print(f"  • {row['feature'].replace('_', ' ').title()}: {row['importance']:.3f}")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🏁 Starting NASCAR Watkins Glen Prediction Analysis")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Engineer features
        self.engineer_features()
        
        # Train model
        self.train_model()
        
        # Make predictions
        self.predict_top_10()
        
        # Print results
        self.print_results()
        
        # Create visualization
        self.create_visualization()
        
        print("\n✅ Analysis complete! Check 'watkins_glen_predictions.png' for visualization.")
        
        return self.predictions

# Run the analysis
if __name__ == "__main__":
    predictor = WatkinsGlenPredictor()
    predictions = predictor.run_full_analysis()