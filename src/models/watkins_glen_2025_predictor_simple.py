#!/usr/bin/env python3
"""
Simple NASCAR Watkins Glen Top 10 Predictor

Usage: python3 src/models/watkins_glen_2025_predictor_simple.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class SimpleWatkinsGlenPredictor:
    
    def __init__(self, csv_path='data/raw/cup_series.csv'):
        self.csv_path = csv_path
        self.model = None
        self.predictions = None
        
    def prepare_data(self):
        print("Loading NASCAR data...")
        data = pd.read_csv(self.csv_path)
        road_data = data[
            (data['Surface'] == 'road') & 
            (data['Season'] >= 2023)
        ].dropna(subset=['Finish', 'Start', 'Driver'])
        
        # Calculate driver statistics
        driver_stats = []
        for driver in road_data['Driver'].unique():
            races = road_data[road_data['Driver'] == driver]
            
            if len(races) < 2:  # Skip drivers with very few races
                continue
                
            # Watkins Glen specific performance
            wg_races = races[races['Track'] == 'Watkins Glen International']
            
            stats = {
                'Driver': driver,
                'avg_finish': races['Finish'].mean(),
                'top_10_rate': (races['Finish'] <= 10).mean(),
                'best_finish': races['Finish'].min(),
                'consistency': races['Finish'].std(),
                'watkins_avg': wg_races['Finish'].mean() if len(wg_races) > 0 else races['Finish'].mean()
            }
            driver_stats.append(stats)
        
        self.driver_data = pd.DataFrame(driver_stats)
        self.driver_data['consistency'].fillna(self.driver_data['consistency'].median(), inplace=True)
        
        print(f"Prepared data for {len(self.driver_data)} drivers")
        return self.driver_data
    
    def train_and_predict(self):
        print("Training model and making predictions...")
        
        # Prepare features
        features = ['avg_finish', 'top_10_rate', 'best_finish', 'consistency']
        X = self.driver_data[features]
        y = self.driver_data['watkins_avg']
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Make predictions
        predicted_finishes = self.model.predict(X)
        
        # Calculate probabilities based on prediction confidence
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        confidence = 1 / (1 + np.std(tree_preds, axis=0))
        
        # Create predictions dataframe
        results = pd.DataFrame({
            'Driver': self.driver_data['Driver'],
            'Predicted_Finish': predicted_finishes,
            'Confidence': confidence
        })
        
        # Get top 10 and calculate probabilities
        self.predictions = results.nsmallest(10, 'Predicted_Finish').reset_index(drop=True)
        self.predictions['Position'] = range(1, 11)
        
        # Normalize confidence to probabilities
        total_confidence = self.predictions['Confidence'].sum()
        self.predictions['Probability'] = self.predictions['Confidence'] / total_confidence
        
        return self.predictions
    
    def create_chart(self):
        print("Creating visualization...")
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        plt.style.use('default')
        
        # Create color gradient (green to red)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 10))
        
        # Create horizontal bar chart
        bars = plt.barh(range(10), self.predictions['Probability'], 
                       color=colors, alpha=0.85, edgecolor='black', linewidth=1)
        
        # Customize the chart
        plt.yticks(range(10), [f"{i+1}. {driver}" for i, driver in 
                              enumerate(self.predictions['Driver'])])
        plt.xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        plt.title('NASCAR Watkins Glen International\nTop 10 Predicted Finishers', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, self.predictions['Probability'])):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', ha='left', fontweight='bold', fontsize=10)
        
        # Style the chart
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        
        # Add subtle background color
        plt.gca().set_facecolor('#f8f9fa')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and show
        plt.savefig('watkins_glen_top10.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return plt.gcf()
    
    def print_results(self):
        print("\n" + "="*50)
        print("WATKINS GLEN TOP 10 PREDICTIONS")
        print("="*50)
        
        for _, row in self.predictions.iterrows():
            print(f"{row['Position']:2d}. {row['Driver']:<25} ({row['Probability']:5.1%})")
        
        print("="*50)
    
    def run(self):
        print("NASCAR Watkins Glen Simple Predictor")
        print("-" * 40)
        
        self.prepare_data()
        self.train_and_predict()
        self.print_results()
        self.create_chart()
        
        print("\n✅ Complete! Chart saved as 'watkins_glen_top10.png'")
        return self.predictions

# Run the predictor
if __name__ == "__main__":
    predictor = SimpleWatkinsGlenPredictor()
    predictions = predictor.run()