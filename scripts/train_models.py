#!/usr/bin/env python3
"""
Complete NASCAR Driver Analysis Training Script

This script orchestrates the complete machine learning pipeline:
1. Data loading and processing
2. Feature engineering
3. Driver clustering analysis
4. LSTM career trajectory prediction
5. Model evaluation and visualization
6. Results export and reporting

Run this from the project root directory.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import our modules
from config import get_config, get_data_paths
from data.data_loader import load_nascar_data
from data.feature_engineering import create_nascar_features
from models.clustering import run_clustering_analysis
from models.lstm_model import NASCARLSTMPredictor


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Set up comprehensive logging for the training pipeline.
    
    Args:
        log_level (str): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'nascar_training_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('nascar_training')
    logger.info(f"Training session started at {datetime.now()}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def check_project_setup() -> bool:
    """
    Verify that the project is properly set up.
    
    Returns:
        bool: True if setup is complete
    """
    required_dirs = [
        'config', 'src', 'src/data', 'src/models', 'src/visualization',
        'data', 'data/raw', 'data/processed', 'data/models', 'outputs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   {dir_path}")
        print("\nCreate them with: mkdir -p " + " ".join(missing_dirs))
        return False
    
    # Check for config file
    if not Path('config/config.yaml').exists():
        print("❌ Missing config/config.yaml")
        return False
    
    return True


def run_complete_analysis(config_path: Optional[str] = None,
                         skip_data_check: bool = False,
                         save_all_results: bool = True) -> Dict:
    """
    Run the complete NASCAR driver analysis pipeline.
    
    Args:
        config_path (Optional[str]): Path to config file
        skip_data_check (bool): Skip data freshness check
        save_all_results (bool): Save all intermediate and final results
        
    Returns:
        Dict: Analysis results and model objects
    """
    logger = logging.getLogger('nascar_training')
    start_time = time.time()
    
    results = {
        'success': False,
        'error': None,
        'data_loader': None,
        'feature_engineer': None,
        'clustering_analyzer': None,
        'lstm_predictor': None,
        'execution_time': 0
    }
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING AND VALIDATION
        # =================================================================
        logger.info("🏁 STEP 1: Loading and validating NASCAR data")
        print("=" * 60)
        print("🏁 NASCAR DRIVER CAREER ANALYSIS PIPELINE")
        print("=" * 60)
        print("\n📊 STEP 1: Data Loading and Validation")
        print("-" * 40)
        
        # Load NASCAR data
        print("Loading NASCAR Cup Series data...")
        data_loader = load_nascar_data(config_path)
        results['data_loader'] = data_loader
        
        # Check data freshness
        if not skip_data_check:
            freshness = data_loader.check_data_freshness()
            print(f"Data status: {freshness.get('status', 'Unknown')}")
            
            if freshness.get('days_old', 0) > 30:
                print("⚠️  Data is more than 30 days old. Consider updating with update_data_from_r()")
        
        # Display data summary
        summary = data_loader.get_data_summary()
        print("\nData Summary:")
        for section, stats in summary.items():
            print(f"  {section.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"    {key.replace('_', ' ').title()}: {value}")
        
        step1_time = time.time() - start_time
        print(f"\n✅ Step 1 completed in {step1_time:.1f} seconds")
        
        # =================================================================
        # STEP 2: FEATURE ENGINEERING
        # =================================================================
        step2_start = time.time()
        logger.info("🔧 STEP 2: Advanced feature engineering")
        print(f"\n🔧 STEP 2: Advanced Feature Engineering")
        print("-" * 40)
        
        # Create engineered features
        print("Engineering advanced features...")
        feature_engineer = create_nascar_features(config_path, save_results=save_all_results)
        results['feature_engineer'] = feature_engineer
        
        # Display feature summary
        feature_summary = feature_engineer.get_feature_summary()
        print(f"\nFeature Engineering Summary:")
        print(f"  Total Features: {feature_summary['total_features']}")
        print(f"  Driver-Seasons: {feature_summary['total_driver_seasons']}")
        print(f"  Season Range: {feature_summary['season_range']}")
        
        print("\n  Feature Categories:")
        for category, count in feature_summary['feature_categories'].items():
            if count > 0:
                print(f"    {category.replace('_', ' ').title()}: {count}")
        
        step2_time = time.time() - step2_start
        print(f"\n✅ Step 2 completed in {step2_time:.1f} seconds")
        
        # =================================================================
        # STEP 3: DRIVER CLUSTERING ANALYSIS
        # =================================================================
        step3_start = time.time()
        logger.info("🏷️ STEP 3: Driver archetype clustering")
        print(f"\n🏷️ STEP 3: Driver Archetype Clustering")
        print("-" * 40)
        
        # Run clustering analysis
        print("Analyzing driver archetypes with K-means clustering...")
        clustering_analyzer = run_clustering_analysis(config_path, save_results=save_all_results)
        results['clustering_analyzer'] = clustering_analyzer
        
        # Display cluster results
        cluster_analysis = clustering_analyzer.cluster_analysis
        print(f"\nDiscovered {len(cluster_analysis)} Driver Archetypes:")
        for idx, row in cluster_analysis.iterrows():
            print(f"  {row['archetype']}: {row['driver_count']} drivers")
            print(f"    Avg Wins/Season: {row['avg_wins_per_season']:.2f}")
            print(f"    Avg Finish: {row['avg_finish']:.1f}")
            print(f"    Top-5 Rate: {row['avg_top5_rate']:.1%}")
        
        step3_time = time.time() - step3_start
        print(f"\n✅ Step 3 completed in {step3_time:.1f} seconds")
        
        # =================================================================
        # STEP 4: LSTM CAREER PREDICTION
        # =================================================================
        step4_start = time.time()
        logger.info("🧠 STEP 4: LSTM career trajectory prediction")
        print(f"\n🧠 STEP 4: LSTM Career Trajectory Prediction")
        print("-" * 40)
        
        # Train LSTM model
        print("Training LSTM neural network for career prediction...")
        print("This may take several minutes depending on your hardware...")
        
        # Train LSTM manually using the class
        print("Training LSTM neural network...")
        
        # Initialize predictor
        lstm_predictor = NASCARLSTMPredictor()
        
        # Load sequences from feature engineer
        sequences, targets, driver_names = feature_engineer.lstm_sequences
        lstm_predictor.load_sequences(sequences, targets, driver_names)
        
        # Prepare targets
        prediction_targets = lstm_predictor.prepare_prediction_targets(feature_engineer.engineered_features)
        
        # Preprocess data
        lstm_predictor.preprocess_data(sequences, prediction_targets)
        
        # Build and train model
        lstm_predictor.build_model()
        history = lstm_predictor.train_model()
        
        # Save model
        if save_all_results:
            lstm_predictor.save_model()
        results['lstm_predictor'] = lstm_predictor
        
        step4_time = time.time() - step4_start
        print(f"\n✅ Step 4 completed in {step4_time:.1f} seconds")

        # =================================================================
        # STEP 5: FINISHING POSITION PREDICTOR  
        # =================================================================
        step5_start = time.time()
        logger.info("STEP 5: Finishing position prediction model")
        print(f"\nSTEP 5: Finishing Position Prediction")
        print("-" * 40)
        
        # Train finishing position predictor
        print("Training linear regression model for finish position prediction...")
        
        # Import the model
        from models.finish_position_predictor import FinishPositionPredictor
        
        # Initialize predictor
        finish_predictor = FinishPositionPredictor()
        
        # Get the raw NASCAR data for training
        raw_data = data_loader.df  # This should be your full cup_series DataFrame
        
        # Train the model
        print("Creating training examples from NASCAR race data...")
        training_results = finish_predictor.train(
            raw_data, 
            min_history_races=10,
            seasons_to_predict=list(range(2015, 2025))  # Use recent seasons
        )
        
        results['finish_predictor'] = finish_predictor
        results['finish_training_results'] = training_results
        
        # Display results
        print(f"\nFinishing Position Predictor Results:")
        print(f"  Training Examples: {training_results['training_samples']}")
        print(f"  Validation Examples: {training_results['validation_samples']}")
        print(f"  Drivers Analyzed: {training_results['drivers_count']}")
        print(f"  Validation MAE: {training_results['val_mae']:.2f} positions")
        print(f"  Validation R²: {training_results['val_r2']:.3f}")
        
        print(f"\n  Top Feature Importance:")
        feature_importance = training_results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"    {feature}: {importance:.3f}")
        
        # Save the model
        model_path = Path('data/models/finish_position_predictor.pkl')
        finish_predictor.save(str(model_path))
        print(f"  Model saved to: {model_path}")
        
        step5_time = time.time() - step5_start
        print(f"\n✅ Step 5 completed in {step5_time:.1f} seconds")
        
        # =================================================================
        # STEP 6: MODEL EXAMPLES AND DEMONSTRATIONS
        # =================================================================
        step6_start = time.time()
        logger.info("📊 STEP 6: Model demonstrations and examples")
        print(f"\n📊 STEP 6: Model Demonstrations")
        print("-" * 40)
        
        # Demonstrate finish position predictor with popular drivers
        popular_drivers = ["Kyle Larson", "Chase Elliott", "Denny Hamlin", "Joey Logano", "Martin Truex Jr."]
        
        print("Finish Position Prediction Examples:")
        for driver in popular_drivers:
            try:
                prediction = finish_predictor.predict_for_driver(
                    raw_data, 
                    driver, 
                    next_track_name="Charlotte Motor Speedway",
                    next_track_length=1.5
                )
                print(f"  {driver}:")
                print(f"    Predicted finish: {prediction['predicted_finish']}")
                print(f"    Confidence: {prediction['confidence']:.1%}")
                print(f"    Position improvement: {prediction['position_improvement']:+d}")
                
            except Exception as e:
                print(f"    Could not predict for {driver}: {str(e)}")
        
        step6_time = time.time() - step6_start
        print(f"\n✅ Step 6 completed in {step6_time:.1f} seconds")

def generate_summary_report(results: Dict, output_path: Path) -> None:
    """Generate comprehensive summary report."""
    report_path = output_path / 'training_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("NASCAR DRIVER ANALYSIS TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data info
        if results['data_loader']:
            summary = results['data_loader'].get_data_summary()
            f.write("DATA SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Records: {summary['raw_data']['total_records']}\n")
            f.write(f"Seasons: {summary['raw_data']['season_range']}\n")
            f.write(f"Unique Drivers: {summary['raw_data']['unique_drivers']}\n")
            f.write(f"Driver-Seasons: {summary['aggregated']['driver_seasons']}\n\n")
        
        # Feature engineering info
        if results['feature_engineer']:
            feature_summary = results['feature_engineer'].get_feature_summary()
            f.write("FEATURE ENGINEERING\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Features: {feature_summary['total_features']}\n")
            f.write(f"Driver-Seasons: {feature_summary['total_driver_seasons']}\n")
            f.write(f"Season Range: {feature_summary['season_range']}\n\n")
        
        # Clustering info
        if results['clustering_analyzer']:
            cluster_analysis = results['clustering_analyzer'].cluster_analysis
            f.write("DRIVER ARCHETYPES (CLUSTERING)\n")
            f.write("-" * 35 + "\n")
            f.write(f"Number of Clusters: {len(cluster_analysis)}\n\n")
            
            for idx, row in cluster_analysis.iterrows():
                f.write(f"{row['archetype'].upper()}\n")
                f.write(f"  Drivers: {row['driver_count']}\n")
                f.write(f"  Avg Wins/Season: {row['avg_wins_per_season']:.2f}\n")
                f.write(f"  Avg Finish: {row['avg_finish']:.1f}\n")
                f.write(f"  Top-5 Rate: {row['avg_top5_rate']:.1%}\n")
                f.write(f"  Representatives: {row['representative_drivers']}\n\n")
        
        # LSTM model info
        if results['lstm_predictor']:
            f.write("LSTM CAREER PREDICTION MODEL\n")
            f.write("-" * 35 + "\n")
            f.write(f"Training Sequences: {len(results['lstm_predictor'].X_train)}\n")
            f.write(f"Validation Sequences: {len(results['lstm_predictor'].X_val)}\n")
            f.write(f"Test Sequences: {len(results['lstm_predictor'].X_test)}\n")
            f.write(f"Sequence Length: {results['lstm_predictor'].lstm_config['sequence_length']} seasons\n")
            f.write(f"Hidden Units: {results['lstm_predictor'].lstm_config['hidden_units']}\n\n")
        
        # Finishing Position Predictor info
        if results.get('finish_predictor') and results.get('finish_training_results'):
            training_results = results['finish_training_results']
            f.write("FINISHING POSITION PREDICTOR\n")
            f.write("-" * 35 + "\n")
            f.write(f"Model Type: Linear Regression\n")
            f.write(f"Training Examples: {training_results['training_samples']}\n")
            f.write(f"Validation Examples: {training_results['validation_samples']}\n")
            f.write(f"Drivers Analyzed: {training_results['drivers_count']}\n")
            f.write(f"Validation MAE: {training_results['val_mae']:.2f} positions\n")
            f.write(f"Validation R²: {training_results['val_r2']:.3f}\n")
            f.write(f"Seasons Used: {training_results['seasons_used'][0]}-{training_results['seasons_used'][-1]}\n\n")
            
            f.write("Top Features by Importance:\n")
            feature_importance = training_results['feature_importance']
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, importance in sorted_features:
                f.write(f"  {feature}: {importance:.3f}\n")
            f.write("\n")
    
    print(f"  Summary report saved to {report_path}")

def main():
    """Main execution function."""
    print("🏁 NASCAR Driver Career Analysis Pipeline")
    print("Starting comprehensive machine learning analysis...")
    
    # Check project setup
    if not check_project_setup():
        print("❌ Project setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    # Parse command line arguments (basic)
    skip_data_check = '--skip-data-check' in sys.argv
    no_save = '--no-save' in sys.argv
    finish_only = '--finish-predictor-only' in sys.argv  # New flag for testing
    
    # Run analysis
    try:
        results = run_complete_analysis(
            skip_data_check=skip_data_check,
            save_all_results=not no_save,
            finish_predictor_only=finish_only  # Pass flag through
        )
        
        if results['success']:
            print("\n🎉 Analysis completed successfully!")
            print("Check the 'outputs' directory for detailed results.")
            print("🎯 Finishing Position Predictor ready for FastAPI integration!")
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            print(f"\n❌ Analysis failed: {results['error']}")
            logger.error(f"Pipeline failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()