#!/usr/bin/env python3
"""
Complete NASCAR Driver Analysis Training Script

This script orchestrates the complete machine learning pipeline:
1. Data loading and processing
2. Feature engineering
3. Driver clustering analysis
4. LSTM career trajectory prediction
5. Finish position prediction (Linear Regression)
6. Driver volatility prediction (Random Forest)
7. Win probability prediction (Logistic Regression)
8. Model evaluation and visualization
9. Results export and reporting

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
from models.finish_position_predictor import FinishPositionPredictor
from models.driver_volatility_predictor import DriverVolatilityPredictor
from models.win_probability_predictor import WinProbabilityPredictor


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
        bool: True if setup is valid
    """
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/models',
        'src/models',
        'outputs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("Please create these directories before running the script.")
        return False
    
    # Check for raw data files (prefer Parquet, fallback to CSV)
    parquet_file = Path('data/raw/cup_series.parquet')
    csv_file = Path('data/raw/cup_series.csv')
    
    if parquet_file.exists():
        print(f"✅ Found NASCAR data: {parquet_file} (Parquet - optimal)")
        return True
    elif csv_file.exists():
        print(f"✅ Found NASCAR data: {csv_file} (CSV - fallback)")
        print("💡 Consider converting to Parquet format for faster loading")
        return True
    else:
        print("❌ Missing raw data files:")
        print(f"   Primary: {parquet_file}")
        print(f"   Fallback: {csv_file}")
        print("Please ensure NASCAR data is available in either format.")
        print("Run 'Rscript scripts/update_data.R' to download and convert data.")
        return False


def run_complete_analysis(skip_data_check: bool = False, save_all_results: bool = True,
                         predictors_only: bool = False) -> Dict:
    """
    Run the complete NASCAR analysis pipeline.
    
    Args:
        skip_data_check: Skip data validation
        save_all_results: Save all intermediate results
        predictors_only: Run only the three new predictors (skip clustering/LSTM)
        
    Returns:
        Dict containing all results and models
    """
    logger = logging.getLogger('nascar_training')
    results = {'success': False}
    total_start_time = time.time()
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING
        # =================================================================
        step1_start = time.time()
        logger.info("🏁 STEP 1: Data loading and validation")
        print(f"\nSTEP 1: Data Loading")
        print("-" * 40)
        
        data_loader = load_nascar_data()
        
        if not skip_data_check:
            print("Validating data integrity...")
            summary = data_loader.get_data_summary()
            print(f"✅ Loaded {summary['raw_data']['total_records']:,} race records")
            print(f"✅ {summary['raw_data']['unique_drivers']} unique drivers")
            print(f"✅ Seasons: {summary['raw_data']['season_range']}")
        
        results['data_loader'] = data_loader
        step1_time = time.time() - step1_start
        print(f"\n✅ Step 1 completed in {step1_time:.1f} seconds")

        # =================================================================
        # STEP 2: FEATURE ENGINEERING  
        # =================================================================
        step2_start = time.time()
        logger.info("🔧 STEP 2: Feature engineering")
        print(f"\nSTEP 2: Feature Engineering")
        print("-" * 40)
        
        print("Creating comprehensive driver features...")
        feature_engineer = create_nascar_features(data_loader)
        
        feature_summary = feature_engineer.get_feature_summary()
        print(f"✅ Generated {feature_summary['total_features']} features")
        print(f"✅ {feature_summary['total_driver_seasons']} driver-seasons")
        
        results['feature_engineer'] = feature_engineer
        step2_time = time.time() - step2_start
        print(f"\n✅ Step 2 completed in {step2_time:.1f} seconds")

        if not predictors_only:
            # =================================================================
            # STEP 3: DRIVER CLUSTERING
            # =================================================================
            step3_start = time.time()
            logger.info("🏷️ STEP 3: Driver archetype clustering")
            print(f"\nSTEP 3: Driver Archetype Clustering")
            print("-" * 40)
            
            print("Analyzing driver performance patterns...")
            clustering_analyzer = run_clustering_analysis(feature_engineer.engineered_features)
            
            print(f"✅ Identified {len(clustering_analyzer.cluster_analysis)} driver archetypes")
            for idx, row in clustering_analyzer.cluster_analysis.iterrows():
                print(f"  - {row['archetype']}: {row['driver_count']} drivers")
            
            results['clustering_analyzer'] = clustering_analyzer
            step3_time = time.time() - step3_start
            print(f"\n✅ Step 3 completed in {step3_time:.1f} seconds")

            # =================================================================
            # STEP 4: LSTM CAREER PREDICTION
            # =================================================================
            step4_start = time.time()
            logger.info("🧠 STEP 4: LSTM career trajectory prediction")
            print(f"\nSTEP 4: LSTM Career Trajectory Prediction")
            print("-" * 40)
            
            print("Training neural network for career prediction...")
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
        # STEP 5: FINISH POSITION PREDICTOR  
        # =================================================================
        step5_start = time.time()
        logger.info("🎯 STEP 5: Finish position prediction model")
        print(f"\nSTEP 5: Finish Position Prediction")
        print("-" * 40)
        
        print("Training linear regression model for finish position prediction...")
        
        # Initialize predictor
        finish_predictor = FinishPositionPredictor()
        
        # Get the raw NASCAR data for training
        raw_data = data_loader.df
        
        # Train the model
        print("Creating training examples from NASCAR race data...")
        finish_training_results = finish_predictor.train(
            raw_data, 
            min_history_races=10,  # Fixed parameter name
            seasons_to_predict=[2020, 2021, 2022, 2023, 2024]
        )
        
        print(f"✅ Finish Position Model trained")
        print(f"  - Training examples: {finish_training_results['training_samples']:,}")
        print(f"  - Validation MAE: {finish_training_results['val_mae']:.2f} positions")
        print(f"  - Validation R²: {finish_training_results['val_r2']:.3f}")
        
        # Save model
        if save_all_results:
            finish_predictor.save('data/models/finish_position_predictor.pkl')
        
        results['finish_predictor'] = finish_predictor
        results['finish_training_results'] = finish_training_results
        
        step5_time = time.time() - step5_start
        print(f"\n✅ Step 5 completed in {step5_time:.1f} seconds")

        # =================================================================
        # STEP 6: DRIVER VOLATILITY PREDICTOR  
        # =================================================================
        step6_start = time.time()
        logger.info("📊 STEP 6: Driver volatility prediction model")
        print(f"\nSTEP 6: Driver Volatility Prediction")
        print("-" * 40)
        
        print("Training random forest model for performance volatility prediction...")
        
        # Initialize predictor
        volatility_predictor = DriverVolatilityPredictor()
        
        # Train the model
        print("Creating training examples from NASCAR race data...")
        volatility_training_results = volatility_predictor.train(
            raw_data,
            min_history_races=15,  # Fixed parameter name
            seasons_to_predict=[2020, 2021, 2022, 2023, 2024]
        )
        
        print(f"✅ Volatility Model trained")
        print(f"  - Training examples: {volatility_training_results['training_samples']:,}")
        print(f"  - Validation MAE: {volatility_training_results['val_mae']:.2f} volatility units")
        print(f"  - Validation R²: {volatility_training_results['val_r2']:.3f}")
        
        # Save model
        if save_all_results:
            volatility_predictor.save('data/models/driver_volatility_predictor.pkl')
        
        results['volatility_predictor'] = volatility_predictor
        results['volatility_training_results'] = volatility_training_results
        
        step6_time = time.time() - step6_start
        print(f"\n✅ Step 6 completed in {step6_time:.1f} seconds")

        # =================================================================
        # STEP 7: WIN PROBABILITY PREDICTOR  
        # =================================================================
        step7_start = time.time()
        logger.info("🏆 STEP 7: Win probability prediction model")
        print(f"\nSTEP 7: Win Probability Prediction")
        print("-" * 40)
        
        print("Training logistic regression model for win probability prediction...")
        
        # Initialize predictor
        win_predictor = WinProbabilityPredictor()
        
        # Train the model
        print("Creating training examples from NASCAR race data...")
        win_training_results = win_predictor.train(
            raw_data,
            min_history_races=10,  # Fixed parameter name
            seasons_to_predict=[2020, 2021, 2022, 2023, 2024]
        )
        
        print(f"✅ Win Probability Model trained")
        print(f"  - Training examples: {win_training_results['training_samples']:,}")
        print(f"  - Validation Accuracy: {win_training_results['val_accuracy']:.3f}")
        print(f"  - Validation AUC: {win_training_results['val_auc']:.3f}")
        print(f"  - Validation Precision: {win_training_results['val_precision']:.3f}")
        
        # Save model
        if save_all_results:
            win_predictor.save('data/models/win_probability_predictor.pkl')
        
        results['win_predictor'] = win_predictor
        results['win_training_results'] = win_training_results
        
        step7_time = time.time() - step7_start
        print(f"\n✅ Step 7 completed in {step7_time:.1f} seconds")

        # =================================================================
        # FINAL STEPS: SAVE RESULTS AND GENERATE REPORTS
        # =================================================================
        total_time = time.time() - total_start_time
        
        if save_all_results:
            print(f"\nSaving comprehensive results...")
            save_training_results(results)
        
        results['success'] = True
        results['total_time'] = total_time
        
        print(f"\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 All models trained and ready for deployment")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed at step: {str(e)}")
        logger.error("Pipeline failed: " + str(e))
        results['error'] = str(e)
        results['success'] = False
        return results


def save_training_results(results: Dict) -> None:
    """
    Save comprehensive training results and generate reports.
    
    Args:
        results: Dictionary containing all training results
    """
    print("\nGenerating comprehensive training report...")
    
    # Create outputs directory
    output_path = Path('outputs')
    output_path.mkdir(exist_ok=True)
    
    # Generate summary report
    report_path = output_path / 'training_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("NASCAR DRIVER ANALYSIS TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data info
        if results.get('data_loader'):
            summary = results['data_loader'].get_data_summary()
            f.write("DATA SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Records: {summary['raw_data']['total_records']}\n")
            f.write(f"Seasons: {summary['raw_data']['season_range']}\n")
            f.write(f"Unique Drivers: {summary['raw_data']['unique_drivers']}\n")
            f.write(f"Driver-Seasons: {summary['aggregated']['driver_seasons']}\n\n")
        
        # Feature engineering info
        if results.get('feature_engineer'):
            feature_summary = results['feature_engineer'].get_feature_summary()
            f.write("FEATURE ENGINEERING\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Features: {feature_summary['total_features']}\n")
            f.write(f"Driver-Seasons: {feature_summary['total_driver_seasons']}\n")
            f.write(f"Season Range: {feature_summary['season_range']}\n\n")
        
        # Clustering info (if run)
        if results.get('clustering_analyzer'):
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
        
        # LSTM model info (if run)
        if results.get('lstm_predictor'):
            f.write("LSTM CAREER PREDICTION MODEL\n")
            f.write("-" * 35 + "\n")
            f.write(f"Training Sequences: {len(results['lstm_predictor'].X_train)}\n")
            f.write(f"Validation Sequences: {len(results['lstm_predictor'].X_val)}\n")
            f.write(f"Test Sequences: {len(results['lstm_predictor'].X_test)}\n")
            f.write(f"Sequence Length: {results['lstm_predictor'].lstm_config['sequence_length']} seasons\n")
            f.write(f"Hidden Units: {results['lstm_predictor'].lstm_config['hidden_units']}\n\n")
        
        # Finish Position Predictor info
        if results.get('finish_predictor') and results.get('finish_training_results'):
            training_results = results['finish_training_results']
            f.write("FINISH POSITION PREDICTOR\n")
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
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for feature, importance in sorted_features:
                f.write(f"  {feature}: {importance:.3f}\n")
            f.write("\n")

        # Driver Volatility Predictor info
        if results.get('volatility_predictor') and results.get('volatility_training_results'):
            training_results = results['volatility_training_results']
            f.write("DRIVER VOLATILITY PREDICTOR\n")
            f.write("-" * 35 + "\n")
            f.write(f"Model Type: Random Forest Regression\n")
            f.write(f"Training Examples: {training_results['training_samples']}\n")
            f.write(f"Validation Examples: {training_results['validation_samples']}\n")
            f.write(f"Drivers Analyzed: {training_results['drivers_count']}\n")
            f.write(f"Validation MAE: {training_results['val_mae']:.2f} volatility units\n")
            f.write(f"Validation R²: {training_results['val_r2']:.3f}\n")
            f.write(f"Validation RMSE: {training_results['val_rmse']:.2f}\n\n")
            
            f.write("Top Features by Importance:\n")
            feature_importance = training_results['feature_importance']
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                f.write(f"  {feature}: {importance:.3f}\n")
            f.write("\n")

        # Win Probability Predictor info
        if results.get('win_predictor') and results.get('win_training_results'):
            training_results = results['win_training_results']
            f.write("WIN PROBABILITY PREDICTOR\n")
            f.write("-" * 35 + "\n")
            f.write(f"Model Type: Logistic Regression\n")
            f.write(f"Training Examples: {training_results['training_samples']}\n")
            f.write(f"Validation Examples: {training_results['validation_samples']}\n")
            f.write(f"Drivers Analyzed: {training_results['drivers_count']}\n")
            f.write(f"Validation Accuracy: {training_results['val_accuracy']:.3f}\n")
            f.write(f"Validation AUC: {training_results['val_auc']:.3f}\n")
            f.write(f"Validation Precision: {training_results['val_precision']:.3f}\n")
            f.write(f"Validation Recall: {training_results['val_recall']:.3f}\n")
            f.write(f"Validation F1: {training_results['val_f1']:.3f}\n\n")
            
            # Win rate in training data
            win_rate = sum(1 for ex in training_results.get('training_examples', []) if ex.get('won', False))
            total_examples = len(training_results.get('training_examples', []))
            if total_examples > 0:
                f.write(f"Training Win Rate: {win_rate/total_examples:.1%}\n\n")

        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 20 + "\n")
        if results.get('total_time'):
            f.write(f"Total Training Time: {results['total_time']:.1f} seconds\n")
            f.write(f"Total Training Time: {results['total_time']/60:.1f} minutes\n")
        f.write(f"Training Status: {'SUCCESS' if results.get('success') else 'FAILED'}\n")
        if not results.get('success') and results.get('error'):
            f.write(f"Error: {results['error']}\n")
    
    print(f"  📄 Summary report saved to {report_path}")
    
    # Save model metadata
    models_info = {}
    for model_key in ['finish_predictor', 'volatility_predictor', 'win_predictor']:
        if results.get(model_key):
            try:
                # Try to get model info if method exists
                if hasattr(results[model_key], 'get_model_info'):
                    models_info[model_key] = results[model_key].get_model_info()
                else:
                    # Create basic info if method doesn't exist
                    model_names = {
                        'finish_predictor': 'Finish Position Predictor',
                        'volatility_predictor': 'Driver Volatility Predictor', 
                        'win_predictor': 'Win Probability Predictor'
                    }
                    models_info[model_key] = {
                        'name': model_names[model_key],
                        'is_trained': getattr(results[model_key], 'is_trained', True),
                        'features': getattr(results[model_key], 'feature_names', []),
                        'training_metrics': getattr(results[model_key], 'training_metrics', {})
                    }
            except Exception as e:
                print(f"  ⚠️  Could not get metadata for {model_key}: {e}")
                continue
    
    if models_info:
        import pickle
        with open(output_path / 'models_metadata.pkl', 'wb') as f:
            pickle.dump(models_info, f)
        print(f"  🔧 Model metadata saved to {output_path / 'models_metadata.pkl'}")


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
    
    # Parse command line arguments
    skip_data_check = '--skip-data-check' in sys.argv
    no_save = '--no-save' in sys.argv
    predictors_only = '--predictors-only' in sys.argv  # New flag to skip clustering/LSTM
    
    # Run analysis
    try:
        results = run_complete_analysis(
            skip_data_check=skip_data_check,
            save_all_results=not no_save,
            predictors_only=predictors_only
        )
        
        if results['success']:
            print("\n🎉 Analysis completed successfully!")
            print("Check the 'outputs' directory for detailed results.")
            
            # Show what models are available
            models_trained = []
            if results.get('finish_predictor'):
                models_trained.append("✅ Finish Position Predictor")
            if results.get('volatility_predictor'):
                models_trained.append("✅ Driver Volatility Predictor")
            if results.get('win_predictor'):
                models_trained.append("✅ Win Probability Predictor")
            if results.get('clustering_analyzer'):
                models_trained.append("✅ Driver Clustering Analysis")
            if results.get('lstm_predictor'):
                models_trained.append("✅ LSTM Career Trajectory Predictor")
            
            print("\nModels ready for deployment:")
            for model in models_trained:
                print(f"  {model}")
            
            print("\n🎯 All predictors ready for FastAPI integration!")
            
        else:
            print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()