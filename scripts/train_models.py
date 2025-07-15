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
        # STEP 5: MODEL EVALUATION AND EXAMPLES
        # =================================================================
        step5_start = time.time()
        logger.info("📈 STEP 5: Model evaluation and examples")
        print(f"\n📈 STEP 5: Model Evaluation and Examples")
        print("-" * 40)
        
        # Generate example predictions
        print("Generating example career predictions...")
        
        example_drivers = ['Kyle Larson', 'Chase Elliott', 'Denny Hamlin', 'William Byron', 'Ryan Blaney']
        prediction_examples = []
        
        for driver in example_drivers:
            try:
                # Skip individual predictions for now - function not implemented
                prediction = {'error': 'Individual prediction function not implemented'}
                
                if 'error' not in prediction:
                    prediction_examples.append(prediction)
                    
                    print(f"\n{driver}:")
                    print(f"  Current Season: {prediction['current_season']}")
                    print(f"  Career Seasons: {prediction['career_seasons']}")
                    print(f"  Current Stats: {prediction['current_stats']['avg_finish']:.1f} avg finish, {prediction['current_stats']['win_rate']:.1%} win rate")
                    print(f"  Performance Trend: {prediction['interpretation']['performance_trend']}")
                    print(f"  Career Stage: {prediction['interpretation']['career_stage']}")
                    
                    # Show clustering archetype
                    try:
                        archetype_info = clustering_analyzer.get_driver_archetype(driver)
                        if 'error' not in archetype_info:
                            print(f"  Driver Archetype: {archetype_info['archetype']}")
                    except:
                        pass
                else:
                    print(f"\n{driver}: {prediction['error']}")
                    
            except Exception as e:
                logger.warning(f"Failed to predict for {driver}: {e}")
                print(f"\n{driver}: Prediction failed - {e}")
        
        step5_time = time.time() - step5_start
        print(f"\n✅ Step 5 completed in {step5_time:.1f} seconds")
        
        # =================================================================
        # STEP 6: RESULTS EXPORT AND REPORTING
        # =================================================================
        if save_all_results:
            step6_start = time.time()
            logger.info("💾 STEP 6: Exporting results and reports")
            print(f"\n💾 STEP 6: Results Export and Reporting")
            print("-" * 40)
            
            # Save comprehensive results
            print("Saving comprehensive analysis results...")
            save_comprehensive_results(results, prediction_examples)
            
            step6_time = time.time() - step6_start
            print(f"\n✅ Step 6 completed in {step6_time:.1f} seconds")
        
        # =================================================================
        # COMPLETION SUMMARY
        # =================================================================
        total_time = time.time() - start_time
        results['execution_time'] = total_time
        results['success'] = True
        
        print(f"\n🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total Execution Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Data: {summary['driver_seasons']['total_driver_seasons']} driver-seasons")
        print(f"Features: {feature_summary['total_features']} engineered features")
        print(f"Clusters: {len(cluster_analysis)} driver archetypes")
        print(f"LSTM: {len(lstm_predictor.X_train)} training sequences")
        
        logger.info(f"Analysis pipeline completed successfully in {total_time:.1f} seconds")
        
        return results
        
    except Exception as e:
        error_msg = f"Pipeline failed at step: {e}"
        logger.error(error_msg, exc_info=True)
        results['error'] = error_msg
        print(f"\n❌ PIPELINE FAILED: {error_msg}")
        return results


def save_comprehensive_results(results: Dict, prediction_examples: List[Dict]) -> None:
    """
    Save comprehensive analysis results to files.
    
    Args:
        results (Dict): Analysis results
        prediction_examples (List[Dict]): Example predictions
    """
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save prediction examples
    if prediction_examples:
        predictions_df = pd.DataFrame(prediction_examples)
        predictions_path = outputs_dir / f'career_predictions_{timestamp}.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  Career predictions saved to {predictions_path}")
    
    # Save cluster analysis
    if results['clustering_analyzer'] is not None:
        cluster_path = outputs_dir / f'cluster_analysis_{timestamp}.csv'
        results['clustering_analyzer'].cluster_analysis.to_csv(cluster_path, index=False)
        print(f"  Cluster analysis saved to {cluster_path}")
    
    # Save model evaluation
    if results['lstm_predictor'] is not None:
        try:
            evaluation = results['lstm_predictor'].evaluate_model()
            eval_df = pd.DataFrame(evaluation).T
            eval_path = outputs_dir / f'lstm_evaluation_{timestamp}.csv'
            eval_df.to_csv(eval_path)
            print(f"  LSTM evaluation saved to {eval_path}")
        except:
            pass
    
    # Create summary report
    create_summary_report(results, outputs_dir / f'analysis_summary_{timestamp}.txt')


def create_summary_report(results: Dict, report_path: Path) -> None:
    """
    Create a comprehensive text summary report.
    
    Args:
        results (Dict): Analysis results
        report_path (Path): Path to save report
    """
    with open(report_path, 'w') as f:
        f.write("NASCAR DRIVER CAREER ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Time: {results['execution_time']:.1f} seconds\n\n")
        
        # Data summary
        if results['data_loader']:
            summary = results['data_loader'].get_data_summary()
            f.write("DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            for section, stats in summary.items():
                f.write(f"{section.replace('_', ' ').title()}:\n")
                for key, value in stats.items():
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
        
        # Feature engineering summary
        if results['feature_engineer']:
            feature_summary = results['feature_engineer'].get_feature_summary()
            f.write("FEATURE ENGINEERING SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Features: {feature_summary['total_features']}\n")
            f.write(f"Driver-Seasons: {feature_summary['total_driver_seasons']}\n")
            f.write(f"Season Range: {feature_summary['season_range']}\n\n")
            
            f.write("Feature Categories:\n")
            for category, count in feature_summary['feature_categories'].items():
                if count > 0:
                    f.write(f"  {category.replace('_', ' ').title()}: {count}\n")
            f.write("\n")
        
        # Clustering results
        if results['clustering_analyzer']:
            cluster_analysis = results['clustering_analyzer'].cluster_analysis
            f.write("DRIVER ARCHETYPE CLUSTERING\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of Clusters: {len(cluster_analysis)}\n\n")
            
            for idx, row in cluster_analysis.iterrows():
                f.write(f"{row['archetype']}:\n")
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
    
    # Run analysis
    try:
        results = run_complete_analysis(
            skip_data_check=skip_data_check,
            save_all_results=not no_save
        )
        
        if results['success']:
            print("\n🎉 Analysis completed successfully!")
            print("Check the 'outputs' directory for detailed results.")
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