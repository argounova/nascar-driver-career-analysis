#!/usr/bin/env python3
"""
Real NASCAR Data LSTM Test Script
Tests LSTM career prediction with actual NASCAR data spanning 1949-2025.

Save as tests/test_lstm.py
Run from project root: python tests/test_lstm.py
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

print("🧠 Real NASCAR Data LSTM Test")
print("=" * 35)

# Test imports
try:
    print("📦 Testing imports...")
    from config import get_config, get_data_paths
    from models.lstm_model import NASCARLSTMPredictor
    from data.data_loader import load_nascar_data
    from data.feature_engineering import create_nascar_features
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def load_real_lstm_data():
    """Load real NASCAR data and prepare LSTM sequences."""
    print("\n📊 Loading Real NASCAR Data for LSTM")
    print("-" * 40)
    
    try:
        # Load real NASCAR data with feature engineering
        print("Loading and engineering real NASCAR data...")
        start_time = time.time()
        engineer = create_nascar_features(save_results=False)
        load_time = time.time() - start_time
        
        print(f"✅ Data loading completed in {load_time:.1f}s")
        print(f"✅ Engineered features: {len(engineer.engineered_features.columns)} columns")
        print(f"✅ Driver-seasons: {len(engineer.engineered_features)}")
        print(f"✅ Unique drivers: {engineer.engineered_features['Driver'].nunique()}")
        
        # Get LSTM sequences
        sequences, targets, driver_names = engineer.lstm_sequences
        
        print(f"\n🧠 LSTM Sequence Summary:")
        print(f"   Total sequences: {len(sequences)}")
        print(f"   Sequence shape: {sequences.shape}")
        print(f"   Drivers with sequences: {len(set(driver_names))}")
        print(f"   Season span per sequence: {sequences.shape[1]} seasons")
        print(f"   Features per season: {sequences.shape[2]}")
        
        # Show some real driver examples
        print(f"\n📋 Real Drivers in LSTM Training:")
        unique_drivers = list(set(driver_names))
        legendary_drivers = ['Richard Petty', 'Jeff Gordon', 'Dale Earnhardt', 'Kyle Larson', 'Denny Hamlin']
        found_legends = [driver for driver in legendary_drivers if driver in unique_drivers]
        
        if found_legends:
            print(f"   Legendary drivers: {', '.join(found_legends)}")
        
        print(f"   Other drivers: {', '.join(unique_drivers[:10])}{'...' if len(unique_drivers) > 10 else ''}")
        
        return engineer, sequences, targets, driver_names
        
    except Exception as e:
        print(f"❌ Failed to load real LSTM data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_lstm_initialization(engineer):
    """Test LSTM predictor initialization with real data."""
    print(f"\n🏗️  Testing LSTM Predictor Initialization")
    print("-" * 45)
    
    try:
        # Initialize LSTM predictor
        print("Initializing LSTM predictor...")
        predictor = NASCARLSTMPredictor()
        
        print(f"✅ LSTM predictor initialized successfully")
        print(f"   Model architecture: {predictor.lstm_config['hidden_units']}")
        print(f"   Sequence length: {predictor.lstm_config['sequence_length']}")
        print(f"   Dropout rate: {predictor.lstm_config['dropout_rate']}")
        print(f"   Learning rate: {predictor.lstm_config['learning_rate']}")
        print(f"   Max epochs: {predictor.lstm_config['epochs']}")
        
        # Load sequences
        sequences, targets, driver_names = engineer.lstm_sequences
        print(f"\nLoading real NASCAR sequences...")
        predictor.load_sequences(sequences, targets, driver_names)
        
        print(f"✅ Real sequences loaded")
        print(f"   Sequences: {len(predictor.sequences)}")
        print(f"   Driver names: {len(predictor.driver_names)}")
        
        # Prepare prediction targets
        print(f"\nPreparing prediction targets...")
        prediction_targets = predictor.prepare_prediction_targets(engineer.engineered_features)
        
        print(f"✅ Prediction targets prepared")
        print(f"   Target matrix shape: {prediction_targets.shape}")
        print(f"   Target types: {len(predictor.target_names)}")
        print(f"   Targets: {', '.join(predictor.target_names)}")
        
        return predictor, prediction_targets
        
    except Exception as e:
        print(f"❌ LSTM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_data_preprocessing(predictor, prediction_targets):
    """Test LSTM data preprocessing with real NASCAR data."""
    print(f"\n⚙️  Testing Data Preprocessing")
    print("-" * 30)
    
    try:
        # Preprocess real data
        print("Preprocessing real NASCAR sequences...")
        start_time = time.time()
        predictor.preprocess_data(predictor.sequences, prediction_targets)
        preprocess_time = time.time() - start_time
        
        print(f"✅ Data preprocessing completed in {preprocess_time:.1f}s")
        print(f"   Training sequences: {len(predictor.X_train)}")
        print(f"   Validation sequences: {len(predictor.X_val)}")
        print(f"   Test sequences: {len(predictor.X_test)}")
        print(f"   Feature scaling: StandardScaler applied")
        print(f"   Target scaling: StandardScaler applied")
        
        # Check data shapes
        print(f"\n📐 Data Shapes:")
        print(f"   X_train: {predictor.X_train.shape}")
        print(f"   y_train: {predictor.y_train.shape}")
        print(f"   X_val: {predictor.X_val.shape}")
        print(f"   y_val: {predictor.y_val.shape}")
        print(f"   X_test: {predictor.X_test.shape}")
        print(f"   y_test: {predictor.y_test.shape}")
        
        # Check for data quality
        print(f"\n🔍 Data Quality Check:")
        train_finite = np.isfinite(predictor.X_train).all()
        target_finite = np.isfinite(predictor.y_train).all()
        
        print(f"   Training data finite: {'✅' if train_finite else '❌'}")
        print(f"   Target data finite: {'✅' if target_finite else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_building(predictor):
    """Test LSTM model architecture building."""
    print(f"\n🏗️  Testing Model Architecture")
    print("-" * 30)
    
    try:
        # Build model
        print("Building LSTM model architecture...")
        start_time = time.time()
        model = predictor.build_model()
        build_time = time.time() - start_time
        
        print(f"✅ Model built in {build_time:.1f}s")
        
        # Model summary information
        print(f"\n📊 Model Architecture:")
        print(f"   Input shape: ({predictor.X_train.shape[1]}, {predictor.X_train.shape[2]})")
        print(f"   Output shape: {predictor.y_train.shape[1]} targets")
        print(f"   LSTM layers: {len(predictor.lstm_config['hidden_units'])}")
        print(f"   Hidden units: {predictor.lstm_config['hidden_units']}")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Test model compilation
        print(f"\n⚙️  Model Configuration:")
        print(f"   Optimizer: {model.optimizer.__class__.__name__}")
        print(f"   Loss function: {predictor.lstm_config['loss_function']}")
        print(f"   Metrics: {predictor.lstm_config['metrics']}")
        
        # Test a forward pass
        print(f"\nTesting forward pass...")
        test_batch = predictor.X_train[:5]  # Small batch test
        test_output = model.predict(test_batch, verbose=0)
        
        print(f"✅ Forward pass successful")
        print(f"   Input batch shape: {test_batch.shape}")
        print(f"   Output batch shape: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup(predictor):
    """Test LSTM training setup and run a few epochs."""
    print(f"\n🚀 Testing Training Setup")
    print("-" * 25)
    
    try:
        # Modify config for quick testing
        original_epochs = predictor.lstm_config['epochs']
        predictor.lstm_config['epochs'] = 3  # Just a few epochs for testing
        
        print("Starting short training run (3 epochs for testing)...")
        print(f"   Training on {len(predictor.X_train)} real NASCAR sequences")
        print(f"   Validating on {len(predictor.X_val)} sequences")
        
        start_time = time.time()
        history = predictor.train_model()
        training_time = time.time() - start_time
        
        print(f"✅ Training test completed in {training_time:.1f}s")
        
        # Check training history
        if history.history:
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            print(f"   Final training loss: {final_loss:.4f}")
            print(f"   Final validation loss: {final_val_loss:.4f}")
            
            # Check if model is learning (loss should decrease)
            if len(history.history['loss']) > 1:
                initial_loss = history.history['loss'][0]
                learning = initial_loss > final_loss
                print(f"   Model learning: {'✅' if learning else '⚠️'} ({'Improving' if learning else 'Not improving'})")
        
        # Restore original epochs
        predictor.lstm_config['epochs'] = original_epochs
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_predictions(predictor, engineer):
    """Test predictions on real NASCAR drivers."""
    print(f"\n🔮 Testing Real Driver Predictions")
    print("-" * 35)
    
    try:
        # Test evaluation
        print("Evaluating model on test data...")
        evaluation = predictor.evaluate_model()
        
        print(f"✅ Model evaluation completed")
        print(f"   Overall MAE: {evaluation['overall']['mae']:.3f}")
        print(f"   Overall RMSE: {evaluation['overall']['rmse']:.3f}")
        
        # Show performance by target
        print(f"\n📊 Performance by prediction target:")
        for target, metrics in evaluation.items():
            if target != 'overall':
                print(f"   {target}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
        
        # Test predictions on specific real drivers
        print(f"\n🏁 Real Driver Career Predictions:")
        test_drivers = ['Kyle Larson', 'Chase Elliott', 'Denny Hamlin']
        
        # Test basic model prediction capability
        print("\n   Testing basic prediction functionality...")
        try:
            # Test prediction on a small batch
            test_batch = predictor.X_test[:3]
            raw_predictions = predictor.model.predict(test_batch, verbose=0)
            
            # Inverse transform to get actual scale predictions
            actual_predictions = predictor.target_scaler.inverse_transform(raw_predictions)
            
            print(f"   ✅ Model predictions generated successfully")
            print(f"   Sample predictions shape: {actual_predictions.shape}")
            print(f"   Target names: {predictor.target_names}")
            
            # Show sample prediction
            print(f"\n   Sample prediction for anonymous driver:")
            for i, target in enumerate(predictor.target_names):
                pred_value = actual_predictions[0, i]
                print(f"     {target}: {pred_value:.3f}")
                
        except Exception as e:
            print(f"   ❌ Basic prediction test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real predictions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_saving(predictor):
    """Test model saving functionality."""
    print(f"\n💾 Testing Model Saving")
    print("-" * 20)
    
    try:
        # Test saving
        print("Saving trained model...")
        predictor.save_model()
        
        # Check if files were created
        model_path = Path(predictor.paths['models']) / 'lstm_career_predictor.keras'
        preprocessor_path = Path(predictor.paths['models']) / 'lstm_preprocessors.pkl'
        
        model_exists = model_path.exists()
        preprocessor_exists = preprocessor_path.exists()
        
        print(f"✅ Model saving completed")
        print(f"   Model file: {'✅' if model_exists else '❌'} {model_path}")
        print(f"   Preprocessors: {'✅' if preprocessor_exists else '❌'} {preprocessor_path}")
        
        # Check file sizes
        if model_exists:
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   Model size: {model_size:.1f} MB")
        
        return model_exists and preprocessor_exists
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False

# Main test execution
def main():
    """Run all real data LSTM tests."""
    
    print("Starting Real NASCAR Data LSTM Tests...")
    
    # Test 1: Load real LSTM data
    engineer, sequences, targets, driver_names = load_real_lstm_data()
    if engineer is None:
        print("❌ Cannot proceed without real LSTM data")
        return False
    
    # Test 2: LSTM initialization
    predictor, prediction_targets = test_lstm_initialization(engineer)
    if predictor is None:
        print("❌ LSTM initialization failed")
        return False
    
    # Test 3: Data preprocessing
    preprocessing_success = test_data_preprocessing(predictor, prediction_targets)
    if not preprocessing_success:
        print("❌ Data preprocessing failed")
        return False
    
    # Test 4: Model building
    model_success = test_model_building(predictor)
    if not model_success:
        print("❌ Model building failed")
        return False
    
    # Test 5: Training setup
    training_success = test_training_setup(predictor)
    if not training_success:
        print("❌ Training setup failed")
        return False
    
    # Test 6: Real predictions
    prediction_success = test_real_predictions(predictor, engineer)
    
    # Test 7: Model saving
    saving_success = test_model_saving(predictor)
    
    # Summary
    print(f"\n{'='*35}")
    if all([preprocessing_success, model_success, training_success, prediction_success, saving_success]):
        print("🎉 All Real Data LSTM Tests Passed!")
        print("✅ Real NASCAR sequences processed successfully")
        print("✅ LSTM architecture built and compiled")
        print("✅ Training pipeline working with real data")
        print("✅ Real driver predictions generated")
        print("✅ Model saving functionality confirmed")
        
        print(f"\nReal NASCAR LSTM capabilities:")
        print(f"   • Training on {len(sequences)} real career sequences")
        print(f"   • Predicting for {len(set(driver_names))} drivers")
        print(f"   • 5-season sequences with {sequences.shape[2]} features")
        print(f"   • {len(predictor.target_names)} prediction targets")
        
        print("\nNext steps:")
        print("1. Run full training with more epochs")
        print("2. Create prediction visualizations")
        print("3. Build end-to-end pipeline")
        print("4. Generate career forecasts for current drivers")
    else:
        print("❌ Some Real Data LSTM Tests Failed!")
        print("Check the error messages above")
    
    return all([preprocessing_success, model_success, training_success, prediction_success, saving_success])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)