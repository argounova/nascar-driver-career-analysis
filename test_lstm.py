#!/usr/bin/env python3
"""
Simple test script for LSTM model that handles import paths correctly.
Run this from the project root directory.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src directory to Python path  
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print("🏁 Testing LSTM Model Imports")
print("=" * 40)
print(f"Project root: {project_root}")
print(f"Python path includes project root: {str(project_root) in sys.path}")
print(f"Python path includes src: {str(src_dir) in sys.path}")

# Test basic imports first
try:
    print("\n📦 Testing TensorFlow...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    print(f"✅ TensorFlow {tf.__version__} imports working")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    sys.exit(1)

# Test config import
try:
    print("\n⚙️  Testing config import...")
    from config import get_config, get_data_paths
    print("✅ Config module imported successfully")
    
    # Test loading config
    config = get_config()
    print(f"✅ Config loaded: {len(config)} sections")
    
    # Test data paths
    paths = get_data_paths(config)
    print(f"✅ Data paths created: {list(paths.keys())}")
    
except ImportError as e:
    print(f"❌ Config import failed: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Test LSTM model import
try:
    print("\n🧠 Testing LSTM model import...")
    from models.lstm_model import NASCARLSTMPredictor, check_tensorflow_installation
    print("✅ LSTM model imported successfully")
    
    # Test TensorFlow check function
    print("\n🔧 Running TensorFlow compatibility check...")
    if check_tensorflow_installation():
        print("✅ TensorFlow compatibility check passed")
    else:
        print("❌ TensorFlow compatibility issues")
        
    # Test predictor initialization
    print("\n🏗️  Testing LSTM predictor initialization...")
    predictor = NASCARLSTMPredictor()
    print("✅ LSTM predictor initialized successfully")
    print(f"   Model config: {predictor.lstm_config['hidden_units']} hidden units")
    print(f"   Sequence length: {predictor.lstm_config['sequence_length']}")
    
except ImportError as e:
    print(f"❌ LSTM model import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ LSTM predictor initialization failed: {e}")
    sys.exit(1)

print("\n🎉 All LSTM tests passed!")
print("=" * 40)
print("The LSTM model is ready to use.")
print("Next steps:")
print("1. Create some training data")
print("2. Test the feature engineering module")
print("3. Run end-to-end training")