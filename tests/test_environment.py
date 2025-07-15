#!/usr/bin/env python3
"""
Environment Test Script for NASCAR Project

This script tests if all required packages are properly installed
and can be imported correctly.
"""

import sys
import subprocess
from pathlib import Path

def test_basic_imports():
    """Test basic Python packages."""
    print("🐍 Testing Basic Python Packages")
    print("-" * 40)
    
    basic_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('plotly', None),
        ('sklearn', None),
        ('yaml', None),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns')
    ]
    
    success_count = 0
    
    for package, alias in basic_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
                print(f"✅ {package} (as {alias})")
            else:
                exec(f"import {package}")
                print(f"✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    print(f"\nBasic packages: {success_count}/{len(basic_packages)} working")
    return success_count == len(basic_packages)


def test_tensorflow_detailed():
    """Detailed TensorFlow testing."""
    print("\n🧠 Testing TensorFlow Installation")
    print("-" * 40)
    
    # Test 1: Basic TensorFlow import
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully")
        print(f"   Version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    # Test 2: Keras imports (multiple methods)
    keras_import_methods = [
        ("tensorflow.keras", "from tensorflow import keras"),
        ("tensorflow.keras.models", "from tensorflow.keras.models import Sequential"),
        ("tensorflow.keras.layers", "from tensorflow.keras.layers import LSTM, Dense"),
        ("standalone keras", "import keras")
    ]
    
    working_methods = []
    
    for method_name, import_statement in keras_import_methods:
        try:
            exec(import_statement)
            print(f"✅ {method_name}")
            working_methods.append(method_name)
        except ImportError as e:
            print(f"❌ {method_name}: {e}")
    
    # Test 3: GPU availability
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✅ GPU available: {len(gpu_devices)} device(s)")
            for i, gpu in enumerate(gpu_devices):
                print(f"   GPU {i}: {gpu}")
        else:
            print("ℹ️  No GPU devices found (using CPU)")
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
    
    # Test 4: Simple TensorFlow operation
    try:
        test_tensor = tf.constant([1, 2, 3, 4])
        result = tf.reduce_sum(test_tensor)
        print(f"✅ TensorFlow operations working: {result.numpy()}")
    except Exception as e:
        print(f"❌ TensorFlow operations failed: {e}")
        return False
    
    return len(working_methods) > 0


def test_project_structure():
    """Test if project structure is correct."""
    print("\n📁 Testing Project Structure")
    print("-" * 40)
    
    required_paths = [
        "config/config.yaml",
        "config/__init__.py",
        "src/",
        "data/",
        "scripts/"
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            missing_paths.append(path)
    
    return len(missing_paths) == 0


def test_python_environment():
    """Test Python environment details."""
    print("\n🔧 Python Environment Details")
    print("-" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {Path.cwd()}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment")
    
    # Check Python path
    print(f"\nPython path includes:")
    for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
        print(f"  {i+1}. {path}")
    if len(sys.path) > 5:
        print(f"  ... and {len(sys.path)-5} more paths")


def install_packages():
    """Attempt to install required packages."""
    print("\n📦 Installing Required Packages")
    print("-" * 40)
    
    try:
        # Check if requirements.txt exists
        if Path("requirements.txt").exists():
            print("Found requirements.txt, installing packages...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package installation completed")
                return True
            else:
                print(f"❌ Package installation failed:")
                print(result.stderr)
                return False
        else:
            print("❌ requirements.txt not found")
            print("Try installing packages manually:")
            print("pip install tensorflow pandas numpy plotly scikit-learn pyyaml")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def main():
    """Run all tests."""
    print("🏁 NASCAR Project Environment Test")
    print("=" * 50)
    
    # Test Python environment
    test_python_environment()
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Test basic imports
    basic_ok = test_basic_imports()
    
    # Test TensorFlow specifically
    tf_ok = test_tensorflow_detailed()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Project Structure: {'✅ OK' if structure_ok else '❌ Issues'}")
    print(f"Basic Packages: {'✅ OK' if basic_ok else '❌ Issues'}")
    print(f"TensorFlow: {'✅ OK' if tf_ok else '❌ Issues'}")
    
    if not basic_ok or not tf_ok:
        print("\n🔧 Suggested Fixes:")
        if not basic_ok:
            print("1. Install missing packages with:")
            print("   pip install -r requirements.txt")
        if not tf_ok:
            print("2. For TensorFlow issues, try:")
            print("   pip uninstall tensorflow")
            print("   pip install tensorflow")
        
        # Offer to install packages
        response = input("\nWould you like me to try installing packages now? (y/n): ")
        if response.lower() == 'y':
            install_packages()
            print("\n🔄 Re-testing after installation...")
            test_basic_imports()
            test_tensorflow_detailed()
    else:
        print("\n🎉 All tests passed! Your environment is ready.")


if __name__ == "__main__":
    main()