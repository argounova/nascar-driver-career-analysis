#!/usr/bin/env python3
"""
Test script to verify NASCAR project setup and data loading.
Run this from the project root directory.
"""

import sys
from pathlib import Path
import os

def test_project_structure():
    """Test that all required directories exist."""
    print("🏗️  Testing Project Structure")
    print("=" * 40)
    
    required_dirs = [
        'config',
        'src',
        'src/data', 
        'src/models',
        'src/visualization',
        'src/utils',
        'scripts',
        'data',
        'data/raw',
        'data/processed',
        'data/models'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ Missing: {dir_path}")
        else:
            print(f"✅ Found: {dir_path}")
    
    if missing_dirs:
        print(f"\n⚠️  Missing {len(missing_dirs)} directories. Create them with:")
        for dir_path in missing_dirs:
            print(f"mkdir -p {dir_path}")
        return False
    else:
        print("\n✅ All directories present!")
        return True


def test_config_files():
    """Test that config files exist and are readable."""
    print("\n📋 Testing Configuration Files")
    print("=" * 40)
    
    config_files = ['config/__init__.py', 'config/config.yaml']
    
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"✅ Found: {file_path}")
            try:
                if file_path.endswith('.py'):
                    # Test Python import
                    sys.path.insert(0, str(Path.cwd()))
                    import config
                    print(f"   ✅ Python import successful")
                elif file_path.endswith('.yaml'):
                    # Test YAML loading
                    import yaml
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)
                    print(f"   ✅ YAML parsing successful")
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
                return False
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True


def test_data_loader():
    """Test the data loader module."""
    print("\n📊 Testing Data Loader")
    print("=" * 40)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path.cwd() / 'src'))
        
        from data.data_loader import NASCARDataLoader
        print("✅ Data loader import successful")
        
        # Initialize loader
        loader = NASCARDataLoader()
        print("✅ Data loader initialization successful")
        
        # Test data freshness check
        freshness = loader.check_data_freshness()
        print(f"✅ Data freshness check: {freshness.get('status', 'Unknown')}")
        
        # Test loading (will use sample data if no real data)
        print("\n📈 Attempting to load data...")
        loader.load_raw_data()
        print(f"✅ Raw data loaded: {len(loader.raw_data)} records")
        
        # Test filtering
        loader.apply_data_filtering()
        print(f"✅ Data filtered: {len(loader.filtered_data)} records")
        
        # Test season summary
        loader.create_driver_season_summary()
        print(f"✅ Season summaries: {len(loader.driver_seasons)} driver-seasons")
        
        # Show summary
        summary = loader.get_data_summary()
        print("\n📊 Data Summary:")
        for section, stats in summary.items():
            print(f"\n{section.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_r_integration():
    """Test R integration for data updates."""
    print("\n🔧 Testing R Integration")
    print("=" * 40)
    
    r_script = Path('scripts/update_data.R')
    if r_script.exists():
        print("✅ R script found")
        
        # Test if R is available
        import subprocess
        try:
            result = subprocess.run(['Rscript', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ R is installed and accessible")
                print(f"   Version: {result.stdout.strip()}")
                return True
            else:
                print("❌ R is installed but not working properly")
                return False
        except FileNotFoundError:
            print("❌ R is not installed or not in PATH")
            return False
        except subprocess.TimeoutExpired:
            print("❌ R command timed out")
            return False
    else:
        print("❌ R script not found")
        return False


def main():
    """Run all tests."""
    print("🏁 NASCAR Project Setup Test")
    print("=" * 50)
    print(f"Testing from: {Path.cwd()}")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Config Files", test_config_files), 
        ("Data Loader", test_data_loader),
        ("R Integration", test_r_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)