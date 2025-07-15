#!/usr/bin/env python3
"""
Fix train_models.py import issues for VS Code execution.
This removes problematic imports and simplifies the script.
"""

import sys
from pathlib import Path

# File path
project_root = Path(__file__).parent
train_models_file = project_root / 'scripts' / 'train_models.py'

print("🔧 Fixing Train Models Import Issues")
print("=" * 40)

if not train_models_file.exists():
    print(f"❌ File not found: {train_models_file}")
    sys.exit(1)

# Read the file
with open(train_models_file, 'r') as f:
    content = f.read()

# Fix 1: Remove problematic imports
old_import_line = "from models.lstm_model import train_lstm_career_predictor, predict_driver_career"
new_import_line = "from models.lstm_model import NASCARLSTMPredictor"

content = content.replace(old_import_line, new_import_line)

# Fix 2: Update the LSTM training call to use the class directly
old_lstm_call = "lstm_predictor = train_lstm_career_predictor(config_path, save_results=save_all_results)"

new_lstm_call = '''# Train LSTM manually using the class
        print("Training LSTM neural network...")
        
        # Initialize predictor
        lstm_predictor = NASCARLSTMPredictor()
        
        # Load sequences from feature engineer
        sequences, targets, driver_names = engineer.lstm_sequences
        lstm_predictor.load_sequences(sequences, targets, driver_names)
        
        # Prepare targets
        prediction_targets = lstm_predictor.prepare_prediction_targets(engineer.engineered_features)
        
        # Preprocess data
        lstm_predictor.preprocess_data(sequences, prediction_targets)
        
        # Build and train model
        lstm_predictor.build_model()
        history = lstm_predictor.train_model()
        
        # Save model
        if save_all_results:
            lstm_predictor.save_model()'''

content = content.replace(old_lstm_call, new_lstm_call)

# Fix 3: Remove the predict_driver_career calls since that function doesn't exist
old_prediction_section = '''                prediction = predict_driver_career(
                    driver, 
                    feature_engineer.engineered_features, 
                    lstm_predictor
                )'''

new_prediction_section = '''                # Skip individual predictions for now - function not implemented
                prediction = {'error': 'Individual prediction function not implemented'}'''

content = content.replace(old_prediction_section, new_prediction_section)

# Write the fixed content back
with open(train_models_file, 'w') as f:
    f.write(content)

print("✅ Fixed train_models.py import issues:")
print("   • Removed non-existent function imports")
print("   • Updated LSTM training to use class directly")
print("   • Simplified prediction section")
print("\nNow you can run the script using VS Code's play button!")
print("The script will train your LSTM with 200 epochs.")

print(f"\nFile: {train_models_file}")