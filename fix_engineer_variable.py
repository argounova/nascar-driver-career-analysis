#!/usr/bin/env python3
"""
Quick fix for the 'engineer' variable name error in train_models.py
"""

import sys
from pathlib import Path

# File path
project_root = Path(__file__).parent
train_models_file = project_root / 'scripts' / 'train_models.py'

print("🔧 Fixing Engineer Variable Name")
print("=" * 32)

if not train_models_file.exists():
    print(f"❌ File not found: {train_models_file}")
    sys.exit(1)

# Read the file
with open(train_models_file, 'r') as f:
    content = f.read()

# Fix the variable name - change 'engineer' to 'feature_engineer'
content = content.replace(
    "sequences, targets, driver_names = engineer.lstm_sequences",
    "sequences, targets, driver_names = feature_engineer.lstm_sequences"
)

content = content.replace(
    "prediction_targets = lstm_predictor.prepare_prediction_targets(engineer.engineered_features)",
    "prediction_targets = lstm_predictor.prepare_prediction_targets(feature_engineer.engineered_features)"
)

# Write the fixed content back
with open(train_models_file, 'w') as f:
    f.write(content)

print("✅ Fixed variable name: engineer → feature_engineer")
print("Now run: python3 scripts/train_models.py")

print(f"\nFile: {train_models_file}")