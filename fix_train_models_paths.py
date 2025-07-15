#!/usr/bin/env python3
"""
Quick fix for train_models.py import paths.
This will update the train_models.py file to handle import paths correctly.
"""

import sys
from pathlib import Path

# File path
project_root = Path(__file__).parent
train_models_file = project_root / 'scripts' / 'train_models.py'

print("🔧 Fixing Train Models Import Paths")
print("=" * 38)

if not train_models_file.exists():
    print(f"❌ File not found: {train_models_file}")
    sys.exit(1)

# Read the file
with open(train_models_file, 'r') as f:
    content = f.read()

# Find the imports section and add path setup
old_imports = '''# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our modules
from config import get_config, get_data_paths'''

new_imports = '''# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import our modules
from config import get_config, get_data_paths'''

# Apply the fix
content_fixed = content.replace(old_imports, new_imports)

# Check if the fix was applied
if content != content_fixed:
    # Write the fixed content back
    with open(train_models_file, 'w') as f:
        f.write(content_fixed)
    
    print("✅ Fixed import paths in train_models.py")
    print("   Added project root and src to Python path")
    print("\nNow run: python scripts/train_models.py")
else:
    print("⚠️  No import section found to fix")
    print("The file might have different structure")
    
    # Show the beginning of the file for debugging
    lines = content.split('\n')
    print("\nFirst 40 lines of train_models.py:")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:2d}: {line}")

print(f"\nFile: {train_models_file}")