import sys
from pathlib import Path

# Add the correct paths
current_dir = Path(__file__).parent
api_dir = current_dir / 'api'
sys.path.append(str(api_dir))

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTesting imports...")

try:
    from api.preprocessing.data_loader import DataLoader
    print("✓ DataLoader imported successfully")
    
    from api.preprocessing.data_cleaning import DataCleaner
    print("✓ DataCleaner imported successfully")
    
    from api.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    print("✓ PreprocessingPipeline imported successfully")
    
    print("\n✓ All imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    
    # Check if files exist
    preprocessing_dir = api_dir / 'preprocessing'
    if preprocessing_dir.exists():
        print(f"Preprocessing directory exists: {preprocessing_dir}")
        files = list(preprocessing_dir.glob('*.py'))
        print("Files in preprocessing directory:")
        for f in files:
            print(f"  {f.name}")
    else:
        print(f"Preprocessing directory not found: {preprocessing_dir}")