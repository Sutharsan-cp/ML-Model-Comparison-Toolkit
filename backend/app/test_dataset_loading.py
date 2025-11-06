"""
Test dataset loading functionality for Streamlit demo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from streamlit_supervised_demo import get_available_datasets, load_dataset

def test_dataset_discovery():
    """Test dataset discovery functionality"""
    print("Testing Dataset Discovery")
    print("=" * 40)
    
    datasets = get_available_datasets()
    
    for task_type, task_datasets in datasets.items():
        print(f"\n{task_type} Datasets:")
        if task_datasets:
            for name, info in task_datasets.items():
                print(f"  - {name}: {info['path']} ({info['type']})")
        else:
            print("  - No datasets found")

def test_dataset_loading():
    """Test loading a few datasets"""
    print("\nTesting Dataset Loading")
    print("=" * 40)
    
    datasets = get_available_datasets()
    
    # Test classification datasets
    classification_datasets = datasets.get("Classification", {})
    if classification_datasets:
        # Test first classification dataset
        dataset_name = list(classification_datasets.keys())[0]
        dataset_info = classification_datasets[dataset_name]
        
        print(f"\nTesting: {dataset_name}")
        try:
            # Load without target column first to see columns
            _, _, columns, df, _ = load_dataset(dataset_info)
            print(f"  Columns: {columns}")
            print(f"  Shape: {df.shape}")
            
            # Try loading with last column as target
            if columns:
                target_col = columns[-1]
                X, y, feature_names, _, info = load_dataset(dataset_info, target_col)
                if X is not None:
                    print(f"  Features shape: {X.shape}")
                    print(f"  Target shape: {y.shape}")
                    print(f"  Feature names: {feature_names[:5]}...")  # First 5 features
                    if info.get('categorical_columns'):
                        print(f"  Categorical columns: {info['categorical_columns']}")
                    print("  ✅ Successfully loaded")
                else:
                    print("  ❌ Failed to load with target")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test regression datasets
    regression_datasets = datasets.get("Regression", {})
    if regression_datasets:
        # Test first regression dataset
        dataset_name = list(regression_datasets.keys())[0]
        dataset_info = regression_datasets[dataset_name]
        
        print(f"\nTesting: {dataset_name}")
        try:
            # Load without target column first to see columns
            _, _, columns, df, _ = load_dataset(dataset_info)
            print(f"  Columns: {columns}")
            print(f"  Shape: {df.shape}")
            
            # Try loading with last column as target
            if columns:
                target_col = columns[-1]
                X, y, feature_names, _, info = load_dataset(dataset_info, target_col)
                if X is not None:
                    print(f"  Features shape: {X.shape}")
                    print(f"  Target shape: {y.shape}")
                    print(f"  Feature names: {feature_names[:5]}...")  # First 5 features
                    if info.get('categorical_columns'):
                        print(f"  Categorical columns: {info['categorical_columns']}")
                    print("  ✅ Successfully loaded")
                else:
                    print("  ❌ Failed to load with target")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("Dataset Loading Test")
    print("=" * 50)
    
    test_dataset_discovery()
    test_dataset_loading()

if __name__ == "__main__":
    main()