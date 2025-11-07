"""
Full integration test for supervised learning models with real datasets
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# ✅ Fix import path (adjusts to your project structure)
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from combine.supervised.supervised import SupervisedModelRegistry, SupervisedModelTrainer
from combine.supervised.streamlit_supervised_demo import (
    get_available_datasets, load_dataset, get_working_models
)

# ------------------------------------------------------------
# ✅ Integration Test Function
# ------------------------------------------------------------
def test_with_real_dataset():
    """Test working models with a real dataset"""
    print("Full Integration Test - Real Dataset")
    print("=" * 50)
    
    # Get available datasets
    datasets = get_available_datasets()
    working_models = get_working_models()
    
    # ----------------------------- REGRESSION TEST -----------------------------
    regression_datasets = datasets.get("Regression", {})
    if regression_datasets:
        dataset_name = "Boston House Prices"
        if dataset_name in regression_datasets:
            dataset_info = regression_datasets[dataset_name]
            
            print(f"Testing with dataset: {dataset_name}")
            
            # Load dataset
            X, y, feature_names, df, info = load_dataset(dataset_info, "MEDV")
            
            if X is not None:
                print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Test working regression models
                test_models = working_models['regression'][:3]  # Test first 3 models
                print(f"Testing models: {test_models}")
                
                trainer = SupervisedModelTrainer()
                results = []
                
                for model_name in test_models:
                    print(f"\nTraining {model_name}...")
                    try:
                        result = trainer.train_model(
                            model_name, X_train_scaled, y_train, 
                            X_test_scaled, y_test, 'regression'
                        )
                        results.append(result)
                        
                        if result['training_successful']:
                            print(f"✅ {result['model_display_name']}: "
                                  f"Train={result['train_score']:.4f}, "
                                  f"Test={result.get('test_score', 'N/A')}")
                        else:
                            print(f"❌ {result['model_display_name']}: {result['error']}")
                    
                    except Exception as e:
                        print(f"❌ {model_name}: {str(e)}")
                
                # Summary
                successful_models = [r for r in results if r['training_successful']]
                print(f"\nSummary: {len(successful_models)}/{len(test_models)} models successful")
                
                if successful_models:
                    best_model = max(successful_models, key=lambda x: x.get('test_score', 0))
                    print(f"Best model: {best_model['model_display_name']} "
                          f"(Test Score: {best_model.get('test_score', 'N/A')})")
            else:
                print("Failed to load dataset")
    
    # ----------------------------- CLASSIFICATION TEST -----------------------------
    print("\n" + "=" * 50)
    classification_datasets = datasets.get("Classification", {})
    if classification_datasets:
        dataset_name = "IRIS"
        if dataset_name in classification_datasets:
            dataset_info = classification_datasets[dataset_name]
            
            print(f"Testing with dataset: {dataset_name}")
            
            # Load dataset - try different possible target columns
            possible_targets = ["species", "Species", "class", "Class", "target", "Target"]
            X, y, feature_names, df, info = None, None, None, None, None
            
            # First load to see columns
            _, _, columns, sample_df, _ = load_dataset(dataset_info)
            print(f"Available columns: {columns}")
            
            # Find target column
            target_col = next((col for col in possible_targets if col in columns), None)
            if target_col is None and columns:
                target_col = columns[-1]
            
            if target_col:
                X, y, feature_names, df, info = load_dataset(dataset_info, target_col)
                
                if X is not None:
                    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
                    print(f"Classes: {len(np.unique(y))}")
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Test working classification models
                    test_models = working_models['classification'][:3]
                    print(f"Testing models: {test_models}")
                    
                    trainer = SupervisedModelTrainer()
                    results = []
                    
                    for model_name in test_models:
                        print(f"\nTraining {model_name}...")
                        try:
                            result = trainer.train_model(
                                model_name, X_train_scaled, y_train, 
                                X_test_scaled, y_test, 'classification'
                            )
                            results.append(result)
                            
                            if result['training_successful']:
                                print(f"✅ {result['model_display_name']}: "
                                      f"Train={result['train_score']:.4f}, "
                                      f"Test={result.get('test_score', 'N/A')}")
                            else:
                                print(f"❌ {result['model_display_name']}: {result['error']}")
                        
                        except Exception as e:
                            print(f"❌ {model_name}: {str(e)}")
                    
                    # Summary
                    successful_models = [r for r in results if r['training_successful']]
                    print(f"\nSummary: {len(successful_models)}/{len(test_models)} models successful")
                    
                    if successful_models:
                        best_model = max(successful_models, key=lambda x: x.get('test_score', 0))
                        print(f"Best model: {best_model['model_display_name']} "
                              f"(Test Score: {best_model.get('test_score', 'N/A')})")
                else:
                    print("Failed to load dataset")
            else:
                print("Could not determine target column")


# ------------------------------------------------------------
# ✅ Main entrypoint
# ------------------------------------------------------------
def main():
    """Main test function"""
    print("Full Integration Test")
    print("=" * 60)
    
    try:
        test_with_real_dataset()
        print("\n🎉 Integration test completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
