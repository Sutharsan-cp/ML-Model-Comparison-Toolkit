"""
Full integration test for neural network models with real datasets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_networks import NeuralNetworkRegistry, NeuralNetworkTrainer
from streamlit_neural_networks_demo import get_available_datasets, load_dataset, get_working_models, generate_synthetic_data

def test_with_real_dataset():
    """Test working models with a real dataset"""
    print("Neural Networks Integration Test - Real Dataset")
    print("=" * 50)
    
    # Get available datasets
    datasets = get_available_datasets()
    working_models = get_working_models()
    
    # Test with a real dataset if available
    all_datasets = {}
    all_datasets.update(datasets.get("Classification", {}))
    all_datasets.update(datasets.get("Regression", {}))
    
    if all_datasets:
        dataset_name = list(all_datasets.keys())[0]
        dataset_info = all_datasets[dataset_name]
        
        print(f"Testing with dataset: {dataset_name}")
        
        # Load dataset to get columns first
        _, _, columns, sample_df, _ = load_dataset(dataset_info)
        target_col = columns[-1] if columns else None
        
        # Load dataset with target
        X, y, feature_names, df, info = load_dataset(dataset_info, target_col)
        
        if X is not None and y is not None:
            print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            
            # Test feedforward models
            test_models = working_models['feedforward'][:2]  # Test first 2 models
            print(f"Testing feedforward models: {test_models}")
            
            trainer = NeuralNetworkTrainer()
            results = []
            
            for model_name in test_models:
                print(f"\nTraining {model_name}...")
                try:
                    # Use simple hyperparameters for testing
                    hyperparams = {'maxEpochs': 10, 'batchSize': 32}
                    if model_name == 'mlp':
                        hyperparams['hiddenLayers'] = [32]
                    elif model_name in ['ann', 'backpropagation']:
                        hyperparams['layers'] = [32]
                    
                    result = trainer.train_model(
                        model_name, X_train, y_train, 'feedforward', hyperparams
                    )
                    results.append(result)
                    
                    if result['training_successful']:
                        final_acc = result.get('final_accuracy', 'N/A')
                        final_loss = result.get('final_loss', 'N/A')
                        if isinstance(final_acc, float):
                            final_acc = f"{final_acc:.3f}"
                        if isinstance(final_loss, float):
                            final_loss = f"{final_loss:.3f}"
                        print(f"✅ {result['model_display_name']}: "
                              f"Final_Acc={final_acc}, Loss={final_loss}")
                    else:
                        print(f"❌ {result['model_display_name']}: {result['error']}")
                
                except Exception as e:
                    print(f"❌ {model_name}: {str(e)}")
            
            # Summary
            successful_models = [r for r in results if r['training_successful']]
            print(f"\nSummary: {len(successful_models)}/{len(test_models)} models successful")
        
        else:
            print("Failed to load dataset or no target column")
    else:
        print("No real datasets found, using synthetic data")

def test_synthetic_datasets():
    """Test models with synthetic datasets"""
    print("\n" + "=" * 50)
    print("Testing with Synthetic Datasets")
    print("=" * 50)
    
    synthetic_tests = [
        ("Binary Classification", "Classification"),
        ("Linear Regression", "Regression"),
        ("Synthetic Images", "Image Classification")
    ]
    
    working_models = get_working_models()
    trainer = NeuralNetworkTrainer()
    
    for dataset_name, task_type in synthetic_tests:
        print(f"\nDataset: {dataset_name} ({task_type})")
        
        try:
            X, y, feature_names = generate_synthetic_data(dataset_name, task_type)
            print(f"  Shape: {X.shape}")
            
            # Scale data if tabular
            if X.ndim == 2:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Test appropriate models based on task
            if task_type == "Classification":
                test_models = working_models['feedforward'][:1]  # Test first model
                category = 'feedforward'
            elif task_type == "Regression":
                test_models = working_models['feedforward'][:1]  # Test first model
                category = 'feedforward'
            elif task_type == "Image Classification":
                test_models = working_models['convolutional'][:1]  # Test CNN
                category = 'convolutional'
            else:
                continue
            
            for model_name in test_models:
                try:
                    hyperparams = {'maxEpochs': 5, 'batchSize': 16}
                    if model_name == 'mlp':
                        hyperparams['hiddenLayers'] = [32]
                    elif model_name in ['ann', 'backpropagation']:
                        hyperparams['layers'] = [32]
                    
                    result = trainer.train_model(model_name, X_scaled, y, category, hyperparams)
                    
                    if result['training_successful']:
                        final_acc = result.get('final_accuracy', 'N/A')
                        if isinstance(final_acc, float):
                            final_acc = f"{final_acc:.3f}"
                        print(f"    ✅ {result['model_display_name']}: Final_Acc={final_acc}")
                    else:
                        print(f"    ❌ {result['model_display_name']}: {result['error']}")
                
                except Exception as e:
                    print(f"    ❌ {model_name}: {str(e)}")
        
        except Exception as e:
            print(f"  ❌ Failed to generate {dataset_name}: {str(e)}")

def test_model_comparison_report():
    """Test the model comparison report functionality"""
    print("\n" + "=" * 50)
    print("Testing Model Comparison Report")
    print("=" * 50)
    
    # Generate test data
    X, y, feature_names = generate_synthetic_data("Binary Classification", "Classification")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train multiple models
    working_models = get_working_models()
    trainer = NeuralNetworkTrainer()
    
    test_models = working_models['feedforward'][:2]
    results = []
    
    for model_name in test_models:
        try:
            hyperparams = {'maxEpochs': 5, 'batchSize': 32}
            if model_name == 'mlp':
                hyperparams['hiddenLayers'] = [32]
            elif model_name in ['ann', 'backpropagation']:
                hyperparams['layers'] = [32]
            
            result = trainer.train_model(model_name, X_scaled, y, 'feedforward', hyperparams)
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from neural_networks import create_model_comparison_report
    
    try:
        comparison_df = create_model_comparison_report(results)
        print("✅ Comparison report generated successfully")
        print(f"Report shape: {comparison_df.shape}")
        print("\nReport columns:", list(comparison_df.columns))
        
        if not comparison_df.empty:
            print("\nSample report:")
            print(comparison_df.head())
        
    except Exception as e:
        print(f"❌ Failed to generate comparison report: {e}")

def main():
    """Main test function"""
    print("Neural Networks Integration Test")
    print("=" * 60)
    
    try:
        test_with_real_dataset()
        test_synthetic_datasets()
        test_model_comparison_report()
        
        print("\n🎉 Integration test completed!")
        
        # Show final summary
        registry = NeuralNetworkRegistry()
        all_models = registry.get_all_models()
        working_models = get_working_models()
        
        print(f"\nFinal Summary:")
        total_models = sum(len(models) for models in all_models.values())
        working_count = sum(len(models) for models in working_models.values())
        
        print(f"Total Models Available: {total_models}")
        print(f"Working Models: {working_count}")
        print(f"Success Rate: {working_count/total_models*100:.1f}%")
        
        for category, models in all_models.items():
            working_in_category = len(working_models.get(category, []))
            print(f"{category.title()}: {working_in_category}/{len(models)} working")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()