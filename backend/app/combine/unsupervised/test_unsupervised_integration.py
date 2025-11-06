"""
Full integration test for unsupervised learning models with real datasets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unsupervised import UnsupervisedModelRegistry, UnsupervisedModelTrainer
from streamlit_unsupervised_demo import get_available_datasets, load_dataset, get_working_models, generate_synthetic_data

def test_with_real_dataset():
    """Test working models with a real dataset"""
    print("Full Integration Test - Real Dataset")
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
        
        # Load dataset
        X, feature_names, df, info = load_dataset(dataset_info)
        
        if X is not None:
            print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Test clustering models
            test_clustering_models = working_models['clustering'][:3]  # Test first 3 models
            print(f"Testing clustering models: {test_clustering_models}")
            
            trainer = UnsupervisedModelTrainer()
            results = []
            
            for model_name in test_clustering_models:
                print(f"\nTraining {model_name}...")
                try:
                    # Set appropriate hyperparameters
                    hyperparams = {}
                    if model_name in ['kmeans', 'hierarchical', 'agglomerative', 'spectral_clustering', 'birch']:
                        hyperparams['n_clusters'] = 3
                    elif model_name == 'gaussian_mixture':
                        hyperparams['n_components'] = 3
                    
                    result = trainer.train_model(
                        model_name, X_scaled, 'clustering', hyperparams
                    )
                    results.append(result)
                    
                    if result['training_successful']:
                        n_clusters = result.get('n_clusters_found', 'N/A')
                        inertia = result.get('inertia', 'N/A')
                        if isinstance(inertia, float):
                            inertia = f"{inertia:.3f}"
                        print(f"✅ {result['model_display_name']}: "
                              f"Clusters={n_clusters}, Inertia={inertia}")
                    else:
                        print(f"❌ {result['model_display_name']}: {result['error']}")
                
                except Exception as e:
                    print(f"❌ {model_name}: {str(e)}")
            
            # Test dimensionality reduction if high-dimensional
            if X.shape[1] > 3:
                print(f"\nTesting dimensionality reduction models...")
                test_dr_models = working_models['dimensionality_reduction'][:2]  # Test first 2 models
                
                for model_name in test_dr_models:
                    print(f"Training {model_name}...")
                    try:
                        hyperparams = {'n_components': min(3, X.shape[1] - 1)}
                        result = trainer.train_model(
                            model_name, X_scaled, 'dimensionality_reduction', hyperparams
                        )
                        
                        if result['training_successful']:
                            n_components = result.get('n_components', 'N/A')
                            explained_var = result.get('explained_variance_ratio', 'N/A')
                            if hasattr(explained_var, '__len__'):
                                total_var = np.sum(explained_var)
                                explained_var = f"{total_var:.3f}"
                            print(f"✅ {result['model_display_name']}: "
                                  f"Components={n_components}, Explained_Var={explained_var}")
                        else:
                            print(f"❌ {result['model_display_name']}: {result['error']}")
                    
                    except Exception as e:
                        print(f"❌ {model_name}: {str(e)}")
            
            # Summary
            successful_models = [r for r in results if r['training_successful']]
            print(f"\nSummary: {len(successful_models)}/{len(test_clustering_models)} clustering models successful")
        
        else:
            print("Failed to load dataset")
    else:
        print("No real datasets found, using synthetic data")

def test_synthetic_datasets():
    """Test models with all synthetic datasets"""
    print("\n" + "=" * 50)
    print("Testing with Synthetic Datasets")
    print("=" * 50)
    
    synthetic_datasets = [
        "Blob Clusters",
        "Concentric Circles", 
        "Two Moons",
        "High-Dimensional Data",
        "Gaussian Mixture",
        "Anomaly Detection Data"
    ]
    
    working_models = get_working_models()
    trainer = UnsupervisedModelTrainer()
    
    for dataset_name in synthetic_datasets:
        print(f"\nDataset: {dataset_name}")
        
        try:
            X, feature_names, y_true = generate_synthetic_data(dataset_name)
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            print(f"  Shape: {X.shape}, Features: {len(feature_names)}")
            
            # Test appropriate models based on dataset
            if dataset_name == "Anomaly Detection Data":
                # Test anomaly detection models
                test_models = working_models['anomaly_detection'][:1]  # Test first model
                task_type = 'anomaly_detection'
            elif dataset_name == "High-Dimensional Data":
                # Test dimensionality reduction
                test_models = working_models['dimensionality_reduction'][:1]  # Test first model
                task_type = 'dimensionality_reduction'
            else:
                # Test clustering models
                test_models = working_models['clustering'][:2]  # Test first 2 models
                task_type = 'clustering'
            
            for model_name in test_models:
                try:
                    hyperparams = {}
                    if task_type == 'clustering':
                        if model_name in ['kmeans', 'hierarchical', 'agglomerative', 'spectral_clustering', 'birch']:
                            hyperparams['n_clusters'] = len(np.unique(y_true)) if 'y_true' in locals() else 3
                        elif model_name == 'gaussian_mixture':
                            hyperparams['n_components'] = len(np.unique(y_true)) if 'y_true' in locals() else 3
                    elif task_type == 'dimensionality_reduction':
                        hyperparams['n_components'] = 2
                    
                    result = trainer.train_model(model_name, X_scaled, task_type, hyperparams)
                    
                    if result['training_successful']:
                        if task_type == 'clustering':
                            n_clusters = result.get('n_clusters_found', 'N/A')
                            print(f"    ✅ {result['model_display_name']}: {n_clusters} clusters")
                        elif task_type == 'dimensionality_reduction':
                            n_components = result.get('n_components', 'N/A')
                            print(f"    ✅ {result['model_display_name']}: {n_components} components")
                        else:
                            print(f"    ✅ {result['model_display_name']}: Success")
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
    X, feature_names, y_true = generate_synthetic_data("Blob Clusters")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train multiple models
    working_models = get_working_models()
    trainer = UnsupervisedModelTrainer()
    
    test_models = working_models['clustering'][:3]
    results = []
    
    for model_name in test_models:
        try:
            hyperparams = {}
            if model_name in ['kmeans', 'hierarchical', 'agglomerative', 'spectral_clustering', 'birch']:
                hyperparams['n_clusters'] = 4
            elif model_name == 'gaussian_mixture':
                hyperparams['n_components'] = 4
            
            result = trainer.train_model(model_name, X_scaled, 'clustering', hyperparams)
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from unsupervised import create_model_comparison_report
    
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
    print("Unsupervised Learning Integration Test")
    print("=" * 60)
    
    try:
        test_with_real_dataset()
        test_synthetic_datasets()
        test_model_comparison_report()
        
        print("\n🎉 Integration test completed!")
        
        # Show final summary
        registry = UnsupervisedModelRegistry()
        all_models = registry.get_all_models()
        
        print(f"\nFinal Summary:")
        print(f"Total Clustering Models: {len(all_models['clustering'])}")
        print(f"Total Dimensionality Reduction Models: {len(all_models['dimensionality_reduction'])}")
        print(f"Total Anomaly Detection Models: {len(all_models['anomaly_detection'])}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()