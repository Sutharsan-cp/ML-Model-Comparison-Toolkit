"""
Test script for unsupervised learning models
"""

import sys
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

# Import our unsupervised module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unsupervised import UnsupervisedModelRegistry, UnsupervisedModelTrainer

def generate_test_data():
    """Generate various test datasets for unsupervised learning"""
    datasets = {}
    
    # Blob data (good for centroid-based clustering)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=1.5)
    datasets['blobs'] = (X_blobs, y_blobs, "Blob clusters")
    
    # Circle data (good for density-based clustering)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    datasets['circles'] = (X_circles, y_circles, "Concentric circles")
    
    # Moon data (good for manifold learning)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    datasets['moons'] = (X_moons, y_moons, "Two moons")
    
    # High-dimensional data (good for dimensionality reduction)
    np.random.seed(42)
    X_high_dim = np.random.randn(200, 20)
    # Add some structure
    X_high_dim[:100, :5] += 3
    X_high_dim[100:, 5:10] += 3
    y_high_dim = np.array([0]*100 + [1]*100)
    datasets['high_dim'] = (X_high_dim, y_high_dim, "High-dimensional data")
    
    return datasets

def test_clustering_models():
    """Test clustering models"""
    print("Testing Clustering Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y_true, description = datasets['blobs']  # Use blob data for clustering
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize registry and trainer
    registry = UnsupervisedModelRegistry()
    trainer = UnsupervisedModelTrainer()
    
    # Get available clustering models
    clustering_models = registry.get_clustering_models()
    print(f"Available clustering models: {len(clustering_models)}")
    print(f"Test data: {description} - {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test all clustering models
    test_models = list(clustering_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        results = trainer.train_multiple_models(test_models, X_scaled, 'clustering')
        
        for result in results:
            if result['training_successful']:
                n_clusters = result.get('n_clusters_found', 'N/A')
                inertia = result.get('inertia', 'N/A')
                if isinstance(inertia, float):
                    inertia = f"{inertia:.3f}"
                print(f"✅ {result['model_display_name']}: Clusters={n_clusters}, Inertia={inertia}")
            else:
                print(f"❌ {result['model_display_name']}: {result['error']}")
    else:
        print("No clustering models available")
    
    print()

def test_dimensionality_reduction_models():
    """Test dimensionality reduction models"""
    print("Testing Dimensionality Reduction Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y_true, description = datasets['high_dim']  # Use high-dim data for DR
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize registry and trainer
    registry = UnsupervisedModelRegistry()
    trainer = UnsupervisedModelTrainer()
    
    # Get available dimensionality reduction models
    dr_models = registry.get_dimensionality_reduction_models()
    print(f"Available dimensionality reduction models: {len(dr_models)}")
    print(f"Test data: {description} - {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test all dimensionality reduction models
    test_models = list(dr_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        results = trainer.train_multiple_models(test_models, X_scaled, 'dimensionality_reduction')
        
        for result in results:
            if result['training_successful']:
                n_components = result.get('n_components', 'N/A')
                explained_var = result.get('explained_variance_ratio', 'N/A')
                if hasattr(explained_var, '__len__') and len(explained_var) > 0:
                    try:
                        total_var = np.sum(explained_var.astype(float))
                        explained_var = f"{total_var:.3f}"
                    except:
                        explained_var = "N/A"
                elif isinstance(explained_var, (int, float)):
                    explained_var = f"{explained_var:.3f}"
                print(f"✅ {result['model_display_name']}: Components={n_components}, Explained_Var={explained_var}")
            else:
                print(f"❌ {result['model_display_name']}: {result['error']}")
    else:
        print("No dimensionality reduction models available")
    
    print()

def test_anomaly_detection_models():
    """Test anomaly detection models"""
    print("Testing Anomaly Detection Models")
    print("=" * 50)
    
    # Generate test data with outliers
    np.random.seed(42)
    X_normal = np.random.randn(200, 5)
    X_outliers = np.random.randn(20, 5) * 3 + 5  # Outliers
    X = np.vstack([X_normal, X_outliers])
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize registry and trainer
    registry = UnsupervisedModelRegistry()
    trainer = UnsupervisedModelTrainer()
    
    # Get available anomaly detection models
    anomaly_models = registry.get_anomaly_detection_models()
    print(f"Available anomaly detection models: {len(anomaly_models)}")
    print(f"Test data: Normal + Outliers - {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test all anomaly detection models
    test_models = list(anomaly_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        results = trainer.train_multiple_models(test_models, X_scaled, 'anomaly_detection')
        
        for result in results:
            if result['training_successful']:
                model = result['model_instance']
                # Try to get anomaly scores or predictions
                try:
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X_scaled)
                        n_outliers = np.sum(predictions == -1) if hasattr(predictions, '__len__') else 'N/A'
                    else:
                        n_outliers = 'N/A'
                    print(f"✅ {result['model_display_name']}: Outliers_detected={n_outliers}")
                except:
                    print(f"✅ {result['model_display_name']}: Training successful")
            else:
                print(f"❌ {result['model_display_name']}: {result['error']}")
    else:
        print("No anomaly detection models available")
    
    print()

def test_different_datasets():
    """Test models with different types of datasets"""
    print("Testing Models with Different Datasets")
    print("=" * 50)
    
    datasets = generate_test_data()
    registry = UnsupervisedModelRegistry()
    trainer = UnsupervisedModelTrainer()
    
    # Test a few representative models with different datasets
    test_models = {
        'clustering': ['kmeans', 'dbscan'],
        'dimensionality_reduction': ['pca'],
    }
    
    for dataset_name, (X, y_true, description) in datasets.items():
        print(f"\nDataset: {description}")
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test clustering models
        clustering_models = registry.get_clustering_models()
        available_clustering = [m for m in test_models['clustering'] if m in clustering_models]
        
        if available_clustering:
            print(f"  Clustering models: {available_clustering}")
            for model_name in available_clustering:
                try:
                    result = trainer.train_model(model_name, X_scaled, 'clustering')
                    if result['training_successful']:
                        n_clusters = result.get('n_clusters_found', 'N/A')
                        print(f"    ✅ {result['model_display_name']}: {n_clusters} clusters")
                    else:
                        print(f"    ❌ {result['model_display_name']}: {result['error']}")
                except Exception as e:
                    print(f"    ❌ {model_name}: {str(e)}")
        
        # Test dimensionality reduction models (only for high-dim data)
        if X.shape[1] > 2:
            dr_models = registry.get_dimensionality_reduction_models()
            available_dr = [m for m in test_models['dimensionality_reduction'] if m in dr_models]
            
            if available_dr:
                print(f"  Dimensionality reduction models: {available_dr}")
                for model_name in available_dr:
                    try:
                        # Use fewer components for small datasets
                        n_components = min(2, X.shape[1] - 1)
                        hyperparams = {'n_components': n_components}
                        result = trainer.train_model(model_name, X_scaled, 'dimensionality_reduction', hyperparams)
                        if result['training_successful']:
                            explained_var = result.get('explained_variance_ratio', 'N/A')
                            if hasattr(explained_var, '__len__'):
                                total_var = np.sum(explained_var)
                                explained_var = f"{total_var:.3f}"
                            print(f"    ✅ {result['model_display_name']}: Explained variance={explained_var}")
                        else:
                            print(f"    ❌ {result['model_display_name']}: {result['error']}")
                    except Exception as e:
                        print(f"    ❌ {model_name}: {str(e)}")

def main():
    """Main test function"""
    print("Unsupervised Learning Models Test")
    print("=" * 60)
    
    try:
        test_clustering_models()
        test_dimensionality_reduction_models()
        test_anomaly_detection_models()
        test_different_datasets()
        
        # Show summary
        registry = UnsupervisedModelRegistry()
        all_models = registry.get_all_models()
        
        print("Summary:")
        print(f"Total Clustering Models: {len(all_models['clustering'])}")
        print(f"Total Dimensionality Reduction Models: {len(all_models['dimensionality_reduction'])}")
        print(f"Total Anomaly Detection Models: {len(all_models['anomaly_detection'])}")
        
        print("\nClustering Models:")
        for name, info in all_models['clustering'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nDimensionality Reduction Models:")
        for name, info in all_models['dimensionality_reduction'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nAnomaly Detection Models:")
        for name, info in all_models['anomaly_detection'].items():
            print(f"  - {info['name']} ({info['type']})")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()