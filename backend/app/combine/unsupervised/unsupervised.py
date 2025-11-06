"""
Unsupervised Learning Models Module
Consolidates all unsupervised learning algorithms for the ML Comparison Toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all unsupervised models with error handling
clustering_imports = {}
dimensionality_reduction_imports = {}
anomaly_detection_imports = {}

# Clustering models
try:
    from models.unsupervised.kmeans import KMeans
    clustering_imports['kmeans'] = KMeans
except ImportError as e:
    print(f"Warning: Could not import KMeans: {e}")

try:
    from models.unsupervised.DBSCAN import DBSCAN
    clustering_imports['dbscan'] = DBSCAN
except ImportError as e:
    print(f"Warning: Could not import DBSCAN: {e}")

try:
    from models.unsupervised.hierarchical import HierarchicalClustering
    clustering_imports['hierarchical'] = HierarchicalClustering
except ImportError as e:
    print(f"Warning: Could not import HierarchicalClustering: {e}")

try:
    from models.unsupervised.gaussian_mixture import GaussianMixture
    clustering_imports['gaussian_mixture'] = GaussianMixture
except ImportError as e:
    print(f"Warning: Could not import GaussianMixture: {e}")

try:
    from models.unsupervised.mean_shift import MeanShift
    clustering_imports['mean_shift'] = MeanShift
except ImportError as e:
    print(f"Warning: Could not import MeanShift: {e}")

try:
    from models.unsupervised.agglomerative import AgglomerativeClustering
    clustering_imports['agglomerative'] = AgglomerativeClustering
except ImportError as e:
    print(f"Warning: Could not import AgglomerativeClustering: {e}")

try:
    from models.unsupervised.spectral_clustering import SpectralClustering
    clustering_imports['spectral_clustering'] = SpectralClustering
except ImportError as e:
    print(f"Warning: Could not import SpectralClustering: {e}")

try:
    from models.unsupervised.birch import BIRCH
    clustering_imports['birch'] = BIRCH
except ImportError as e:
    print(f"Warning: Could not import BIRCH: {e}")

try:
    from models.unsupervised.optics import OPTICS
    clustering_imports['optics'] = OPTICS
except ImportError as e:
    print(f"Warning: Could not import OPTICS: {e}")

# Dimensionality Reduction models
try:
    from models.unsupervised.PCA import PCA
    dimensionality_reduction_imports['pca'] = PCA
except ImportError as e:
    print(f"Warning: Could not import PCA: {e}")

try:
    from models.unsupervised.ICA import FastICA
    dimensionality_reduction_imports['ica'] = FastICA
except ImportError as e:
    print(f"Warning: Could not import FastICA: {e}")

try:
    from models.unsupervised.tsne import TSNE
    dimensionality_reduction_imports['tsne'] = TSNE
except ImportError as e:
    print(f"Warning: Could not import TSNE: {e}")

try:
    from models.unsupervised.umap import UMAP
    dimensionality_reduction_imports['umap'] = UMAP
except ImportError as e:
    print(f"Warning: Could not import UMAP: {e}")

# Anomaly Detection models
try:
    from models.unsupervised.isolation_forest import IsolationForest
    anomaly_detection_imports['isolation_forest'] = IsolationForest
except ImportError as e:
    print(f"Warning: Could not import IsolationForest: {e}")

try:
    from models.unsupervised.one_class_svm import OneClassSVM
    anomaly_detection_imports['one_class_svm'] = OneClassSVM
except ImportError as e:
    print(f"Warning: Could not import OneClassSVM: {e}")

class UnsupervisedModelRegistry:
    """Registry for all unsupervised learning models"""
    
    def __init__(self):
        # Build clustering models dictionary from successfully imported classes
        self.clustering_models = {}
        
        # Define clustering model configurations
        clustering_configs = {
            'kmeans': {
                'name': 'K-Means',
                'type': 'centroid',
                'description': 'Partitions data into k clusters using centroids',
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10],
                    'max_iters': [100, 200, 300, 500],
                    'init': ['k-means++', 'random'],
                    'tol': [1e-4, 1e-3, 1e-2]
                }
            },
            'dbscan': {
                'name': 'DBSCAN',
                'type': 'density',
                'description': 'Density-based clustering algorithm',
                'hyperparameters': {
                    'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
                    'min_samples': [3, 5, 7, 10, 15],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                }
            },
            'hierarchical': {
                'name': 'Hierarchical Clustering',
                'type': 'hierarchical',
                'description': 'Hierarchical clustering using linkage criteria',
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10],
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                }
            },
            'gaussian_mixture': {
                'name': 'Gaussian Mixture Model',
                'type': 'probabilistic',
                'description': 'Probabilistic clustering using Gaussian mixtures',
                'hyperparameters': {
                    'n_components': [2, 3, 4, 5, 6, 7, 8, 10],
                    'max_iters': [50, 100, 200],
                    'tol': [1e-3, 1e-4, 1e-5],
                    'init_params': ['kmeans', 'random']
                }
            },
            'mean_shift': {
                'name': 'Mean Shift',
                'type': 'density',
                'description': 'Clustering using mean shift algorithm',
                'hyperparameters': {
                    'bandwidth': [None, 0.5, 1.0, 1.5, 2.0],
                    'max_iter': [100, 200, 300],
                    'tol': [1e-3, 1e-4, 1e-5]
                }
            },
            'agglomerative': {
                'name': 'Agglomerative Clustering',
                'type': 'hierarchical',
                'description': 'Bottom-up hierarchical clustering',
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10],
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                }
            },
            'spectral_clustering': {
                'name': 'Spectral Clustering',
                'type': 'graph',
                'description': 'Clustering using spectral graph theory',
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10],
                    'affinity': ['rbf', 'nearest_neighbors', 'polynomial'],
                    'gamma': [0.1, 1.0, 10.0, 100.0],
                    'n_neighbors': [5, 10, 15, 20]
                }
            },
            'birch': {
                'name': 'BIRCH',
                'type': 'hierarchical',
                'description': 'Balanced Iterative Reducing and Clustering using Hierarchies',
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10],
                    'threshold': [0.1, 0.3, 0.5, 0.7, 1.0],
                    'branching_factor': [20, 30, 50, 70, 100]
                }
            },
            'optics': {
                'name': 'OPTICS',
                'type': 'density',
                'description': 'Ordering Points To Identify Clustering Structure',
                'hyperparameters': {
                    'min_samples': [3, 5, 7, 10, 15],
                    'max_eps': [1.0, 2.0, 5.0, 10.0, np.inf],
                    'metric': ['euclidean', 'manhattan', 'cosine'],
                    'xi': [0.01, 0.05, 0.1, 0.2]
                }
            }
        }
        
        # Add successfully imported clustering models
        for model_key, model_class in clustering_imports.items():
            if model_key in clustering_configs:
                config = clustering_configs[model_key].copy()
                config['class'] = model_class
                self.clustering_models[model_key] = config        

        # Build dimensionality reduction models dictionary
        self.dimensionality_reduction_models = {}
        
        # Define dimensionality reduction model configurations
        dr_configs = {
            'pca': {
                'name': 'Principal Component Analysis',
                'type': 'linear',
                'description': 'Linear dimensionality reduction using SVD',
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10, 20, None],
                    'svd_solver': ['auto', 'full', 'arpack', 'randomized']
                }
            },
            'ica': {
                'name': 'Independent Component Analysis',
                'type': 'linear',
                'description': 'Separates multivariate signal into independent components',
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10, 20, None],
                    'algorithm': ['parallel', 'deflation'],
                    'fun': ['logcosh', 'exp', 'cube'],
                    'max_iter': [100, 200, 500]
                }
            },
            'tsne': {
                'name': 't-SNE',
                'type': 'manifold',
                'description': 't-distributed Stochastic Neighbor Embedding',
                'hyperparameters': {
                    'n_components': [2, 3],
                    'perplexity': [5, 10, 30, 50, 100],
                    'learning_rate': [10, 50, 100, 200, 500],
                    'n_iter': [500, 1000, 2000],
                    'early_exaggeration': [4, 8, 12, 20]
                }
            },
            'umap': {
                'name': 'UMAP',
                'type': 'manifold',
                'description': 'Uniform Manifold Approximation and Projection',
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10],
                    'n_neighbors': [5, 10, 15, 30, 50],
                    'min_dist': [0.01, 0.1, 0.3, 0.5, 0.8],
                    'metric': ['euclidean', 'manhattan', 'cosine'],
                    'learning_rate': [0.5, 1.0, 2.0]
                }
            }
        }
        
        # Add successfully imported dimensionality reduction models
        for model_key, model_class in dimensionality_reduction_imports.items():
            if model_key in dr_configs:
                config = dr_configs[model_key].copy()
                config['class'] = model_class
                self.dimensionality_reduction_models[model_key] = config
        
        # Build anomaly detection models dictionary
        self.anomaly_detection_models = {}
        
        # Define anomaly detection model configurations
        anomaly_configs = {
            'isolation_forest': {
                'name': 'Isolation Forest',
                'type': 'ensemble',
                'description': 'Anomaly detection using isolation trees',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 500],
                    'max_samples': ['auto', 0.5, 0.7, 1.0],
                    'contamination': ['auto', 0.01, 0.05, 0.1, 0.2],
                    'max_features': [0.5, 0.7, 1.0]
                }
            },
            'one_class_svm': {
                'name': 'One-Class SVM',
                'type': 'kernel',
                'description': 'Anomaly detection using one-class SVM',
                'hyperparameters': {
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'nu': [0.01, 0.05, 0.1, 0.2, 0.5],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'degree': [2, 3, 4, 5]
                }
            }
        }
        
        # Add successfully imported anomaly detection models
        for model_key, model_class in anomaly_detection_imports.items():
            if model_key in anomaly_configs:
                config = anomaly_configs[model_key].copy()
                config['class'] = model_class
                self.anomaly_detection_models[model_key] = config
    
    def get_clustering_models(self) -> Dict[str, Dict]:
        """Get all available clustering models"""
        return self.clustering_models
    
    def get_dimensionality_reduction_models(self) -> Dict[str, Dict]:
        """Get all available dimensionality reduction models"""
        return self.dimensionality_reduction_models
    
    def get_anomaly_detection_models(self) -> Dict[str, Dict]:
        """Get all available anomaly detection models"""
        return self.anomaly_detection_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all unsupervised models"""
        return {
            'clustering': self.clustering_models,
            'dimensionality_reduction': self.dimensionality_reduction_models,
            'anomaly_detection': self.anomaly_detection_models
        }
    
    def get_model_by_name(self, model_name: str, task_type: str = None):
        """Get a specific model by name"""
        if task_type == 'clustering':
            return self.clustering_models.get(model_name)
        elif task_type == 'dimensionality_reduction':
            return self.dimensionality_reduction_models.get(model_name)
        elif task_type == 'anomaly_detection':
            return self.anomaly_detection_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.clustering_models, self.dimensionality_reduction_models, self.anomaly_detection_models]:
                if model_name in models:
                    return models[model_name]
            return None
    
    def get_models_by_type(self, model_type: str, task_type: str = None) -> Dict[str, Dict]:
        """Get models by their type (e.g., 'density', 'centroid', 'linear')"""
        models = {}
        
        if task_type is None or task_type == 'clustering':
            for name, info in self.clustering_models.items():
                if info['type'] == model_type:
                    models[name] = info
        
        if task_type is None or task_type == 'dimensionality_reduction':
            for name, info in self.dimensionality_reduction_models.items():
                if info['type'] == model_type:
                    models[name] = info
        
        if task_type is None or task_type == 'anomaly_detection':
            for name, info in self.anomaly_detection_models.items():
                if info['type'] == model_type:
                    models[name] = info
        
        return models

class UnsupervisedModelTrainer:
    """Trainer class for unsupervised learning models"""
    
    def __init__(self):
        self.registry = UnsupervisedModelRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def train_model(self, model_name: str, X, task_type: str = None, 
                   hyperparameters: Dict = None) -> Dict:
        """Train a single unsupervised model"""
        
        model_info = self.registry.get_model_by_name(model_name, task_type)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            model = model_info['class'](**hyperparameters)
            
            # Train the model
            model.fit(X)
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': task_type,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None
            }
            
            # Get model-specific results
            if hasattr(model, 'labels_'):
                result['labels'] = model.labels_
                if hasattr(model.labels_, '__len__'):
                    unique_labels = np.unique(model.labels_)
                    result['n_clusters_found'] = len(unique_labels[unique_labels != -1])  # Exclude noise (-1)
            
            if hasattr(model, 'cluster_centers_'):
                result['cluster_centers'] = model.cluster_centers_
            
            if hasattr(model, 'inertia_'):
                result['inertia'] = model.inertia_
            
            if hasattr(model, 'explained_variance_ratio_'):
                result['explained_variance_ratio'] = model.explained_variance_ratio_
                result['cumulative_variance_ratio'] = np.cumsum(model.explained_variance_ratio_)
            
            if hasattr(model, 'components_'):
                if hasattr(model.components_, 'shape'):
                    result['n_components'] = model.components_.shape[0]
                else:
                    result['n_components'] = len(model.components_) if model.components_ is not None else 0
            
            # Get model parameters
            if hasattr(model, 'get_params'):
                try:
                    result['model_params'] = model.get_params()
                except:
                    pass
            
            # Store trained model
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': task_type,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e)
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], X, task_type: str = None) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(model_name, X, task_type)
            results.append(result)
        
        return results
    
    def train_all_models(self, X, task_type: str = 'clustering') -> List[Dict]:
        """Train all available models for a given task type"""
        
        if task_type == 'clustering':
            models = list(self.registry.get_clustering_models().keys())
        elif task_type == 'dimensionality_reduction':
            models = list(self.registry.get_dimensionality_reduction_models().keys())
        elif task_type == 'anomaly_detection':
            models = list(self.registry.get_anomaly_detection_models().keys())
        else:
            raise ValueError("task_type must be 'clustering', 'dimensionality_reduction', or 'anomaly_detection'")
        
        return self.train_multiple_models(models, X, task_type)
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_history:
            summary_row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('model_category', 'Unknown'),
                'Training_Successful': result['training_successful'],
                'N_Clusters': result.get('n_clusters_found', 'N/A'),
                'Inertia': result.get('inertia', 'N/A'),
                'Explained_Variance': result.get('explained_variance_ratio', 'N/A'),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_unsupervised_models_info() -> Dict:
    """Get information about all available unsupervised models"""
    registry = UnsupervisedModelRegistry()
    return registry.get_all_models()

def create_model_comparison_report(training_results: List[Dict]) -> pd.DataFrame:
    """Create a comparison report from training results"""
    if not training_results:
        return pd.DataFrame()
    
    comparison_data = []
    for result in training_results:
        if result['training_successful']:
            row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('model_category', 'Unknown'),
                'N_Clusters': result.get('n_clusters_found', 'N/A'),
                'Inertia': result.get('inertia', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            
            # Add explained variance for dimensionality reduction
            if 'explained_variance_ratio' in result:
                variance_ratio = result['explained_variance_ratio']
                if hasattr(variance_ratio, '__len__'):
                    row['Explained_Variance'] = f"{np.sum(variance_ratio):.3f}"
                else:
                    row['Explained_Variance'] = f"{variance_ratio:.3f}"
            else:
                row['Explained_Variance'] = 'N/A'
            
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = UnsupervisedModelRegistry()
    trainer = UnsupervisedModelTrainer()
    
    print("Available Clustering Models:")
    for name, info in registry.get_clustering_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Dimensionality Reduction Models:")
    for name, info in registry.get_dimensionality_reduction_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Anomaly Detection Models:")
    for name, info in registry.get_anomaly_detection_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")