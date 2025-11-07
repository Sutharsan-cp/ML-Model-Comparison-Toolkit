"""
Unsupervised Learning Models Module
Consolidates all unsupervised learning algorithms for the ML Comparison Toolkit
Uses actual models from backend/app/models/unsupervised directory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Add models path to sys.path
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unsupervised')
if models_path not in sys.path:
    sys.path.append(models_path)

# Import dictionaries for different categories
clustering_imports = {}
dimensionality_reduction_imports = {}
anomaly_detection_imports = {}
manifold_learning_imports = {}

# Clustering algorithms
try:
    from kmeans import KMeans
    clustering_imports['kmeans'] = KMeans
except ImportError as e:
    print(f"Warning: Could not import KMeans: {e}")

try:
    from DBSCAN import DBSCAN
    clustering_imports['dbscan'] = DBSCAN
except ImportError as e:
    print(f"Warning: Could not import DBSCAN: {e}")

# Create wrapper for Gaussian Mixture to fix import issues
class GaussianMixtureWrapper:
    """Wrapper for Gaussian Mixture Model"""
    def __init__(self, n_components=3, max_iters=100, tol=1e-3, 
                 init_params='kmeans', random_state=None, reg_covar=1e-6):
        from sklearn.mixture import GaussianMixture as SKGaussianMixture
        self.model = SKGaussianMixture(
            n_components=n_components,
            max_iter=max_iters,
            tol=tol,
            init_params=init_params,
            random_state=random_state,
            reg_covar=reg_covar
        )
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.init_params = init_params
        self.random_state = random_state
        self.reg_covar = reg_covar
    
    def fit(self, X):
        self.model.fit(X)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def fit_predict(self, X):
        return self.model.fit_predict(X)
    
    def score_samples(self, X):
        return self.model.score_samples(X)
    
    def bic(self, X):
        return self.model.bic(X)
    
    @property
    def weights_(self):
        return self.model.weights_
    
    @property
    def means_(self):
        return self.model.means_
    
    @property
    def covariances_(self):
        return self.model.covariances_
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'max_iters': self.max_iters,
            'tol': self.tol,
            'init_params': self.init_params,
            'random_state': self.random_state,
            'reg_covar': self.reg_covar
        }

clustering_imports['gaussian_mixture'] = GaussianMixtureWrapper

# Create wrapper for Hierarchical Clustering to fix issues
class HierarchicalClusteringWrapper:
    """Wrapper for Hierarchical Clustering"""
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        from sklearn.cluster import AgglomerativeClustering
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric if linkage != 'ward' else 'euclidean'
        )
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
    
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'metric': self.metric
        }

clustering_imports['hierarchical'] = HierarchicalClusteringWrapper

try:
    from agglomerative import AgglomerativeClustering
    clustering_imports['agglomerative'] = AgglomerativeClustering
except ImportError as e:
    print(f"Warning: Could not import AgglomerativeClustering: {e}")

# Create wrapper for Spectral Clustering to fix import issues
class SpectralClusteringWrapper:
    """Wrapper for Spectral Clustering"""
    def __init__(self, n_clusters=8, gamma=1.0, random_state=None):
        from sklearn.cluster import SpectralClustering as SKSpectralClustering
        self.model = SKSpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            random_state=random_state,
            affinity='rbf'
        )
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'gamma': self.gamma,
            'random_state': self.random_state
        }

clustering_imports['spectral_clustering'] = SpectralClusteringWrapper

try:
    from mean_shift import MeanShift
    clustering_imports['mean_shift'] = MeanShift
except ImportError as e:
    print(f"Warning: Could not import MeanShift: {e}")

# Dimensionality reduction algorithms
try:
    from PCA import PCA
    dimensionality_reduction_imports['pca'] = PCA
except ImportError as e:
    print(f"Warning: Could not import PCA: {e}")

# Create wrapper for ICA to fix shape issues
class FastICAWrapper:
    """Wrapper for FastICA"""
    def __init__(self, n_components=None, algorithm='parallel', fun='logcosh',
                 max_iter=200, tol=1e-4, random_state=None):
        from sklearn.decomposition import FastICA as SKFastICA
        self.n_components = n_components
        self.algorithm = algorithm
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.model = SKFastICA(
            n_components=n_components,
            algorithm=algorithm,
            fun=fun,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    
    def fit(self, X):
        self.model.fit(X)
        self.components_ = self.model.components_
        self.mixing_ = self.model.mixing_
        self.mean_ = self.model.mean_
        self.n_iter_ = self.model.n_iter_
        return self
    
    def transform(self, X):
        return self.model.transform(X)
    
    def fit_transform(self, X):
        return self.model.fit_transform(X)
    
    def inverse_transform(self, X_transformed):
        return self.model.inverse_transform(X_transformed)
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'algorithm': self.algorithm,
            'fun': self.fun,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_iter': getattr(self, 'n_iter_', None),
            'components_shape': self.components_.shape if hasattr(self, 'components_') else None
        }

dimensionality_reduction_imports['ica'] = FastICAWrapper

# Manifold learning algorithms
try:
    from tsne import TSNE
    manifold_learning_imports['tsne'] = TSNE
except ImportError as e:
    print(f"Warning: Could not import TSNE: {e}")

# Anomaly detection algorithms
try:
    from isolation_forest import IsolationForest
    anomaly_detection_imports['isolation_forest'] = IsolationForest
except ImportError as e:
    print(f"Warning: Could not import IsolationForest: {e}")

class UnsupervisedLearningRegistry:
    """Registry for all unsupervised learning models"""
    
    def __init__(self):
        # Build clustering models dictionary
        self.clustering_models = {}
        
        # Define clustering model configurations
        clustering_configs = {
            'kmeans': {
                'name': 'K-Means',
                'type': 'clustering',
                'description': 'Partitional clustering algorithm using centroids',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 8, 10],
                    'max_iters': [100, 300, 500],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'init': ['k-means++', 'random'],
                    'random_state': [42, 123, 456]
                }
            },
            'dbscan': {
                'name': 'DBSCAN',
                'type': 'clustering',
                'description': 'Density-based clustering algorithm',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'eps': [0.1, 0.3, 0.5, 1.0, 2.0],
                    'min_samples': [3, 5, 10, 15],
                    'metric': ['euclidean', 'manhattan'],
                    'algorithm': ['auto']
                }
            },
            'gaussian_mixture': {
                'name': 'Gaussian Mixture Model',
                'type': 'clustering',
                'description': 'Probabilistic clustering using Gaussian distributions',
                'data_types': ['tabular'],
                'task_types': ['clustering', 'density_estimation'],
                'hyperparameters': {
                    'n_components': [2, 3, 4, 5, 8],
                    'max_iters': [50, 100, 200],
                    'tol': [1e-3, 1e-4],
                    'init_params': ['kmeans', 'random'],
                    'random_state': [42, 123, 456]
                }
            },
            'hierarchical': {
                'name': 'Hierarchical Clustering',
                'type': 'clustering',
                'description': 'Tree-based hierarchical clustering',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 8],
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                }
            },
            'agglomerative': {
                'name': 'Agglomerative Clustering',
                'type': 'clustering',
                'description': 'Bottom-up hierarchical clustering',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 8],
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'spectral_clustering': {
                'name': 'Spectral Clustering',
                'type': 'clustering',
                'description': 'Graph-based clustering using eigenvalues',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'n_clusters': [2, 3, 4, 5, 8],
                    'gamma': [0.1, 1.0, 10.0],
                    'random_state': [42, 123, 456]
                }
            },
            'mean_shift': {
                'name': 'Mean Shift',
                'type': 'clustering',
                'description': 'Mode-seeking clustering algorithm',
                'data_types': ['tabular'],
                'task_types': ['clustering'],
                'hyperparameters': {
                    'bandwidth': [0.5, 1.0, 2.0, 'auto'],
                    'max_iter': [100, 300, 500]
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
        dimensionality_reduction_configs = {
            'pca': {
                'name': 'Principal Component Analysis',
                'type': 'dimensionality_reduction',
                'description': 'Linear dimensionality reduction using SVD',
                'data_types': ['tabular'],
                'task_types': ['dimensionality_reduction', 'feature_extraction'],
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10, None],
                    'svd_solver': ['auto', 'full'],
                    'random_state': [42, 123, 456]
                }
            },
            'ica': {
                'name': 'Independent Component Analysis',
                'type': 'dimensionality_reduction',
                'description': 'Blind source separation using statistical independence',
                'data_types': ['tabular'],
                'task_types': ['dimensionality_reduction', 'source_separation'],
                'hyperparameters': {
                    'n_components': [2, 3, 5, 10, None],
                    'algorithm': ['parallel', 'deflation'],
                    'fun': ['logcosh', 'exp', 'cube'],
                    'max_iter': [100, 200, 500],
                    'tol': [1e-4, 1e-3],
                    'random_state': [42, 123, 456]
                }
            }
        }
        
        # Add successfully imported dimensionality reduction models
        for model_key, model_class in dimensionality_reduction_imports.items():
            if model_key in dimensionality_reduction_configs:
                config = dimensionality_reduction_configs[model_key].copy()
                config['class'] = model_class
                self.dimensionality_reduction_models[model_key] = config
        
        # Build manifold learning models dictionary
        self.manifold_learning_models = {}
        
        # Define manifold learning model configurations
        manifold_learning_configs = {
            'tsne': {
                'name': 't-SNE',
                'type': 'manifold_learning',
                'description': 'Non-linear dimensionality reduction for visualization',
                'data_types': ['tabular'],
                'task_types': ['dimensionality_reduction', 'visualization'],
                'hyperparameters': {
                    'n_components': [2, 3],
                    'perplexity': [5.0, 10.0, 30.0, 50.0],
                    'early_exaggeration': [4.0, 12.0, 20.0],
                    'learning_rate': [50.0, 200.0, 500.0],
                    'n_iter': [250, 500, 1000],
                    'random_state': [42, 123, 456]
                }
            }
        }
        
        # Add successfully imported manifold learning models
        for model_key, model_class in manifold_learning_imports.items():
            if model_key in manifold_learning_configs:
                config = manifold_learning_configs[model_key].copy()
                config['class'] = model_class
                self.manifold_learning_models[model_key] = config
        
        # Build anomaly detection models dictionary
        self.anomaly_detection_models = {}
        
        # Define anomaly detection model configurations
        anomaly_detection_configs = {
            'isolation_forest': {
                'name': 'Isolation Forest',
                'type': 'anomaly_detection',
                'description': 'Ensemble method for anomaly detection',
                'data_types': ['tabular'],
                'task_types': ['anomaly_detection', 'outlier_detection'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'max_samples': ['auto', 256, 512],
                    'contamination': ['auto', 0.05, 0.1, 0.2],
                    'max_features': [0.5, 1.0],
                    'random_state': [42, 123, 456]
                }
            }
        }
        
        # Add successfully imported anomaly detection models
        for model_key, model_class in anomaly_detection_imports.items():
            if model_key in anomaly_detection_configs:
                config = anomaly_detection_configs[model_key].copy()
                config['class'] = model_class
                self.anomaly_detection_models[model_key] = config
    
    def get_clustering_models(self) -> Dict[str, Dict]:
        """Get all available clustering models"""
        return self.clustering_models
    
    def get_dimensionality_reduction_models(self) -> Dict[str, Dict]:
        """Get all available dimensionality reduction models"""
        return self.dimensionality_reduction_models
    
    def get_manifold_learning_models(self) -> Dict[str, Dict]:
        """Get all available manifold learning models"""
        return self.manifold_learning_models
    
    def get_anomaly_detection_models(self) -> Dict[str, Dict]:
        """Get all available anomaly detection models"""
        return self.anomaly_detection_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all unsupervised learning models"""
        return {
            'clustering': self.clustering_models,
            'dimensionality_reduction': self.dimensionality_reduction_models,
            'manifold_learning': self.manifold_learning_models,
            'anomaly_detection': self.anomaly_detection_models
        }
    
    def get_model_by_name(self, model_name: str, category: str = None):
        """Get a specific model by name"""
        if category == 'clustering':
            return self.clustering_models.get(model_name)
        elif category == 'dimensionality_reduction':
            return self.dimensionality_reduction_models.get(model_name)
        elif category == 'manifold_learning':
            return self.manifold_learning_models.get(model_name)
        elif category == 'anomaly_detection':
            return self.anomaly_detection_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.clustering_models, self.dimensionality_reduction_models, 
                          self.manifold_learning_models, self.anomaly_detection_models]:
                if model_name in models:
                    return models[model_name]
            return None
    
    def get_models_by_data_type(self, data_type: str) -> Dict[str, Dict]:
        """Get models suitable for a specific data type"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for category, models in all_models.items():
            for name, info in models.items():
                if data_type in info.get('data_types', []):
                    suitable_models[f"{category}_{name}"] = info
        
        return suitable_models
    
    def get_models_by_task_type(self, task_type: str) -> Dict[str, Dict]:
        """Get models suitable for a specific task type"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for category, models in all_models.items():
            for name, info in models.items():
                if task_type in info.get('task_types', []):
                    suitable_models[f"{category}_{name}"] = info
        
        return suitable_models

class UnsupervisedLearningTrainer:
    """Trainer class for unsupervised learning models"""
    
    def __init__(self):
        self.registry = UnsupervisedLearningRegistry()
        self.trained_models = {}
        self.training_history = []
        self.scaler = StandardScaler()
    
    def prepare_data(self, X, test_size: float = 0.2, random_state: int = 42, 
                    scale_features: bool = True) -> Tuple:
        """Prepare data for unsupervised learning"""
        
        X = np.array(X)
        
        # Scale features if requested
        if scale_features and X.ndim == 2:
            X = self.scaler.fit_transform(X)
        
        # Split data for evaluation (even though unsupervised)
        if len(X) > 10:  # Only split if we have enough data
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test = X, X
        
        return X_train, X_test
    
    def train_model(self, model_name: str, X_train: np.ndarray, X_test: np.ndarray = None,
                   category: str = None, hyperparameters: Dict = None, 
                   verbose: bool = False) -> Dict:
        """Train a single unsupervised learning model"""
        
        model_info = self.registry.get_model_by_name(model_name, category)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            model = model_info['class'](**hyperparameters)
            
            if verbose:
                print(f"Training {model_info['name']}...")
                print(f"Training samples: {len(X_train)}")
            
            # Train the model
            model.fit(X_train)
            
            # Evaluate the model
            evaluation_results = self._evaluate_model(model, model_info, X_train, X_test)
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'training_samples': len(X_train),
                'task_types': model_info.get('task_types', []),
                'data_types': model_info.get('data_types', []),
                **evaluation_results
            }
            
            # Store trained model
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e),
                'training_samples': len(X_train)
            }
            self.training_history.append(error_result)
            return error_result    

    def _evaluate_model(self, model, model_info, X_train, X_test):
        """Evaluate unsupervised model performance"""
        evaluation_results = {}
        
        try:
            model_type = model_info['type']
            
            if model_type == 'clustering':
                # Clustering evaluation
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                elif hasattr(model, 'fit_predict'):
                    labels = model.fit_predict(X_train)
                else:
                    labels = model.predict(X_train)
                
                # Calculate clustering metrics
                if len(np.unique(labels)) > 1:
                    try:
                        silhouette = silhouette_score(X_train, labels)
                        evaluation_results['silhouette_score'] = silhouette
                    except:
                        evaluation_results['silhouette_score'] = None
                
                evaluation_results['n_clusters'] = len(np.unique(labels[labels != -1])) if -1 in labels else len(np.unique(labels))
                evaluation_results['n_noise'] = np.sum(labels == -1) if -1 in labels else 0
                
                # Inertia for K-means like models
                if hasattr(model, 'inertia_'):
                    evaluation_results['inertia'] = model.inertia_
                
            elif model_type in ['dimensionality_reduction', 'manifold_learning']:
                # Dimensionality reduction evaluation
                if hasattr(model, 'transform'):
                    X_transformed = model.transform(X_train)
                    evaluation_results['output_dimensions'] = X_transformed.shape[1]
                    
                    # Explained variance for PCA
                    if hasattr(model, 'explained_variance_ratio_'):
                        evaluation_results['explained_variance_ratio'] = np.sum(model.explained_variance_ratio_)
                        evaluation_results['cumulative_variance'] = model.explained_variance_ratio_.tolist()
                
                # Reconstruction error for models that support it
                if hasattr(model, 'inverse_transform'):
                    try:
                        X_reconstructed = model.inverse_transform(model.transform(X_train))
                        reconstruction_error = np.mean((X_train - X_reconstructed) ** 2)
                        evaluation_results['reconstruction_error'] = reconstruction_error
                    except:
                        evaluation_results['reconstruction_error'] = None
                
            elif model_type == 'anomaly_detection':
                # Anomaly detection evaluation
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_train)
                    evaluation_results['anomaly_scores_mean'] = np.mean(scores)
                    evaluation_results['anomaly_scores_std'] = np.std(scores)
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_train)
                    evaluation_results['n_outliers'] = np.sum(predictions == -1)
                    evaluation_results['outlier_fraction'] = np.mean(predictions == -1)
            
            # Test set evaluation if available
            if X_test is not None and len(X_test) > 0:
                try:
                    if model_type == 'clustering' and hasattr(model, 'predict'):
                        test_labels = model.predict(X_test)
                        if len(np.unique(test_labels)) > 1:
                            test_silhouette = silhouette_score(X_test, test_labels)
                            evaluation_results['test_silhouette_score'] = test_silhouette
                    
                    elif model_type in ['dimensionality_reduction', 'manifold_learning'] and hasattr(model, 'transform'):
                        X_test_transformed = model.transform(X_test)
                        evaluation_results['test_output_dimensions'] = X_test_transformed.shape[1]
                    
                    elif model_type == 'anomaly_detection':
                        if hasattr(model, 'decision_function'):
                            test_scores = model.decision_function(X_test)
                            evaluation_results['test_anomaly_scores_mean'] = np.mean(test_scores)
                        
                        if hasattr(model, 'predict'):
                            test_predictions = model.predict(X_test)
                            evaluation_results['test_outlier_fraction'] = np.mean(test_predictions == -1)
                
                except Exception as e:
                    evaluation_results['test_evaluation_error'] = str(e)
        
        except Exception as e:
            evaluation_results['evaluation_error'] = str(e)
        
        return evaluation_results
    
    def train_multiple_models(self, model_names: List[str], X_train: np.ndarray, 
                            X_test: np.ndarray = None, category: str = None, 
                            verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(
                model_name, X_train, X_test, category, verbose=verbose
            )
            results.append(result)
        
        return results
    
    def train_category_models(self, category: str, X_train: np.ndarray, 
                            X_test: np.ndarray = None, verbose: bool = False) -> List[Dict]:
        """Train all models in a specific category"""
        
        category_models = getattr(self.registry, f'get_{category}_models')()
        model_names = list(category_models.keys())
        
        return self.train_multiple_models(
            model_names, X_train, X_test, category, verbose
        )
    
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
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Silhouette_Score': result.get('silhouette_score', 'N/A'),
                'N_Clusters': result.get('n_clusters', 'N/A'),
                'Explained_Variance': result.get('explained_variance_ratio', 'N/A'),
                'Outlier_Fraction': result.get('outlier_fraction', 'N/A'),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Data_Types': ', '.join(result.get('data_types', [])),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'silhouette_score') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
        
        if not models_with_metric:
            return None
        
        # For most metrics, higher is better
        return max(models_with_metric, key=lambda x: x[metric])
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_unsupervised_learning_info() -> Dict:
    """Get information about all available unsupervised learning models"""
    registry = UnsupervisedLearningRegistry()
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
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Silhouette_Score': result.get('silhouette_score', 'N/A'),
                'N_Clusters': result.get('n_clusters', 'N/A'),
                'Explained_Variance': result.get('explained_variance_ratio', 'N/A'),
                'Outlier_Fraction': result.get('outlier_fraction', 'N/A'),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Data_Types': ', '.join(result.get('data_types', [])),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = UnsupervisedLearningRegistry()
    trainer = UnsupervisedLearningTrainer()
    
    print("Available Clustering Models:")
    for name, info in registry.get_clustering_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Dimensionality Reduction Models:")
    for name, info in registry.get_dimensionality_reduction_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Manifold Learning Models:")
    for name, info in registry.get_manifold_learning_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Anomaly Detection Models:")
    for name, info in registry.get_anomaly_detection_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")