"""
Semi-Supervised Learning Models Module
Consolidates all semi-supervised learning algorithms for the ML Comparison Toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# Add models path to sys.path
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'semi_supervised')
if models_path not in sys.path:
    sys.path.append(models_path)

# Import dictionaries
graph_based_imports = {}
self_training_imports = {}
consistency_imports = {}
contrastive_imports = {}
deep_learning_imports = {}

# Graph-based methods
try:
    from label_propagation import LabelPropagation
    graph_based_imports['label_propagation'] = LabelPropagation
except ImportError as e:
    print(f"Warning: Could not import LabelPropagation: {e}")

try:
    from label_spreading import LabelSpreading
    graph_based_imports['label_spreading'] = LabelSpreading
except ImportError as e:
    print(f"Warning: Could not import LabelSpreading: {e}")

# Self-training methods
try:
    from self_training import SelfTraining
    self_training_imports['self_training'] = SelfTraining
except ImportError as e:
    print(f"Warning: Could not import SelfTraining: {e}")

try:
    from semi_supervised_svm import SemiSupervisedSVM
    self_training_imports['semi_supervised_svm'] = SemiSupervisedSVM
except ImportError as e:
    print(f"Warning: Could not import SemiSupervisedSVM: {e}")

# Consistency regularization methods
try:
    from mean_teacher import MeanTeacher
    consistency_imports['mean_teacher'] = MeanTeacher
except ImportError as e:
    print(f"Warning: Could not import MeanTeacher: {e}")

try:
    from virtual_adversarial import VirtualAdversarial
    consistency_imports['virtual_adversarial'] = VirtualAdversarial
except ImportError as e:
    print(f"Warning: Could not import VirtualAdversarial: {e}")

try:
    from fixmatch import FixMatch
    consistency_imports['fixmatch'] = FixMatch
except ImportError as e:
    print(f"Warning: Could not import FixMatch: {e}")

try:
    from mixmatch import MixMatch
    consistency_imports['mixmatch'] = MixMatch
except ImportError as e:
    print(f"Warning: Could not import MixMatch: {e}")

# Contrastive learning methods
try:
    from simclr import SimCLR
    contrastive_imports['simclr'] = SimCLR
except ImportError as e:
    print(f"Warning: Could not import SimCLR: {e}")

try:
    from byol import BYOL
    contrastive_imports['byol'] = BYOL
except ImportError as e:
    print(f"Warning: Could not import BYOL: {e}")

try:
    from vicreg import VICReg
    contrastive_imports['vicreg'] = VICReg
except ImportError as e:
    print(f"Warning: Could not import VICReg: {e}")

try:
    from DINO import DINO
    contrastive_imports['dino'] = DINO
except ImportError as e:
    print(f"Warning: Could not import DINO: {e}")

# Deep learning methods
try:
    from ladder_network import LadderNetwork
    deep_learning_imports['ladder_network'] = LadderNetwork
except ImportError as e:
    print(f"Warning: Could not import LadderNetwork: {e}")

try:
    from masked_autoencoder import MaskedAutoencoder
    deep_learning_imports['masked_autoencoder'] = MaskedAutoencoder
except ImportError as e:
    print(f"Warning: Could not import MaskedAutoencoder: {e}")

try:
    from sway import SWAY
    deep_learning_imports['sway'] = SWAY
except ImportError as e:
    print(f"Warning: Could not import SWAY: {e}")

class SemiSupervisedLearningRegistry:
    """Registry for all semi-supervised learning models"""
    
    def __init__(self):
        # Build graph-based models dictionary
        self.graph_based_models = {}
        
        # Define graph-based model configurations
        graph_based_configs = {
            'label_propagation': {
                'name': 'Label Propagation',
                'type': 'graph_based',
                'description': 'Graph-based label propagation algorithm',
                'data_types': ['tabular', 'graph'],
                'hyperparameters': {
                    'kernel': ['rbf', 'knn'],
                    'gamma': [0.1, 1.0, 10.0],
                    'n_neighbors': [5, 10, 20],
                    'max_iter': [100, 300, 1000]
                }
            },
            'label_spreading': {
                'name': 'Label Spreading',
                'type': 'graph_based',
                'description': 'Graph-based label spreading algorithm',
                'data_types': ['tabular', 'graph'],
                'hyperparameters': {
                    'kernel': ['rbf', 'knn'],
                    'gamma': [0.1, 1.0, 10.0],
                    'alpha': [0.1, 0.2, 0.5],
                    'n_neighbors': [5, 10, 20],
                    'max_iter': [100, 300, 1000]
                }
            }
        }
        
        # Add successfully imported graph-based models
        for model_key, model_class in graph_based_imports.items():
            if model_key in graph_based_configs:
                config = graph_based_configs[model_key].copy()
                config['class'] = model_class
                self.graph_based_models[model_key] = config
        
        # Build self-training models dictionary
        self.self_training_models = {}
        
        # Define self-training model configurations
        self_training_configs = {
            'self_training': {
                'name': 'Self-Training',
                'type': 'self_training',
                'description': 'Self-training with confidence threshold',
                'data_types': ['tabular', 'image', 'text'],
                'hyperparameters': {
                    'threshold': [0.7, 0.8, 0.9],
                    'max_iter': [10, 20, 50],
                    'base_classifier': ['svm', 'random_forest', 'logistic_regression']
                }
            },
            'semi_supervised_svm': {
                'name': 'Semi-Supervised SVM',
                'type': 'self_training',
                'description': 'SVM with semi-supervised learning',
                'data_types': ['tabular'],
                'hyperparameters': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 1.0],
                    'max_iter': [100, 500, 1000]
                }
            }
        }
        
        # Add successfully imported self-training models
        for model_key, model_class in self_training_imports.items():
            if model_key in self_training_configs:
                config = self_training_configs[model_key].copy()
                config['class'] = model_class
                self.self_training_models[model_key] = config
        
        # Build consistency regularization models dictionary
        self.consistency_models = {}
        
        # Define consistency regularization model configurations
        consistency_configs = {
            'mean_teacher': {
                'name': 'Mean Teacher',
                'type': 'consistency',
                'description': 'Mean teacher consistency regularization',
                'data_types': ['image', 'tabular'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'ema_decay': [0.99, 0.999, 0.9999],
                    'consistency_weight': [0.1, 1.0, 10.0],
                    'epochs': [50, 100, 200]
                }
            },
            'virtual_adversarial': {
                'name': 'Virtual Adversarial Training',
                'type': 'consistency',
                'description': 'Virtual adversarial training for semi-supervised learning',
                'data_types': ['image', 'tabular'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'xi': [1e-6, 1e-5, 1e-4],
                    'epsilon': [0.1, 1.0, 10.0],
                    'epochs': [50, 100, 200]
                }
            },
            'fixmatch': {
                'name': 'FixMatch',
                'type': 'consistency',
                'description': 'FixMatch semi-supervised learning',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'threshold': [0.9, 0.95, 0.99],
                    'lambda_u': [1.0, 5.0, 10.0],
                    'epochs': [100, 200, 500]
                }
            },
            'mixmatch': {
                'name': 'MixMatch',
                'type': 'consistency',
                'description': 'MixMatch semi-supervised learning',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'alpha': [0.75, 1.0, 2.0],
                    'lambda_u': [75, 100, 150],
                    'T': [0.5, 1.0, 2.0],
                    'epochs': [100, 200, 500]
                }
            }
        }
        
        # Add successfully imported consistency models
        for model_key, model_class in consistency_imports.items():
            if model_key in consistency_configs:
                config = consistency_configs[model_key].copy()
                config['class'] = model_class
                self.consistency_models[model_key] = config
        
        # Build contrastive learning models dictionary
        self.contrastive_models = {}
        
        # Define contrastive learning model configurations
        contrastive_configs = {
            'simclr': {
                'name': 'SimCLR',
                'type': 'contrastive',
                'description': 'Simple contrastive learning of visual representations',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'temperature': [0.1, 0.5, 1.0],
                    'batch_size': [64, 128, 256],
                    'epochs': [100, 200, 500]
                }
            },
            'byol': {
                'name': 'BYOL',
                'type': 'contrastive',
                'description': 'Bootstrap Your Own Latent representation learning',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'tau': [0.99, 0.996, 0.999],
                    'batch_size': [64, 128, 256],
                    'epochs': [100, 200, 500]
                }
            },
            'vicreg': {
                'name': 'VICReg',
                'type': 'contrastive',
                'description': 'Variance-Invariance-Covariance Regularization',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'sim_coeff': [25.0, 50.0, 100.0],
                    'std_coeff': [25.0, 50.0, 100.0],
                    'cov_coeff': [1.0, 5.0, 10.0],
                    'epochs': [100, 200, 500]
                }
            },
            'dino': {
                'name': 'DINO',
                'type': 'contrastive',
                'description': 'Self-distillation with no labels',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.0005, 0.001, 0.003],
                    'teacher_temp': [0.04, 0.07, 0.1],
                    'student_temp': [0.1, 0.2, 0.5],
                    'center_momentum': [0.9, 0.99, 0.999],
                    'epochs': [100, 200, 500]
                }
            }
        }
        
        # Add successfully imported contrastive models
        for model_key, model_class in contrastive_imports.items():
            if model_key in contrastive_configs:
                config = contrastive_configs[model_key].copy()
                config['class'] = model_class
                self.contrastive_models[model_key] = config
        
        # Build deep learning models dictionary
        self.deep_learning_models = {}
        
        # Define deep learning model configurations
        deep_learning_configs = {
            'ladder_network': {
                'name': 'Ladder Network',
                'type': 'deep_learning',
                'description': 'Ladder networks for semi-supervised learning',
                'data_types': ['image', 'tabular'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'noise_std': [0.1, 0.3, 0.5],
                    'denoising_cost': [0.1, 1.0, 10.0],
                    'epochs': [50, 100, 200]
                }
            },
            'masked_autoencoder': {
                'name': 'Masked Autoencoder',
                'type': 'deep_learning',
                'description': 'Masked autoencoder for self-supervised learning',
                'data_types': ['image'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'mask_ratio': [0.5, 0.75, 0.9],
                    'patch_size': [8, 16, 32],
                    'epochs': [100, 200, 500]
                }
            },
            'sway': {
                'name': 'SWAY',
                'type': 'deep_learning',
                'description': 'Self-supervised learning with augmented views',
                'data_types': ['image', 'tabular'],
                'hyperparameters': {
                    'learning_rate': [0.001, 0.003, 0.01],
                    'temperature': [0.1, 0.5, 1.0],
                    'projection_dim': [64, 128, 256],
                    'epochs': [100, 200, 500]
                }
            }
        }
        
        # Add successfully imported deep learning models
        for model_key, model_class in deep_learning_imports.items():
            if model_key in deep_learning_configs:
                config = deep_learning_configs[model_key].copy()
                config['class'] = model_class
                self.deep_learning_models[model_key] = config
    
    def get_graph_based_models(self) -> Dict[str, Dict]:
        """Get all available graph-based models"""
        return self.graph_based_models
    
    def get_self_training_models(self) -> Dict[str, Dict]:
        """Get all available self-training models"""
        return self.self_training_models
    
    def get_consistency_models(self) -> Dict[str, Dict]:
        """Get all available consistency regularization models"""
        return self.consistency_models
    
    def get_contrastive_models(self) -> Dict[str, Dict]:
        """Get all available contrastive learning models"""
        return self.contrastive_models
    
    def get_deep_learning_models(self) -> Dict[str, Dict]:
        """Get all available deep learning models"""
        return self.deep_learning_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all semi-supervised models"""
        return {
            'graph_based': self.graph_based_models,
            'self_training': self.self_training_models,
            'consistency': self.consistency_models,
            'contrastive': self.contrastive_models,
            'deep_learning': self.deep_learning_models
        }
    
    def get_model_by_name(self, model_name: str, category: str = None):
        """Get a specific model by name"""
        if category == 'graph_based':
            return self.graph_based_models.get(model_name)
        elif category == 'self_training':
            return self.self_training_models.get(model_name)
        elif category == 'consistency':
            return self.consistency_models.get(model_name)
        elif category == 'contrastive':
            return self.contrastive_models.get(model_name)
        elif category == 'deep_learning':
            return self.deep_learning_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.graph_based_models, self.self_training_models, 
                          self.consistency_models, self.contrastive_models, 
                          self.deep_learning_models]:
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

class SemiSupervisedLearningTrainer:
    """Trainer class for semi-supervised learning models"""
    
    def __init__(self):
        self.registry = SemiSupervisedLearningRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def prepare_semi_supervised_data(self, X, y, labeled_ratio: float = 0.1, 
                                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for semi-supervised learning by creating labeled/unlabeled splits"""
        
        # Split data into labeled and unlabeled portions
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=labeled_ratio, random_state=random_state, stratify=y
        )
        
        return X_labeled, X_unlabeled, y_labeled, y_unlabeled
    
    def train_model(self, model_name: str, X_labeled: np.ndarray, y_labeled: np.ndarray,
                   X_unlabeled: np.ndarray, y_unlabeled: np.ndarray = None,
                   X_test: np.ndarray = None, y_test: np.ndarray = None,
                   category: str = None, hyperparameters: Dict = None, 
                   verbose: bool = False) -> Dict:
        """Train a single semi-supervised model"""
        
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
                print(f"Labeled samples: {len(X_labeled)}")
                print(f"Unlabeled samples: {len(X_unlabeled)}")
            
            # Train the model
            if hasattr(model, 'fit'):
                # For sklearn-like models
                if model_name in ['label_propagation', 'label_spreading']:
                    # Graph-based methods need combined data
                    X_combined = np.vstack([X_labeled, X_unlabeled])
                    y_combined = np.hstack([y_labeled, np.full(len(X_unlabeled), -1)])
                    model.fit(X_combined, y_combined)
                elif model_name == 'semi_supervised_svm':
                    # Semi-supervised SVM has different interface
                    model.fit(X_labeled, y_labeled)
                else:
                    # Other methods
                    try:
                        model.fit(X_labeled, y_labeled, X_unlabeled)
                    except TypeError:
                        # Fallback to standard fit
                        model.fit(X_labeled, y_labeled)
            elif hasattr(model, 'train'):
                # For custom training methods
                model.train(X_labeled, y_labeled, X_unlabeled)
            else:
                raise AttributeError(f"Model {model_name} has no fit or train method")
            
            # Make predictions on test set if provided
            if X_test is not None and y_test is not None:
                try:
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_test)
                    elif hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred = None
                    
                    if y_pred is not None:
                        accuracy = accuracy_score(y_test, y_pred)
                        try:
                            classification_rep = classification_report(y_test, y_pred, output_dict=True)
                        except:
                            classification_rep = None
                    else:
                        accuracy = None
                        classification_rep = None
                except Exception as pred_error:
                    if verbose:
                        print(f"Prediction error: {pred_error}")
                    y_pred = None
                    accuracy = None
                    classification_rep = None
            else:
                y_pred = None
                accuracy = None
                classification_rep = None
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'labeled_samples': len(X_labeled),
                'unlabeled_samples': len(X_unlabeled),
                'test_accuracy': accuracy,
                'classification_report': classification_rep,
                'predictions': y_pred
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
                'labeled_samples': len(X_labeled),
                'unlabeled_samples': len(X_unlabeled)
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], X_labeled: np.ndarray, 
                            y_labeled: np.ndarray, X_unlabeled: np.ndarray,
                            y_unlabeled: np.ndarray = None, X_test: np.ndarray = None, 
                            y_test: np.ndarray = None, category: str = None, 
                            verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(
                model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                X_test, y_test, category, verbose=verbose
            )
            results.append(result)
        
        return results
    
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
                'Labeled_Samples': result.get('labeled_samples', 'N/A'),
                'Unlabeled_Samples': result.get('unlabeled_samples', 'N/A'),
                'Test_Accuracy': result.get('test_accuracy', 'N/A'),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'test_accuracy') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
        
        if not models_with_metric:
            return None
        
        return max(models_with_metric, key=lambda x: x[metric])
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_semi_supervised_models_info() -> Dict:
    """Get information about all available semi-supervised models"""
    registry = SemiSupervisedLearningRegistry()
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
                'Labeled_Samples': result.get('labeled_samples', 'N/A'),
                'Unlabeled_Samples': result.get('unlabeled_samples', 'N/A'),
                'Test_Accuracy': result.get('test_accuracy', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    print("Available Graph-Based Models:")
    for name, info in registry.get_graph_based_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Self-Training Models:")
    for name, info in registry.get_self_training_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Consistency Models:")
    for name, info in registry.get_consistency_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Contrastive Models:")
    for name, info in registry.get_contrastive_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Deep Learning Models:")
    for name, info in registry.get_deep_learning_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")