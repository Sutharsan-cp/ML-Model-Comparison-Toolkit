"""
Ensemble Learning Models Module
Consolidates all ensemble learning algorithms for the ML Comparison Toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Add models path to sys.path
classification_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble', 'classification')
regression_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble', 'regression')

if classification_path not in sys.path:
    sys.path.append(classification_path)
if regression_path not in sys.path:
    sys.path.append(regression_path)

# Import dictionaries
classification_imports = {}
regression_imports = {}

# Classification ensemble methods
try:
    from random_forest_classifier import RandomForestClassifier
    classification_imports['random_forest'] = RandomForestClassifier
except ImportError as e:
    print(f"Warning: Could not import RandomForestClassifier: {e}")

try:
    from gradient_boosting_classifier import GradientBoostingClassifier
    classification_imports['gradient_boosting'] = GradientBoostingClassifier
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingClassifier: {e}")

try:
    from xgboost_classifier import XGBoostClassifier
    classification_imports['xgboost'] = XGBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import XGBoostClassifier: {e}")

try:
    from lightgbm_classifier import LightGBMClassifier
    classification_imports['lightgbm'] = LightGBMClassifier
except ImportError as e:
    print(f"Warning: Could not import LightGBMClassifier: {e}")

try:
    from catboost_classifier import CatBoostClassifier
    classification_imports['catboost'] = CatBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import CatBoostClassifier: {e}")

try:
    from bagging_classifier import BaggingClassifier
    classification_imports['bagging'] = BaggingClassifier
except ImportError as e:
    print(f"Warning: Could not import BaggingClassifier: {e}")

try:
    from stacking_classifier import StackingClassifier
    classification_imports['stacking'] = StackingClassifier
except ImportError as e:
    print(f"Warning: Could not import StackingClassifier: {e}")

# Regression ensemble methods
try:
    from random_forest_regressor import RandomForestRegressor
    regression_imports['random_forest'] = RandomForestRegressor
except ImportError as e:
    print(f"Warning: Could not import RandomForestRegressor: {e}")

try:
    from gradient_boosting_regressor import GradientBoostingRegressor
    regression_imports['gradient_boosting'] = GradientBoostingRegressor
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingRegressor: {e}")

try:
    from xgboost_regressor import XGBoostRegressor
    regression_imports['xgboost'] = XGBoostRegressor
except ImportError as e:
    print(f"Warning: Could not import XGBoostRegressor: {e}")

try:
    from lightgbm_regressor import LightGBMRegressor
    regression_imports['lightgbm'] = LightGBMRegressor
except ImportError as e:
    print(f"Warning: Could not import LightGBMRegressor: {e}")

try:
    from bagging_regressor import BaggingRegressor
    regression_imports['bagging'] = BaggingRegressor
except ImportError as e:
    print(f"Warning: Could not import BaggingRegressor: {e}")

try:
    from stacking_regressor import StackingRegressor
    regression_imports['stacking'] = StackingRegressor
except ImportError as e:
    print(f"Warning: Could not import StackingRegressor: {e}")

class EnsembleLearningRegistry:
    """Registry for all ensemble learning models"""
    
    def __init__(self):
        # Build classification models dictionary
        self.classification_models = {}
        
        # Define classification model configurations
        classification_configs = {
            'random_forest': {
                'name': 'Random Forest Classifier',
                'type': 'bagging',
                'task': 'classification',
                'description': 'Random forest ensemble for classification',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting Classifier',
                'type': 'boosting',
                'task': 'classification',
                'description': 'Gradient boosting ensemble for classification',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'name': 'XGBoost Classifier',
                'type': 'boosting',
                'task': 'classification',
                'description': 'Extreme gradient boosting for classification',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'name': 'LightGBM Classifier',
                'type': 'boosting',
                'task': 'classification',
                'description': 'Light gradient boosting for classification',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'catboost': {
                'name': 'CatBoost Classifier',
                'type': 'boosting',
                'task': 'classification',
                'description': 'Categorical boosting for classification',
                'hyperparameters': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                }
            },
            'bagging': {
                'name': 'Bagging Classifier',
                'type': 'bagging',
                'task': 'classification',
                'description': 'Bootstrap aggregating for classification',
                'hyperparameters': {
                    'n_estimators': [10, 50, 100],
                    'max_samples': [0.5, 0.7, 1.0],
                    'max_features': [0.5, 0.7, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False]
                }
            },
            'stacking': {
                'name': 'Stacking Classifier',
                'type': 'stacking',
                'task': 'classification',
                'description': 'Stacking ensemble for classification',
                'hyperparameters': {
                    'cv': [3, 5, 10],
                    'stack_method': ['auto', 'predict_proba', 'decision_function'],
                    'n_jobs': [1, -1],
                    'passthrough': [True, False]
                }
            }
        }
        
        # Add successfully imported classification models
        for model_key, model_class in classification_imports.items():
            if model_key in classification_configs:
                config = classification_configs[model_key].copy()
                config['class'] = model_class
                self.classification_models[model_key] = config
        
        # Build regression models dictionary
        self.regression_models = {}
        
        # Define regression model configurations
        regression_configs = {
            'random_forest': {
                'name': 'Random Forest Regressor',
                'type': 'bagging',
                'task': 'regression',
                'description': 'Random forest ensemble for regression',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting Regressor',
                'type': 'boosting',
                'task': 'regression',
                'description': 'Gradient boosting ensemble for regression',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'name': 'XGBoost Regressor',
                'type': 'boosting',
                'task': 'regression',
                'description': 'Extreme gradient boosting for regression',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'name': 'LightGBM Regressor',
                'type': 'boosting',
                'task': 'regression',
                'description': 'Light gradient boosting for regression',
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'bagging': {
                'name': 'Bagging Regressor',
                'type': 'bagging',
                'task': 'regression',
                'description': 'Bootstrap aggregating for regression',
                'hyperparameters': {
                    'n_estimators': [10, 50, 100],
                    'max_samples': [0.5, 0.7, 1.0],
                    'max_features': [0.5, 0.7, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False]
                }
            },
            'stacking': {
                'name': 'Stacking Regressor',
                'type': 'stacking',
                'task': 'regression',
                'description': 'Stacking ensemble for regression',
                'hyperparameters': {
                    'cv': [3, 5, 10],
                    'n_jobs': [1, -1],
                    'passthrough': [True, False]
                }
            }
        }
        
        # Add successfully imported regression models
        for model_key, model_class in regression_imports.items():
            if model_key in regression_configs:
                config = regression_configs[model_key].copy()
                config['class'] = model_class
                self.regression_models[model_key] = config
    
    def get_classification_models(self) -> Dict[str, Dict]:
        """Get all available classification models"""
        return self.classification_models
    
    def get_regression_models(self) -> Dict[str, Dict]:
        """Get all available regression models"""
        return self.regression_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all ensemble models"""
        return {
            'classification': self.classification_models,
            'regression': self.regression_models
        }
    
    def get_model_by_name(self, model_name: str, task: str = None):
        """Get a specific model by name"""
        if task == 'classification':
            return self.classification_models.get(model_name)
        elif task == 'regression':
            return self.regression_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.classification_models, self.regression_models]:
                if model_name in models:
                    return models[model_name]
            return None
    
    def get_models_by_type(self, ensemble_type: str) -> Dict[str, Dict]:
        """Get models by ensemble type (bagging, boosting, stacking)"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for task, models in all_models.items():
            for name, info in models.items():
                if info.get('type') == ensemble_type:
                    suitable_models[f"{task}_{name}"] = info
        
        return suitable_models

class EnsembleLearningTrainer:
    """Trainer class for ensemble learning models"""
    
    def __init__(self):
        self.registry = EnsembleLearningRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray = None, y_test: np.ndarray = None,
                   task: str = 'classification', hyperparameters: Dict = None, 
                   cv_folds: int = 5, verbose: bool = False) -> Dict:
        """Train a single ensemble model"""
        
        model_info = self.registry.get_model_by_name(model_name, task)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found for task '{task}'")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            model = model_info['class'](**hyperparameters)
            
            if verbose:
                print(f"Training {model_info['name']}...")
                print(f"Training samples: {len(X_train)}")
                if X_test is not None:
                    print(f"Test samples: {len(X_test)}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            try:
                if task == 'classification':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            except Exception as cv_error:
                if verbose:
                    print(f"Cross-validation error: {cv_error}")
                cv_mean = None
                cv_std = None
            
            # Make predictions on test set if provided
            if X_test is not None and y_test is not None:
                try:
                    y_pred = model.predict(X_test)
                    
                    if task == 'classification':
                        test_score = accuracy_score(y_test, y_pred)
                        try:
                            classification_rep = classification_report(y_test, y_pred, output_dict=True)
                        except:
                            classification_rep = None
                        regression_metrics = None
                    else:
                        test_score = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        regression_metrics = {'mse': mse, 'rmse': rmse, 'r2': test_score}
                        classification_rep = None
                    
                except Exception as pred_error:
                    if verbose:
                        print(f"Prediction error: {pred_error}")
                    y_pred = None
                    test_score = None
                    classification_rep = None
                    regression_metrics = None
            else:
                y_pred = None
                test_score = None
                classification_rep = None
                regression_metrics = None
            
            # Feature importance (if available)
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    feature_importance = np.abs(model.coef_).flatten()
            except:
                pass
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'task': task,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0,
                'cv_mean_score': cv_mean,
                'cv_std_score': cv_std,
                'test_score': test_score,
                'classification_report': classification_rep,
                'regression_metrics': regression_metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
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
                'task': task,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e),
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], X_train: np.ndarray, 
                            y_train: np.ndarray, X_test: np.ndarray = None, 
                            y_test: np.ndarray = None, task: str = 'classification',
                            cv_folds: int = 5, verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(
                model_name, X_train, y_train, X_test, y_test, 
                task, cv_folds=cv_folds, verbose=verbose
            )
            results.append(result)
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        task: str = 'classification', cv_folds: int = 5, 
                        verbose: bool = False) -> List[Dict]:
        """Train all available models for a given task"""
        
        if task == 'classification':
            models = list(self.registry.get_classification_models().keys())
        else:
            models = list(self.registry.get_regression_models().keys())
        
        return self.train_multiple_models(
            models, X_train, y_train, X_test, y_test, task, cv_folds, verbose
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
                'Task': result['task'],
                'Training_Successful': result['training_successful'],
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Test_Samples': result.get('test_samples', 'N/A'),
                'CV_Mean_Score': result.get('cv_mean_score', 'N/A'),
                'CV_Std_Score': result.get('cv_std_score', 'N/A'),
                'Test_Score': result.get('test_score', 'N/A'),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'test_score') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
        
        if not models_with_metric:
            return None
        
        return max(models_with_metric, key=lambda x: x[metric])
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary for models that support it"""
        importance_data = []
        
        for result in self.training_history:
            if result['training_successful'] and result.get('feature_importance') is not None:
                importance = result['feature_importance']
                for i, imp in enumerate(importance):
                    importance_data.append({
                        'Model': result['model_display_name'],
                        'Feature_Index': i,
                        'Importance': imp
                    })
        
        return pd.DataFrame(importance_data)
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_ensemble_models_info() -> Dict:
    """Get information about all available ensemble models"""
    registry = EnsembleLearningRegistry()
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
                'Task': result['task'],
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Test_Samples': result.get('test_samples', 'N/A'),
                'CV_Mean_Score': result.get('cv_mean_score', 'N/A'),
                'Test_Score': result.get('test_score', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = EnsembleLearningRegistry()
    trainer = EnsembleLearningTrainer()
    
    print("Available Classification Models:")
    for name, info in registry.get_classification_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Regression Models:")
    for name, info in registry.get_regression_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nEnsemble Types:")
    for ensemble_type in ['bagging', 'boosting', 'stacking']:
        models = registry.get_models_by_type(ensemble_type)
        print(f"- {ensemble_type.title()}: {len(models)} models")