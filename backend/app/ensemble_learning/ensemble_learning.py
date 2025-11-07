"""
Ensemble Learning Module
Consolidates all ensemble learning algorithms for the ML Comparison Toolkit
Uses actual models from backend/app/models/ensemble directory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
warnings.filterwarnings('ignore')

# Add models path to sys.path
classification_models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble', 'classification')
regression_models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble', 'regression')

if classification_models_path not in sys.path:
    sys.path.append(classification_models_path)
if regression_models_path not in sys.path:
    sys.path.append(regression_models_path)

# Import dictionaries for different categories
classification_imports = {}
regression_imports = {}

# Classification Models
try:
    from random_forest_classifier import RandomForestClassifier
    classification_imports['randomforest'] = RandomForestClassifier
except ImportError as e:
    print(f"Warning: Could not import RandomForestClassifier: {e}")

try:
    from xgboost_classifier import XGBoostClassifier
    classification_imports['xgboost'] = XGBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import XGBoostClassifier: {e}")

try:
    from gradient_boosting_classifier import GradientBoostingClassifier
    classification_imports['gradientboosting'] = GradientBoostingClassifier
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingClassifier: {e}")

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

# Regression Models
try:
    from random_forest_regressor import RandomForestRegressor
    regression_imports['randomforest'] = RandomForestRegressor
except ImportError as e:
    print(f"Warning: Could not import RandomForestRegressor: {e}")

try:
    from xgboost_regressor import XGBoostRegressor
    regression_imports['xgboost'] = XGBoostRegressor
except ImportError as e:
    print(f"Warning: Could not import XGBoostRegressor: {e}")

try:
    from gradient_boosting_regressor import GradientBoostingRegressor
    regression_imports['gradientboosting'] = GradientBoostingRegressor
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingRegressor: {e}")

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

try:
    from lightgbm_regressor import LightGBMRegressor
    regression_imports['lightgbm'] = LightGBMRegressor
except ImportError as e:
    print(f"Warning: Could not import LightGBMRegressor: {e}")

class EnsembleLearningRegistry:
    """Registry for all ensemble learning models"""
    
    def __init__(self):
        # Build classification models dictionary
        self.classification_models = {}
        
        # Define classification model configurations
        classification_configs = {
            'randomforest': {
                'name': 'Random Forest Classifier',
                'type': 'ensemble_classification',
                'description': 'Ensemble of decision trees using bootstrap aggregating (bagging)',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification', 'multiclass_classification'],
                'hyperparameters': {
                    'n_estimators': [10, 50, 100, 200, 300, 500],
                    'max_depth': [None, 3, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],
                    'bootstrap': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'xgboost': {
                'name': 'XGBoost Classifier',
                'type': 'ensemble_classification',
                'description': 'Extreme Gradient Boosting for classification tasks',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.5],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_lambda': [0, 0.1, 1.0, 10.0],
                    'reg_alpha': [0, 0.1, 1.0, 10.0],
                    'random_state': [42, 123, 456]
                }
            },
            'gradientboosting': {
                'name': 'Gradient Boosting Classifier',
                'type': 'ensemble_classification',
                'description': 'Sequential ensemble that builds models to correct previous errors',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'random_state': [42, 123, 456]
                }
            },
            'bagging': {
                'name': 'Bagging Classifier',
                'type': 'ensemble_classification',
                'description': 'Bootstrap Aggregating with customizable base estimators',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification', 'multiclass_classification'],
                'hyperparameters': {
                    'n_estimators': [10, 20, 50, 100, 200],
                    'max_samples': [0.5, 0.7, 0.8, 1.0],
                    'max_features': [0.5, 0.7, 0.8, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'stacking': {
                'name': 'Stacking Classifier',
                'type': 'ensemble_classification',
                'description': 'Meta-learning approach combining multiple base models',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification', 'multiclass_classification'],
                'hyperparameters': {
                    'cv': [3, 5, 10],
                    'use_probas': [True, False],
                    'passthrough': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'lightgbm': {
                'name': 'LightGBM Classifier',
                'type': 'ensemble_classification',
                'description': 'Light Gradient Boosting Machine for fast training',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification', 'multiclass_classification'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'num_leaves': [15, 31, 63, 127],
                    'min_child_samples': [10, 20, 30, 50],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_lambda': [0, 0.1, 1.0, 10.0],
                    'reg_alpha': [0, 0.1, 1.0, 10.0],
                    'random_state': [42, 123, 456]
                }
            },
            'catboost': {
                'name': 'CatBoost Classifier',
                'type': 'ensemble_classification',
                'description': 'Categorical Boosting for handling categorical features automatically',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['classification', 'binary_classification', 'multiclass_classification'],
                'hyperparameters': {
                    'iterations': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'depth': [3, 4, 5, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 254],
                    'random_state': [42, 123, 456]
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
            'randomforest': {
                'name': 'Random Forest Regressor',
                'type': 'ensemble_regression',
                'description': 'Ensemble of decision trees using bootstrap aggregating for regression',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'n_estimators': [10, 50, 100, 200, 300, 500],
                    'max_depth': [None, 3, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],
                    'bootstrap': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'xgboost': {
                'name': 'XGBoost Regressor',
                'type': 'ensemble_regression',
                'description': 'Extreme Gradient Boosting for regression tasks',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.5],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_lambda': [0, 0.1, 1.0, 10.0],
                    'reg_alpha': [0, 0.1, 1.0, 10.0],
                    'random_state': [42, 123, 456]
                }
            },
            'gradientboosting': {
                'name': 'Gradient Boosting Regressor',
                'type': 'ensemble_regression',
                'description': 'Sequential ensemble that builds models to correct previous errors for regression',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'random_state': [42, 123, 456]
                }
            },
            'bagging': {
                'name': 'Bagging Regressor',
                'type': 'ensemble_regression',
                'description': 'Bootstrap Aggregating with customizable base estimators for regression',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'n_estimators': [10, 20, 50, 100, 200],
                    'max_samples': [0.5, 0.7, 0.8, 1.0],
                    'max_features': [0.5, 0.7, 0.8, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'stacking': {
                'name': 'Stacking Regressor',
                'type': 'ensemble_regression',
                'description': 'Meta-learning approach combining multiple base models for regression',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'cv': [3, 5, 10],
                    'passthrough': [True, False],
                    'random_state': [42, 123, 456]
                }
            },
            'lightgbm': {
                'name': 'LightGBM Regressor',
                'type': 'ensemble_regression',
                'description': 'Light Gradient Boosting Machine for fast regression training',
                'data_types': ['tabular', 'numerical', 'categorical'],
                'task_types': ['regression', 'continuous_prediction'],
                'hyperparameters': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'num_leaves': [15, 31, 63, 127],
                    'min_child_samples': [10, 20, 30, 50],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_lambda': [0, 0.1, 1.0, 10.0],
                    'reg_alpha': [0, 0.1, 1.0, 10.0],
                    'random_state': [42, 123, 456]
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
        """Get all ensemble learning models"""
        return {
            'classification': self.classification_models,
            'regression': self.regression_models
        }
    
    def get_model_by_name(self, model_name: str, task_type: str = None):
        """Get a specific model by name"""
        if task_type == 'classification':
            return self.classification_models.get(model_name)
        elif task_type == 'regression':
            return self.regression_models.get(model_name)
        else:
            # Search in both categories
            if model_name in self.classification_models:
                return self.classification_models[model_name]
            elif model_name in self.regression_models:
                return self.regression_models[model_name]
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

class EnsembleLearningTrainer:
    """Trainer class for ensemble learning models"""
    
    def __init__(self):
        self.registry = EnsembleLearningRegistry()
        self.trained_models = {}
        self.training_history = []
        self.scaler = StandardScaler()
    
    def prepare_data(self, X, y, test_size: float = 0.2, random_state: int = 42, 
                    scale_features: bool = True, stratify: bool = None) -> Tuple:
        """Prepare data for ensemble learning"""
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features if requested
        if scale_features and X.ndim == 2:
            X = self.scaler.fit_transform(X)
        
        # Determine if we should stratify (only for classification with enough samples per class)
        stratify_param = None
        if stratify is None:
            # Auto-detect: check if y looks like classification data
            unique_values = np.unique(y)
            if len(unique_values) < len(y) * 0.5:  # Likely classification
                # Check if each class has at least 2 samples
                min_class_count = min([np.sum(y == val) for val in unique_values])
                if min_class_count >= 2:
                    stratify_param = y
        elif stratify:
            stratify_param = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_base_estimators_for_ensemble(self, model_name: str, task_type: str):
        """Prepare base estimators for ensemble methods like bagging and stacking"""
        base_estimators = []
        
        if task_type == 'classification':
            # Default base estimators for classification
            base_estimators = [
                DecisionTreeClassifier(max_depth=5, random_state=42),
                DecisionTreeClassifier(max_depth=10, random_state=123),
                DecisionTreeClassifier(max_depth=15, random_state=456)
            ]
            
            if model_name == 'stacking':
                # For stacking, use diverse base estimators
                base_estimators = [
                    ('dt1', DecisionTreeClassifier(max_depth=5, random_state=42)),
                    ('dt2', DecisionTreeClassifier(max_depth=10, random_state=123)),
                    ('dt3', DecisionTreeClassifier(max_depth=15, random_state=456))
                ]
        
        elif task_type == 'regression':
            # Default base estimators for regression
            base_estimators = [
                DecisionTreeRegressor(max_depth=5, random_state=42),
                DecisionTreeRegressor(max_depth=10, random_state=123),
                DecisionTreeRegressor(max_depth=15, random_state=456)
            ]
            
            if model_name == 'stacking':
                # For stacking, use diverse base estimators
                base_estimators = [
                    ('dt1', DecisionTreeRegressor(max_depth=5, random_state=42)),
                    ('dt2', DecisionTreeRegressor(max_depth=10, random_state=123)),
                    ('dt3', DecisionTreeRegressor(max_depth=15, random_state=456))
                ]
        
        return base_estimators
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, task_type: str,
                   hyperparameters: Dict = None, verbose: bool = False) -> Dict:
        """Train a single ensemble learning model"""
        
        model_info = self.registry.get_model_by_name(model_name, task_type)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found for task type '{task_type}'")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            # Special handling for ensemble methods that need base estimators
            if model_name in ['bagging', 'stacking']:
                if model_name == 'bagging':
                    base_estimators = self._prepare_base_estimators_for_ensemble(model_name, task_type)
                    # Use first base estimator for bagging
                    hyperparameters['base_estimator'] = base_estimators[0]
                elif model_name == 'stacking':
                    base_estimators = self._prepare_base_estimators_for_ensemble(model_name, task_type)
                    hyperparameters['estimators'] = base_estimators
                    
                    # Set final estimator
                    if task_type == 'classification':
                        hyperparameters['final_estimator'] = LogisticRegression(random_state=42)
                    else:
                        hyperparameters['final_estimator'] = LinearRegression()
            
            model = model_info['class'](**hyperparameters)
            
            if verbose:
                print(f"Training {model_info['name']}...")
                print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics based on task type
            if task_type == 'classification':
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Additional classification metrics
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Get probabilities if available
                train_probas = None
                test_probas = None
                if hasattr(model, 'predict_proba'):
                    try:
                        train_probas = model.predict_proba(X_train)
                        test_probas = model.predict_proba(X_test)
                    except:
                        pass
                
                metrics = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_precision': train_precision,
                    'test_precision': test_precision,
                    'train_recall': train_recall,
                    'test_recall': test_recall,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'train_probas': train_probas,
                    'test_probas': test_probas
                }
                
            else:  # regression
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                metrics = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'get_feature_importances'):
                try:
                    feature_importance = model.get_feature_importances()
                except:
                    pass
            elif hasattr(model, 'get_feature_importance'):
                try:
                    feature_importance = model.get_feature_importance()
                except:
                    pass
            elif hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'task_type': task_type,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'feature_importance': feature_importance,
                'task_types': model_info.get('task_types', []),
                'data_types': model_info.get('data_types', []),
                **metrics
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
                'task_type': task_type,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], X_train: np.ndarray, 
                            y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                            task_type: str, verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(
                model_name, X_train, y_train, X_test, y_test, task_type, verbose=verbose
            )
            results.append(result)
        
        return results
    
    def train_all_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray,
                                      verbose: bool = False) -> List[Dict]:
        """Train all classification models"""
        classification_models = list(self.registry.get_classification_models().keys())
        
        return self.train_multiple_models(
            classification_models, X_train, y_train, X_test, y_test, 
            'classification', verbose
        )
    
    def train_all_regression_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   verbose: bool = False) -> List[Dict]:
        """Train all regression models"""
        regression_models = list(self.registry.get_regression_models().keys())
        
        return self.train_multiple_models(
            regression_models, X_train, y_train, X_test, y_test, 
            'regression', verbose
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
                'Task_Type': result.get('task_type', 'Unknown'),
                'Training_Successful': result['training_successful'],
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Test_Samples': result.get('test_samples', 'N/A'),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Data_Types': ', '.join(result.get('data_types', [])),
                'Error': result.get('error', None)
            }
            
            # Add task-specific metrics
            if result.get('task_type') == 'classification':
                summary_row.update({
                    'Train_Accuracy': result.get('train_accuracy', 'N/A'),
                    'Test_Accuracy': result.get('test_accuracy', 'N/A'),
                    'Test_F1': result.get('test_f1', 'N/A'),
                    'Test_Precision': result.get('test_precision', 'N/A'),
                    'Test_Recall': result.get('test_recall', 'N/A')
                })
            elif result.get('task_type') == 'regression':
                summary_row.update({
                    'Train_R2': result.get('train_r2', 'N/A'),
                    'Test_R2': result.get('test_r2', 'N/A'),
                    'Test_RMSE': result.get('test_rmse', 'N/A'),
                    'Test_MAE': result.get('test_mae', 'N/A')
                })
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, task_type: str, metric: str = None) -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history 
                           if r['training_successful'] and r.get('task_type') == task_type]
        
        if not successful_models:
            return None
        
        # Set default metric based on task type
        if metric is None:
            if task_type == 'classification':
                metric = 'test_accuracy'
            else:
                metric = 'test_r2'
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
        
        if not models_with_metric:
            return None
        
        # For most metrics, higher is better (except for error metrics like MSE, MAE, RMSE)
        if metric in ['test_mse', 'test_mae', 'test_rmse', 'train_mse', 'train_mae', 'train_rmse']:
            return min(models_with_metric, key=lambda x: x[metric])
        else:
            return max(models_with_metric, key=lambda x: x[metric])
    
    def compare_models(self, task_type: str, metric: str = None) -> pd.DataFrame:
        """Compare models based on performance metrics"""
        successful_models = [r for r in self.training_history 
                           if r['training_successful'] and r.get('task_type') == task_type]
        
        if not successful_models:
            return pd.DataFrame()
        
        comparison_data = []
        for result in successful_models:
            row = {
                'Model': result['model_display_name'],
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Test_Samples': result.get('test_samples', 'N/A')
            }
            
            if task_type == 'classification':
                row.update({
                    'Train_Accuracy': result.get('train_accuracy', 'N/A'),
                    'Test_Accuracy': result.get('test_accuracy', 'N/A'),
                    'Test_F1': result.get('test_f1', 'N/A'),
                    'Test_Precision': result.get('test_precision', 'N/A'),
                    'Test_Recall': result.get('test_recall', 'N/A')
                })
            else:  # regression
                row.update({
                    'Train_R2': result.get('train_r2', 'N/A'),
                    'Test_R2': result.get('test_r2', 'N/A'),
                    'Test_RMSE': result.get('test_rmse', 'N/A'),
                    'Test_MAE': result.get('test_mae', 'N/A'),
                    'Test_MSE': result.get('test_mse', 'N/A')
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric or default
        if metric and metric in df.columns:
            ascending = metric in ['Test_RMSE', 'Test_MAE', 'Test_MSE']  # Lower is better for these
            df = df.sort_values(metric, ascending=ascending)
        
        return df
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_ensemble_learning_info() -> Dict:
    """Get information about all available ensemble learning models"""
    registry = EnsembleLearningRegistry()
    return registry.get_all_models()

def create_model_comparison_report(training_results: List[Dict], task_type: str) -> pd.DataFrame:
    """Create a comparison report from training results"""
    if not training_results:
        return pd.DataFrame()
    
    comparison_data = []
    for result in training_results:
        if result['training_successful']:
            row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Task_Type': result.get('task_type', 'Unknown'),
                'Training_Samples': result.get('training_samples', 'N/A'),
                'Test_Samples': result.get('test_samples', 'N/A'),
                'Task_Types': ', '.join(result.get('task_types', [])),
                'Data_Types': ', '.join(result.get('data_types', [])),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            
            # Add task-specific metrics
            if task_type == 'classification':
                row.update({
                    'Train_Accuracy': result.get('train_accuracy', 'N/A'),
                    'Test_Accuracy': result.get('test_accuracy', 'N/A'),
                    'Test_F1': result.get('test_f1', 'N/A'),
                    'Test_Precision': result.get('test_precision', 'N/A'),
                    'Test_Recall': result.get('test_recall', 'N/A')
                })
            else:  # regression
                row.update({
                    'Train_R2': result.get('train_r2', 'N/A'),
                    'Test_R2': result.get('test_r2', 'N/A'),
                    'Test_RMSE': result.get('test_rmse', 'N/A'),
                    'Test_MAE': result.get('test_mae', 'N/A'),
                    'Test_MSE': result.get('test_mse', 'N/A')
                })
            
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
    
    # Create sample data for testing
    print("\nTesting with sample data...")
    
    # Classification data
    X_class, y_class = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = trainer.prepare_data(X_class, y_class)
    
    # Regression data
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=10, noise=0.1, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = trainer.prepare_data(X_reg, y_reg)
    
    print(f"Classification data: {X_train_c.shape[0]} train, {X_test_c.shape[0]} test")
    print(f"Regression data: {X_train_r.shape[0]} train, {X_test_r.shape[0]} test")