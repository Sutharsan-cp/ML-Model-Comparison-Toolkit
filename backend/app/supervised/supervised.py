"""
Supervised Learning Models Module
Consolidates all supervised learning algorithms for the ML Comparison Toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
import numpy as np
from typing import Optional, Dict


# Import all classification models with error handling
classification_imports = {}
regression_imports = {}

# Classification models
try:
    from models.supervised.classification.logistic_regression import LogisticRegression
    classification_imports['logistic_regression'] = LogisticRegression
except ImportError as e:
    print(f"Warning: Could not import LogisticRegression: {e}")

try:
    from models.supervised.classification.random_forest_classifier import RandomForestClassifier
    classification_imports['random_forest_classifier'] = RandomForestClassifier
except ImportError as e:
    print(f"Warning: Could not import RandomForestClassifier: {e}")

try:
    from models.supervised.classification.svm_classifier import SVMClassifier
    classification_imports['svm_classifier'] = SVMClassifier
except ImportError as e:
    print(f"Warning: Could not import SVMClassifier: {e}")

try:
    from models.supervised.classification.naive_bayes_classifier import NaiveBayesClassifier
    classification_imports['naive_bayes'] = NaiveBayesClassifier
except ImportError as e:
    print(f"Warning: Could not import NaiveBayesClassifier: {e}")

try:
    from models.supervised.classification.knn_classifier import KNNClassifier
    classification_imports['knn_classifier'] = KNNClassifier
except ImportError as e:
    print(f"Warning: Could not import KNNClassifier: {e}")

try:
    from models.supervised.classification.decision_tree_cart import DecisionTreeCART
    classification_imports['decision_tree_cart'] = DecisionTreeCART
except ImportError as e:
    print(f"Warning: Could not import DecisionTreeCART: {e}")

try:
    from models.supervised.classification.gradient_boosting_classifier import GradientBoostingClassifier
    classification_imports['gradient_boosting'] = GradientBoostingClassifier
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingClassifier: {e}")

try:
    from models.supervised.classification.adaboost_classifier import AdaBoostClassifier
    classification_imports['adaboost'] = AdaBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import AdaBoostClassifier: {e}")

try:
    from models.supervised.classification.xgboost_classifier import XGBoostClassifier
    classification_imports['xgboost'] = XGBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import XGBoostClassifier: {e}")

try:
    from models.supervised.classification.linear_discriminant_analysis import LinearDiscriminantAnalysis
    classification_imports['lda'] = LinearDiscriminantAnalysis
except ImportError as e:
    print(f"Warning: Could not import LinearDiscriminantAnalysis: {e}")

try:
    from models.supervised.classification.quadratic_discriminant_analysis import QuadraticDiscriminantAnalysis
    classification_imports['qda'] = QuadraticDiscriminantAnalysis
except ImportError as e:
    print(f"Warning: Could not import QuadraticDiscriminantAnalysis: {e}")

try:
    from models.supervised.classification.ridge_classifier import RidgeClassifier
    classification_imports['ridge_classifier'] = RidgeClassifier
except ImportError as e:
    print(f"Warning: Could not import RidgeClassifier: {e}")

try:
    from models.supervised.classification.lasso_classifier import LassoClassifier
    classification_imports['lasso_classifier'] = LassoClassifier
except ImportError as e:
    print(f"Warning: Could not import LassoClassifier: {e}")

try:
    from models.supervised.classification.elastic_net_classifier import ElasticNetClassifier
    classification_imports['elastic_net_classifier'] = ElasticNetClassifier
except ImportError as e:
    print(f"Warning: Could not import ElasticNetClassifier: {e}")

try:
    from models.supervised.classification.catboost_classifier import CatBoostClassifier
    classification_imports['catboost_classifier'] = CatBoostClassifier
except ImportError as e:
    print(f"Warning: Could not import CatBoostClassifier: {e}")

try : 
    from models.supervised.classification.decision_stump import DecisionStump
    classification_imports['decision_stump'] = DecisionStump
except ImportError as e:
    print(f"Warning: Could not import DecisionStump: {e}")

try : 
    from models.supervised.classification.decision_tree_c45 import DecisionTreeC45
    classification_imports['decision_tree_c45'] = DecisionTreeC45
except ImportError as e:
    print(f"Warning: Could not import DecisionTreeC45: {e}")

try : 
    from models.supervised.classification.decision_tree_id3 import DecisionTreeID3
    classification_imports['decision_tree_id3'] = DecisionTreeID3
except ImportError as e:
    print(f"Warning: Could not import DecisionTreeID3: {e}")

try:
    from models.supervised.classification.decision_tree_chaid import DecisionTreeCHAID
    classification_imports['decision_tree_chaid'] = DecisionTreeCHAID
except ImportError as e:
    print(f"Warning: Could not import DecisionTreeCHAID: {e}")



# Regression models
try:
    from models.supervised.regression.linear_regression import LinearRegression
    regression_imports['linear_regression'] = LinearRegression
except ImportError as e:
    print(f"Warning: Could not import LinearRegression: {e}")

try:
    from models.supervised.regression.random_forest_regressor import RandomForestRegressor
    regression_imports['random_forest_regressor'] = RandomForestRegressor
except ImportError as e:
    print(f"Warning: Could not import RandomForestRegressor: {e}")

try:
    from models.supervised.regression.svm_regressor import SVMRegressor
    regression_imports['svm_regressor'] = SVMRegressor
except ImportError as e:
    print(f"Warning: Could not import SVMRegressor: {e}")

try:
    from models.supervised.regression.ridge_regression import RidgeRegression
    regression_imports['ridge_regression'] = RidgeRegression
except ImportError as e:
    print(f"Warning: Could not import RidgeRegression: {e}")

try:
    from models.supervised.regression.lasso_regression import LassoRegression
    regression_imports['lasso_regression'] = LassoRegression
except ImportError as e:
    print(f"Warning: Could not import LassoRegression: {e}")

try:
    from models.supervised.regression.elastic_net_regression import ElasticNetRegression
    regression_imports['elastic_net_regression'] = ElasticNetRegression
except ImportError as e:
    print(f"Warning: Could not import ElasticNetRegression: {e}")

try:
    from models.supervised.regression.decision_tree_regressor import DecisionTreeRegressor
    regression_imports['decision_tree_regressor'] = DecisionTreeRegressor
except ImportError as e:
    print(f"Warning: Could not import DecisionTreeRegressor: {e}")

try:
    from models.supervised.regression.gradient_boosting_regressor import GradientBoostingRegressor
    regression_imports['gradient_boosting_regressor'] = GradientBoostingRegressor
except ImportError as e:
    print(f"Warning: Could not import GradientBoostingRegressor: {e}")

try:
    from models.supervised.regression.adaboost_regressor import AdaBoostRegressor
    regression_imports['adaboost_regressor'] = AdaBoostRegressor
except ImportError as e:
    print(f"Warning: Could not import AdaBoostRegressor: {e}")

try:
    from models.supervised.regression.xgboost_regressor import XGBoostRegressor
    regression_imports['xgboost_regressor'] = XGBoostRegressor
except ImportError as e:
    print(f"Warning: Could not import XGBoostRegressor: {e}")

try:
    from models.supervised.regression.polynomial_regression import PolynomialRegression
    regression_imports['polynomial_regression'] = PolynomialRegression
except ImportError as e:
    print(f"Warning: Could not import PolynomialRegression: {e}")

try:
    from models.supervised.regression.bayesian_ridge import BayesianRidge
    regression_imports['bayesian_ridge'] = BayesianRidge
except ImportError as e:
    print(f"Warning: Could not import BayesianRidge: {e}")

try:
    from models.supervised.regression.catboost_regressor import CatBoostRegressor
    regression_imports['catboost_regressor'] = CatBoostRegressor
except ImportError as e:
    print(f"Warning: Could not import CatBoostRegressor: {e}")



class SupervisedModelRegistry:
    """Registry for all supervised learning models"""
    
    def __init__(self):
        # Build classification models dictionary from successfully imported classes
        self.classification_models = {}
        
        # Define model configurations
        classification_configs = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'type': 'linear',
                'description': 'Linear model for binary and multiclass classification',
                'hyperparameters': {
                    'learningRate': [0.001, 0.01, 0.1, 1.0],
                    'maxIter': [100, 500, 1000, 2000],
                    'regularization': [None, 'l1', 'l2', 'elasticnet'],
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'random_forest_classifier': {
                'name': 'Random Forest Classifier',
                'type': 'ensemble',
                'description': 'Ensemble of decision trees for classification',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200, 500],
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10],
                    'minSamplesLeaf': [1, 2, 4],
                    'maxFeatures': ['sqrt', 'log2', None]
                }
            },
            'svm_classifier': {
                'name': 'Support Vector Machine',
                'type': 'kernel',
                'description': 'Support Vector Machine for classification',
                'hyperparameters': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'degree': [2, 3, 4, 5]
                }
            },
            'naive_bayes': {
                'name': 'Naive Bayes',
                'type': 'probabilistic',
                'description': 'Probabilistic classifier based on Bayes theorem',
                'hyperparameters': {
                    'alpha': [0.1, 0.5, 1.0, 2.0],
                    'fitPrior': [True, False]
                }
            },
            'knn_classifier': {
                'name': 'K-Nearest Neighbors',
                'type': 'instance_based',
                'description': 'Instance-based learning algorithm',
                'hyperparameters': {
                    'nNeighbors': [3, 5, 7, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'decision_tree_cart': {
                'name': 'Decision Tree (CART)',
                'type': 'tree',
                'description': 'Decision tree using CART algorithm',
                'hyperparameters': {
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10, 20],
                    'minSamplesLeaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'type': 'ensemble',
                'description': 'Gradient boosting for classification',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 0.2],
                    'maxDepth': [3, 5, 7, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'adaboost': {
                'name': 'AdaBoost',
                'type': 'ensemble',
                'description': 'Adaptive boosting classifier',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 1.0, 2.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            },
            'xgboost': {
                'name': 'XGBoost',
                'type': 'ensemble',
                'description': 'Extreme gradient boosting classifier',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 0.2],
                    'maxDepth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsampleBytree': [0.8, 0.9, 1.0]
                }
            },
            'lda': {
                'name': 'Linear Discriminant Analysis',
                'type': 'linear',
                'description': 'Linear discriminant analysis for classification',
                'hyperparameters': {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
                }
            },
            'qda': {
                'name': 'Quadratic Discriminant Analysis',
                'type': 'quadratic',
                'description': 'Quadratic discriminant analysis for classification',
                'hyperparameters': {
                    'regParam': [0.0, 0.01, 0.1, 0.5]
                }
            },
            'ridge_classifier': {
                'name': 'Ridge Classifier',
                'type': 'linear',
                'description': 'Ridge regression for classification',
                'hyperparameters': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'fitIntercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            },
            'lasso_classifier': {
                'name': 'Lasso Classifier',
                'type': 'linear',
                'description': 'Lasso regression for classification',
                'hyperparameters': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'maxIter': [100, 500, 1000],
                    'tol': [1e-4, 1e-3, 1e-2]
                }
            },
            'elastic_net_classifier': {
                'name': 'Elastic Net Classifier',
                'type': 'linear',
                'description': 'Elastic net regression for classification',
                'hyperparameters': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1Ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'maxIter': [100, 500, 1000]
                }
            },
            'catboost_classifier': {
                'name': 'CatBoost Classifier',
                'type': 'ensemble',
                'description': 'Gradient boosting with categorical features support',
                'hyperparameters': {
                    'iterations': [100, 200, 500],
                    'learningRate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7, 10],
                    'l2LeafReg': [1, 3, 5, 7]
                }
            },
            'decision_stump': {
                'name': 'Decision Stump',
                'type': 'tree',
                'description': 'One-level decision tree for classification',
                'hyperparameters': {
                    'criterion': ['gini', 'entropy']
                }
            },
            'decision_tree_c45': {
                'name': 'Decision Tree C4.5',
                'type': 'tree',
                'description': 'Decision tree using C4.5 algorithm',
                'hyperparameters': {
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10, 20],
                    'minSamplesLeaf': [1, 2, 5, 10]
                }
            },
            'decision_tree_id3': {
                'name': 'Decision Tree ID3',
                'type': 'tree',
                'description': 'Decision tree using ID3 algorithm',
                'hyperparameters': {
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10, 20],
                    'minSamplesLeaf': [1, 2, 5, 10]
                }
            },
            'decision_tree_chaid': {
                'name': 'Decision Tree CHAID',
                'type': 'tree',
                'description': 'Decision tree using CHAID algorithm',
                'hyperparameters': {
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10, 20],
                    'minSamplesLeaf': [1, 2, 5, 10]
                }
            },


        }
        
        # Add successfully imported classification models
        for model_key, model_class in classification_imports.items():
            if model_key in classification_configs:
                config = classification_configs[model_key].copy()
                config['class'] = model_class
                self.classification_models[model_key] = config
        
        # Build regression models dictionary from successfully imported classes
        self.regression_models = {}
        
        # Define regression model configurations
        regression_configs = {
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'linear',
                'description': 'Ordinary least squares linear regression',
                'hyperparameters': {
                    'fitIntercept': [True, False],
                    'method': ['normal', 'gradient']
                }
            },
            'random_forest_regressor': {
                'name': 'Random Forest Regressor',
                'type': 'ensemble',
                'description': 'Ensemble of decision trees for regression',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200, 500],
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10],
                    'minSamplesLeaf': [1, 2, 4],
                    'maxFeatures': ['auto', 'sqrt', 'log2', None]
                }
            },
            'svm_regressor': {
                'name': 'Support Vector Regression',
                'type': 'kernel',
                'description': 'Support Vector Machine for regression',
                'hyperparameters': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                }
            },
            'ridge_regression': {
                'name': 'Ridge Regression',
                'type': 'linear',
                'description': 'Linear regression with L2 regularization',
                'hyperparameters': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
                    'fitIntercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            },
            'lasso_regression': {
                'name': 'Lasso Regression',
                'type': 'linear',
                'description': 'Linear regression with L1 regularization',
                'hyperparameters': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'maxIter': [100, 500, 1000, 2000],
                    'tol': [1e-4, 1e-3, 1e-2]
                }
            },
            'elastic_net_regression': {
                'name': 'Elastic Net Regression',
                'type': 'linear',
                'description': 'Linear regression with L1 and L2 regularization',
                'hyperparameters': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1Ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'maxIter': [100, 500, 1000]
                }
            },
            'decision_tree_regressor': {
                'name': 'Decision Tree Regressor',
                'type': 'tree',
                'description': 'Decision tree for regression',
                'hyperparameters': {
                    'maxDepth': [None, 5, 10, 20, 50],
                    'minSamplesSplit': [2, 5, 10, 20],
                    'minSamplesLeaf': [1, 2, 5, 10],
                    'criterion': ['mse', 'mae']
                }
            },
            'gradient_boosting_regressor': {
                'name': 'Gradient Boosting Regressor',
                'type': 'ensemble',
                'description': 'Gradient boosting for regression',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 0.2],
                    'maxDepth': [3, 5, 7, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'adaboost_regressor': {
                'name': 'AdaBoost Regressor',
                'type': 'ensemble',
                'description': 'Adaptive boosting regressor',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 1.0, 2.0],
                    'lossFunction': ['linear', 'square', 'exponential']
                }
            },
            'xgboost_regressor': {
                'name': 'XGBoost Regressor',
                'type': 'ensemble',
                'description': 'Extreme gradient boosting regressor',
                'hyperparameters': {
                    'nEstimators': [50, 100, 200],
                    'learningRate': [0.01, 0.1, 0.2],
                    'maxDepth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsampleBytree': [0.8, 0.9, 1.0]
                }
            },
            'polynomial_regression': {
                'name': 'Polynomial Regression',
                'type': 'polynomial',
                'description': 'Polynomial regression with feature expansion',
                'hyperparameters': {
                    'degree': [2, 3, 4, 5],
                    'fitIntercept': [True, False],
                    'includeBias': [True, False]
                }
            },
            'bayesian_ridge': {
                'name': 'Bayesian Ridge Regression',
                'type': 'bayesian',
                'description': 'Bayesian ridge regression',
                'hyperparameters': {
                    'alpha1': [1e-6, 1e-5, 1e-4, 1e-3],
                    'alpha2': [1e-6, 1e-5, 1e-4, 1e-3],
                    'lambda1': [1e-6, 1e-5, 1e-4, 1e-3],
                    'lambda2': [1e-6, 1e-5, 1e-4, 1e-3]
                }
            },
            'catboost_regressor': {
                'name': 'CatBoost Regressor',
                'type': 'ensemble',
                'description': 'Gradient boosting with categorical features support',
                'hyperparameters': {
                    'iterations': [100, 200, 500],
                    'learningRate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7, 10],
                    'l2LeafReg': [1, 3, 5, 7]
                }
            },
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
        """Get all supervised models"""
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
            # Search in both
            model = self.classification_models.get(model_name)
            if model is None:
                model = self.regression_models.get(model_name)
            return model
    
    def get_models_by_type(self, model_type: str, task_type: str = None) -> Dict[str, Dict]:
        """Get models by their type (e.g., 'linear', 'ensemble', 'tree')"""
        models = {}
        
        if task_type is None or task_type == 'classification':
            for name, info in self.classification_models.items():
                if info['type'] == model_type:
                    models[name] = info
        
        if task_type is None or task_type == 'regression':
            for name, info in self.regression_models.items():
                if info['type'] == model_type:
                    models[name] = info
        
        return models

class SupervisedModelTrainer:
    """Trainer class for supervised learning models"""
    
    def __init__(self):
        self.registry = SupervisedModelRegistry()
        self.trained_models = {}
        self.training_history = []
    


    def train_model(self, model_name: str, X_train, y_train, X_test=None, y_test=None, 
                    task_type: str = None, hyperparameters: Dict = None) -> Dict:
        """Train a single model"""
        
        model_info = self.registry.get_model_by_name(model_name, task_type)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            model = model_info['class'](**hyperparameters)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on training data
            train_predictions = model.predict(X_train)
            train_score = model.score(X_train, y_train)
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'train_score': train_score,
                'train_predictions': train_predictions,
                'training_successful': True,
                'error': None
            }
            
            # Test predictions if test data provided
            if X_test is not None and y_test is not None:
                test_predictions = model.predict(X_test)
                test_score = model.score(X_test, y_test)
                result.update({
                    'test_score': test_score,
                    'test_predictions': test_predictions
                })

                # ==============================
                # 🔹 ADDITIONAL METRICS SECTION
                # ==============================
                if task_type == 'classification':
                    result['accuracy'] = accuracy_score(y_test, test_predictions)
                    result['precision'] = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
                    result['recall'] = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
                    result['f1_score'] = f1_score(y_test, test_predictions, average='weighted', zero_division=0)

                    # ROC-AUC (for binary classification)
                    if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            result['roc_auc'] = roc_auc_score(y_test, y_proba)
                        except Exception:
                            result['roc_auc'] = None

                elif task_type == 'regression':
                    result['r2'] = r2_score(y_test, test_predictions)
                    result['mae'] = mean_absolute_error(y_test, test_predictions)
                    result['rmse'] = np.sqrt(mean_squared_error(y_test, test_predictions))
                # ==============================
            
            # Get probability predictions if available
            if hasattr(model, 'predict_proba'):
                try:
                    train_probabilities = model.predict_proba(X_train)
                    result['train_probabilities'] = train_probabilities
                    
                    if X_test is not None:
                        test_probabilities = model.predict_proba(X_test)
                        result['test_probabilities'] = test_probabilities
                except:
                    pass
            
            # Get feature importance if available
            if hasattr(model, 'get_feature_importance') or hasattr(model, 'feature_importances_'):
                try:
                    if hasattr(model, 'get_feature_importance'):
                        feature_importance = model.get_feature_importance()
                    else:
                        feature_importance = model.feature_importances_
                    result['feature_importance'] = feature_importance
                except:
                    pass
            
            # Store trained model and record history
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e)
            }
            self.training_history.append(error_result)
            return error_result

    def train_multiple_models(self, model_names: List[str], X_train, y_train, 
                            X_test=None, y_test=None, task_type: str = None) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(model_name, X_train, y_train, X_test, y_test, task_type)
            results.append(result)
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test=None, y_test=None, 
                        task_type: str = 'classification') -> List[Dict]:
        """Train all available models for a given task type"""
        
        if task_type == 'classification':
            models = list(self.registry.get_classification_models().keys())
        elif task_type == 'regression':
            models = list(self.registry.get_regression_models().keys())
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        return self.train_multiple_models(models, X_train, y_train, X_test, y_test, task_type)
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_history:
            summary_row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Training_Successful': result['training_successful'],
                'Train_Score': result.get('train_score', None),
                'Test_Score': result.get('test_score', None),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    

    def get_best_model(self, task_type: str = None) -> Optional[Dict]:
        """
        Automatically detect the best model based on task type and dataset balance.
        For classification:
            - If dataset is imbalanced → use F1 or ROC-AUC
            - If balanced → use Accuracy
        For regression:
            - Use R² by default, fallback to MAE or RMSE
        """
        successful_models = [r for r in self.training_history if r.get('training_successful')]
        if not successful_models:
            return None

        # Try to infer task type
        if not task_type:
            task_type = successful_models[0].get('model_type', 'classification').lower()

        # Classification logic — detect imbalance if possible
        if task_type == 'classification':
            imbalance_ratio = 1.0
            for m in successful_models:
                y_true = m.get('y_test')
                if y_true is not None:
                    class_counts = np.bincount(y_true) if len(np.unique(y_true)) == 2 else []
                    if len(class_counts) > 1:
                        imbalance_ratio = min(class_counts) / max(class_counts)
                        break

            if imbalance_ratio < 0.5:
                # Imbalanced dataset → prefer F1 or ROC-AUC
                metric_candidates = ['f1_score', 'roc_auc', 'accuracy']
            else:
                # Balanced → prefer accuracy or F1
                metric_candidates = ['accuracy', 'f1_score']
        else:
            # Regression → prefer R², fallback to MAE or RMSE
            metric_candidates = ['r2', 'mae', 'rmse']

        # Select the first available metric in priority order
        for metric in metric_candidates:
            models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None]
            if models_with_metric:
                higher_is_better = not (metric in ['mae', 'rmse'])
                best_model = max(models_with_metric, key=lambda x: x[metric]) if higher_is_better \
                            else min(models_with_metric, key=lambda x: x[metric])
                best_model['selected_metric'] = metric
                return best_model

        # Fallback if no metrics found
        models_with_test = [r for r in successful_models if 'test_score' in r]
        if models_with_test:
            best_model = max(models_with_test, key=lambda x: x['test_score'])
            best_model['selected_metric'] = 'test_score'
            return best_model

        return None

    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_supervised_models_info() -> Dict:
    """Get information about all available supervised models"""
    registry = SupervisedModelRegistry()
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
                'Train_Score': result.get('train_score', 'N/A'),
                'Test_Score': result.get('test_score', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data).sort_values('Test_Score', ascending=False)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = SupervisedModelRegistry()
    trainer = SupervisedModelTrainer()
    
    print("Available Classification Models:")
    for name, info in registry.get_classification_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Regression Models:")
    for name, info in registry.get_regression_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")