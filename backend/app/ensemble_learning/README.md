# Ensemble Learning Module

A comprehensive ensemble learning module for the ML Comparison Toolkit that provides a unified interface for training and comparing various ensemble learning algorithms.

## Overview

This module consolidates all ensemble learning algorithms from the `backend/app/models/ensemble` directory and provides:

- **Unified Registry**: Access to all ensemble learning models with their configurations
- **Automated Training**: Easy training and evaluation of multiple models
- **Performance Comparison**: Built-in comparison and ranking of models
- **Flexible Configuration**: Support for custom hyperparameters

## Available Models

### Classification Models (7)

1. **Random Forest Classifier** - Ensemble of decision trees using bootstrap aggregating
2. **XGBoost Classifier** - Extreme Gradient Boosting for classification
3. **Gradient Boosting Classifier** - Sequential ensemble correcting previous errors
4. **Bagging Classifier** - Bootstrap Aggregating with customizable base estimators
5. **Stacking Classifier** - Meta-learning combining multiple base models
6. **LightGBM Classifier** - Light Gradient Boosting Machine for fast training
7. **CatBoost Classifier** - Categorical Boosting handling categorical features automatically

### Regression Models (6)

1. **Random Forest Regressor** - Ensemble of decision trees for regression
2. **XGBoost Regressor** - Extreme Gradient Boosting for regression
3. **Gradient Boosting Regressor** - Sequential ensemble for regression
4. **Bagging Regressor** - Bootstrap Aggregating for regression
5. **Stacking Regressor** - Meta-learning for regression
6. **LightGBM Regressor** - Light Gradient Boosting for regression

## Quick Start

### Basic Usage

```python
from ensemble_learning import EnsembleLearningRegistry, EnsembleLearningTrainer
from sklearn.datasets import make_classification, make_regression

# Initialize
registry = EnsembleLearningRegistry()
trainer = EnsembleLearningTrainer()

# Classification Example
X_class, y_class = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = trainer.prepare_data(X_class, y_class)

# Train a single model
result = trainer.train_model(
    'randomforest', 
    X_train, y_train, 
    X_test, y_test,
    task_type='classification',
    hyperparameters={'n_estimators': 100, 'max_depth': 10},
    verbose=True
)

print(f"Test Accuracy: {result['test_accuracy']:.3f}")

# Regression Example
X_reg, y_reg = make_regression(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = trainer.prepare_data(X_reg, y_reg)

result = trainer.train_model(
    'xgboost',
    X_train, y_train,
    X_test, y_test,
    task_type='regression',
    hyperparameters={'n_estimators': 100, 'learning_rate': 0.1},
    verbose=True
)

print(f"Test R²: {result['test_r2']:.3f}")
```

### Training Multiple Models

```python
# Train all classification models
results = trainer.train_all_classification_models(
    X_train, y_train, X_test, y_test, verbose=True
)

# Get training summary
summary = trainer.get_training_summary()
print(summary)

# Get best model
best_model = trainer.get_best_model('classification', 'test_accuracy')
print(f"Best Model: {best_model['model_display_name']}")
print(f"Test Accuracy: {best_model['test_accuracy']:.3f}")

# Compare models
comparison = trainer.compare_models('classification', 'test_accuracy')
print(comparison)
```

### Model Registry

```python
# Get all available models
all_models = registry.get_all_models()

# Get classification models only
classification_models = registry.get_classification_models()

# Get regression models only
regression_models = registry.get_regression_models()

# Get model by name
model_info = registry.get_model_by_name('randomforest', 'classification')
print(f"Model: {model_info['name']}")
print(f"Description: {model_info['description']}")
print(f"Hyperparameters: {model_info['hyperparameters']}")

# Get models by data type
tabular_models = registry.get_models_by_data_type('tabular')

# Get models by task type
classification_models = registry.get_models_by_task_type('classification')
```

## Features

### 1. Unified Model Registry

The `EnsembleLearningRegistry` class provides:
- Centralized access to all ensemble models
- Model metadata (name, type, description, supported data types)
- Hyperparameter configurations
- Filtering by data type or task type

### 2. Automated Training

The `EnsembleLearningTrainer` class provides:
- Data preparation with automatic scaling and stratification
- Single model training with custom hyperparameters
- Batch training of multiple models
- Training all models in a category
- Automatic metric calculation

### 3. Performance Metrics

#### Classification Metrics
- Accuracy (train and test)
- Precision (weighted average)
- Recall (weighted average)
- F1 Score (weighted average)
- Predicted probabilities (if available)

#### Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### 4. Model Comparison

- Training summary with all metrics
- Best model selection based on any metric
- Side-by-side comparison of models
- Automatic sorting by performance

### 5. Feature Importance

For models that support it:
- Feature importance scores
- Ranking of features by importance

## Advanced Usage

### Custom Hyperparameters

```python
# Random Forest with custom parameters
hyperparameters = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42
}

result = trainer.train_model(
    'randomforest',
    X_train, y_train,
    X_test, y_test,
    task_type='classification',
    hyperparameters=hyperparameters
)
```

### Ensemble Methods (Bagging, Stacking)

```python
# Bagging automatically uses DecisionTree as base estimator
result = trainer.train_model(
    'bagging',
    X_train, y_train,
    X_test, y_test,
    task_type='classification',
    hyperparameters={
        'n_estimators': 50,
        'max_samples': 0.8,
        'max_features': 0.8
    }
)

# Stacking automatically uses diverse base estimators
result = trainer.train_model(
    'stacking',
    X_train, y_train,
    X_test, y_test,
    task_type='classification',
    hyperparameters={
        'cv': 5,
        'use_probas': True,
        'passthrough': False
    }
)
```

### Model Comparison Report

```python
# Train multiple models
results = trainer.train_multiple_models(
    ['randomforest', 'xgboost', 'gradientboosting'],
    X_train, y_train,
    X_test, y_test,
    task_type='classification'
)

# Create comparison report
from ensemble_learning import create_model_comparison_report
report = create_model_comparison_report(results, 'classification')
print(report)
```

## Model-Specific Notes

### Random Forest
- Good for both classification and regression
- Handles non-linear relationships well
- Provides feature importance
- Less prone to overfitting than single trees

### XGBoost
- State-of-the-art gradient boosting
- Excellent performance on structured data
- Supports regularization (L1 and L2)
- Handles missing values

### Gradient Boosting
- Sequential ensemble method
- Builds trees to correct previous errors
- Slower training than Random Forest
- Often achieves high accuracy

### Bagging
- Reduces variance through bootstrap aggregating
- Can use any base estimator
- Parallel training of base models
- Good for reducing overfitting

### Stacking
- Meta-learning approach
- Combines predictions from multiple models
- Uses cross-validation for meta-features
- Can achieve better performance than individual models

### LightGBM
- Fast training speed
- Low memory usage
- Handles large datasets efficiently
- Supports categorical features

### CatBoost
- Handles categorical features automatically
- Reduces overfitting
- Fast prediction speed
- Good default parameters

## Testing

Run the test suite:

```bash
python backend/app/ensemble_learning/test_ensemble_learning.py
```

The test suite includes:
- Registry initialization tests
- Model training tests (classification and regression)
- Data preparation tests
- Performance metric tests
- Comparison and ranking tests

## Performance Tips

1. **Start with Random Forest**: Good baseline model with reasonable defaults
2. **Try XGBoost for best performance**: Often achieves highest accuracy
3. **Use Stacking for maximum performance**: Combines strengths of multiple models
4. **Tune hyperparameters**: Use grid search or random search for optimization
5. **Consider training time**: LightGBM is fastest, Stacking is slowest
6. **Feature scaling**: Automatically handled by the trainer
7. **Cross-validation**: Use for more robust performance estimates

## Integration with ML Comparison Toolkit

This module integrates seamlessly with the ML Comparison Toolkit:

```python
# Use with the main comparison framework
from ensemble_learning import get_ensemble_learning_info

# Get all model information
model_info = get_ensemble_learning_info()

# Use in automated model selection
trainer = EnsembleLearningTrainer()
results = trainer.train_all_classification_models(X_train, y_train, X_test, y_test)
best = trainer.get_best_model('classification')
```

## Requirements

- numpy
- pandas
- scikit-learn
- Custom ensemble models from `backend/app/models/ensemble`

## License

Part of the ML Comparison Toolkit project.

## Contributing

To add new ensemble models:
1. Implement the model in `backend/app/models/ensemble/classification` or `regression`
2. Add import statement in `ensemble_learning.py`
3. Add model configuration in the registry
4. Update this README

## Support

For issues or questions, please refer to the main ML Comparison Toolkit documentation.