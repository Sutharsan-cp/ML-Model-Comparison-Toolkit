# Supervised Learning Models Module

This module provides a comprehensive collection of supervised learning algorithms for the ML Model Comparison Toolkit with an interactive Streamlit dashboard.

## Features

- **29 Total Models**: 14 Classification + 15 Regression algorithms
- **Unified Interface**: Consistent API across all models
- **Error Handling**: Graceful handling of import failures
- **Model Registry**: Centralized model management
- **Training Pipeline**: Automated training and evaluation
- **Hyperparameter Definitions**: Pre-defined hyperparameter grids
- **Interactive Dashboard**: Streamlit web interface for model comparison
- **Dataset Integration**: Supports your custom datasets from data directory
- **Real-time Comparison**: Side-by-side model performance analysis

## Available Models

### Classification Models (14)
- **Linear Models**: Logistic Regression, Ridge Classifier, Lasso Classifier, Elastic Net Classifier
- **Tree-based**: Decision Tree (CART), Random Forest Classifier
- **Ensemble**: Gradient Boosting, AdaBoost, XGBoost
- **Instance-based**: K-Nearest Neighbors
- **Probabilistic**: Naive Bayes
- **Kernel Methods**: Support Vector Machine
- **Discriminant Analysis**: Linear Discriminant Analysis, Quadratic Discriminant Analysis

### Regression Models (15)
- **Linear Models**: Linear Regression, Ridge Regression, Lasso Regression, Elastic Net Regression
- **Tree-based**: Decision Tree Regressor, Random Forest Regressor
- **Ensemble**: Gradient Boosting Regressor, AdaBoost Regressor, XGBoost Regressor
- **Kernel Methods**: Support Vector Regression
- **Polynomial**: Polynomial Regression
- **Bayesian**: Bayesian Ridge Regression
- **Robust**: Huber Regressor, Quantile Regression
- **Online**: Passive Aggressive Regressor

## Usage

### Basic Usage

```python
from supervised import SupervisedModelRegistry, SupervisedModelTrainer

# Initialize registry and trainer
registry = SupervisedModelRegistry()
trainer = SupervisedModelTrainer()

# Get available models
classification_models = registry.get_classification_models()
regression_models = registry.get_regression_models()

# Train a single model
result = trainer.train_model('logistic_regression', X_train, y_train, X_test, y_test, 'classification')

# Train multiple models
results = trainer.train_multiple_models(
    ['logistic_regression', 'random_forest_classifier'], 
    X_train, y_train, X_test, y_test, 'classification'
)

# Train all models for a task
all_results = trainer.train_all_models(X_train, y_train, X_test, y_test, 'classification')
```

### Model Information

```python
# Get model by name
model_info = registry.get_model_by_name('logistic_regression', 'classification')

# Get models by type
linear_models = registry.get_models_by_type('linear', 'classification')

# Get training summary
summary_df = trainer.get_training_summary()

# Get best model
best_model = trainer.get_best_model('test_score')
```

### Interactive Streamlit Dashboard

The module includes a comprehensive Streamlit dashboard for interactive model comparison.

#### Quick Start
```bash
cd backend/app
python run_streamlit_demo.py
```

#### Manual Start
```bash
cd backend/app
pip install -r requirements_streamlit.txt
streamlit run streamlit_supervised_demo.py
```

#### Dashboard Features
- **Dataset Selection**: Choose from your datasets or synthetic data
- **Model Comparison**: Select multiple models to compare side-by-side
- **Real-time Training**: Watch models train with progress indicators
- **Performance Visualization**: Interactive charts and metrics
- **Dataset Preview**: Explore your data before training
- **Model Information**: Detailed information about each algorithm

#### Using Your Own Datasets
1. Place CSV files in `data/raw/classification/` or `data/raw/regression/`
2. The dashboard will automatically detect and list your datasets
3. Select target column from dropdown
4. Categorical features are automatically encoded

## Testing

### Basic Model Testing
```bash
cd backend/app
python test_supervised.py
```

### Dataset Loading Testing
```bash
cd backend/app
python test_dataset_loading.py
```

### Full Integration Testing
```bash
cd backend/app
python test_full_integration.py
```

## Test Results Summary

Based on comprehensive testing with synthetic and real datasets:

### ✅ Working Models (20/29)

**Classification (10/14)**:
- Logistic Regression (binary only)
- Random Forest Classifier
- Support Vector Machine (binary only)
- K-Nearest Neighbors
- XGBoost
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Ridge Classifier
- Lasso Classifier
- Elastic Net Classifier

**Regression (10/15)**:
- Linear Regression
- Random Forest Regressor
- Support Vector Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Decision Tree Regressor
- XGBoost Regressor
- Bayesian Ridge Regression
- Passive Aggressive Regressor

### ⚠️ Models with Issues (9/29)

**Classification Issues**:
- Naive Bayes: Requires non-negative features
- Decision Tree (CART): Missing score method
- Gradient Boosting: Fitting issues
- AdaBoost: Missing DecisionStump dependency

**Regression Issues**:
- Gradient Boosting Regressor: Fitting issues
- AdaBoost Regressor: Base estimator issues
- Polynomial Regression: Negative R² scores
- Huber Regressor: Array dimension issues
- Quantile Regression: Poor performance metrics

## Model Structure

Each model in the registry contains:
- `class`: The model class
- `name`: Display name
- `type`: Model category (linear, ensemble, tree, etc.)
- `description`: Brief description
- `hyperparameters`: Dictionary of hyperparameter options

## Error Handling

The module gracefully handles import errors for individual models. If a model fails to import, it will be excluded from the registry but won't break the entire module.

## Extending the Module

To add new models:

1. Create the model class in the appropriate directory
2. Add the import statement with error handling
3. Add the model configuration to the registry
4. Update the model counts in this README

## Dependencies

- numpy
- pandas
- scikit-learn (for datasets and utilities)
- scipy (for optimization)
- Custom model implementations in `models/supervised/`

## Performance Notes

- All models use custom implementations for educational purposes
- For production use, consider using scikit-learn or other optimized libraries
- Some models may be slower than their scikit-learn counterparts
- Memory usage varies by model complexity and dataset size