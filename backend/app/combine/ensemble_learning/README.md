# Ensemble Learning Module

This directory contains the comprehensive ensemble learning module for the ML Model Comparison Toolkit.

## Structure

- `ensemble_learning.py` - Main ensemble module with registry and trainer
- `test_ensemble_learning.py` - Comprehensive test suite
- `streamlit_ensemble_demo.py` - Interactive Streamlit dashboard
- `examples/` - Example usage and tutorials

## Quick Start

```bash
cd backend/app/ensemble_learning
python test_ensemble_learning.py
```

## Available Algorithms

### Classification Ensembles
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Bagging Classifier
- Stacking Classifier

### Regression Ensembles
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- Bagging Regressor
- Stacking Regressor

## Features

- 13 different ensemble algorithms
- Both classification and regression support
- Advanced boosting techniques (XGBoost, LightGBM, CatBoost)
- Meta-learning with stacking
- Interactive training visualization
- Performance comparison tools
- Hyperparameter optimization
- Feature importance analysis

## Ensemble Methods

### Bagging (Bootstrap Aggregating)
- **Random Forest**: Uses decision trees with random feature selection
- **Bagging**: General bagging approach with any base estimator

### Boosting
- **Gradient Boosting**: Sequential weak learners with gradient descent
- **XGBoost**: Extreme gradient boosting with regularization
- **LightGBM**: Light gradient boosting machine (fast and efficient)
- **CatBoost**: Categorical boosting (handles categorical features well)

### Stacking
- **Stacking**: Meta-learning approach combining multiple models
- Uses cross-validation to train meta-learner

## Key Advantages

1. **Improved Accuracy**: Combines multiple models for better predictions
2. **Reduced Overfitting**: Averaging reduces variance
3. **Robustness**: Less sensitive to outliers and noise
4. **Feature Importance**: Provides insights into feature relevance
5. **Versatility**: Works with various base estimators