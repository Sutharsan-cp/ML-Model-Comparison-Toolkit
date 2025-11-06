"""
Test script for ensemble learning models
"""

import sys
import os
import numpy as np
import warnings
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Import our ensemble module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_learning import EnsembleLearningRegistry, EnsembleLearningTrainer

def create_classification_dataset(n_samples=1000, n_features=10, n_classes=3, random_state=42):
    """Create a test classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def create_regression_dataset(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Create a test regression dataset"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        noise=noise,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def test_classification_models():
    """Test ensemble classification models"""
    print("Testing Ensemble Classification Models")
    print("=" * 50)
    
    # Create test dataset
    X, y = create_classification_dataset(n_samples=500, n_features=8, n_classes=3)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = EnsembleLearningRegistry()
    trainer = EnsembleLearningTrainer()
    
    # Get available classification models
    classification_models = registry.get_classification_models()
    print(f"Available classification models: {len(classification_models)}")
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test classification models
    test_models = list(classification_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'random_forest':
                    hyperparams = {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
                elif model_name == 'gradient_boosting':
                    hyperparams = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                elif model_name in ['xgboost', 'lightgbm']:
                    hyperparams = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                elif model_name == 'catboost':
                    hyperparams = {'iterations': 50, 'learning_rate': 0.1, 'depth': 3, 'verbose': False, 'random_state': 42}
                elif model_name == 'bagging':
                    hyperparams = {'n_estimators': 10, 'random_state': 42}
                elif model_name == 'stacking':
                    hyperparams = {'cv': 3, 'n_jobs': 1}
                
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test, 
                    'classification', hyperparams, cv_folds=3
                )
                
                if result['training_successful']:
                    cv_score = result.get('cv_mean_score', 'N/A')
                    test_score = result.get('test_score', 'N/A')
                    
                    if isinstance(cv_score, float):
                        cv_score = f"{cv_score:.3f}"
                    if isinstance(test_score, float):
                        test_score = f"{test_score:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: CV={cv_score}, Test={test_score}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No classification models available")
    
    print()

def test_regression_models():
    """Test ensemble regression models"""
    print("Testing Ensemble Regression Models")
    print("=" * 50)
    
    # Create test dataset
    X, y = create_regression_dataset(n_samples=500, n_features=8, noise=0.1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = EnsembleLearningRegistry()
    trainer = EnsembleLearningTrainer()
    
    # Get available regression models
    regression_models = registry.get_regression_models()
    print(f"Available regression models: {len(regression_models)}")
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test regression models
    test_models = list(regression_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'random_forest':
                    hyperparams = {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
                elif model_name == 'gradient_boosting':
                    hyperparams = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                elif model_name in ['xgboost', 'lightgbm']:
                    hyperparams = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                elif model_name == 'bagging':
                    hyperparams = {'n_estimators': 10, 'random_state': 42}
                elif model_name == 'stacking':
                    hyperparams = {'cv': 3, 'n_jobs': 1}
                
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test, 
                    'regression', hyperparams, cv_folds=3
                )
                
                if result['training_successful']:
                    cv_score = result.get('cv_mean_score', 'N/A')
                    test_score = result.get('test_score', 'N/A')
                    
                    if isinstance(cv_score, float):
                        cv_score = f"{cv_score:.3f}"
                    if isinstance(test_score, float):
                        test_score = f"{test_score:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: CV={cv_score}, Test={test_score}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No regression models available")
    
    print()

def test_ensemble_types():
    """Test filtering models by ensemble type"""
    print("Testing Ensemble Type Filtering")
    print("=" * 50)
    
    registry = EnsembleLearningRegistry()
    
    # Test different ensemble types
    ensemble_types = ['bagging', 'boosting', 'stacking']
    
    for ensemble_type in ensemble_types:
        suitable_models = registry.get_models_by_type(ensemble_type)
        print(f"\n{ensemble_type.title()} models: {len(suitable_models)}")
        for model_key, model_info in suitable_models.items():
            print(f"  - {model_info['name']} ({model_info['task']})")
    
    print()

def test_model_comparison():
    """Test model comparison functionality"""
    print("Testing Model Comparison")
    print("=" * 50)
    
    # Create test dataset
    X, y = create_classification_dataset(n_samples=300, n_features=5, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize trainer
    trainer = EnsembleLearningTrainer()
    
    # Train a few models for comparison
    test_models = ['random_forest', 'gradient_boosting']
    results = []
    
    for model_name in test_models:
        try:
            hyperparams = {'n_estimators': 30, 'random_state': 42}
            if model_name == 'gradient_boosting':
                hyperparams['learning_rate'] = 0.1
                hyperparams['max_depth'] = 3
            
            result = trainer.train_model(
                model_name, X_train, y_train, X_test, y_test, 
                'classification', hyperparams, cv_folds=3
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from ensemble_learning import create_model_comparison_report
    
    try:
        comparison_df = create_model_comparison_report(results)
        print("✅ Comparison report generated successfully")
        print(f"Report shape: {comparison_df.shape}")
        if not comparison_df.empty:
            print("\nComparison results:")
            print(comparison_df.to_string(index=False))
    except Exception as e:
        print(f"❌ Failed to generate comparison report: {e}")
    
    print()

def test_feature_importance():
    """Test feature importance functionality"""
    print("Testing Feature Importance")
    print("=" * 50)
    
    # Create test dataset
    X, y = create_classification_dataset(n_samples=300, n_features=5, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize trainer
    trainer = EnsembleLearningTrainer()
    
    # Train models that support feature importance
    test_models = ['random_forest', 'gradient_boosting']
    
    for model_name in test_models:
        try:
            hyperparams = {'n_estimators': 30, 'random_state': 42}
            if model_name == 'gradient_boosting':
                hyperparams['learning_rate'] = 0.1
                hyperparams['max_depth'] = 3
            
            result = trainer.train_model(
                model_name, X_train, y_train, X_test, y_test, 
                'classification', hyperparams, cv_folds=3
            )
            
            if result['training_successful'] and result.get('feature_importance') is not None:
                importance = result['feature_importance']
                print(f"\n{result['model_display_name']} Feature Importance:")
                for i, imp in enumerate(importance):
                    print(f"  Feature {i}: {imp:.4f}")
        except Exception as e:
            print(f"Failed to get feature importance for {model_name}: {e}")
    
    # Get feature importance summary
    try:
        importance_df = trainer.get_feature_importance_summary()
        print(f"\n✅ Feature importance summary generated: {importance_df.shape}")
    except Exception as e:
        print(f"❌ Failed to generate feature importance summary: {e}")
    
    print()

def run_all_tests():
    """Run all ensemble learning tests"""
    print("Ensemble Learning Models Test Suite")
    print("=" * 60)
    print()
    
    test_classification_models()
    test_regression_models()
    test_ensemble_types()
    test_model_comparison()
    test_feature_importance()
    
    print("=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests()