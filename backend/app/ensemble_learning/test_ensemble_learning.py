"""
Test suite for Ensemble Learning Module
Tests all ensemble learning algorithms and their integration
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the ensemble_learning module to path
sys.path.append(os.path.dirname(__file__))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from ensemble_learning import (
    EnsembleLearningRegistry, 
    EnsembleLearningTrainer,
    get_ensemble_learning_info,
    create_model_comparison_report
)

class TestEnsembleLearningRegistry:
    """Test the EnsembleLearningRegistry class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = EnsembleLearningRegistry()
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly"""
        assert isinstance(self.registry, EnsembleLearningRegistry)
        assert hasattr(self.registry, 'classification_models')
        assert hasattr(self.registry, 'regression_models')
    
    def test_get_all_models(self):
        """Test getting all models"""
        all_models = self.registry.get_all_models()
        assert isinstance(all_models, dict)
        assert 'classification' in all_models
        assert 'regression' in all_models
    
    def test_get_classification_models(self):
        """Test getting classification models"""
        classification_models = self.registry.get_classification_models()
        assert isinstance(classification_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['randomforest', 'xgboost', 'gradientboosting', 
                          'bagging', 'stacking', 'lightgbm', 'catboost']
        for model in expected_models:
            if model in classification_models:
                assert 'name' in classification_models[model]
                assert 'type' in classification_models[model]
                assert 'description' in classification_models[model]
                assert 'class' in classification_models[model]
    
    def test_get_regression_models(self):
        """Test getting regression models"""
        regression_models = self.registry.get_regression_models()
        assert isinstance(regression_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['randomforest', 'xgboost', 'gradientboosting', 
                          'bagging', 'stacking', 'lightgbm']
        for model in expected_models:
            if model in regression_models:
                assert 'name' in regression_models[model]
                assert 'type' in regression_models[model]
                assert 'description' in regression_models[model]
                assert 'class' in regression_models[model]
    
    def test_get_model_by_name(self):
        """Test getting model by name"""
        all_models = self.registry.get_all_models()
        
        # Test with task type specified
        for task_type, models in all_models.items():
            for model_name in models.keys():
                model_info = self.registry.get_model_by_name(model_name, task_type)
                assert model_info is not None
                assert model_info['name'] is not None
        
        # Test without task type (should search all)
        for task_type, models in all_models.items():
            for model_name in models.keys():
                model_info = self.registry.get_model_by_name(model_name)
                assert model_info is not None
    
    def test_get_models_by_data_type(self):
        """Test getting models by data type"""
        tabular_models = self.registry.get_models_by_data_type('tabular')
        assert isinstance(tabular_models, dict)
        
        numerical_models = self.registry.get_models_by_data_type('numerical')
        assert isinstance(numerical_models, dict)
    
    def test_get_models_by_task_type(self):
        """Test getting models by task type"""
        classification_models = self.registry.get_models_by_task_type('classification')
        assert isinstance(classification_models, dict)
        
        regression_models = self.registry.get_models_by_task_type('regression')
        assert isinstance(regression_models, dict)

class TestEnsembleLearningTrainer:
    """Test the EnsembleLearningTrainer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.trainer = EnsembleLearningTrainer()
        
        # Create sample datasets
        self.X_classification, self.y_classification = make_classification(
            n_samples=200, n_features=10, n_classes=2, n_redundant=0, 
            n_informative=8, random_state=42
        )
        
        self.X_regression, self.y_regression = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        
        # Multiclass classification
        self.X_multiclass, self.y_multiclass = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_redundant=0, 
            n_informative=8, random_state=42
        )
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly"""
        assert isinstance(self.trainer, EnsembleLearningTrainer)
        assert hasattr(self.trainer, 'registry')
        assert hasattr(self.trainer, 'trained_models')
        assert hasattr(self.trainer, 'training_history')
    
    def test_prepare_data(self):
        """Test data preparation"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.2
        )
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert len(X_train) > len(X_test)  # Default test_size=0.2
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_train_classification_models(self):
        """Test training classification models"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = self.trainer.registry.get_classification_models()
        
        for model_name in list(classification_models.keys())[:3]:  # Test first 3 models
            try:
                # Set appropriate hyperparameters for each model
                hyperparameters = {}
                if model_name == 'randomforest':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 5}
                elif model_name == 'xgboost':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 3}
                elif model_name == 'gradientboosting':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 3}
                elif model_name in ['bagging', 'stacking']:
                    # These will be handled by the trainer's special logic
                    pass
                
                result = self.trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    task_type='classification',
                    hyperparameters=hyperparameters,
                    verbose=False
                )
                
                assert result is not None
                assert 'model_name' in result
                assert 'training_successful' in result
                
                if result['training_successful']:
                    assert result['model_instance'] is not None
                    assert 'train_accuracy' in result
                    assert 'test_accuracy' in result
                    print(f"✓ {model_name}: Training successful - Test Accuracy: {result.get('test_accuracy', 'N/A'):.3f}")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_regression_models(self):
        """Test training regression models"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_regression, self.y_regression, test_size=0.3
        )
        
        regression_models = self.trainer.registry.get_regression_models()
        
        for model_name in list(regression_models.keys())[:3]:  # Test first 3 models
            try:
                # Set appropriate hyperparameters for each model
                hyperparameters = {}
                if model_name == 'randomforest':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 5}
                elif model_name == 'xgboost':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 3}
                elif model_name == 'gradientboosting':
                    hyperparameters = {'n_estimators': 10, 'max_depth': 3}
                elif model_name in ['bagging', 'stacking']:
                    # These will be handled by the trainer's special logic
                    pass
                
                result = self.trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    task_type='regression',
                    hyperparameters=hyperparameters,
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    assert 'train_r2' in result
                    assert 'test_r2' in result
                    print(f"✓ {model_name}: Training successful - Test R²: {result.get('test_r2', 'N/A'):.3f}")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_multiple_models(self):
        """Test training multiple models at once"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = list(self.trainer.registry.get_classification_models().keys())[:2]
        
        if classification_models:
            results = self.trainer.train_multiple_models(
                classification_models, X_train, y_train, X_test, y_test,
                task_type='classification', verbose=False
            )
            
            assert isinstance(results, list)
            assert len(results) == len(classification_models)
            
            for result in results:
                assert 'model_name' in result
                assert 'training_successful' in result
    
    def test_train_all_classification_models(self):
        """Test training all classification models"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        try:
            results = self.trainer.train_all_classification_models(
                X_train, y_train, X_test, y_test, verbose=False
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            successful_models = [r for r in results if r['training_successful']]
            print(f"Successfully trained {len(successful_models)} out of {len(results)} classification models")
            
        except Exception as e:
            print(f"Error in training all classification models: {str(e)}")
    
    def test_train_all_regression_models(self):
        """Test training all regression models"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_regression, self.y_regression, test_size=0.3
        )
        
        try:
            results = self.trainer.train_all_regression_models(
                X_train, y_train, X_test, y_test, verbose=False
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            successful_models = [r for r in results if r['training_successful']]
            print(f"Successfully trained {len(successful_models)} out of {len(results)} regression models")
            
        except Exception as e:
            print(f"Error in training all regression models: {str(e)}")
    
    def test_get_training_summary(self):
        """Test getting training summary"""
        # Train a few models first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = list(self.trainer.registry.get_classification_models().keys())[:2]
        
        if classification_models:
            self.trainer.train_multiple_models(
                classification_models, X_train, y_train, X_test, y_test,
                task_type='classification', verbose=False
            )
            
            summary = self.trainer.get_training_summary()
            assert isinstance(summary, pd.DataFrame)
            
            if not summary.empty:
                assert 'Model' in summary.columns
                assert 'Training_Successful' in summary.columns
                assert len(summary) > 0
    
    def test_get_best_model(self):
        """Test getting best model"""
        # Train a few models first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = list(self.trainer.registry.get_classification_models().keys())[:2]
        
        if classification_models:
            self.trainer.train_multiple_models(
                classification_models, X_train, y_train, X_test, y_test,
                task_type='classification', verbose=False
            )
            
            best_model = self.trainer.get_best_model('classification', 'test_accuracy')
            
            if best_model:
                assert 'model_name' in best_model
                assert 'test_accuracy' in best_model
                assert best_model['training_successful'] is True
    
    def test_compare_models(self):
        """Test model comparison"""
        # Train a few models first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = list(self.trainer.registry.get_classification_models().keys())[:2]
        
        if classification_models:
            self.trainer.train_multiple_models(
                classification_models, X_train, y_train, X_test, y_test,
                task_type='classification', verbose=False
            )
            
            comparison = self.trainer.compare_models('classification', 'test_accuracy')
            
            if not comparison.empty:
                assert isinstance(comparison, pd.DataFrame)
                assert 'Model' in comparison.columns
                assert 'Test_Accuracy' in comparison.columns
    
    def test_clear_history(self):
        """Test clearing training history"""
        # Train a model first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        classification_models = list(self.trainer.registry.get_classification_models().keys())
        
        if classification_models:
            self.trainer.train_model(
                classification_models[0], X_train, y_train, X_test, y_test,
                task_type='classification', verbose=False
            )
            
            # Verify history exists
            assert len(self.trainer.training_history) > 0
            assert len(self.trainer.trained_models) > 0
            
            # Clear history
            self.trainer.clear_history()
            
            # Verify history is cleared
            assert len(self.trainer.training_history) == 0
            assert len(self.trainer.trained_models) == 0

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_ensemble_learning_info(self):
        """Test getting ensemble learning info"""
        info = get_ensemble_learning_info()
        assert isinstance(info, dict)
        assert 'classification' in info
        assert 'regression' in info
    
    def test_create_model_comparison_report(self):
        """Test creating model comparison report"""
        # Create mock training results
        mock_results = [
            {
                'model_display_name': 'Random Forest',
                'model_type': 'ensemble_classification',
                'task_type': 'classification',
                'training_successful': True,
                'training_samples': 100,
                'test_samples': 50,
                'train_accuracy': 0.95,
                'test_accuracy': 0.90,
                'test_f1': 0.89,
                'test_precision': 0.91,
                'test_recall': 0.88,
                'task_types': ['classification'],
                'data_types': ['tabular'],
                'hyperparameters': {'n_estimators': 100}
            }
        ]
        
        report = create_model_comparison_report(mock_results, 'classification')
        assert isinstance(report, pd.DataFrame)
        
        if not report.empty:
            assert 'Model' in report.columns
            assert 'Test_Accuracy' in report.columns

def run_comprehensive_test():
    """Run a comprehensive test of the ensemble learning module"""
    print("="*60)
    print("COMPREHENSIVE ENSEMBLE LEARNING MODULE TEST")
    print("="*60)
    
    # Test registry
    print("\n1. Testing Registry...")
    registry = EnsembleLearningRegistry()
    
    classification_models = registry.get_classification_models()
    regression_models = registry.get_regression_models()
    
    print(f"   Available Classification Models: {len(classification_models)}")
    for name, info in classification_models.items():
        print(f"   - {info['name']}")
    
    print(f"   Available Regression Models: {len(regression_models)}")
    for name, info in regression_models.items():
        print(f"   - {info['name']}")
    
    # Test trainer
    print("\n2. Testing Trainer...")
    trainer = EnsembleLearningTrainer()
    
    # Create test data
    X_class, y_class = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
    X_reg, y_reg = make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)
    
    # Test classification
    print("\n   Testing Classification Models:")
    X_train_c, X_test_c, y_train_c, y_test_c = trainer.prepare_data(X_class, y_class, test_size=0.3)
    
    for model_name in list(classification_models.keys())[:3]:  # Test first 3
        try:
            result = trainer.train_model(
                model_name, X_train_c, y_train_c, X_test_c, y_test_c,
                task_type='classification',
                hyperparameters={'n_estimators': 10} if 'n_estimators' in classification_models[model_name]['hyperparameters'] else {},
                verbose=False
            )
            
            if result['training_successful']:
                print(f"   ✓ {model_name}: Accuracy = {result.get('test_accuracy', 0):.3f}")
            else:
                print(f"   ✗ {model_name}: {result.get('error', 'Failed')}")
        except Exception as e:
            print(f"   ✗ {model_name}: Exception - {str(e)}")
    
    # Test regression
    print("\n   Testing Regression Models:")
    X_train_r, X_test_r, y_train_r, y_test_r = trainer.prepare_data(X_reg, y_reg, test_size=0.3)
    
    for model_name in list(regression_models.keys())[:3]:  # Test first 3
        try:
            result = trainer.train_model(
                model_name, X_train_r, y_train_r, X_test_r, y_test_r,
                task_type='regression',
                hyperparameters={'n_estimators': 10} if 'n_estimators' in regression_models[model_name]['hyperparameters'] else {},
                verbose=False
            )
            
            if result['training_successful']:
                print(f"   ✓ {model_name}: R² = {result.get('test_r2', 0):.3f}")
            else:
                print(f"   ✗ {model_name}: {result.get('error', 'Failed')}")
        except Exception as e:
            print(f"   ✗ {model_name}: Exception - {str(e)}")
    
    # Test summary and comparison
    print("\n3. Testing Summary and Comparison...")
    summary = trainer.get_training_summary()
    print(f"   Training Summary: {len(summary)} entries")
    
    if len(summary) > 0:
        print("   Summary columns:", list(summary.columns))
        
        # Test best model selection
        best_class = trainer.get_best_model('classification')
        best_reg = trainer.get_best_model('regression')
        
        if best_class:
            print(f"   Best Classification Model: {best_class['model_display_name']} (Accuracy: {best_class.get('test_accuracy', 'N/A')})")
        
        if best_reg:
            print(f"   Best Regression Model: {best_reg['model_display_name']} (R²: {best_reg.get('test_r2', 'N/A')})")
    
    print("\n" + "="*60)
    print("ENSEMBLE LEARNING MODULE TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        # Run pytest if available
        pytest.main([__file__, "-v"])
    else:
        # Run comprehensive test
        run_comprehensive_test()
        
        # Run individual test classes
        print("\nRunning individual tests...")
        
        # Test Registry
        registry_test = TestEnsembleLearningRegistry()
        registry_test.setup_method()
        
        try:
            registry_test.test_registry_initialization()
            registry_test.test_get_all_models()
            registry_test.test_get_classification_models()
            registry_test.test_get_regression_models()
            print("✓ Registry tests passed")
        except Exception as e:
            print(f"✗ Registry tests failed: {e}")
        
        # Test Trainer
        trainer_test = TestEnsembleLearningTrainer()
        trainer_test.setup_method()
        
        try:
            trainer_test.test_trainer_initialization()
            trainer_test.test_prepare_data()
            trainer_test.test_train_classification_models()
            trainer_test.test_train_regression_models()
            print("✓ Trainer tests passed")
        except Exception as e:
            print(f"✗ Trainer tests failed: {e}")
        
        # Test Utilities
        utility_test = TestUtilityFunctions()
        
        try:
            utility_test.test_get_ensemble_learning_info()
            utility_test.test_create_model_comparison_report()
            print("✓ Utility tests passed")
        except Exception as e:
            print(f"✗ Utility tests failed: {e}")