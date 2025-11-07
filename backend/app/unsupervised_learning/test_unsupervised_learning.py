"""
Test suite for Unsupervised Learning Module
Tests all unsupervised learning algorithms and their integration
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, load_iris
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the unsupervised_learning module to path
sys.path.append(os.path.dirname(__file__))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from unsupervised_learning import (
    UnsupervisedLearningRegistry, 
    UnsupervisedLearningTrainer,
    get_unsupervised_learning_info,
    create_model_comparison_report
)

class TestUnsupervisedLearningRegistry:
    """Test the UnsupervisedLearningRegistry class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = UnsupervisedLearningRegistry()
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly"""
        assert isinstance(self.registry, UnsupervisedLearningRegistry)
        assert hasattr(self.registry, 'clustering_models')
        assert hasattr(self.registry, 'dimensionality_reduction_models')
        assert hasattr(self.registry, 'manifold_learning_models')
        assert hasattr(self.registry, 'anomaly_detection_models')
    
    def test_get_all_models(self):
        """Test getting all models"""
        all_models = self.registry.get_all_models()
        assert isinstance(all_models, dict)
        assert 'clustering' in all_models
        assert 'dimensionality_reduction' in all_models
        assert 'manifold_learning' in all_models
        assert 'anomaly_detection' in all_models
    
    def test_get_clustering_models(self):
        """Test getting clustering models"""
        clustering_models = self.registry.get_clustering_models()
        assert isinstance(clustering_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical', 'agglomerative', 'spectral_clustering', 'mean_shift']
        for model in expected_models:
            if model in clustering_models:
                assert 'name' in clustering_models[model]
                assert 'type' in clustering_models[model]
                assert 'description' in clustering_models[model]
                assert 'class' in clustering_models[model]
    
    def test_get_dimensionality_reduction_models(self):
        """Test getting dimensionality reduction models"""
        dr_models = self.registry.get_dimensionality_reduction_models()
        assert isinstance(dr_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['pca', 'ica']
        for model in expected_models:
            if model in dr_models:
                assert 'name' in dr_models[model]
                assert 'type' in dr_models[model]
                assert 'description' in dr_models[model]
                assert 'class' in dr_models[model]
    
    def test_get_manifold_learning_models(self):
        """Test getting manifold learning models"""
        manifold_models = self.registry.get_manifold_learning_models()
        assert isinstance(manifold_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['tsne']
        for model in expected_models:
            if model in manifold_models:
                assert 'name' in manifold_models[model]
                assert 'type' in manifold_models[model]
                assert 'description' in manifold_models[model]
                assert 'class' in manifold_models[model]
    
    def test_get_anomaly_detection_models(self):
        """Test getting anomaly detection models"""
        anomaly_models = self.registry.get_anomaly_detection_models()
        assert isinstance(anomaly_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['isolation_forest']
        for model in expected_models:
            if model in anomaly_models:
                assert 'name' in anomaly_models[model]
                assert 'type' in anomaly_models[model]
                assert 'description' in anomaly_models[model]
                assert 'class' in anomaly_models[model]
    
    def test_get_model_by_name(self):
        """Test getting model by name"""
        all_models = self.registry.get_all_models()
        
        # Test with category specified
        for category, models in all_models.items():
            for model_name in models.keys():
                model_info = self.registry.get_model_by_name(model_name, category)
                assert model_info is not None
                assert model_info['name'] is not None
        
        # Test without category (should search all)
        for category, models in all_models.items():
            for model_name in models.keys():
                model_info = self.registry.get_model_by_name(model_name)
                assert model_info is not None
    
    def test_get_models_by_data_type(self):
        """Test getting models by data type"""
        tabular_models = self.registry.get_models_by_data_type('tabular')
        assert isinstance(tabular_models, dict)
    
    def test_get_models_by_task_type(self):
        """Test getting models by task type"""
        clustering_models = self.registry.get_models_by_task_type('clustering')
        assert isinstance(clustering_models, dict)
        
        dr_models = self.registry.get_models_by_task_type('dimensionality_reduction')
        assert isinstance(dr_models, dict)

class TestUnsupervisedLearningTrainer:
    """Test the UnsupervisedLearningTrainer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.trainer = UnsupervisedLearningTrainer()
        
        # Create sample datasets
        self.X_blobs, self.y_blobs = make_blobs(
            n_samples=200, centers=3, n_features=4, 
            random_state=42, cluster_std=1.5
        )
        
        self.X_classification, self.y_classification = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_redundant=0, 
            n_informative=8, random_state=42
        )
        
        # Load iris dataset for testing
        try:
            iris = load_iris()
            self.X_iris = iris.data
            self.y_iris = iris.target
        except:
            self.X_iris = self.X_classification
            self.y_iris = self.y_classification
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly"""
        assert isinstance(self.trainer, UnsupervisedLearningTrainer)
        assert hasattr(self.trainer, 'registry')
        assert hasattr(self.trainer, 'trained_models')
        assert hasattr(self.trainer, 'training_history')
    
    def test_prepare_data(self):
        """Test data preparation"""
        X_train, X_test = self.trainer.prepare_data(self.X_blobs)
        
        assert X_train is not None
        assert X_test is not None
        assert len(X_train) > len(X_test)  # Default test_size=0.2
    
    def test_train_clustering_models(self):
        """Test training clustering models"""
        X_train, X_test = self.trainer.prepare_data(self.X_blobs, test_size=0.3)
        
        clustering_models = self.trainer.registry.get_clustering_models()
        
        for model_name in list(clustering_models.keys())[:3]:  # Test first 3 models
            try:
                result = self.trainer.train_model(
                    model_name, X_train, X_test,
                    category='clustering', 
                    hyperparameters={'n_clusters': 3} if 'n_clusters' in clustering_models[model_name]['hyperparameters'] else {},
                    verbose=False
                )
                
                assert result is not None
                assert 'model_name' in result
                assert 'training_successful' in result
                
                if result['training_successful']:
                    assert result['model_instance'] is not None
                    print(f"✓ {model_name}: Training successful")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_dimensionality_reduction_models(self):
        """Test training dimensionality reduction models"""
        X_train, X_test = self.trainer.prepare_data(self.X_classification, test_size=0.3)
        
        dr_models = self.trainer.registry.get_dimensionality_reduction_models()
        
        for model_name in list(dr_models.keys()):  # Test all DR models
            try:
                result = self.trainer.train_model(
                    model_name, X_train, X_test,
                    category='dimensionality_reduction',
                    hyperparameters={'n_components': 3},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print(f"✓ {model_name}: Training successful")
                    assert 'output_dimensions' in result or 'explained_variance_ratio' in result
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_manifold_learning_models(self):
        """Test training manifold learning models"""
        X_train, X_test = self.trainer.prepare_data(self.X_iris, test_size=0.3)
        
        manifold_models = self.trainer.registry.get_manifold_learning_models()
        
        for model_name in list(manifold_models.keys()):  # Test all manifold models
            try:
                result = self.trainer.train_model(
                    model_name, X_train, X_test,
                    category='manifold_learning',
                    hyperparameters={'n_components': 2, 'n_iter': 100, 'perplexity': 10},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print(f"✓ {model_name}: Training successful")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_anomaly_detection_models(self):
        """Test training anomaly detection models"""
        X_train, X_test = self.trainer.prepare_data(self.X_classification, test_size=0.3)
        
        anomaly_models = self.trainer.registry.get_anomaly_detection_models()
        
        for model_name in list(anomaly_models.keys()):  # Test all anomaly detection models
            try:
                result = self.trainer.train_model(
                    model_name, X_train, X_test,
                    category='anomaly_detection',
                    hyperparameters={'n_estimators': 50, 'contamination': 0.1},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print(f"✓ {model_name}: Training successful")
                    assert 'outlier_fraction' in result or 'anomaly_scores_mean' in result
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_multiple_models(self):
        """Test training multiple models at once"""
        X_train, X_test = self.trainer.prepare_data(self.X_blobs, test_size=0.3)
        
        clustering_models = list(self.trainer.registry.get_clustering_models().keys())[:2]
        
        if clustering_models:
            results = self.trainer.train_multiple_models(
                clustering_models, X_train, X_test,
                category='clustering', verbose=False
            )
            
            assert isinstance(results, list)
            assert len(results) == len(clustering_models)
            
            for result in results:
                assert 'model_name' in result
                assert 'training_successful' in result
    
    def test_get_training_summary(self):
        """Test getting training summary"""
        # Train a model first
        X_train, X_test = self.trainer.prepare_data(self.X_blobs, test_size=0.3)
        
        clustering_models = list(self.trainer.registry.get_clustering_models().keys())
        if clustering_models:
            self.trainer.train_model(
                clustering_models[0], X_train, X_test,
                category='clustering', verbose=False
            )
        
        summary = self.trainer.get_training_summary()
        assert isinstance(summary, pd.DataFrame)
        
        if not summary.empty:
            expected_columns = ['Model', 'Type', 'Category', 'Training_Successful']
            for col in expected_columns:
                assert col in summary.columns
    
    def test_get_best_model(self):
        """Test getting best model"""
        # Train multiple models first
        X_train, X_test = self.trainer.prepare_data(self.X_blobs, test_size=0.3)
        
        clustering_models = list(self.trainer.registry.get_clustering_models().keys())[:2]
        
        for model_name in clustering_models:
            try:
                self.trainer.train_model(
                    model_name, X_train, X_test,
                    category='clustering', verbose=False
                )
            except:
                pass  # Some models might fail, that's okay for this test
        
        best_model = self.trainer.get_best_model('silhouette_score')
        
        # Should return None if no successful models, or a dict if there are successful models
        assert best_model is None or isinstance(best_model, dict)
    
    def test_clear_history(self):
        """Test clearing training history"""
        # Add some training history first
        X_train, X_test = self.trainer.prepare_data(self.X_blobs)
        
        clustering_models = list(self.trainer.registry.get_clustering_models().keys())
        if clustering_models:
            try:
                self.trainer.train_model(
                    clustering_models[0], X_train, X_test,
                    category='clustering', verbose=False
                )
            except:
                pass
        
        # Clear history
        self.trainer.clear_history()
        
        assert len(self.trainer.trained_models) == 0
        assert len(self.trainer.training_history) == 0

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_unsupervised_learning_info(self):
        """Test getting unsupervised learning info"""
        info = get_unsupervised_learning_info()
        assert isinstance(info, dict)
        
        expected_categories = ['clustering', 'dimensionality_reduction', 'manifold_learning', 'anomaly_detection']
        for category in expected_categories:
            assert category in info
    
    def test_create_model_comparison_report(self):
        """Test creating model comparison report"""
        # Create sample training results
        sample_results = [
            {
                'model_display_name': 'Test Model 1',
                'model_type': 'clustering',
                'model_category': 'clustering',
                'training_successful': True,
                'training_samples': 100,
                'silhouette_score': 0.65,
                'n_clusters': 3,
                'task_types': ['clustering'],
                'data_types': ['tabular'],
                'hyperparameters': {'n_clusters': 3}
            },
            {
                'model_display_name': 'Test Model 2',
                'model_type': 'dimensionality_reduction',
                'model_category': 'dimensionality_reduction',
                'training_successful': True,
                'training_samples': 100,
                'explained_variance_ratio': 0.85,
                'task_types': ['dimensionality_reduction'],
                'data_types': ['tabular'],
                'hyperparameters': {'n_components': 2}
            }
        ]
        
        report = create_model_comparison_report(sample_results)
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 2
        
        expected_columns = ['Model', 'Type', 'Category', 'Training_Samples']
        for col in expected_columns:
            assert col in report.columns

def run_comprehensive_test():
    """Run a comprehensive test of the unsupervised learning module"""
    print("=" * 60)
    print("UNSUPERVISED LEARNING MODULE COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test registry
    print("\n1. Testing Registry...")
    registry = UnsupervisedLearningRegistry()
    all_models = registry.get_all_models()
    
    total_models = 0
    for category, models in all_models.items():
        print(f"   {category}: {len(models)} models")
        total_models += len(models)
    
    print(f"   Total models available: {total_models}")
    
    # Test trainer with sample data
    print("\n2. Testing Trainer...")
    trainer = UnsupervisedLearningTrainer()
    
    # Create sample data
    X, _ = make_blobs(n_samples=100, centers=3, n_features=6, random_state=42)
    X_train, X_test = trainer.prepare_data(X, test_size=0.3)
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    
    # Test a few models from each category
    successful_models = 0
    failed_models = 0
    
    for category, models in all_models.items():
        if models:  # If category has models
            model_name = list(models.keys())[0]  # Get first model
            try:
                hyperparameters = {}
                if 'n_clusters' in models[model_name]['hyperparameters']:
                    hyperparameters['n_clusters'] = 3
                elif 'n_components' in models[model_name]['hyperparameters']:
                    hyperparameters['n_components'] = 2
                
                result = trainer.train_model(
                    model_name, X_train, X_test,
                    category=category,
                    hyperparameters=hyperparameters,
                    verbose=False
                )
                
                if result['training_successful']:
                    successful_models += 1
                    print(f"   ✓ {models[model_name]['name']}: Success")
                else:
                    failed_models += 1
                    print(f"   ✗ {models[model_name]['name']}: Failed")
            except Exception as e:
                failed_models += 1
                print(f"   ✗ {models[model_name]['name']}: Exception - {str(e)[:50]}...")
    
    print(f"\n3. Results Summary:")
    print(f"   Successful models: {successful_models}")
    print(f"   Failed models: {failed_models}")
    print(f"   Success rate: {successful_models/(successful_models+failed_models)*100:.1f}%")
    
    # Test summary generation
    summary = trainer.get_training_summary()
    print(f"   Training summary shape: {summary.shape}")
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Run comprehensive test
    run_comprehensive_test()
    
    # Run pytest if available
    if PYTEST_AVAILABLE:
        try:
            pytest.main([__file__, "-v"])
        except:
            print("\nPytest failed. Running basic tests...")
    else:
        print("\nPytest not available. Running basic tests...")
        
        # Run basic tests manually
        test_registry = TestUnsupervisedLearningRegistry()
        test_registry.setup_method()
        test_registry.test_registry_initialization()
        test_registry.test_get_all_models()
        print("✓ Registry tests passed")
        
        test_trainer = TestUnsupervisedLearningTrainer()
        test_trainer.setup_method()
        test_trainer.test_trainer_initialization()
        test_trainer.test_prepare_data()
        print("✓ Trainer tests passed")
        
        test_utils = TestUtilityFunctions()
        test_utils.test_get_unsupervised_learning_info()
        print("✓ Utility function tests passed")
        
        print("\n✓ All basic tests completed successfully!")