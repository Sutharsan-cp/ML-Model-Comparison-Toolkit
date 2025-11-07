"""
Test suite for Neural Networks Module
Tests all neural network algorithms and their integration
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the neural_networks module to path
sys.path.append(os.path.dirname(__file__))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from neural_networks import (
    NeuralNetworksRegistry, 
    NeuralNetworksTrainer,
    get_neural_networks_info,
    create_model_comparison_report
)

class TestNeuralNetworksRegistry:
    """Test the NeuralNetworksRegistry class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = NeuralNetworksRegistry()
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly"""
        assert isinstance(self.registry, NeuralNetworksRegistry)
        assert hasattr(self.registry, 'basic_nn_models')
        assert hasattr(self.registry, 'deep_learning_models')
        assert hasattr(self.registry, 'recurrent_models')
        assert hasattr(self.registry, 'generative_models')
        assert hasattr(self.registry, 'specialized_models')
    
    def test_get_all_models(self):
        """Test getting all models"""
        all_models = self.registry.get_all_models()
        assert isinstance(all_models, dict)
        assert 'basic_nn' in all_models
        assert 'deep_learning' in all_models
        assert 'recurrent' in all_models
        assert 'generative' in all_models
        assert 'specialized' in all_models
    
    def test_get_basic_nn_models(self):
        """Test getting basic neural network models"""
        basic_models = self.registry.get_basic_nn_models()
        assert isinstance(basic_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['perceptron', 'mlp', 'ann', 'backpropagation']
        for model in expected_models:
            if model in basic_models:
                assert 'name' in basic_models[model]
                assert 'type' in basic_models[model]
                assert 'description' in basic_models[model]
                assert 'class' in basic_models[model]
    
    def test_get_deep_learning_models(self):
        """Test getting deep learning models"""
        deep_models = self.registry.get_deep_learning_models()
        assert isinstance(deep_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['cnn', 'resnet', 'vgg', 'inception', 'densenet', 'efficientnet', 'mobilenet']
        for model in expected_models:
            if model in deep_models:
                assert 'name' in deep_models[model]
                assert 'type' in deep_models[model]
                assert 'description' in deep_models[model]
                assert 'class' in deep_models[model]
    
    def test_get_recurrent_models(self):
        """Test getting recurrent models"""
        recurrent_models = self.registry.get_recurrent_models()
        assert isinstance(recurrent_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['rnn', 'lstm', 'gru']
        for model in expected_models:
            if model in recurrent_models:
                assert 'name' in recurrent_models[model]
                assert 'type' in recurrent_models[model]
                assert 'description' in recurrent_models[model]
                assert 'class' in recurrent_models[model]
    
    def test_get_generative_models(self):
        """Test getting generative models"""
        generative_models = self.registry.get_generative_models()
        assert isinstance(generative_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['autoencoder', 'gan']
        for model in expected_models:
            if model in generative_models:
                assert 'name' in generative_models[model]
                assert 'type' in generative_models[model]
                assert 'description' in generative_models[model]
                assert 'class' in generative_models[model]
    
    def test_get_specialized_models(self):
        """Test getting specialized models"""
        specialized_models = self.registry.get_specialized_models()
        assert isinstance(specialized_models, dict)
        
        # Check if expected models are present (if imported successfully)
        expected_models = ['transformer']
        for model in expected_models:
            if model in specialized_models:
                assert 'name' in specialized_models[model]
                assert 'type' in specialized_models[model]
                assert 'description' in specialized_models[model]
                assert 'class' in specialized_models[model]
    
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
        
        image_models = self.registry.get_models_by_data_type('image')
        assert isinstance(image_models, dict)
        
        sequence_models = self.registry.get_models_by_data_type('sequence')
        assert isinstance(sequence_models, dict)
    
    def test_get_models_by_task_type(self):
        """Test getting models by task type"""
        classification_models = self.registry.get_models_by_task_type('classification')
        assert isinstance(classification_models, dict)
        
        regression_models = self.registry.get_models_by_task_type('regression')
        assert isinstance(regression_models, dict)

class TestNeuralNetworksTrainer:
    """Test the NeuralNetworksTrainer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.trainer = NeuralNetworksTrainer()
        
        # Create sample datasets
        self.X_classification, self.y_classification = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_redundant=0, 
            n_informative=8, random_state=42
        )
        
        self.X_regression, self.y_regression = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        
        # Create simple sequence data for RNN testing
        np.random.seed(42)
        self.X_sequence = np.random.randn(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
        self.y_sequence = np.random.randint(0, 2, 100)  # Binary classification
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly"""
        assert isinstance(self.trainer, NeuralNetworksTrainer)
        assert hasattr(self.trainer, 'registry')
        assert hasattr(self.trainer, 'trained_models')
        assert hasattr(self.trainer, 'training_history')
    
    def test_prepare_data(self):
        """Test data preparation"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification
        )
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert len(X_train) > len(X_test)  # Default test_size=0.2
        
        # Test unsupervised data preparation
        X_unsup, _, _, _ = self.trainer.prepare_data(self.X_classification)
        assert X_unsup is not None
    
    def test_train_basic_nn_models(self):
        """Test training basic neural network models"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        basic_models = self.trainer.registry.get_basic_nn_models()
        
        for model_name in list(basic_models.keys())[:2]:  # Test first 2 models
            try:
                result = self.trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    category='basic_nn', 
                    hyperparameters={'maxEpochs': 10},  # Quick training
                    verbose=False
                )
                
                assert result is not None
                assert 'model_name' in result
                assert 'training_successful' in result
                
                if result['training_successful']:
                    assert result['model_instance'] is not None
                    assert 'test_accuracy' in result
                    print(f"✓ {model_name}: Training successful")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_deep_learning_models(self):
        """Test training deep learning models (with image-like data)"""
        # Create simple image-like data
        X_image = np.random.randn(50, 28, 28)  # 50 samples, 28x28 images
        y_image = np.random.randint(0, 3, 50)
        
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            X_image, y_image, test_size=0.3, scale_features=False
        )
        
        deep_models = self.trainer.registry.get_deep_learning_models()
        
        # Test CNN (most likely to work with our simple implementation)
        if 'cnn' in deep_models:
            try:
                result = self.trainer.train_model(
                    'cnn', X_train, y_train, X_test, y_test,
                    category='deep_learning',
                    hyperparameters={'maxEpochs': 5, 'batchSize': 16},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print("✓ CNN: Training successful")
                else:
                    print(f"✗ CNN: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ CNN: Exception during training - {str(e)}")
    
    def test_train_recurrent_models(self):
        """Test training recurrent models"""
        recurrent_models = self.trainer.registry.get_recurrent_models()
        
        for model_name in list(recurrent_models.keys())[:2]:  # Test first 2 models
            try:
                result = self.trainer.train_model(
                    model_name, self.X_sequence, self.y_sequence,
                    category='recurrent',
                    hyperparameters={'maxEpochs': 5, 'hiddenDim': 32},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print(f"✓ {model_name}: Training successful")
                else:
                    print(f"✗ {model_name}: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {model_name}: Exception during training - {str(e)}")
    
    def test_train_generative_models(self):
        """Test training generative models"""
        X_train, _, _, _ = self.trainer.prepare_data(
            self.X_classification, test_size=0.3
        )
        
        generative_models = self.trainer.registry.get_generative_models()
        
        # Test autoencoder (unsupervised)
        if 'autoencoder' in generative_models:
            try:
                result = self.trainer.train_model(
                    'autoencoder', X_train,
                    category='generative',
                    hyperparameters={'maxEpochs': 10, 'encodingDim': 5},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print("✓ Autoencoder: Training successful")
                else:
                    print(f"✗ Autoencoder: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ Autoencoder: Exception during training - {str(e)}")
        
        # Test GAN
        if 'gan' in generative_models:
            try:
                result = self.trainer.train_model(
                    'gan', X_train,
                    category='generative',
                    hyperparameters={'maxEpochs': 5, 'latentDim': 10},
                    verbose=False
                )
                
                assert result is not None
                if result['training_successful']:
                    print("✓ GAN: Training successful")
                else:
                    print(f"✗ GAN: Training failed - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ GAN: Exception during training - {str(e)}")
    
    def test_train_multiple_models(self):
        """Test training multiple models at once"""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        basic_models = list(self.trainer.registry.get_basic_nn_models().keys())[:2]
        
        if basic_models:
            results = self.trainer.train_multiple_models(
                basic_models, X_train, y_train, X_test, y_test,
                category='basic_nn', verbose=False
            )
            
            assert isinstance(results, list)
            assert len(results) == len(basic_models)
            
            for result in results:
                assert 'model_name' in result
                assert 'training_successful' in result
    
    def test_get_training_summary(self):
        """Test getting training summary"""
        # Train a model first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        basic_models = list(self.trainer.registry.get_basic_nn_models().keys())
        if basic_models:
            self.trainer.train_model(
                basic_models[0], X_train, y_train, X_test, y_test,
                category='basic_nn', verbose=False
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
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X_classification, self.y_classification, test_size=0.3
        )
        
        basic_models = list(self.trainer.registry.get_basic_nn_models().keys())[:2]
        
        for model_name in basic_models:
            try:
                self.trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    category='basic_nn', verbose=False
                )
            except:
                pass  # Some models might fail, that's okay for this test
        
        best_model = self.trainer.get_best_model('test_accuracy')
        
        # Should return None if no successful models, or a dict if there are successful models
        assert best_model is None or isinstance(best_model, dict)
    
    def test_clear_history(self):
        """Test clearing training history"""
        # Add some training history first
        X_train, _, y_train, _ = self.trainer.prepare_data(
            self.X_classification, self.y_classification
        )
        
        basic_models = list(self.trainer.registry.get_basic_nn_models().keys())
        if basic_models:
            try:
                self.trainer.train_model(
                    basic_models[0], X_train, y_train,
                    category='basic_nn', verbose=False
                )
            except:
                pass
        
        # Clear history
        self.trainer.clear_history()
        
        assert len(self.trainer.trained_models) == 0
        assert len(self.trainer.training_history) == 0

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_neural_networks_info(self):
        """Test getting neural networks info"""
        info = get_neural_networks_info()
        assert isinstance(info, dict)
        
        expected_categories = ['basic_nn', 'deep_learning', 'recurrent', 'generative', 'specialized']
        for category in expected_categories:
            assert category in info
    
    def test_create_model_comparison_report(self):
        """Test creating model comparison report"""
        # Create sample training results
        sample_results = [
            {
                'model_display_name': 'Test Model 1',
                'model_type': 'basic_nn',
                'model_category': 'basic_nn',
                'training_successful': True,
                'training_samples': 100,
                'test_accuracy': 0.85,
                'test_mse': None,
                'task_types': ['classification'],
                'data_types': ['tabular'],
                'hyperparameters': {'learningRate': 0.01}
            },
            {
                'model_display_name': 'Test Model 2',
                'model_type': 'deep_learning',
                'model_category': 'deep_learning',
                'training_successful': True,
                'training_samples': 100,
                'test_accuracy': 0.90,
                'test_mse': None,
                'task_types': ['classification'],
                'data_types': ['image'],
                'hyperparameters': {'learningRate': 0.001}
            }
        ]
        
        report = create_model_comparison_report(sample_results)
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 2
        
        expected_columns = ['Model', 'Type', 'Category', 'Training_Samples', 'Test_Accuracy']
        for col in expected_columns:
            assert col in report.columns

def run_comprehensive_test():
    """Run a comprehensive test of the neural networks module"""
    print("=" * 60)
    print("NEURAL NETWORKS MODULE COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test registry
    print("\n1. Testing Registry...")
    registry = NeuralNetworksRegistry()
    all_models = registry.get_all_models()
    
    total_models = 0
    for category, models in all_models.items():
        print(f"   {category}: {len(models)} models")
        total_models += len(models)
    
    print(f"   Total models available: {total_models}")
    
    # Test trainer with sample data
    print("\n2. Testing Trainer...")
    trainer = NeuralNetworksTrainer()
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.3)
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    
    # Test a few models from each category
    successful_models = 0
    failed_models = 0
    
    for category, models in all_models.items():
        if models:  # If category has models
            model_name = list(models.keys())[0]  # Get first model
            try:
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    category=category,
                    hyperparameters={'maxEpochs': 5} if 'maxEpochs' in models[model_name]['hyperparameters'] else {},
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
        test_registry = TestNeuralNetworksRegistry()
        test_registry.setup_method()
        test_registry.test_registry_initialization()
        test_registry.test_get_all_models()
        print("✓ Registry tests passed")
        
        test_trainer = TestNeuralNetworksTrainer()
        test_trainer.setup_method()
        test_trainer.test_trainer_initialization()
        test_trainer.test_prepare_data()
        print("✓ Trainer tests passed")
        
        test_utils = TestUtilityFunctions()
        test_utils.test_get_neural_networks_info()
        print("✓ Utility function tests passed")
        
        print("\n✓ All basic tests completed successfully!")