"""
Test script for semi-supervised learning models
"""

import sys
import os
import numpy as np
import warnings
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Import our semi-supervised module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semi_supervised_learning import SemiSupervisedLearningRegistry, SemiSupervisedLearningTrainer

def create_test_dataset(n_samples=1000, n_features=10, n_classes=3, random_state=42):
    """Create a test dataset for semi-supervised learning"""
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

def test_graph_based_models():
    """Test graph-based semi-supervised models"""
    print("Testing Graph-Based Semi-Supervised Models")
    print("=" * 60)
    
    # Create test dataset
    X, y = create_test_dataset(n_samples=300, n_features=5, n_classes=3)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data (10% labeled, 90% unlabeled)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.1, random_state=42
    )
    
    # Get available graph-based models
    graph_based_models = registry.get_graph_based_models()
    print(f"Available graph-based models: {len(graph_based_models)}")
    print(f"Dataset: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test samples")
    
    # Test graph-based models
    test_models = list(graph_based_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'label_propagation':
                    hyperparams = {
                        'kernel': 'rbf',
                        'gamma': 20,
                        'max_iter': 100
                    }
                elif model_name == 'label_spreading':
                    hyperparams = {
                        'kernel': 'rbf',
                        'gamma': 20,
                        'alpha': 0.2,
                        'max_iter': 100
                    }
                
                result = trainer.train_model(
                    model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                    X_test, y_test, 'graph_based', hyperparams
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    if isinstance(accuracy, float):
                        accuracy = f"{accuracy:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Accuracy={accuracy}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No graph-based models available")
    
    print()

def test_self_training_models():
    """Test self-training semi-supervised models"""
    print("Testing Self-Training Semi-Supervised Models")
    print("=" * 60)
    
    # Create test dataset - binary classification for better results
    X, y = create_test_dataset(n_samples=300, n_features=8, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data (20% labeled, 80% unlabeled)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.2, random_state=42
    )
    
    # Get available self-training models
    self_training_models = registry.get_self_training_models()
    print(f"Available self-training models: {len(self_training_models)}")
    print(f"Dataset: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test samples")
    
    # Test self-training models
    test_models = list(self_training_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'self_training':
                    from sklearn.ensemble import RandomForestClassifier
                    hyperparams = {
                        'base_estimator': RandomForestClassifier(n_estimators=50, random_state=42),
                        'threshold': 0.75,
                        'max_iter': 5
                    }
                elif model_name == 'co_training':
                    hyperparams = {
                        'classifier1': 'random_forest',
                        'classifier2': 'svm',
                        'max_iter': 3
                    }
                
                result = trainer.train_model(
                    model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                    X_test, y_test, 'self_training', hyperparams
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    if isinstance(accuracy, float):
                        accuracy = f"{accuracy:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Accuracy={accuracy}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No self-training models available")
    
    print()

def test_model_comparison():
    """Test model comparison functionality"""
    print("Testing Model Comparison")
    print("=" * 60)
    
    # Create test dataset
    X, y = create_test_dataset(n_samples=200, n_features=5, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize trainer
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.15, random_state=42
    )
    
    # Train a few models for comparison
    test_models = ['label_propagation', 'label_spreading', 'self_training']
    results = []
    
    for model_name in test_models:
        try:
            if model_name in ['label_propagation', 'label_spreading']:
                hyperparams = {
                    'kernel': 'rbf',
                    'gamma': 20,
                    'max_iter': 50
                }
                if model_name == 'label_spreading':
                    hyperparams['alpha'] = 0.2
                category = 'graph_based'
            else:
                hyperparams = {
                    'base_classifier': 'random_forest',
                    'threshold': 0.75,
                    'max_iter': 3
                }
                category = 'self_training'
            
            result = trainer.train_model(
                model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                X_test, y_test, category, hyperparams
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from semi_supervised_learning import create_model_comparison_report
    
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

def test_data_type_filtering():
    """Test filtering models by data type"""
    print("Testing Data Type Filtering")
    print("=" * 60)
    
    registry = SemiSupervisedLearningRegistry()
    
    # Test different data types
    data_types = ['tabular', 'image', 'text', 'graph']
    
    for data_type in data_types:
        suitable_models = registry.get_models_by_data_type(data_type)
        print(f"\nModels suitable for {data_type} data: {len(suitable_models)}")
        for model_key, model_info in suitable_models.items():
            print(f"  - {model_info['name']} ({model_info['type']})")
    
    print()

def test_consistency_models():
    """Test consistency regularization semi-supervised models"""
    print("Testing Consistency Regularization Semi-Supervised Models")
    print("=" * 60)
    
    # Create test dataset - binary classification for better results
    X, y = create_test_dataset(n_samples=200, n_features=6, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data (25% labeled, 75% unlabeled)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.25, random_state=42
    )
    
    # Get available consistency models
    consistency_models = registry.get_consistency_models()
    print(f"Available consistency models: {len(consistency_models)}")
    print(f"Dataset: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test samples")
    
    # Test consistency models
    test_models = list(consistency_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'mean_teacher':
                    hyperparams = {
                        'alpha': 0.99,
                        'consistency_weight': 1.0,
                        'epochs': 10
                    }
                elif model_name == 'fixmatch':
                    hyperparams = {
                        'threshold': 0.95,
                        'lambda_u': 1.0,
                        'epochs': 10
                    }
                elif model_name == 'mixmatch':
                    hyperparams = {
                        'T': 0.5,
                        'alpha': 0.75,
                        'lambda_u': 10,
                        'epochs': 10
                    }
                
                result = trainer.train_model(
                    model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                    X_test, y_test, 'consistency', hyperparams
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    if isinstance(accuracy, float):
                        accuracy = f"{accuracy:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Accuracy={accuracy}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No consistency models available")
    
    print()

def test_contrastive_models():
    """Test contrastive learning semi-supervised models"""
    print("Testing Contrastive Learning Semi-Supervised Models")
    print("=" * 60)
    
    # Create test dataset
    X, y = create_test_dataset(n_samples=200, n_features=8, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.2, random_state=42
    )
    
    # Get available contrastive models
    contrastive_models = registry.get_contrastive_models()
    print(f"Available contrastive models: {len(contrastive_models)}")
    print(f"Dataset: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test samples")
    
    # Test contrastive models
    test_models = list(contrastive_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'simclr':
                    hyperparams = {
                        'projection_dim': 64,
                        'temperature': 0.5,
                        'epochs': 10
                    }
                elif model_name == 'byol':
                    hyperparams = {
                        'moving_average_decay': 0.99,
                        'projection_dim': 64,
                        'epochs': 10
                    }
                
                result = trainer.train_model(
                    model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                    X_test, y_test, 'contrastive', hyperparams
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    if isinstance(accuracy, float):
                        accuracy = f"{accuracy:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Accuracy={accuracy}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No contrastive models available")
    
    print()

def test_deep_learning_models():
    """Test deep learning semi-supervised models"""
    print("Testing Deep Learning Semi-Supervised Models")
    print("=" * 60)
    
    # Create test dataset
    X, y = create_test_dataset(n_samples=200, n_features=6, n_classes=2)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SemiSupervisedLearningRegistry()
    trainer = SemiSupervisedLearningTrainer()
    
    # Prepare semi-supervised data
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = trainer.prepare_semi_supervised_data(
        X_train, y_train, labeled_ratio=0.2, random_state=42
    )
    
    # Get available deep learning models
    deep_learning_models = registry.get_deep_learning_models()
    print(f"Available deep learning models: {len(deep_learning_models)}")
    print(f"Dataset: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test samples")
    
    # Test deep learning models
    test_models = list(deep_learning_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'ladder_network':
                    hyperparams = {
                        'noise_std': 0.3,
                        'supervised_weight': 1.0,
                        'unsupervised_weight': 1.0,
                        'epochs': 10
                    }
                elif model_name == 'masked_autoencoder':
                    hyperparams = {
                        'mask_ratio': 0.75,
                        'epochs': 10
                    }
                
                result = trainer.train_model(
                    model_name, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                    X_test, y_test, 'deep_learning', hyperparams
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    if isinstance(accuracy, float):
                        accuracy = f"{accuracy:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Accuracy={accuracy}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No deep learning models available")
    
    print()

def run_all_tests():
    """Run all semi-supervised learning tests"""
    print("Semi-Supervised Learning Models Test Suite")
    print("=" * 70)
    print()
    
    test_graph_based_models()
    test_self_training_models()
    test_consistency_models()
    test_contrastive_models()
    test_deep_learning_models()
    test_model_comparison()
    test_data_type_filtering()
    
    print("=" * 70)
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests()