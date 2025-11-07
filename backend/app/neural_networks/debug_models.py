"""
Debug script to test individual neural network models and identify issues
"""

import numpy as np
import sys
import os
from sklearn.datasets import make_classification, make_regression

# Add the neural_networks module to path
sys.path.append(os.path.dirname(__file__))

from neural_networks import NeuralNetworksRegistry, NeuralNetworksTrainer

def test_individual_models():
    """Test each model individually to identify specific issues"""
    
    print("=" * 80)
    print("INDIVIDUAL MODEL TESTING - DETAILED ERROR ANALYSIS")
    print("=" * 80)
    
    registry = NeuralNetworksRegistry()
    trainer = NeuralNetworksTrainer()
    
    # Prepare different types of data
    print("\n📊 Preparing test datasets...")
    
    # Tabular data for basic NN
    X_tabular, y_tabular = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=8, n_redundant=0, random_state=42)
    X_train_tab, X_test_tab, y_train_tab, y_test_tab = trainer.prepare_data(X_tabular, y_tabular, test_size=0.3)
    print(f"   Tabular data: {X_train_tab.shape} train, {X_test_tab.shape} test")
    
    # Image-like data for CNN
    X_image = np.random.randn(50, 28, 28)
    y_image = np.random.randint(0, 3, 50)
    X_train_img, X_test_img, y_train_img, y_test_img = trainer.prepare_data(X_image, y_image, test_size=0.3, scale_features=False)
    print(f"   Image data: {X_train_img.shape} train, {X_test_img.shape} test")
    
    # Sequence data for RNN
    X_seq = np.random.randn(50, 10, 5)  # 50 samples, 10 timesteps, 5 features
    y_seq = np.random.randint(0, 2, (50, 10))  # Sequence labels
    print(f"   Sequence data: {X_seq.shape} input, {y_seq.shape} labels")
    
    # Unsupervised data for autoencoder
    X_unsup = X_train_tab  # Use tabular data
    print(f"   Unsupervised data: {X_unsup.shape}")
    
    all_models = registry.get_all_models()
    
    for category, models in all_models.items():
        print(f"\n🔍 Testing {category.upper()} models:")
        print("-" * 50)
        
        for model_name, model_info in models.items():
            print(f"\n🧠 Testing {model_info['name']} ({model_name})...")
            
            try:
                # Choose appropriate data based on model type
                if category == 'basic_nn':
                    X_train, X_test, y_train, y_test = X_train_tab, X_test_tab, y_train_tab, y_test_tab
                elif category == 'deep_learning':
                    X_train, X_test, y_train, y_test = X_train_img, X_test_img, y_train_img, y_test_img
                elif category == 'recurrent':
                    X_train, X_test, y_train, y_test = X_seq, X_seq[:15], y_seq, y_seq[:15]
                elif category == 'generative':
                    X_train, X_test, y_train, y_test = X_unsup, None, None, None
                elif category == 'specialized':
                    # Transformer needs special sequence data
                    X_train = np.random.randint(1, 100, (20, 10))  # Token sequences
                    y_train = np.random.randint(1, 100, (20, 10))  # Target sequences
                    X_test, y_test = X_train[:5], y_train[:5]
                
                # Set minimal hyperparameters for quick testing
                hyperparameters = {'maxEpochs': 3}
                
                # Add model-specific hyperparameters
                if model_name in ['cnn', 'resnet', 'vgg']:
                    hyperparameters['batchSize'] = 8
                elif model_name in ['rnn', 'lstm', 'gru']:
                    hyperparameters['hiddenDim'] = 16
                elif model_name == 'autoencoder':
                    hyperparameters['encodingDim'] = 5
                elif model_name == 'gan':
                    hyperparameters['latentDim'] = 10
                elif model_name == 'transformer':
                    hyperparameters.update({
                        'vocabSize': 100,
                        'dModel': 64,
                        'nHeads': 2,
                        'nLayers': 1,
                        'batchSize': 4
                    })
                
                print(f"   Data shape: {X_train.shape if X_train is not None else 'None'}")
                print(f"   Hyperparameters: {hyperparameters}")
                
                # Train the model
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    category=category, hyperparameters=hyperparameters,
                    verbose=False
                )
                
                if result['training_successful']:
                    accuracy = result.get('test_accuracy', 'N/A')
                    print(f"   ✅ SUCCESS - Accuracy: {accuracy}")
                else:
                    print(f"   ❌ FAILED - Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"   💥 EXCEPTION - {type(e).__name__}: {str(e)}")
                
                # Print more detailed error info
                import traceback
                print(f"   📋 Traceback: {traceback.format_exc().split('\\n')[-3:-1]}")
    
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL TESTING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_individual_models()