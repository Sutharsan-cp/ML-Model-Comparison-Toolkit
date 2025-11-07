"""
Final comprehensive test with appropriate data for each model type
"""

import numpy as np
from sklearn.datasets import make_classification
import sys
import os

sys.path.append(os.path.dirname(__file__))

from neural_networks import NeuralNetworksRegistry, NeuralNetworksTrainer

def test_all_models():
    """Test all 17 neural network models with appropriate data"""
    
    print("=" * 80)
    print("FINAL COMPREHENSIVE TEST - ALL 17 NEURAL NETWORK MODELS")
    print("=" * 80)
    
    registry = NeuralNetworksRegistry()
    trainer = NeuralNetworksTrainer()
    
    # Prepare different types of data
    print("\n📊 Preparing test datasets...")
    
    # Tabular data for basic NN
    X_tabular, y_tabular = make_classification(n_samples=100, n_features=10, n_classes=3, 
                                               n_informative=8, n_redundant=0, random_state=42)
    X_train_tab, X_test_tab, y_train_tab, y_test_tab = trainer.prepare_data(X_tabular, y_tabular, test_size=0.3)
    
    # Image-like data for CNN
    X_image = np.random.randn(50, 28, 28)
    y_image = np.random.randint(0, 3, 50)
    X_train_img, X_test_img, y_train_img, y_test_img = trainer.prepare_data(X_image, y_image, test_size=0.3, scale_features=False)
    
    # Sequence data for RNN
    X_seq = np.random.randn(50, 10, 5)
    y_seq = np.random.randint(0, 2, (50, 10))
    
    # Unsupervised data for autoencoder
    X_unsup = X_train_tab
    
    # Sequence tokens for transformer
    X_tokens = np.random.randint(1, 100, (20, 10))
    y_tokens = np.random.randint(1, 100, (20, 10))
    
    all_models = registry.get_all_models()
    
    total_models = 0
    successful_models = 0
    failed_models = 0
    
    print("\n🧪 Testing all models...")
    print("-" * 80)
    
    for category, models in all_models.items():
        print(f"\n📁 Category: {category.upper()}")
        
        for model_name, model_info in models.items():
            total_models += 1
            print(f"   {total_models}. {model_info['name']} ({model_name})...", end=" ")
            
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
                    X_train, X_test, y_train, y_test = X_tokens, X_tokens[:5], y_tokens, y_tokens[:5]
                
                # Set minimal hyperparameters for quick testing
                hyperparameters = {'maxEpochs': 2}
                
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
                
                # Train the model
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    category=category, hyperparameters=hyperparameters,
                    verbose=False
                )
                
                if result['training_successful']:
                    successful_models += 1
                    accuracy = result.get('test_accuracy', 'N/A')
                    if accuracy != 'N/A' and accuracy is not None:
                        print(f"✅ SUCCESS (Accuracy: {accuracy:.3f})")
                    else:
                        print(f"✅ SUCCESS")
                else:
                    failed_models += 1
                    error = result.get('error', 'Unknown error')
                    print(f"❌ FAILED ({error[:50]}...)")
                
            except Exception as e:
                failed_models += 1
                print(f"💥 EXCEPTION ({type(e).__name__}: {str(e)[:50]}...)")
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Total models tested: {total_models}")
    print(f"✅ Successful: {successful_models}")
    print(f"❌ Failed: {failed_models}")
    print(f"📊 Success rate: {successful_models/total_models*100:.1f}%")
    print("=" * 80)
    
    # Print summary by category
    print("\n📋 Summary by Category:")
    for category, models in all_models.items():
        print(f"   {category}: {len(models)} models")
    
    if successful_models == total_models:
        print("\n🎉 ALL MODELS WORKING! 🎉")
    elif successful_models >= total_models * 0.8:
        print("\n✨ Most models working! Great job!")
    else:
        print("\n⚠️  Some models need attention")
    
    return successful_models, total_models

if __name__ == "__main__":
    successful, total = test_all_models()