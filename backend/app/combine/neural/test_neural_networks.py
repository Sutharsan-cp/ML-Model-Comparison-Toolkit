"""
Test script for neural network models
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our neural networks module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neural_networks import NeuralNetworkRegistry, NeuralNetworkTrainer

def generate_test_data():
    """Generate various test datasets for neural networks"""
    datasets = {}
    
    # Binary classification data
    X_binary, y_binary = make_classification(n_samples=200, n_features=10, n_classes=2, 
                                            n_informative=5, random_state=42)
    datasets['binary_classification'] = (X_binary, y_binary, "Binary classification")
    
    # Multi-class classification data
    X_multi, y_multi = make_classification(n_samples=200, n_features=10, n_classes=3, 
                                          n_informative=7, random_state=42)
    datasets['multi_classification'] = (X_multi, y_multi, "Multi-class classification")
    
    # Regression data
    X_reg, y_reg = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    datasets['regression'] = (X_reg, y_reg, "Regression")
    
    # Image-like data (for CNN testing)
    np.random.seed(42)
    X_image = np.random.randn(100, 28, 28)  # 28x28 images
    y_image = np.random.randint(0, 3, 100)  # 3 classes
    datasets['image_classification'] = (X_image, y_image, "Image classification")
    
    # Sequence data (for RNN testing)
    X_seq = np.random.randn(100, 20, 5)  # 100 sequences, 20 timesteps, 5 features
    y_seq = np.random.randint(0, 2, 100)  # Binary classification
    datasets['sequence_classification'] = (X_seq, y_seq, "Sequence classification")
    
    # Autoencoder data (unsupervised)
    X_ae = np.random.randn(200, 20)
    datasets['autoencoder'] = (X_ae, None, "Autoencoder reconstruction")
    
    return datasets

def test_feedforward_models():
    """Test feedforward neural network models"""
    print("Testing Feedforward Neural Network Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y, description = datasets['binary_classification']  # Use binary classification
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = NeuralNetworkRegistry()
    trainer = NeuralNetworkTrainer()
    
    # Get available feedforward models
    feedforward_models = registry.get_feedforward_models()
    print(f"Available feedforward models: {len(feedforward_models)}")
    print(f"Test data: {description} - {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test all feedforward models
    test_models = list(feedforward_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {}
                if model_name == 'mlp':
                    hyperparams = {'hiddenLayers': [50], 'maxEpochs': 10, 'batchSize': 32}
                elif model_name == 'ann':
                    hyperparams = {'layers': [64, 32], 'maxEpochs': 10, 'batchSize': 32}
                elif model_name == 'backpropagation':
                    hyperparams = {'layers': [64, 32], 'maxEpochs': 10, 'batchSize': 32}
                elif model_name == 'perceptron':
                    hyperparams = {'maxEpochs': 100}
                
                result = trainer.train_model(
                    model_name, X_train, y_train, 'feedforward', hyperparams
                )
                
                if result['training_successful']:
                    final_acc = result.get('final_accuracy', 'N/A')
                    train_acc = result.get('training_accuracy', 'N/A')
                    final_loss = result.get('final_loss', 'N/A')
                    
                    if isinstance(final_acc, float):
                        final_acc = f"{final_acc:.3f}"
                    if isinstance(train_acc, float):
                        train_acc = f"{train_acc:.3f}"
                    if isinstance(final_loss, float):
                        final_loss = f"{final_loss:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final_Acc={final_acc}, Train_Acc={train_acc}, Loss={final_loss}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No feedforward models available")
    
    print()

def test_convolutional_models():
    """Test convolutional neural network models"""
    print("Testing Convolutional Neural Network Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y, description = datasets['image_classification']  # Use image data
    
    # Initialize registry and trainer
    registry = NeuralNetworkRegistry()
    trainer = NeuralNetworkTrainer()
    
    # Get available convolutional models
    convolutional_models = registry.get_convolutional_models()
    print(f"Available convolutional models: {len(convolutional_models)}")
    print(f"Test data: {description} - {X.shape[0]} samples, {X.shape[1]}x{X.shape[2]} images")
    
    # Test basic CNN model only (others might be too complex for simple testing)
    test_models = ['cnn'] if 'cnn' in convolutional_models else []
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {'maxEpochs': 5, 'batchSize': 16}
                
                result = trainer.train_model(
                    model_name, X, y, 'convolutional', hyperparams
                )
                
                if result['training_successful']:
                    final_acc = result.get('final_accuracy', 'N/A')
                    final_loss = result.get('final_loss', 'N/A')
                    
                    if isinstance(final_acc, float):
                        final_acc = f"{final_acc:.3f}"
                    if isinstance(final_loss, float):
                        final_loss = f"{final_loss:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final_Acc={final_acc}, Loss={final_loss}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No convolutional models available for testing")
    
    print()

def test_recurrent_models():
    """Test recurrent neural network models"""
    print("Testing Recurrent Neural Network Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y, description = datasets['sequence_classification']  # Use sequence data
    
    # Initialize registry and trainer
    registry = NeuralNetworkRegistry()
    trainer = NeuralNetworkTrainer()
    
    # Get available recurrent models
    recurrent_models = registry.get_recurrent_models()
    print(f"Available recurrent models: {len(recurrent_models)}")
    print(f"Test data: {description} - {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
    
    # Test recurrent models
    test_models = list(recurrent_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                hyperparams = {'hiddenDim': 32, 'maxEpochs': 5, 'batchSize': 16}
                
                result = trainer.train_model(
                    model_name, X, y, 'recurrent', hyperparams
                )
                
                if result['training_successful']:
                    final_acc = result.get('final_accuracy', 'N/A')
                    final_loss = result.get('final_loss', 'N/A')
                    
                    if isinstance(final_acc, float):
                        final_acc = f"{final_acc:.3f}"
                    if isinstance(final_loss, float):
                        final_loss = f"{final_loss:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: "
                          f"Final_Acc={final_acc}, Loss={final_loss}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No recurrent models available for testing")
    
    print()

def test_generative_models():
    """Test generative models"""
    print("Testing Generative Models")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, _, description = datasets['autoencoder']  # Use unsupervised data
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize registry and trainer
    registry = NeuralNetworkRegistry()
    trainer = NeuralNetworkTrainer()
    
    # Get available generative models
    generative_models = registry.get_generative_models()
    print(f"Available generative models: {len(generative_models)}")
    print(f"Test data: {description} - {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test generative models
    test_models = list(generative_models.keys())
    
    if test_models:
        print(f"Testing models: {test_models}")
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            try:
                # Use simple hyperparameters for testing
                if model_name == 'autoencoder':
                    hyperparams = {'encodingDim': 8, 'hiddenLayers': [32], 'maxEpochs': 10, 'batchSize': 32}
                elif model_name == 'gan':
                    hyperparams = {'latentDim': 10, 'maxEpochs': 5, 'batchSize': 32}
                else:
                    hyperparams = {}
                
                # For autoencoders, we don't need y (unsupervised)
                result = trainer.train_model(
                    model_name, X_scaled, None, 'generative', hyperparams
                )
                
                if result['training_successful']:
                    final_loss = result.get('final_loss', 'N/A')
                    
                    if isinstance(final_loss, float):
                        final_loss = f"{final_loss:.3f}"
                    
                    print(f"✅ {result['model_display_name']}: Loss={final_loss}")
                else:
                    print(f"❌ {result['model_display_name']}: {result['error']}")
            
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
    else:
        print("No generative models available for testing")
    
    print()

def test_model_comparison():
    """Test model comparison functionality"""
    print("Testing Model Comparison")
    print("=" * 50)
    
    # Generate test data
    datasets = generate_test_data()
    X, y, description = datasets['binary_classification']
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize trainer
    trainer = NeuralNetworkTrainer()
    
    # Train a few models for comparison
    test_models = ['perceptron', 'mlp']
    results = []
    
    for model_name in test_models:
        try:
            hyperparams = {}
            if model_name == 'perceptron':
                hyperparams = {'maxEpochs': 50}
            elif model_name == 'mlp':
                hyperparams = {'hiddenLayers': [32], 'maxEpochs': 10, 'batchSize': 32}
            
            result = trainer.train_model(model_name, X_scaled, y, 'feedforward', hyperparams)
            results.append(result)
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Generate comparison report
    from neural_networks import create_model_comparison_report
    
    try:
        comparison_df = create_model_comparison_report(results)
        print("✅ Comparison report generated successfully")
        print(f"Report shape: {comparison_df.shape}")
        
        if not comparison_df.empty:
            print("\nComparison Report:")
            print(comparison_df.to_string(index=False))
        
        # Get training summary
        summary_df = trainer.get_training_summary()
        print(f"\nTraining Summary shape: {summary_df.shape}")
        
        # Get best model
        best_model = trainer.get_best_model('training_accuracy')
        if best_model:
            print(f"\nBest model: {best_model['model_display_name']}")
        
    except Exception as e:
        print(f"❌ Failed to generate comparison report: {e}")

def main():
    """Main test function"""
    print("Neural Network Models Test")
    print("=" * 60)
    
    try:
        test_feedforward_models()
        test_convolutional_models()
        test_recurrent_models()
        test_generative_models()
        test_model_comparison()
        
        # Show summary
        registry = NeuralNetworkRegistry()
        all_models = registry.get_all_models()
        
        print("Summary:")
        print(f"Total Feedforward Models: {len(all_models['feedforward'])}")
        print(f"Total Convolutional Models: {len(all_models['convolutional'])}")
        print(f"Total Recurrent Models: {len(all_models['recurrent'])}")
        print(f"Total Generative Models: {len(all_models['generative'])}")
        print(f"Total Specialized Models: {len(all_models['specialized'])}")
        
        print("\nFeedforward Models:")
        for name, info in all_models['feedforward'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nConvolutional Models:")
        for name, info in all_models['convolutional'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nRecurrent Models:")
        for name, info in all_models['recurrent'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nGenerative Models:")
        for name, info in all_models['generative'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nSpecialized Models:")
        for name, info in all_models['specialized'].items():
            print(f"  - {info['name']} ({info['type']})")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()