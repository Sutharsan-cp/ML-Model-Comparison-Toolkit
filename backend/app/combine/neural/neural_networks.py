"""
Neural Networks Models Module
Consolidates all neural network algorithms for the ML Comparison Toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all neural network models with error handling
feedforward_imports = {}
convolutional_imports = {}
recurrent_imports = {}
generative_imports = {}
specialized_imports = {}

# Add neural networks path to sys.path
import sys
import os
neural_networks_path = os.path.join(os.path.dirname(__file__), 'models', 'neural networks')
if neural_networks_path not in sys.path:
    sys.path.append(neural_networks_path)

# Feedforward Neural Networks
try:
    from perceptron import Perceptron
    feedforward_imports['perceptron'] = Perceptron
except ImportError as e:
    print(f"Warning: Could not import Perceptron: {e}")

try:
    from MLP import MLP
    feedforward_imports['mlp'] = MLP
except ImportError as e:
    print(f"Warning: Could not import MLP: {e}")

try:
    from ANN import ANN
    feedforward_imports['ann'] = ANN
except ImportError as e:
    print(f"Warning: Could not import ANN: {e}")

try:
    from backpropagation import Backpropagation
    feedforward_imports['backpropagation'] = Backpropagation
except ImportError as e:
    print(f"Warning: Could not import Backpropagation: {e}")

# Convolutional Neural Networks
try:
    from CNN import CNN
    convolutional_imports['cnn'] = CNN
except ImportError as e:
    print(f"Warning: Could not import CNN: {e}")

try:
    from VGG import VGG
    convolutional_imports['vgg'] = VGG
except ImportError as e:
    print(f"Warning: Could not import VGG: {e}")

try:
    from resnet import ResNet
    convolutional_imports['resnet'] = ResNet
except ImportError as e:
    print(f"Warning: Could not import ResNet: {e}")

try:
    from inception import Inception
    convolutional_imports['inception'] = Inception
except ImportError as e:
    print(f"Warning: Could not import Inception: {e}")

try:
    from densenet import DenseNet
    convolutional_imports['densenet'] = DenseNet
except ImportError as e:
    print(f"Warning: Could not import DenseNet: {e}")

try:
    from efficientnet import EfficientNet
    convolutional_imports['efficientnet'] = EfficientNet
except ImportError as e:
    print(f"Warning: Could not import EfficientNet: {e}")

try:
    from mobilenet import MobileNet
    convolutional_imports['mobilenet'] = MobileNet
except ImportError as e:
    print(f"Warning: Could not import MobileNet: {e}")

# Recurrent Neural Networks
try:
    from RNN import RNN
    recurrent_imports['rnn'] = RNN
except ImportError as e:
    print(f"Warning: Could not import RNN: {e}")

try:
    from LSTM import LSTM
    recurrent_imports['lstm'] = LSTM
except ImportError as e:
    print(f"Warning: Could not import LSTM: {e}")

try:
    from GRU import GRU
    recurrent_imports['gru'] = GRU
except ImportError as e:
    print(f"Warning: Could not import GRU: {e}")

# Generative Models
try:
    from autoencoder import Autoencoder
    generative_imports['autoencoder'] = Autoencoder
except ImportError as e:
    print(f"Warning: Could not import Autoencoder: {e}")

try:
    from GAN import GAN
    generative_imports['gan'] = GAN
except ImportError as e:
    print(f"Warning: Could not import GAN: {e}")

# Specialized Architectures
try:
    from transformer import Transformer
    specialized_imports['transformer'] = Transformer
except ImportError as e:
    print(f"Warning: Could not import Transformer: {e}")

class NeuralNetworkRegistry:
    """Registry for all neural network models"""
    
    def __init__(self):
        # Build feedforward models dictionary
        self.feedforward_models = {}
        
        # Define feedforward model configurations
        feedforward_configs = {
            'perceptron': {
                'name': 'Perceptron',
                'type': 'linear',
                'description': 'Single-layer linear classifier',
                'task_types': ['binary_classification'],
                'hyperparameters': {
                    'learningRate': [0.001, 0.01, 0.1, 1.0],
                    'maxEpochs': [100, 500, 1000, 2000]
                }
            },
            'mlp': {
                'name': 'Multi-Layer Perceptron',
                'type': 'feedforward',
                'description': 'Multi-layer feedforward neural network',
                'task_types': ['classification', 'regression'],
                'hyperparameters': {
                    'hiddenLayers': [[50], [100], [50, 25], [100, 50], [128, 64, 32]],
                    'activation': ['relu', 'sigmoid', 'tanh'],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            },
            'ann': {
                'name': 'Artificial Neural Network',
                'type': 'feedforward',
                'description': 'General artificial neural network',
                'task_types': ['classification', 'regression'],
                'hyperparameters': {
                    'layers': [[64, 32], [128, 64], [100, 50, 25]],
                    'activation': ['relu', 'sigmoid', 'tanh'],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            },
            'backpropagation': {
                'name': 'Backpropagation Network',
                'type': 'feedforward',
                'description': 'Neural network trained with backpropagation',
                'task_types': ['classification', 'regression'],
                'hyperparameters': {
                    'layers': [[64, 32], [128, 64], [100, 50, 25]],
                    'activation': ['sigmoid', 'relu', 'tanh'],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            }
        }
        
        # Add successfully imported feedforward models
        for model_key, model_class in feedforward_imports.items():
            if model_key in feedforward_configs:
                config = feedforward_configs[model_key].copy()
                config['class'] = model_class
                self.feedforward_models[model_key] = config
        
        # Build convolutional models dictionary
        self.convolutional_models = {}
        
        # Define convolutional model configurations
        convolutional_configs = {
            'cnn': {
                'name': 'Convolutional Neural Network',
                'type': 'convolutional',
                'description': 'Basic CNN for image classification',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'vgg': {
                'name': 'VGG Network',
                'type': 'convolutional',
                'description': 'VGG architecture for image classification',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'architecture': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'resnet': {
                'name': 'ResNet',
                'type': 'convolutional',
                'description': 'Residual Network with skip connections',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'layers': [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 23, 3]],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'inception': {
                'name': 'Inception Network',
                'type': 'convolutional',
                'description': 'Inception architecture with multiple filter sizes',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'densenet': {
                'name': 'DenseNet',
                'type': 'convolutional',
                'description': 'Densely connected convolutional network',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'growthRate': [12, 24, 32, 48],
                    'blockConfig': [[6, 12, 24, 16], [6, 12, 32, 32]],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'efficientnet': {
                'name': 'EfficientNet',
                'type': 'convolutional',
                'description': 'Efficient convolutional neural network',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'compoundCoeff': [0, 1, 2, 3],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            },
            'mobilenet': {
                'name': 'MobileNet',
                'type': 'convolutional',
                'description': 'Lightweight CNN for mobile devices',
                'task_types': ['image_classification'],
                'hyperparameters': {
                    'alpha': [0.25, 0.5, 0.75, 1.0],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [20, 50, 100],
                    'batchSize': [16, 32, 64]
                }
            }
        }
        
        # Add successfully imported convolutional models
        for model_key, model_class in convolutional_imports.items():
            if model_key in convolutional_configs:
                config = convolutional_configs[model_key].copy()
                config['class'] = model_class
                self.convolutional_models[model_key] = config
        
        # Build recurrent models dictionary
        self.recurrent_models = {}
        
        # Define recurrent model configurations
        recurrent_configs = {
            'rnn': {
                'name': 'Recurrent Neural Network',
                'type': 'recurrent',
                'description': 'Basic RNN for sequence processing',
                'task_types': ['sequence_classification', 'time_series'],
                'hyperparameters': {
                    'hiddenDim': [32, 64, 128, 256],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            },
            'lstm': {
                'name': 'Long Short-Term Memory',
                'type': 'recurrent',
                'description': 'LSTM for long sequence processing',
                'task_types': ['sequence_classification', 'time_series'],
                'hyperparameters': {
                    'hiddenDim': [32, 64, 128, 256],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            },
            'gru': {
                'name': 'Gated Recurrent Unit',
                'type': 'recurrent',
                'description': 'GRU for efficient sequence processing',
                'task_types': ['sequence_classification', 'time_series'],
                'hyperparameters': {
                    'hiddenDim': [32, 64, 128, 256],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            }
        }
        
        # Add successfully imported recurrent models
        for model_key, model_class in recurrent_imports.items():
            if model_key in recurrent_configs:
                config = recurrent_configs[model_key].copy()
                config['class'] = model_class
                self.recurrent_models[model_key] = config
        
        # Build generative models dictionary
        self.generative_models = {}
        
        # Define generative model configurations
        generative_configs = {
            'autoencoder': {
                'name': 'Autoencoder',
                'type': 'generative',
                'description': 'Neural network for dimensionality reduction and reconstruction',
                'task_types': ['dimensionality_reduction', 'reconstruction'],
                'hyperparameters': {
                    'encodingDim': [8, 16, 32, 64, 128],
                    'hiddenLayers': [[64, 32], [128, 64], [256, 128, 64]],
                    'activation': ['relu', 'sigmoid', 'tanh'],
                    'learningRate': [0.001, 0.01, 0.1],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            },
            'gan': {
                'name': 'Generative Adversarial Network',
                'type': 'generative',
                'description': 'GAN for generating synthetic data',
                'task_types': ['data_generation'],
                'hyperparameters': {
                    'latentDim': [50, 100, 200],
                    'generatorLayers': [[128, 256], [128, 256, 512]],
                    'discriminatorLayers': [[512, 256], [512, 256, 128]],
                    'learningRate': [0.0001, 0.0002, 0.001],
                    'maxEpochs': [50, 100, 200],
                    'batchSize': [16, 32, 64]
                }
            }
        }
        
        # Add successfully imported generative models
        for model_key, model_class in generative_imports.items():
            if model_key in generative_configs:
                config = generative_configs[model_key].copy()
                config['class'] = model_class
                self.generative_models[model_key] = config
        
        # Build specialized models dictionary
        self.specialized_models = {}
        
        # Define specialized model configurations
        specialized_configs = {
            'transformer': {
                'name': 'Transformer',
                'type': 'attention',
                'description': 'Transformer architecture with self-attention',
                'task_types': ['sequence_to_sequence', 'language_modeling'],
                'hyperparameters': {
                    'vocabSize': [1000, 5000, 10000],
                    'dModel': [256, 512, 768],
                    'nHeads': [4, 8, 12],
                    'dFf': [1024, 2048, 3072],
                    'nLayers': [2, 4, 6],
                    'maxSeqLength': [50, 100, 200],
                    'learningRate': [0.0001, 0.001, 0.01],
                    'maxEpochs': [50, 100, 200]
                }
            }
        }
        
        # Add successfully imported specialized models
        for model_key, model_class in specialized_imports.items():
            if model_key in specialized_configs:
                config = specialized_configs[model_key].copy()
                config['class'] = model_class
                self.specialized_models[model_key] = config
    
    def get_feedforward_models(self) -> Dict[str, Dict]:
        """Get all available feedforward models"""
        return self.feedforward_models
    
    def get_convolutional_models(self) -> Dict[str, Dict]:
        """Get all available convolutional models"""
        return self.convolutional_models
    
    def get_recurrent_models(self) -> Dict[str, Dict]:
        """Get all available recurrent models"""
        return self.recurrent_models
    
    def get_generative_models(self) -> Dict[str, Dict]:
        """Get all available generative models"""
        return self.generative_models
    
    def get_specialized_models(self) -> Dict[str, Dict]:
        """Get all available specialized models"""
        return self.specialized_models
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get all neural network models"""
        return {
            'feedforward': self.feedforward_models,
            'convolutional': self.convolutional_models,
            'recurrent': self.recurrent_models,
            'generative': self.generative_models,
            'specialized': self.specialized_models
        }
    
    def get_model_by_name(self, model_name: str, category: str = None):
        """Get a specific model by name"""
        if category == 'feedforward':
            return self.feedforward_models.get(model_name)
        elif category == 'convolutional':
            return self.convolutional_models.get(model_name)
        elif category == 'recurrent':
            return self.recurrent_models.get(model_name)
        elif category == 'generative':
            return self.generative_models.get(model_name)
        elif category == 'specialized':
            return self.specialized_models.get(model_name)
        else:
            # Search in all categories
            for models in [self.feedforward_models, self.convolutional_models, 
                          self.recurrent_models, self.generative_models, self.specialized_models]:
                if model_name in models:
                    return models[model_name]
            return None
    
    def get_models_by_task(self, task_type: str) -> Dict[str, Dict]:
        """Get models suitable for a specific task type"""
        suitable_models = {}
        
        all_models = self.get_all_models()
        for category, models in all_models.items():
            for name, info in models.items():
                if task_type in info.get('task_types', []):
                    suitable_models[f"{category}_{name}"] = info
        
        return suitable_models

class NeuralNetworkTrainer:
    """Trainer class for neural network models"""
    
    def __init__(self):
        self.registry = NeuralNetworkRegistry()
        self.trained_models = {}
        self.training_history = []
    
    def train_model(self, model_name: str, X, y=None, category: str = None, 
                   hyperparameters: Dict = None, verbose: bool = False) -> Dict:
        """Train a single neural network model"""
        
        model_info = self.registry.get_model_by_name(model_name, category)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Initialize model with hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            model = model_info['class'](**hyperparameters)
            
            # Train the model
            if verbose:
                print(f"Training {model_info['name']}...")
            
            # Handle different training scenarios
            if y is not None:
                model.fit(X, y)
            else:
                # For unsupervised models like autoencoders
                model.fit(X)
            
            result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': model,
                'hyperparameters': hyperparameters,
                'training_successful': True,
                'error': None
            }
            
            # Get model-specific results
            if hasattr(model, 'lossHistory'):
                result['loss_history'] = model.lossHistory
                result['final_loss'] = model.lossHistory[-1] if model.lossHistory else None
            
            if hasattr(model, 'accuracyHistory'):
                result['accuracy_history'] = model.accuracyHistory
                result['final_accuracy'] = model.accuracyHistory[-1] if model.accuracyHistory else None
            
            if hasattr(model, 'converged'):
                result['converged'] = model.converged
            
            # Try to get predictions for evaluation
            if y is not None:
                try:
                    predictions = model.predict(X)
                    if hasattr(predictions, '__len__'):
                        accuracy = np.mean(predictions == y) if len(predictions) == len(y) else None
                        result['training_accuracy'] = accuracy
                except:
                    pass
            
            # Store trained model
            self.trained_models[f"{model_name}_{len(self.trained_models)}"] = result
            self.training_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'model_display_name': model_info['name'],
                'model_type': model_info['type'],
                'model_category': category,
                'model_instance': None,
                'hyperparameters': hyperparameters,
                'training_successful': False,
                'error': str(e)
            }
            self.training_history.append(error_result)
            return error_result
    
    def train_multiple_models(self, model_names: List[str], X, y=None, 
                            category: str = None, verbose: bool = False) -> List[Dict]:
        """Train multiple models"""
        results = []
        
        for model_name in model_names:
            result = self.train_model(model_name, X, y, category, verbose=verbose)
            results.append(result)
        
        return results
    
    def train_all_models(self, X, y=None, category: str = 'feedforward', 
                        verbose: bool = False) -> List[Dict]:
        """Train all available models for a given category"""
        
        if category == 'feedforward':
            models = list(self.registry.get_feedforward_models().keys())
        elif category == 'convolutional':
            models = list(self.registry.get_convolutional_models().keys())
        elif category == 'recurrent':
            models = list(self.registry.get_recurrent_models().keys())
        elif category == 'generative':
            models = list(self.registry.get_generative_models().keys())
        elif category == 'specialized':
            models = list(self.registry.get_specialized_models().keys())
        else:
            raise ValueError("category must be 'feedforward', 'convolutional', 'recurrent', 'generative', or 'specialized'")
        
        return self.train_multiple_models(models, X, y, category, verbose=verbose)
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_history:
            summary_row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('model_category', 'Unknown'),
                'Training_Successful': result['training_successful'],
                'Final_Loss': result.get('final_loss', 'N/A'),
                'Final_Accuracy': result.get('final_accuracy', 'N/A'),
                'Training_Accuracy': result.get('training_accuracy', 'N/A'),
                'Converged': result.get('converged', 'N/A'),
                'Error': result.get('error', None)
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'final_accuracy') -> Optional[Dict]:
        """Get the best performing model based on a metric"""
        successful_models = [r for r in self.training_history if r['training_successful']]
        
        if not successful_models:
            return None
        
        # Filter models that have the requested metric
        models_with_metric = [r for r in successful_models if metric in r and r[metric] is not None and r[metric] != 'N/A']
        
        if not models_with_metric:
            return None
        
        if 'loss' in metric.lower():
            # For loss metrics, lower is better
            return min(models_with_metric, key=lambda x: x[metric])
        else:
            # For accuracy metrics, higher is better
            return max(models_with_metric, key=lambda x: x[metric])
    
    def clear_history(self):
        """Clear training history and trained models"""
        self.trained_models.clear()
        self.training_history.clear()

# Utility functions
def get_neural_network_models_info() -> Dict:
    """Get information about all available neural network models"""
    registry = NeuralNetworkRegistry()
    return registry.get_all_models()

def create_model_comparison_report(training_results: List[Dict]) -> pd.DataFrame:
    """Create a comparison report from training results"""
    if not training_results:
        return pd.DataFrame()
    
    comparison_data = []
    for result in training_results:
        if result['training_successful']:
            row = {
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Category': result.get('model_category', 'Unknown'),
                'Final_Loss': result.get('final_loss', 'N/A'),
                'Final_Accuracy': result.get('final_accuracy', 'N/A'),
                'Training_Accuracy': result.get('training_accuracy', 'N/A'),
                'Converged': result.get('converged', 'N/A'),
                'Hyperparameters': str(result.get('hyperparameters', {}))
            }
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    registry = NeuralNetworkRegistry()
    trainer = NeuralNetworkTrainer()
    
    print("Available Feedforward Models:")
    for name, info in registry.get_feedforward_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Convolutional Models:")
    for name, info in registry.get_convolutional_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Recurrent Models:")
    for name, info in registry.get_recurrent_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Generative Models:")
    for name, info in registry.get_generative_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")
    
    print("\nAvailable Specialized Models:")
    for name, info in registry.get_specialized_models().items():
        print(f"- {info['name']} ({info['type']}): {info['description']}")