"""
Neural Networks Module for ML Comparison Toolkit

This module provides a comprehensive collection of neural network algorithms
organized into categories: basic neural networks, deep learning, recurrent,
generative, and specialized networks.

Available Models:
- Basic Neural Networks: Perceptron, MLP, ANN, Backpropagation (4 models)
- Deep Learning: CNN, ResNet, VGG, Inception, DenseNet, EfficientNet, MobileNet (7 models)
- Recurrent: RNN, LSTM, GRU (3 models)
- Generative: Autoencoder, GAN (2 models)
- Specialized: Transformer (1 model)

Total: 17 neural network models
"""

from .neural_networks import (
    NeuralNetworksRegistry,
    NeuralNetworksTrainer,
    get_neural_networks_info,
    create_model_comparison_report
)

__version__ = "1.0.0"
__author__ = "ML Comparison Toolkit Team"

__all__ = [
    'NeuralNetworksRegistry',
    'NeuralNetworksTrainer', 
    'get_neural_networks_info',
    'create_model_comparison_report'
]