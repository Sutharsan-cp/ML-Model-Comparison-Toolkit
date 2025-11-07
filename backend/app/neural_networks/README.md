# Neural Networks Module

A comprehensive neural networks module for the ML Comparison Toolkit that provides access to 17 different neural network architectures organized into 5 categories.

## 🧠 Available Models

### Basic Neural Networks (4 models)
- **Perceptron**: Single-layer linear classifier, the simplest form of neural network
- **Multi-Layer Perceptron (MLP)**: Feedforward network with multiple hidden layers
- **Artificial Neural Network (ANN)**: General feedforward network with customizable architecture
- **Backpropagation Network**: Network trained using backpropagation algorithm

### Deep Learning Networks (7 models)
- **Convolutional Neural Network (CNN)**: Specialized for image processing with convolutional layers
- **ResNet**: Deep network with residual connections to avoid vanishing gradients
- **VGG**: Deep network with small convolutional filters
- **Inception**: Network with multi-scale convolutions in parallel
- **DenseNet**: Network where each layer connects to every other layer
- **EfficientNet**: Optimized network balancing depth, width, and resolution
- **MobileNet**: Lightweight network optimized for mobile devices

### Recurrent Networks (3 models)
- **RNN**: Basic recurrent network for sequence data
- **LSTM**: Long Short-Term Memory network for long sequences
- **GRU**: Gated Recurrent Unit, simpler alternative to LSTM

### Generative Networks (2 models)
- **Autoencoder**: Network for dimensionality reduction and reconstruction
- **GAN**: Generative Adversarial Network for generating synthetic data

### Specialized Networks (1 model)
- **Transformer**: Attention-based network for sequence-to-sequence tasks

## 🚀 Quick Start

### Basic Usage

```python
from neural_networks import NeuralNetworksRegistry, NeuralNetworksTrainer
from sklearn.datasets import make_classification

# Initialize registry and trainer
registry = NeuralNetworksRegistry()
trainer = NeuralNetworksTrainer()

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)

# Train a model
result = trainer.train_model(
    'perceptron', X_train, y_train, X_test, y_test,
    category='basic_nn',
    hyperparameters={'learningRate': 0.01, 'maxEpochs': 100}
)

print(f"Model: {result['model_display_name']}")
print(f"Accuracy: {result['test_accuracy']:.4f}")
```

### Training Multiple Models

```python
# Train multiple basic neural network models
basic_models = ['perceptron', 'mlp', 'ann']
results = trainer.train_multiple_models(
    basic_models, X_train, y_train, X_test, y_test,
    category='basic_nn'
)

# Get training summary
summary = trainer.get_training_summary()
print(summary)

# Get best model
best_model = trainer.get_best_model('test_accuracy')
print(f"Best model: {best_model['model_display_name']}")
```

### Training by Category

```python
# Train all basic neural network models
results = trainer.train_category_models(
    'basic_nn', X_train, y_train, X_test, y_test
)

# Train all deep learning models (for image data)
# Note: Deep learning models expect image-like data
X_image = np.random.randn(100, 28, 28)  # 100 samples, 28x28 images
y_image = np.random.randint(0, 10, 100)
X_train_img, X_test_img, y_train_img, y_test_img = trainer.prepare_data(
    X_image, y_image, scale_features=False
)

deep_results = trainer.train_category_models(
    'deep_learning', X_train_img, y_train_img, X_test_img, y_test_img
)
```

## 📊 Data Types and Tasks

### Supported Data Types
- **Tabular**: Standard structured data (Basic NN, some Generative)
- **Image**: 2D/3D image data (Deep Learning, CNN-based models)
- **Sequence**: Sequential data like time series or text (Recurrent, Transformer)
- **Text**: Natural language data (Recurrent, Transformer)
- **Time Series**: Temporal data (Recurrent models)

### Supported Task Types
- **Classification**: Predicting discrete classes
- **Regression**: Predicting continuous values
- **Unsupervised**: Learning patterns without labels (Autoencoder)
- **Generative**: Creating new data samples (GAN, Autoencoder)
- **Dimensionality Reduction**: Reducing feature space (Autoencoder)

## 🎛️ Hyperparameters

### Common Hyperparameters
- `learningRate`: Learning rate for optimization (0.001, 0.01, 0.1)
- `maxEpochs`: Maximum training epochs (50, 100, 200)
- `batchSize`: Batch size for training (16, 32, 64)
- `randomState`: Random seed for reproducibility

### Model-Specific Hyperparameters

#### Basic Neural Networks
```python
# MLP/ANN hyperparameters
hyperparameters = {
    'hiddenLayers': [64, 32],  # Hidden layer sizes
    'activation': 'relu',      # Activation function
    'learningRate': 0.01,
    'maxEpochs': 100,
    'batchSize': 32
}
```

#### Deep Learning Networks
```python
# CNN hyperparameters
hyperparameters = {
    'learningRate': 0.001,
    'maxEpochs': 50,
    'batchSize': 32
}

# ResNet hyperparameters
hyperparameters = {
    'layers': [2, 2, 2, 2],    # Layers per block
    'learningRate': 0.001,
    'maxEpochs': 50
}
```

#### Recurrent Networks
```python
# LSTM/GRU hyperparameters
hyperparameters = {
    'hiddenDim': 128,          # Hidden dimension size
    'learningRate': 0.01,
    'maxEpochs': 100,
    'batchSize': 32
}
```

#### Generative Networks
```python
# Autoencoder hyperparameters
hyperparameters = {
    'encodingDim': 32,         # Encoding dimension
    'hiddenLayers': [64, 32],  # Hidden layers
    'activation': 'relu',
    'learningRate': 0.001
}

# GAN hyperparameters
hyperparameters = {
    'latentDim': 100,          # Latent space dimension
    'generatorLayers': [128, 256, 512],
    'discriminatorLayers': [512, 256, 128],
    'learningRate': 0.0002
}
```

## 🎯 Model Selection Guide

### For Tabular Data
- **Small datasets**: Perceptron, MLP
- **Medium datasets**: ANN, Backpropagation Network
- **Feature learning**: Autoencoder

### For Image Data
- **Simple images**: CNN
- **Complex images**: ResNet, VGG, Inception
- **Mobile deployment**: MobileNet, EfficientNet
- **Dense connections**: DenseNet

### For Sequential Data
- **Short sequences**: RNN
- **Long sequences**: LSTM
- **Efficient processing**: GRU
- **Attention-based**: Transformer

### For Generative Tasks
- **Dimensionality reduction**: Autoencoder
- **Data generation**: GAN
- **Representation learning**: Autoencoder

## 📈 Performance Metrics

The module automatically calculates appropriate metrics based on task type:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed classification results

### Regression Metrics
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Square root of MSE

### Unsupervised Metrics
- **Reconstruction Loss**: For autoencoders
- **Generator/Discriminator Loss**: For GANs

## 🖥️ Streamlit Demo

Launch the interactive demo:

```bash
streamlit run backend/app/neural_networks/streamlit_neural_networks_demo.py --server.port 8503
```

### Demo Features
- **Dataset Selection**: Built-in datasets or custom CSV upload
- **Model Comparison**: Train and compare multiple models
- **Interactive Visualizations**: Performance charts and model comparisons
- **Real-time Training**: Watch models train with progress indicators
- **Detailed Results**: Comprehensive training reports and metrics

### Demo Datasets
- **Classification (Synthetic)**: Multi-class synthetic dataset
- **Regression (Synthetic)**: Continuous target synthetic dataset
- **Wine Dataset**: Classic wine classification dataset
- **Digits Dataset**: Handwritten digits recognition
- **Custom Upload**: Upload your own CSV files

## 🧪 Testing

Run the comprehensive test suite:

```bash
python backend/app/neural_networks/test_neural_networks.py
```

### Test Coverage
- **Registry Tests**: Model registration and retrieval
- **Trainer Tests**: Model training and evaluation
- **Integration Tests**: End-to-end workflows
- **Error Handling**: Graceful failure handling
- **Data Preparation**: Data preprocessing and splitting

## 🔧 Advanced Usage

### Custom Model Integration

```python
# Add a custom neural network model
class CustomNN:
    def __init__(self, **kwargs):
        # Initialize your model
        pass
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return predictions

# Register with the system
registry = NeuralNetworksRegistry()
# Add to appropriate category dictionary
```

### Batch Processing

```python
# Process multiple datasets
datasets = [dataset1, dataset2, dataset3]
all_results = []

for X, y in datasets:
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    results = trainer.train_multiple_models(
        ['perceptron', 'mlp'], X_train, y_train, X_test, y_test
    )
    all_results.extend(results)

# Analyze results across datasets
comparison_report = create_model_comparison_report(all_results)
```

### Model Persistence

```python
# Save trained model
import pickle

result = trainer.train_model('mlp', X_train, y_train, X_test, y_test)
if result['training_successful']:
    with open('trained_mlp.pkl', 'wb') as f:
        pickle.dump(result['model_instance'], f)

# Load and use model
with open('trained_mlp.pkl', 'rb') as f:
    model = pickle.load(f)
    predictions = model.predict(new_data)
```

## 📚 Architecture Details

### Model Implementation
- All models use the actual implementations from `backend/app/models/neural networks/`
- Consistent interface with `fit()` and `predict()` methods
- Proper error handling and validation
- Standardized hyperparameter naming

### Registry System
- Automatic model discovery and registration
- Category-based organization
- Metadata storage (data types, task types, hyperparameters)
- Flexible model retrieval

### Training Pipeline
- Automatic data preprocessing (scaling, encoding)
- Consistent train/test splitting
- Comprehensive error handling
- Detailed result tracking

## 🤝 Contributing

To add new neural network models:

1. Implement the model in `backend/app/models/neural networks/`
2. Follow the standard interface (`fit`, `predict` methods)
3. Add import and configuration in `neural_networks.py`
4. Update tests and documentation
5. Test with the demo application

## 📄 License

This module is part of the ML Comparison Toolkit and follows the same license terms.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all model files are in the correct directory
2. **Memory Issues**: Reduce batch size or dataset size for large models
3. **Convergence Issues**: Adjust learning rate or increase epochs
4. **Data Shape Issues**: Check input data dimensions for each model type

### Performance Tips

1. **Use appropriate data scaling** for tabular data
2. **Reduce epochs for quick testing** during development
3. **Use smaller batch sizes** for limited memory
4. **Choose models appropriate** for your data type and size

For more help, check the test files for usage examples or run the Streamlit demo for interactive exploration.