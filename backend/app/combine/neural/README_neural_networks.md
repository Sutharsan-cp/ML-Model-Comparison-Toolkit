# Neural Networks Models Module

This module provides a comprehensive collection of neural network algorithms for the ML Model Comparison Toolkit with an interactive Streamlit dashboard.

## Features

- **17 Total Models**: 4 Feedforward + 7 Convolutional + 3 Recurrent + 2 Generative + 1 Specialized
- **Multiple Categories**: Feedforward, Convolutional, Recurrent, Generative, and Specialized architectures
- **Unified Interface**: Consistent API across all neural network types
- **Error Handling**: Graceful handling of import failures
- **Model Registry**: Centralized model management with hyperparameter grids
- **Training Pipeline**: Automated training with progress tracking
- **Interactive Dashboard**: Streamlit web interface with training visualization
- **Dataset Integration**: Supports custom datasets and synthetic data generation
- **Training History**: Real-time loss and accuracy tracking

## Available Models

### Feedforward Models (4)
- **Perceptron**: Single-layer linear classifier
- **Multi-Layer Perceptron (MLP)**: Multi-layer feedforward network
- **Artificial Neural Network (ANN)**: General neural network
- **Backpropagation Network**: Network trained with backpropagation

### Convolutional Models (7)
- **CNN**: Basic Convolutional Neural Network
- **VGG**: VGG architecture (VGG11, VGG13, VGG16, VGG19)
- **ResNet**: Residual Network with skip connections
- **Inception**: Inception architecture with multiple filter sizes
- **DenseNet**: Densely connected convolutional network
- **EfficientNet**: Efficient convolutional neural network
- **MobileNet**: Lightweight CNN for mobile devices

### Recurrent Models (3)
- **RNN**: Basic Recurrent Neural Network
- **LSTM**: Long Short-Term Memory network
- **GRU**: Gated Recurrent Unit

### Generative Models (2)
- **Autoencoder**: Neural network for dimensionality reduction and reconstruction
- **GAN**: Generative Adversarial Network for data generation

### Specialized Models (1)
- **Transformer**: Transformer architecture with self-attention

## Usage

### Basic Usage

```python
from neural_networks import NeuralNetworkRegistry, NeuralNetworkTrainer

# Initialize registry and trainer
registry = NeuralNetworkRegistry()
trainer = NeuralNetworkTrainer()

# Get available models by category
feedforward_models = registry.get_feedforward_models()
convolutional_models = registry.get_convolutional_models()

# Train a single model
result = trainer.train_model('mlp', X_train, y_train, 'feedforward', 
                            {'hiddenLayers': [64, 32], 'maxEpochs': 50})

# Train multiple models
results = trainer.train_multiple_models(
    ['perceptron', 'mlp'], X_train, y_train, 'feedforward'
)

# Train all models in a category
all_results = trainer.train_all_models(X_train, y_train, 'feedforward')
```

### Model Information

```python
# Get model by name and category
model_info = registry.get_model_by_name('mlp', 'feedforward')

# Get models suitable for a task
classification_models = registry.get_models_by_task('classification')

# Get training summary
summary_df = trainer.get_training_summary()

# Get best model
best_model = trainer.get_best_model('final_accuracy')
```

### Interactive Streamlit Dashboard

The module includes a comprehensive Streamlit dashboard for interactive neural network training and comparison.

#### Quick Start
```bash
cd backend/app
python -c "import streamlit; streamlit.run('streamlit_neural_networks_demo.py')"
```

#### Dashboard Features
- **Model Categories**: Choose between Feedforward, Convolutional, and Generative models
- **Task Types**: Classification, Regression, Image Classification, Unsupervised
- **Dataset Options**: Use synthetic data or your own datasets
- **Training Parameters**: Adjust epochs, batch size, learning rate
- **Real-time Visualization**: Training loss and accuracy plots
- **Model Comparison**: Side-by-side performance comparison
- **Training History**: Interactive plots of training progress

#### Using Your Own Datasets
1. Place CSV files in `data/raw/classification/` or `data/raw/regression/`
2. The dashboard will automatically detect and list your datasets
3. Select target column from dropdown
4. Categorical features are automatically encoded

## Testing

### Basic Model Testing
```bash
cd backend/app
python test_neural_networks.py
```

### Test Results Summary

Based on comprehensive testing with synthetic datasets:

### ✅ Working Models (6/17)

**Feedforward (4/4)**:
- ✅ Perceptron: Training accuracy: 0.564
- ✅ Multi-Layer Perceptron: Final accuracy: 0.521, Loss: 0.399
- ✅ Artificial Neural Network: Final accuracy: 0.671, Loss: 0.579
- ✅ Backpropagation Network: Final accuracy: 0.529, Loss: 0.693

**Convolutional (1/7)**:
- ✅ CNN: Final accuracy: 0.320, Loss: 1.200 (on 28x28 images)
- ⚠️ VGG, ResNet, Inception, DenseNet, EfficientNet, MobileNet: Not tested (complex architectures)

**Recurrent (0/3)**:
- ❌ RNN, LSTM, GRU: Broadcasting shape issues with sequence data

**Generative (1/2)**:
- ✅ GAN: Successfully trains generator and discriminator
- ❌ Autoencoder: Broadcasting shape issues

**Specialized (0/1)**:
- ⚠️ Transformer: Not tested (complex architecture)

### 📊 Performance Examples

**Feedforward on Binary Classification (200 samples, 10 features)**:
- Perceptron: 56.4% training accuracy (simple linear model)
- MLP: 52.1% final accuracy, converged in 10 epochs
- ANN: 67.1% final accuracy, best performer
- Backpropagation: 52.9% final accuracy

**CNN on Image Classification (100 samples, 28x28 images)**:
- CNN: 32.0% final accuracy on 3-class problem

**GAN on Synthetic Data**:
- Successfully trains both generator and discriminator networks
- Discriminator loss: 1.12, Generator loss: 2.68

## Model Structure

Each model in the registry contains:
- `class`: The model class
- `name`: Display name
- `type`: Model architecture type
- `description`: Brief description
- `task_types`: Supported task types
- `hyperparameters`: Dictionary of hyperparameter options

## Synthetic Datasets

The module includes several built-in synthetic datasets:

### Classification Datasets
- **Binary Classification**: 2-class problem with 10 features
- **Multi-class Classification**: 3-class problem with 10 features
- **Complex Classification**: 4-class problem with 20 features

### Regression Datasets
- **Linear Regression**: Linear relationship with 10 features
- **Nonlinear Regression**: Nonlinear relationships with interactions
- **Complex Regression**: High-dimensional regression with 20 features

### Image Datasets
- **Synthetic Images**: 28x28 grayscale images for CNN testing

### Unsupervised Datasets
- **High-Dimensional Data**: 20-dimensional data for autoencoders

## Training Features

The neural network trainer provides:
- **Progress Tracking**: Real-time training progress
- **Loss History**: Track loss over epochs
- **Accuracy History**: Track accuracy over epochs
- **Early Stopping**: Convergence detection for some models
- **Batch Training**: Configurable batch sizes
- **Learning Rate Control**: Adjustable learning rates

## Visualization Features

The Streamlit dashboard provides:
- **Training History Plots**: Loss and accuracy over epochs
- **Model Comparison Charts**: Side-by-side performance comparison
- **Dataset Visualization**: Sample images for CNN tasks
- **Interactive Parameters**: Real-time parameter adjustment
- **Progress Indicators**: Training progress bars

## Error Handling

The module gracefully handles:
- Import errors for individual models
- Training failures with informative error messages
- Data shape mismatches
- Invalid hyperparameter combinations

## Extending the Module

To add new neural network models:

1. Create the model class in `models/neural networks/`
2. Add the import statement with error handling
3. Add the model configuration to the appropriate registry
4. Update the working models list after testing
5. Update the model counts in this README

## Dependencies

- numpy
- pandas
- scikit-learn (for datasets and utilities)
- matplotlib, seaborn (for visualization)
- streamlit (for dashboard)
- Custom neural network implementations in `models/neural networks/`

## Performance Notes

- All models use custom implementations for educational purposes
- Training times vary significantly by model complexity
- CNN models require more computational resources
- RNN models currently have implementation issues
- For production use, consider TensorFlow/PyTorch implementations
- Memory usage scales with batch size and model complexity

## Known Issues

- **RNN Models**: Broadcasting shape issues with sequence data
- **Autoencoder**: Shape mismatch in encoding/decoding layers
- **Complex Architectures**: VGG, ResNet, etc. not tested due to complexity
- **Transformer**: May require specific input formatting

## Future Improvements

- Fix RNN broadcasting issues
- Add more robust error handling for complex architectures
- Implement proper sequence data preprocessing
- Add support for custom loss functions
- Include model checkpointing and resuming
- Add hyperparameter optimization