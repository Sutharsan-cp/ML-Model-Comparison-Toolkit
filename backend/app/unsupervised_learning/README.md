# Unsupervised Learning Module

A comprehensive unsupervised learning module for the ML Comparison Toolkit that provides access to 11 different unsupervised learning algorithms organized into 4 categories.

## 🔍 Available Models

### Clustering Algorithms (7 models)
- **K-Means**: Partitional clustering algorithm using centroids
- **DBSCAN**: Density-based clustering that can find arbitrary shaped clusters
- **Gaussian Mixture Model**: Probabilistic clustering using Gaussian distributions
- **Hierarchical Clustering**: Tree-based hierarchical clustering
- **Agglomerative Clustering**: Bottom-up hierarchical clustering
- **Spectral Clustering**: Graph-based clustering using eigenvalues
- **Mean Shift**: Mode-seeking clustering algorithm

### Dimensionality Reduction (2 models)
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **ICA**: Independent Component Analysis for blind source separation

### Manifold Learning (1 model)
- **t-SNE**: Non-linear dimensionality reduction for visualization

### Anomaly Detection (1 model)
- **Isolation Forest**: Ensemble method for anomaly detection

## 🚀 Quick Start

### Basic Usage

```python
from unsupervised_learning import UnsupervisedLearningRegistry, UnsupervisedLearningTrainer
from sklearn.datasets import make_blobs

# Initialize registry and trainer
registry = UnsupervisedLearningRegistry()
trainer = UnsupervisedLearningTrainer()

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, n_features=4, random_state=42)

# Prepare data
X_train, X_test = trainer.prepare_data(X, test_size=0.2)

# Train a clustering model
result = trainer.train_model(
    'kmeans', X_train, X_test,
    category='clustering',
    hyperparameters={'n_clusters': 4}
)

print(f"Model: {result['model_display_name']}")
print(f"Silhouette Score: {result['silhouette_score']:.4f}")
print(f"Number of Clusters: {result['n_clusters']}")
```

### Training Multiple Models

```python
# Train multiple clustering models
clustering_models = ['kmeans', 'dbscan', 'gaussian_mixture']
results = trainer.train_multiple_models(
    clustering_models, X_train, X_test,
    category='clustering'
)

# Get training summary
summary = trainer.get_training_summary()
print(summary)

# Get best model
best_model = trainer.get_best_model('silhouette_score')
print(f"Best model: {best_model['model_display_name']}")
```

### Training by Category

```python
# Train all clustering models
clustering_results = trainer.train_category_models(
    'clustering', X_train, X_test
)

# Train all dimensionality reduction models
dr_results = trainer.train_category_models(
    'dimensionality_reduction', X_train, X_test
)
```

## 📊 Supported Tasks and Metrics

### Task Types
- **Clustering**: Grouping similar data points together
- **Dimensionality Reduction**: Reducing the number of features while preserving information
- **Feature Extraction**: Extracting meaningful features from data
- **Visualization**: Creating 2D/3D representations of high-dimensional data
- **Anomaly Detection**: Identifying outliers and anomalous data points
- **Density Estimation**: Estimating probability distributions
- **Source Separation**: Separating mixed signals into components

### Evaluation Metrics

#### Clustering Metrics
- **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters
- **Number of Clusters**: Detected number of clusters
- **Inertia**: Within-cluster sum of squares (for K-means)
- **Noise Points**: Number of points classified as noise (for DBSCAN)

#### Dimensionality Reduction Metrics
- **Explained Variance Ratio**: Proportion of variance explained by components
- **Reconstruction Error**: Error when reconstructing original data
- **Output Dimensions**: Number of dimensions in reduced space

#### Anomaly Detection Metrics
- **Outlier Fraction**: Proportion of data points classified as outliers
- **Anomaly Scores**: Numerical scores indicating anomaly likelihood

## 🎛️ Hyperparameters

### Clustering Models

#### K-Means
```python
hyperparameters = {
    'n_clusters': 3,        # Number of clusters
    'max_iters': 300,       # Maximum iterations
    'tol': 1e-4,           # Convergence tolerance
    'init': 'k-means++',   # Initialization method
    'random_state': 42     # Random seed
}
```

#### DBSCAN
```python
hyperparameters = {
    'eps': 0.5,            # Maximum distance between samples
    'min_samples': 5,      # Minimum samples in neighborhood
    'metric': 'euclidean', # Distance metric
    'algorithm': 'auto'    # Algorithm for nearest neighbors
}
```

#### Gaussian Mixture Model
```python
hyperparameters = {
    'n_components': 3,     # Number of mixture components
    'max_iters': 100,      # Maximum EM iterations
    'tol': 1e-3,          # Convergence tolerance
    'init_params': 'kmeans', # Initialization method
    'random_state': 42     # Random seed
}
```

### Dimensionality Reduction Models

#### PCA
```python
hyperparameters = {
    'n_components': 2,     # Number of components to keep
    'svd_solver': 'auto',  # SVD solver algorithm
    'random_state': 42     # Random seed
}
```

#### ICA
```python
hyperparameters = {
    'n_components': 2,     # Number of components
    'algorithm': 'parallel', # FastICA algorithm
    'fun': 'logcosh',      # Contrast function
    'max_iter': 200,       # Maximum iterations
    'tol': 1e-4,          # Convergence tolerance
    'random_state': 42     # Random seed
}
```

### Manifold Learning Models

#### t-SNE
```python
hyperparameters = {
    'n_components': 2,     # Number of dimensions for embedding
    'perplexity': 30.0,    # Perplexity parameter
    'early_exaggeration': 12.0, # Early exaggeration factor
    'learning_rate': 200.0, # Learning rate
    'n_iter': 1000,        # Maximum iterations
    'random_state': 42     # Random seed
}
```

### Anomaly Detection Models

#### Isolation Forest
```python
hyperparameters = {
    'n_estimators': 100,   # Number of trees
    'max_samples': 'auto', # Samples per tree
    'contamination': 0.1,  # Expected outlier fraction
    'max_features': 1.0,   # Features per tree
    'random_state': 42     # Random seed
}
```

## 🎯 Model Selection Guide

### For Clustering Tasks
- **Spherical clusters**: K-Means, Gaussian Mixture Model
- **Arbitrary shapes**: DBSCAN, Spectral Clustering
- **Hierarchical structure**: Hierarchical/Agglomerative Clustering
- **Unknown cluster count**: DBSCAN, Mean Shift
- **Probabilistic clustering**: Gaussian Mixture Model

### For Dimensionality Reduction
- **Linear reduction**: PCA
- **Source separation**: ICA
- **Visualization**: t-SNE, PCA
- **Feature extraction**: PCA, ICA

### For Anomaly Detection
- **General outlier detection**: Isolation Forest
- **High-dimensional data**: Isolation Forest

## 📈 Performance Evaluation

### Clustering Evaluation
```python
# Silhouette analysis
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, labels)

# If true labels are available
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)
```

### Dimensionality Reduction Evaluation
```python
# Explained variance (for PCA)
explained_variance = model.explained_variance_ratio_.sum()

# Reconstruction error
X_reconstructed = model.inverse_transform(model.transform(X))
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
```

### Anomaly Detection Evaluation
```python
# Outlier detection
outliers = model.predict(X)
outlier_fraction = np.mean(outliers == -1)

# Anomaly scores
scores = model.decision_function(X)
```

## 🖥️ Streamlit Demo

Launch the interactive demo:

```bash
streamlit run backend/app/unsupervised_learning/streamlit_unsupervised_demo.py --server.port 8505
```

### Demo Features
- **Dataset Selection**: Built-in datasets (Blobs, Iris, Wine, etc.) or custom CSV upload
- **Model Comparison**: Train and compare multiple models across categories
- **Interactive Visualizations**: 2D scatter plots of clustering results and dimensionality reduction
- **Real-time Training**: Watch models train with progress indicators
- **Performance Metrics**: Comprehensive evaluation metrics for each model type
- **Quick Test All**: One-click testing of all 11 models with optimal parameters

### Demo Datasets
- **Blobs**: Synthetic clustering dataset with natural clusters
- **Classification Data**: Multi-class synthetic dataset
- **Iris Dataset**: Classic flower classification dataset
- **Wine Dataset**: Wine recognition dataset
- **Digits Dataset**: Handwritten digits (subsampled for speed)
- **Custom Upload**: Upload your own CSV files

## 🧪 Testing

Run the comprehensive test suite:

```bash
python backend/app/unsupervised_learning/test_unsupervised_learning.py
```

### Test Coverage
- **Registry Tests**: Model registration and retrieval
- **Trainer Tests**: Model training and evaluation
- **Integration Tests**: End-to-end workflows
- **Error Handling**: Graceful failure handling
- **Data Preparation**: Data preprocessing and splitting
- **Evaluation Metrics**: Metric calculation validation

## 🔧 Advanced Usage

### Custom Model Integration

```python
# Add a custom clustering model
class CustomClustering:
    def __init__(self, **kwargs):
        # Initialize your model
        pass
    
    def fit(self, X):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return labels

# Register with the system
registry = UnsupervisedLearningRegistry()
# Add to appropriate category dictionary
```

### Batch Processing

```python
# Process multiple datasets
datasets = [dataset1, dataset2, dataset3]
all_results = []

for X in datasets:
    X_train, X_test = trainer.prepare_data(X)
    results = trainer.train_multiple_models(
        ['kmeans', 'dbscan'], X_train, X_test
    )
    all_results.extend(results)

# Analyze results across datasets
comparison_report = create_model_comparison_report(all_results)
```

### Model Persistence

```python
# Save trained model
import pickle

result = trainer.train_model('kmeans', X_train, X_test)
if result['training_successful']:
    with open('trained_kmeans.pkl', 'wb') as f:
        pickle.dump(result['model_instance'], f)

# Load and use model
with open('trained_kmeans.pkl', 'rb') as f:
    model = pickle.load(f)
    labels = model.predict(new_data)
```

## 📚 Algorithm Details

### Clustering Algorithms

#### K-Means
- **Pros**: Fast, simple, works well with spherical clusters
- **Cons**: Requires specifying k, sensitive to initialization
- **Best for**: Well-separated, spherical clusters

#### DBSCAN
- **Pros**: Finds arbitrary shapes, handles noise, no need to specify cluster count
- **Cons**: Sensitive to hyperparameters, struggles with varying densities
- **Best for**: Arbitrary shaped clusters with noise

#### Gaussian Mixture Model
- **Pros**: Probabilistic, soft clustering, handles overlapping clusters
- **Cons**: Assumes Gaussian distributions, can overfit
- **Best for**: Overlapping clusters, probabilistic assignments

### Dimensionality Reduction

#### PCA
- **Pros**: Linear, interpretable, preserves global structure
- **Cons**: Linear assumptions, may lose non-linear patterns
- **Best for**: Linear relationships, feature extraction

#### ICA
- **Pros**: Finds independent sources, good for signal separation
- **Cons**: Assumes non-Gaussian sources, order ambiguity
- **Best for**: Blind source separation, signal processing

### Manifold Learning

#### t-SNE
- **Pros**: Excellent for visualization, preserves local structure
- **Cons**: Computationally expensive, non-deterministic, only for visualization
- **Best for**: 2D/3D visualization of high-dimensional data

## 🤝 Contributing

To add new unsupervised learning models:

1. Implement the model in `backend/app/models/unsupervised/`
2. Follow the standard interface (`fit`, `predict`/`transform` methods)
3. Add import and configuration in `unsupervised_learning.py`
4. Update tests and documentation
5. Test with the demo application

## 📄 License

This module is part of the ML Comparison Toolkit and follows the same license terms.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all model files are in the correct directory
2. **Convergence Issues**: Adjust tolerance or increase iterations
3. **Memory Issues**: Reduce dataset size or use sampling
4. **Poor Clustering**: Try different algorithms or adjust hyperparameters

### Performance Tips

1. **Scale your data** for distance-based algorithms
2. **Choose appropriate metrics** for your data type
3. **Use dimensionality reduction** before clustering for high-dimensional data
4. **Validate results** with multiple metrics and visualizations

For more help, check the test files for usage examples or run the Streamlit demo for interactive exploration.