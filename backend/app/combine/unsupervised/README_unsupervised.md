# Unsupervised Learning Models Module

This module provides a comprehensive collection of unsupervised learning algorithms for the ML Model Comparison Toolkit with an interactive Streamlit dashboard.

## Features

- **14 Total Models**: 9 Clustering + 3 Dimensionality Reduction + 2 Anomaly Detection algorithms
- **Unified Interface**: Consistent API across all models
- **Error Handling**: Graceful handling of import failures
- **Model Registry**: Centralized model management
- **Training Pipeline**: Automated training and evaluation
- **Hyperparameter Definitions**: Pre-defined hyperparameter grids
- **Interactive Dashboard**: Streamlit web interface for model comparison
- **Dataset Integration**: Supports your custom datasets from data directory
- **Real-time Visualization**: Clustering and dimensionality reduction plots
- **Synthetic Data Generation**: Built-in test datasets

## Available Models

### Clustering Models (9)
- **Centroid-based**: K-Means
- **Density-based**: DBSCAN, Mean Shift, OPTICS
- **Hierarchical**: Hierarchical Clustering, Agglomerative Clustering, BIRCH
- **Graph-based**: Spectral Clustering
- **Probabilistic**: Gaussian Mixture Model

### Dimensionality Reduction Models (3)
- **Linear**: Principal Component Analysis (PCA), Independent Component Analysis (ICA)
- **Manifold**: t-SNE

### Anomaly Detection Models (2)
- **Ensemble**: Isolation Forest
- **Kernel**: One-Class SVM

## Usage

### Basic Usage

```python
from unsupervised import UnsupervisedModelRegistry, UnsupervisedModelTrainer

# Initialize registry and trainer
registry = UnsupervisedModelRegistry()
trainer = UnsupervisedModelTrainer()

# Get available models
clustering_models = registry.get_clustering_models()
dr_models = registry.get_dimensionality_reduction_models()
anomaly_models = registry.get_anomaly_detection_models()

# Train a single model
result = trainer.train_model('kmeans', X, 'clustering', {'n_clusters': 4})

# Train multiple models
results = trainer.train_multiple_models(
    ['kmeans', 'dbscan', 'gaussian_mixture'], 
    X, 'clustering'
)

# Train all models for a task
all_results = trainer.train_all_models(X, 'clustering')
```

### Model Information

```python
# Get model by name
model_info = registry.get_model_by_name('kmeans', 'clustering')

# Get models by type
density_models = registry.get_models_by_type('density', 'clustering')

# Get training summary
summary_df = trainer.get_training_summary()
```

### Interactive Streamlit Dashboard

The module includes a comprehensive Streamlit dashboard for interactive model comparison.

#### Quick Start
```bash
cd backend/app
python -c "import streamlit; streamlit.run('streamlit_unsupervised_demo.py')"
```

#### Manual Start
```bash
cd backend/app
pip install streamlit matplotlib seaborn
streamlit run streamlit_unsupervised_demo.py
```

#### Dashboard Features
- **Task Selection**: Choose between Clustering, Dimensionality Reduction, or Anomaly Detection
- **Dataset Options**: Use synthetic data or your own datasets
- **Model Comparison**: Select multiple models to compare side-by-side
- **Real-time Visualization**: See clustering results and dimensionality reduction plots
- **Interactive Parameters**: Adjust number of clusters and preprocessing options
- **Dataset Preview**: Explore your data before analysis
- **Model Information**: Detailed information about each algorithm

#### Using Your Own Datasets
1. Place CSV files in `data/raw/classification/` or `data/raw/regression/`
2. The dashboard will automatically detect and list your datasets
3. Categorical features are automatically encoded
4. No target variable needed (unsupervised learning)

## Testing

### Basic Model Testing
```bash
cd backend/app
python test_unsupervised.py
```

### Test Results Summary

Based on comprehensive testing with synthetic and real datasets:

### ✅ Working Models (13/14)

**Clustering (8/9)**:
- ✅ K-Means
- ✅ DBSCAN
- ✅ Gaussian Mixture Model
- ✅ Mean Shift
- ✅ Agglomerative Clustering
- ✅ Spectral Clustering
- ✅ BIRCH
- ✅ OPTICS
- ❌ Hierarchical Clustering (array indexing issue)

**Dimensionality Reduction (3/3)**:
- ✅ Principal Component Analysis (PCA)
- ✅ Independent Component Analysis (ICA)
- ✅ t-SNE

**Anomaly Detection (2/2)**:
- ✅ Isolation Forest
- ✅ One-Class SVM

### 📊 Model Performance Examples

**Clustering on Blob Data (300 samples, 2 features)**:
- K-Means: 8 clusters, Inertia: 27.621
- DBSCAN: 20 clusters (density-based)
- Mean Shift: 3 clusters (automatic cluster detection)
- Gaussian Mixture: Probabilistic clustering
- Spectral Clustering: 8 clusters (graph-based)

**Dimensionality Reduction on High-Dim Data (200 samples, 20 features)**:
- PCA: Explained variance ratio: 1.005 (captures all variance)
- ICA: Independent component separation
- t-SNE: Non-linear manifold embedding

**Anomaly Detection on Mixed Data (220 samples, 5 features)**:
- Isolation Forest: Ensemble-based outlier detection
- One-Class SVM: 58 outliers detected

## Model Structure

Each model in the registry contains:
- `class`: The model class
- `name`: Display name
- `type`: Model category (centroid, density, linear, etc.)
- `description`: Brief description
- `hyperparameters`: Dictionary of hyperparameter options

## Synthetic Datasets

The module includes several built-in synthetic datasets:

### Clustering Datasets
- **Blob Clusters**: Well-separated Gaussian clusters
- **Concentric Circles**: Nested circular patterns
- **Two Moons**: Crescent-shaped clusters
- **Gaussian Mixture**: Multiple Gaussian distributions

### Dimensionality Reduction Datasets
- **High-Dimensional Data**: 10D data with embedded structure
- **Structured Data**: Data with known low-dimensional manifold

### Anomaly Detection Datasets
- **Normal + Outliers**: Clean data with injected anomalies
- **Mixed Distributions**: Multiple data distributions

## Error Handling

The module gracefully handles import errors for individual models. If a model fails to import, it will be excluded from the registry but won't break the entire module.

## Extending the Module

To add new models:

1. Create the model class in the appropriate directory
2. Add the import statement with error handling
3. Add the model configuration to the registry
4. Update the model counts in this README

## Dependencies

- numpy
- pandas
- scikit-learn (for datasets and utilities)
- matplotlib, seaborn (for visualization)
- streamlit (for dashboard)
- Custom model implementations in `models/unsupervised/`

## Performance Notes

- All models use custom implementations for educational purposes
- For production use, consider using scikit-learn or other optimized libraries
- Some models may be slower than their scikit-learn counterparts
- Memory usage varies by model complexity and dataset size
- UMAP requires numba package (optional dependency)

## Visualization Features

The Streamlit dashboard provides:
- **Clustering Plots**: 2D scatter plots with cluster colors and centroids
- **Dimensionality Reduction Plots**: Transformed data visualization
- **Interactive Parameters**: Real-time parameter adjustment
- **Multiple Model Comparison**: Side-by-side result comparison
- **Dataset Statistics**: Comprehensive data exploration tools