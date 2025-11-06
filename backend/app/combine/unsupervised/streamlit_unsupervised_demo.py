"""
Streamlit Demo for Unsupervised Learning Models
ML Model Comparison Toolkit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA as SklearnPCA
import warnings
warnings.filterwarnings('ignore')

# Import our unsupervised module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unsupervised import UnsupervisedModelRegistry, UnsupervisedModelTrainer, create_model_comparison_report

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Toolkit - Unsupervised Learning",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_available_datasets():
    """Get available datasets from the data directory"""
    # Try different possible paths for the data directory
    possible_paths = [
        Path("data"),
        Path("backend/app/data"),
        Path("app/data"),
        Path("../data"),
        Path("./data")
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            break
    
    datasets = {
        "Classification": {},
        "Regression": {}
    }
    
    if data_dir is None:
        return datasets
    
    # Check raw datasets (we'll use these for unsupervised learning)
    for task_type in ["classification", "regression"]:
        task_dir = data_dir / "raw" / task_type
        if task_dir.exists():
            for file_path in task_dir.glob("*.csv"):
                datasets[task_type.title()][file_path.stem] = {
                    "path": str(file_path),
                    "type": "raw"
                }
            for file_path in task_dir.glob("*.xls*"):
                datasets[task_type.title()][file_path.stem] = {
                    "path": str(file_path),
                    "type": "raw"
                }
    
    return datasets

def load_dataset(dataset_info):
    """Load dataset from file for unsupervised learning (no target needed)"""
    file_path = dataset_info["path"]
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Handle missing values
        df = df.dropna()
        
        # Convert categorical features to numeric
        categorical_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_columns.append(col)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Convert to numpy array
        X = df.values.astype(float)
        feature_names = df.columns.tolist()
        
        # Add info about categorical columns
        info = {
            'categorical_columns': categorical_columns,
            'original_shape': df.shape,
            'processed_shape': X.shape
        }
        
        return X, feature_names, df, info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, [], pd.DataFrame(), {}

def generate_synthetic_data(dataset_name):
    """Generate synthetic datasets for unsupervised learning"""
    np.random.seed(42)
    
    if dataset_name == "Blob Clusters":
        X, y = make_blobs(n_samples=300, centers=4, n_features=2, 
                         random_state=42, cluster_std=1.5)
        feature_names = ['Feature_1', 'Feature_2']
        
    elif dataset_name == "Concentric Circles":
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
        feature_names = ['Feature_1', 'Feature_2']
        
    elif dataset_name == "Two Moons":
        X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
        feature_names = ['Feature_1', 'Feature_2']
        
    elif dataset_name == "High-Dimensional Data":
        X = np.random.randn(200, 10)
        # Add some structure
        X[:100, :3] += 3
        X[100:, 3:6] += 3
        y = np.array([0]*100 + [1]*100)
        feature_names = [f'Feature_{i+1}' for i in range(10)]
        
    elif dataset_name == "Gaussian Mixture":
        # Create 3 Gaussian clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 100)
        cluster2 = np.random.multivariate_normal([-2, -2], [[1, -0.3], [-0.3, 1]], 100)
        cluster3 = np.random.multivariate_normal([2, -2], [[1.5, 0], [0, 1.5]], 100)
        X = np.vstack([cluster1, cluster2, cluster3])
        y = np.array([0]*100 + [1]*100 + [2]*100)
        feature_names = ['Feature_1', 'Feature_2']
        
    else:  # Anomaly Detection Data
        # Normal data + outliers
        X_normal = np.random.randn(200, 3)
        X_outliers = np.random.randn(20, 3) * 3 + 5
        X = np.vstack([X_normal, X_outliers])
        y = np.array([0]*200 + [1]*20)  # 1 for outliers
        feature_names = ['Feature_1', 'Feature_2', 'Feature_3']
    
    return X, feature_names, y

def get_working_models():
    """Get only the models that are known to work based on test results"""
    working_models = {
        'clustering': [
            'kmeans',
            'dbscan',
            'gaussian_mixture',
            'mean_shift',
            'agglomerative',
            'spectral_clustering',
            'birch',
            'optics'
        ],
        'dimensionality_reduction': [
            'pca',
            'ica',
            'tsne'
        ],
        'anomaly_detection': [
            'isolation_forest',
            'one_class_svm'
        ]
    }
    return working_models

def plot_clustering_results(X, results, feature_names):
    """Plot clustering results"""
    if X.shape[1] < 2:
        st.warning("Cannot plot clustering results for 1D data")
        return None
    
    # Use first 2 dimensions for plotting
    X_plot = X[:, :2]
    
    n_models = len([r for r in results if r['training_successful']])
    if n_models == 0:
        return None
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for result in results:
        if result['training_successful'] and 'labels' in result:
            ax = axes[plot_idx]
            labels = result['labels']
            
            # Handle different label types
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise
            
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, 
                               cmap='tab10', alpha=0.7, s=50)
            ax.set_title(f"{result['model_display_name']}\n{n_clusters} clusters")
            ax.set_xlabel(feature_names[0] if len(feature_names) > 0 else 'Feature 1')
            ax.set_ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2')
            
            # Add cluster centers if available
            if 'cluster_centers' in result and result['cluster_centers'] is not None:
                centers = result['cluster_centers']
                if centers.shape[1] >= 2:
                    ax.scatter(centers[:, 0], centers[:, 1], 
                             marker='x', s=200, linewidths=3, color='red')
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_dimensionality_reduction_results(X, results, feature_names):
    """Plot dimensionality reduction results"""
    n_models = len([r for r in results if r['training_successful']])
    if n_models == 0:
        return None
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for result in results:
        if result['training_successful']:
            ax = axes[plot_idx]
            model = result['model_instance']
            
            try:
                # Transform the data
                if hasattr(model, 'transform'):
                    X_transformed = model.transform(X)
                elif hasattr(model, 'fit_transform'):
                    X_transformed = model.fit_transform(X)
                else:
                    continue
                
                # Plot first 2 components
                if X_transformed.shape[1] >= 2:
                    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7, s=50)
                    ax.set_title(f"{result['model_display_name']}")
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Component 2')
                else:
                    # 1D plot
                    ax.hist(X_transformed[:, 0], bins=30, alpha=0.7)
                    ax.set_title(f"{result['model_display_name']}")
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Frequency')
                
                plot_idx += 1
                
            except Exception as e:
                st.warning(f"Could not plot {result['model_display_name']}: {str(e)}")
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 ML Model Comparison Toolkit</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Unsupervised Learning Models</h2>', unsafe_allow_html=True)
    
    # Initialize registry
    registry = UnsupervisedModelRegistry()
    working_models = get_working_models()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Task type selection
    task_type = st.sidebar.selectbox(
        "Select Task Type",
        ["Clustering", "Dimensionality Reduction", "Anomaly Detection"]
    )
    
    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Dataset Source",
        ["Synthetic Data", "Your Datasets"]
    )
    
    # Dataset selection
    if dataset_source == "Your Datasets":
        available_datasets = get_available_datasets()
        all_datasets = {}
        all_datasets.update(available_datasets.get("Classification", {}))
        all_datasets.update(available_datasets.get("Regression", {}))
        
        if all_datasets:
            dataset_name = st.sidebar.selectbox(
                "Select Dataset", 
                list(all_datasets.keys())
            )
            dataset_info = all_datasets[dataset_name]
        else:
            st.sidebar.warning("No datasets found in data directory")
            dataset_source = "Synthetic Data"
    
    if dataset_source == "Synthetic Data":
        if task_type == "Clustering":
            dataset_options = ["Blob Clusters", "Concentric Circles", "Two Moons", "Gaussian Mixture"]
        elif task_type == "Dimensionality Reduction":
            dataset_options = ["High-Dimensional Data", "Blob Clusters", "Gaussian Mixture"]
        else:  # Anomaly Detection
            dataset_options = ["Anomaly Detection Data", "High-Dimensional Data"]
        
        dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    # Model selection - only show working models
    task_key = task_type.lower().replace(' ', '_')
    if task_key == 'clustering':
        available_models = registry.get_clustering_models()
        working_model_keys = working_models['clustering']
    elif task_key == 'dimensionality_reduction':
        available_models = registry.get_dimensionality_reduction_models()
        working_model_keys = working_models['dimensionality_reduction']
    else:  # anomaly_detection
        available_models = registry.get_anomaly_detection_models()
        working_model_keys = working_models['anomaly_detection']
    
    # Filter to only working models
    filtered_models = {k: v for k, v in available_models.items() if k in working_model_keys}
    model_names = list(filtered_models.keys())
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_names,
        default=model_names[:3] if len(model_names) >= 3 else model_names
    )
    
    # Data preprocessing options
    st.sidebar.subheader("Preprocessing")
    scale_features = st.sidebar.checkbox("Scale Features", value=True)
    
    # Hyperparameter options for clustering
    if task_type == "Clustering":
        st.sidebar.subheader("Clustering Parameters")
        n_clusters = st.sidebar.slider("Number of Clusters (for applicable models)", 2, 10, 4)
    
    # Main content
    if st.sidebar.button("Run Analysis", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to analyze.")
            return
        
        # Load data
        with st.spinner("Loading dataset..."):
            if dataset_source == "Your Datasets" and 'dataset_info' in locals():
                X, feature_names, df, info = load_dataset(dataset_info)
                if X is None:
                    st.error("Failed to load dataset.")
                    return
                
                # Show preprocessing info
                if info.get('categorical_columns'):
                    st.info(f"Converted categorical columns to numeric: {', '.join(info['categorical_columns'])}")
            else:
                X, feature_names, y_true = generate_synthetic_data(dataset_name)
                df = pd.DataFrame(X, columns=feature_names)
                info = {}
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", X.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Task Type", task_type)
        with col4:
            if dataset_source == "Synthetic Data" and 'y_true' in locals():
                st.metric("True Clusters", len(np.unique(y_true)))
            else:
                st.metric("Data Source", "Real Dataset")
        
        # Show dataset preview
        with st.expander("Dataset Preview"):
            preview_df = df.head(10)
            st.dataframe(preview_df, use_container_width=True)
            
            # Basic statistics
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Data preprocessing
        if scale_features:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
        else:
            X_processed = X.copy()
        
        # Train models
        st.subheader("Model Analysis Results")
        
        trainer = UnsupervisedModelTrainer()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(selected_models))
            
            try:
                # Set hyperparameters for clustering models
                hyperparams = {}
                if task_type == "Clustering" and model_name in ['kmeans', 'hierarchical', 'agglomerative', 'spectral_clustering', 'birch']:
                    hyperparams['n_clusters'] = n_clusters
                elif task_type == "Clustering" and model_name == 'gaussian_mixture':
                    hyperparams['n_components'] = n_clusters
                
                result = trainer.train_model(
                    model_name, X_processed, task_key, hyperparams
                )
                results.append(result)
            except Exception as e:
                st.warning(f"Failed to train {model_name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        successful_results = [r for r in results if r['training_successful']]
        failed_results = [r for r in results if not r['training_successful']]
        
        if successful_results:
            # Create comparison report
            comparison_df = create_model_comparison_report(successful_results)
            
            # Display results table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Plot results
            if task_type == "Clustering":
                st.subheader("Clustering Visualization")
                fig = plot_clustering_results(X_processed, successful_results, feature_names)
                if fig:
                    st.pyplot(fig)
            
            elif task_type == "Dimensionality Reduction":
                st.subheader("Dimensionality Reduction Visualization")
                fig = plot_dimensionality_reduction_results(X_processed, successful_results, feature_names)
                if fig:
                    st.pyplot(fig)
            
            # Detailed results
            with st.expander("Detailed Results"):
                for result in successful_results:
                    st.write(f"**{result['model_display_name']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'n_clusters_found' in result:
                            st.write(f"- Clusters Found: {result['n_clusters_found']}")
                        if 'inertia' in result and result['inertia'] != 'N/A':
                            st.write(f"- Inertia: {result['inertia']:.3f}")
                        if 'explained_variance_ratio' in result:
                            var_ratio = result['explained_variance_ratio']
                            if hasattr(var_ratio, '__len__'):
                                total_var = np.sum(var_ratio)
                                st.write(f"- Explained Variance: {total_var:.3f}")
                    with col2:
                        st.write(f"- Model Type: {result['model_type']}")
                        st.write(f"- Category: {result['model_category']}")
                    st.write("---")
        
        # Failed models
        if failed_results:
            st.subheader("⚠️ Failed Models")
            for result in failed_results:
                st.error(f"**{result['model_display_name']}**: {result['error']}")
    
    # Model Information
    st.sidebar.markdown("---")
    if st.sidebar.button("Show Model Information"):
        st.subheader("Available Models")
        
        if task_type == "Clustering":
            models_info = {k: v for k, v in registry.get_clustering_models().items() 
                          if k in working_models['clustering']}
        elif task_type == "Dimensionality Reduction":
            models_info = {k: v for k, v in registry.get_dimensionality_reduction_models().items() 
                          if k in working_models['dimensionality_reduction']}
        else:  # Anomaly Detection
            models_info = {k: v for k, v in registry.get_anomaly_detection_models().items() 
                          if k in working_models['anomaly_detection']}
        
        for model_name, info in models_info.items():
            status = "✅ Working"
            with st.expander(f"{info['name']} ({info['type']}) - {status}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Type:** {info['type']}")
                st.write("**Hyperparameters:**")
                for param, values in info['hyperparameters'].items():
                    st.write(f"- {param}: {values}")
    
    # Dataset Information
    if st.sidebar.button("Show Available Datasets"):
        st.subheader("Available Datasets")
        
        datasets = get_available_datasets()
        
        for task, task_datasets in datasets.items():
            if task_datasets:
                st.write(f"**{task} Datasets:**")
                for name, info in task_datasets.items():
                    st.write(f"- {name} ({info['type']})")
            else:
                st.write(f"**{task} Datasets:** None found")
        
        st.info("💡 Add your datasets to the `data/raw/classification/` or `data/raw/regression/` directories")

if __name__ == "__main__":
    main()