"""
Streamlit Demo for Unsupervised Learning Module
Interactive web interface for testing and comparing unsupervised learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the unsupervised_learning module to path
sys.path.append(os.path.dirname(__file__))

try:
    from unsupervised_learning import (
        UnsupervisedLearningRegistry, 
        UnsupervisedLearningTrainer,
        get_unsupervised_learning_info,
        create_model_comparison_report
    )
except ImportError as e:
    st.error(f"Error importing unsupervised learning module: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Unsupervised Learning Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #ff7f0e;
    }
    .metric-card {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .clustering-card {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
    .dr-card {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title
    st.markdown('<h1 class="main-header">🔍 Unsupervised Learning Comparison Toolkit</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'trainer' not in st.session_state:
        st.session_state.trainer = UnsupervisedLearningTrainer()
    if 'registry' not in st.session_state:
        st.session_state.registry = UnsupervisedLearningRegistry()
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Dataset selection
        st.subheader("📊 Dataset")
        dataset_type = st.selectbox(
            "Choose dataset type:",
            ["Built-in Dataset", "Custom Dataset Upload"]
        )
        
        if dataset_type == "Built-in Dataset":
            dataset_name = st.selectbox(
                "Select dataset:",
                ["Blobs (Clustering)", "Classification Data", "Iris Dataset", "Wine Dataset", "Digits Dataset"]
            )
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        # Model selection
        st.subheader("🤖 Model Selection")
        
        # Get available models
        all_models = st.session_state.registry.get_all_models()
        
        # Category selection
        selected_categories = st.multiselect(
            "Select model categories:",
            list(all_models.keys()),
            default=['clustering']
        )
        
        # Model selection within categories
        selected_models = []
        for category in selected_categories:
            if category in all_models and all_models[category]:
                st.write(f"**{category.replace('_', ' ').title()} Models:**")
                category_models = st.multiselect(
                    f"Select {category} models:",
                    list(all_models[category].keys()),
                    key=f"models_{category}"
                )
                selected_models.extend([(model, category) for model in category_models])
        
        # Training parameters
        st.subheader("⚙️ Training Parameters")
        test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)
        
        # Category-specific parameters
        if any(cat in selected_categories for cat in ['clustering']):
            n_clusters = st.slider("Number of clusters (for clustering):", 2, 10, 3)
        
        if any(cat in selected_categories for cat in ['dimensionality_reduction', 'manifold_learning']):
            n_components = st.slider("Number of components (for DR):", 2, 10, 2)
        
        # Action buttons
        st.subheader("🚀 Actions")
        train_button = st.button("🏋️ Train Selected Models", type="primary")
        
        # Quick test all models button
        if st.button("⚡ Quick Test All Models", help="Test all 11 models with optimal parameters"):
            st.session_state.quick_test = True
        
        clear_button = st.button("🗑️ Clear Results")
        
        if clear_button:
            st.session_state.trainer.clear_history()
            st.session_state.training_results = []
            if 'quick_test' in st.session_state:
                del st.session_state.quick_test
            st.success("Results cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset preparation
        st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)
        
        X, y, dataset_info = prepare_dataset(dataset_type, dataset_name if dataset_type == "Built-in Dataset" else uploaded_file)
        
        if X is not None:
            # Display dataset info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f'<div class="metric-card"><strong>Samples:</strong> {len(X)}</div>', unsafe_allow_html=True)
            with col_info2:
                st.markdown(f'<div class="metric-card"><strong>Features:</strong> {X.shape[1] if X.ndim > 1 else 1}</div>', unsafe_allow_html=True)
            with col_info3:
                if y is not None:
                    unique_labels = len(np.unique(y))
                    st.markdown(f'<div class="metric-card"><strong>True Classes:</strong> {unique_labels}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><strong>Type:</strong> Unlabeled</div>', unsafe_allow_html=True)
            
            # Dataset preview and visualization
            if st.checkbox("Show dataset preview"):
                preview_df = pd.DataFrame(X[:10])
                if y is not None:
                    preview_df['True_Label'] = y[:10]
                st.dataframe(preview_df)
            
            # Dataset visualization (if 2D or can be reduced to 2D)
            if st.checkbox("Show dataset visualization"):
                visualize_dataset(X, y)
            
            # Quick test all models
            if st.session_state.get('quick_test', False):
                st.markdown('<h2 class="sub-header">⚡ Quick Test All Models</h2>', unsafe_allow_html=True)
                
                # Get all models
                all_models = st.session_state.registry.get_all_models()
                all_model_list = []
                for category, models in all_models.items():
                    for model_name in models.keys():
                        all_model_list.append((model_name, category))
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                X_train, X_test = st.session_state.trainer.prepare_data(X, test_size=test_size)
                
                for i, (model_name, category) in enumerate(all_model_list):
                    status_text.text(f"Quick testing {model_name} ({i+1}/{len(all_model_list)})...")
                    
                    # Set optimal hyperparameters for quick testing
                    hyperparameters = get_optimal_hyperparameters(model_name, category, X)
                    
                    try:
                        result = st.session_state.trainer.train_model(
                            model_name, X_train, X_test,
                            category=category, hyperparameters=hyperparameters,
                            verbose=False
                        )
                        results.append(result)
                        
                    except Exception as e:
                        # Create error result
                        error_result = {
                            'model_name': model_name,
                            'model_display_name': f"{model_name} (Error)",
                            'model_type': category,
                            'training_successful': False,
                            'error': str(e)
                        }
                        results.append(error_result)
                    
                    progress_bar.progress((i + 1) / len(all_model_list))
                
                status_text.text("Quick test completed!")
                st.session_state.training_results = results
                st.session_state.quick_test = False
                
                # Display results
                display_training_results(results, X_train, y)
            
            # Training section
            elif train_button and selected_models:
                st.markdown('<h2 class="sub-header">🏋️ Training Results</h2>', unsafe_allow_html=True)
                
                # Prepare data
                X_train, X_test = st.session_state.trainer.prepare_data(X, test_size=test_size)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, (model_name, category) in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    
                    # Set hyperparameters
                    hyperparameters = {}
                    model_info = st.session_state.registry.get_model_by_name(model_name, category)
                    
                    if model_info and 'hyperparameters' in model_info:
                        if 'n_clusters' in model_info['hyperparameters'] and category == 'clustering':
                            hyperparameters['n_clusters'] = n_clusters
                        if 'n_components' in model_info['hyperparameters'] and category in ['dimensionality_reduction', 'manifold_learning']:
                            hyperparameters['n_components'] = n_components
                    
                    # Add model-specific hyperparameters
                    if model_name == 'dbscan':
                        hyperparameters.update({'eps': 0.5, 'min_samples': 5})
                    elif model_name == 'tsne':
                        hyperparameters.update({'n_iter': 300, 'perplexity': min(30, len(X_train)//4)})
                    elif model_name == 'isolation_forest':
                        hyperparameters.update({'n_estimators': 100, 'contamination': 0.1})
                    
                    try:
                        result = st.session_state.trainer.train_model(
                            model_name, X_train, X_test,
                            category=category, hyperparameters=hyperparameters,
                            verbose=False
                        )
                        results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                status_text.text("Training completed!")
                st.session_state.training_results = results
                
                # Display results
                display_training_results(results, X_train, y)
        
        else:
            st.warning("Please select a dataset or upload a CSV file to begin.")
    
    with col2:
        # Model information panel
        st.markdown('<h2 class="sub-header">📋 Model Information</h2>', unsafe_allow_html=True)
        
        if selected_models:
            for model_name, category in selected_models:
                model_info = st.session_state.registry.get_model_by_name(model_name, category)
                if model_info:
                    with st.expander(f"ℹ️ {model_info['name']}"):
                        st.write(f"**Type:** {model_info['type']}")
                        st.write(f"**Description:** {model_info['description']}")
                        st.write(f"**Data Types:** {', '.join(model_info.get('data_types', []))}")
                        st.write(f"**Task Types:** {', '.join(model_info.get('task_types', []))}")
        
        # Model Status Overview
        st.markdown('<h2 class="sub-header">📊 Model Status</h2>', unsafe_allow_html=True)
        
        # Show total models available
        all_models = st.session_state.registry.get_all_models()
        total_models = sum(len(models) for models in all_models.values())
        
        st.markdown(f'<div class="metric-card"><strong>Total Available Models:</strong> {total_models}</div>', unsafe_allow_html=True)
        
        # Show models by category
        for category, models in all_models.items():
            if models:
                color_class = "clustering-card" if category == "clustering" else "dr-card" if "reduction" in category else "metric-card"
                st.markdown(f'<div class="{color_class}"><strong>{category.replace("_", " ").title()}:</strong> {len(models)} models</div>', unsafe_allow_html=True)
        
        # Training history
        if st.session_state.training_results:
            st.markdown('<h2 class="sub-header">📈 Training History</h2>', unsafe_allow_html=True)
            
            # Calculate success rate
            successful = sum(1 for r in st.session_state.training_results if r['training_successful'])
            total = len(st.session_state.training_results)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f'<div class="success-card"><strong>✅ Successful:</strong> {successful}/{total}</div>', unsafe_allow_html=True)
            with col_s2:
                st.markdown(f'<div class="metric-card"><strong>📊 Success Rate:</strong> {success_rate:.1f}%</div>', unsafe_allow_html=True)
            
            summary_df = st.session_state.trainer.get_training_summary()
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
                
                # Best model
                best_model = st.session_state.trainer.get_best_model('silhouette_score')
                if best_model:
                    silhouette = best_model.get("silhouette_score", "N/A")
                    if silhouette != "N/A" and silhouette is not None:
                        st.markdown(f'<div class="success-card"><strong>🏆 Best Model:</strong><br>{best_model["model_display_name"]}<br><strong>Silhouette Score:</strong> {silhouette:.4f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-card"><strong>🏆 Best Model:</strong><br>{best_model["model_display_name"]}<br><strong>Status:</strong> Trained Successfully</div>', unsafe_allow_html=True)

def prepare_dataset(dataset_type, dataset_selection):
    """Prepare dataset based on user selection"""
    
    if dataset_type == "Built-in Dataset":
        if dataset_selection == "Blobs (Clustering)":
            X, y = make_blobs(
                n_samples=300, centers=4, n_features=4, 
                random_state=42, cluster_std=1.5
            )
            info = "Synthetic clustering dataset with 4 natural clusters"
            
        elif dataset_selection == "Classification Data":
            X, y = make_classification(
                n_samples=500, n_features=8, n_classes=3, 
                n_redundant=0, n_informative=6, random_state=42
            )
            info = "Synthetic classification dataset (labels for reference only)"
            
        elif dataset_selection == "Iris Dataset":
            iris = load_iris()
            X, y = iris.data, iris.target
            info = "Iris flower dataset (4 features, 3 species)"
            
        elif dataset_selection == "Wine Dataset":
            wine = load_wine()
            X, y = wine.data, wine.target
            info = "Wine recognition dataset (13 features, 3 classes)"
            
        elif dataset_selection == "Digits Dataset":
            digits = load_digits()
            X, y = digits.data, digits.target
            # Subsample for faster processing
            indices = np.random.choice(len(X), 500, replace=False)
            X, y = X[indices], y[indices]
            info = "Handwritten digits dataset (64 features, 10 classes) - subsampled"
        
        return X, y, info
    
    else:  # Custom dataset upload
        if dataset_selection is not None:
            try:
                df = pd.read_csv(dataset_selection)
                
                # Let user select target column (optional for unsupervised)
                st.write("**Dataset Preview:**")
                st.dataframe(df.head())
                
                use_target = st.checkbox("Use a target column for reference?")
                
                if use_target:
                    target_column = st.selectbox("Select target column:", df.columns)
                    X = df.drop(columns=[target_column]).values
                    y = df[target_column].values
                    
                    # Handle categorical targets
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                else:
                    X = df.values
                    y = None
                
                info = f"Custom dataset: {len(df)} samples, {X.shape[1]} features"
                return X, y, info
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return None, None, None
        
        return None, None, None

def get_optimal_hyperparameters(model_name, category, X):
    """Get optimal hyperparameters for quick testing"""
    hyperparameters = {}
    
    if category == 'clustering':
        # Estimate optimal number of clusters
        n_samples = len(X)
        optimal_clusters = min(max(2, int(np.sqrt(n_samples/2))), 8)
        hyperparameters['n_clusters'] = optimal_clusters
        
        if model_name == 'dbscan':
            # Estimate eps based on data scale
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors.fit(X)
            distances, _ = neighbors.kneighbors(X)
            eps = np.percentile(distances[:, 4], 90)
            hyperparameters = {'eps': eps, 'min_samples': 5}
        elif model_name == 'mean_shift':
            hyperparameters = {'bandwidth': 'auto'}
    
    elif category in ['dimensionality_reduction', 'manifold_learning']:
        # Use 2 components for visualization
        hyperparameters['n_components'] = 2
        
        if model_name == 'tsne':
            perplexity = min(30, len(X) // 4)
            hyperparameters.update({
                'perplexity': perplexity,
                'n_iter': 300,
                'learning_rate': 200.0
            })
    
    elif category == 'anomaly_detection':
        hyperparameters = {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': 42
        }
    
    return hyperparameters

def visualize_dataset(X, y=None):
    """Visualize the dataset"""
    
    if X.shape[1] >= 2:
        # Use first two dimensions or PCA if more dimensions
        if X.shape[1] == 2:
            X_vis = X
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        
        # Create scatter plot
        if y is not None:
            fig = px.scatter(
                x=X_vis[:, 0], y=X_vis[:, 1], 
                color=y.astype(str),
                title="Dataset Visualization (True Labels)" if X.shape[1] > 2 else "Dataset Visualization",
                labels={'x': 'Feature 1' if X.shape[1] == 2 else 'PC1', 
                       'y': 'Feature 2' if X.shape[1] == 2 else 'PC2',
                       'color': 'True Label'}
            )
        else:
            fig = px.scatter(
                x=X_vis[:, 0], y=X_vis[:, 1],
                title="Dataset Visualization" if X.shape[1] == 2 else "Dataset Visualization (PCA)",
                labels={'x': 'Feature 1' if X.shape[1] == 2 else 'PC1', 
                       'y': 'Feature 2' if X.shape[1] == 2 else 'PC2'}
            )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset has only 1 feature. Showing histogram.")
        fig = px.histogram(x=X.flatten(), title="Feature Distribution")
        st.plotly_chart(fig, use_container_width=True)

def display_training_results(results, X_train, y_true=None):
    """Display training results with visualizations"""
    
    if not results:
        return
    
    # Results summary table
    st.subheader("📊 Results Summary")
    
    summary_data = []
    for result in results:
        if result['training_successful']:
            summary_data.append({
                'Model': result['model_display_name'],
                'Category': result['model_type'],
                'Silhouette Score': result.get('silhouette_score', 'N/A'),
                'N Clusters': result.get('n_clusters', 'N/A'),
                'Explained Variance': result.get('explained_variance_ratio', 'N/A'),
                'Outlier Fraction': result.get('outlier_fraction', 'N/A'),
                'Training Samples': result.get('training_samples', 'N/A'),
                'Status': '✅ Success'
            })
        else:
            summary_data.append({
                'Model': result['model_display_name'],
                'Category': result['model_type'],
                'Silhouette Score': 'N/A',
                'N Clusters': 'N/A',
                'Explained Variance': 'N/A',
                'Outlier Fraction': 'N/A',
                'Training Samples': result.get('training_samples', 'N/A'),
                'Status': '❌ Failed'
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance visualization
    successful_results = [r for r in results if r['training_successful']]
    
    if successful_results:
        st.subheader("📈 Performance Comparison")
        
        # Clustering performance
        clustering_results = [r for r in successful_results if r['model_type'] == 'clustering' and r.get('silhouette_score') is not None]
        
        if clustering_results:
            model_names = [r['model_display_name'] for r in clustering_results]
            silhouette_scores = [r.get('silhouette_score', 0) for r in clustering_results]
            
            fig = px.bar(
                x=model_names, y=silhouette_scores,
                title="Clustering Performance (Silhouette Score)",
                labels={'x': 'Model', 'y': 'Silhouette Score'},
                color=silhouette_scores,
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Dimensionality reduction performance
        dr_results = [r for r in successful_results if r['model_type'] in ['dimensionality_reduction', 'manifold_learning']]
        
        if dr_results:
            dr_names = [r['model_display_name'] for r in dr_results]
            explained_var = [r.get('explained_variance_ratio', 0) if r.get('explained_variance_ratio') is not None else 0 for r in dr_results]
            
            if any(ev > 0 for ev in explained_var):
                fig = px.bar(
                    x=dr_names, y=explained_var,
                    title="Dimensionality Reduction Performance (Explained Variance)",
                    labels={'x': 'Model', 'y': 'Explained Variance Ratio'},
                    color=explained_var,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Category performance summary
        category_stats = {}
        for result in successful_results:
            category = result['model_type']
            if category not in category_stats:
                category_stats[category] = {'count': 0, 'silhouette_scores': [], 'explained_variances': []}
            
            category_stats[category]['count'] += 1
            if result.get('silhouette_score') is not None:
                category_stats[category]['silhouette_scores'].append(result['silhouette_score'])
            if result.get('explained_variance_ratio') is not None:
                category_stats[category]['explained_variances'].append(result['explained_variance_ratio'])
        
        if len(category_stats) > 1:
            st.subheader("📊 Performance by Category")
            
            category_summary = []
            for category, stats in category_stats.items():
                row = {
                    'Category': category.replace('_', ' ').title(),
                    'Models Tested': stats['count'],
                    'Avg Silhouette': np.mean(stats['silhouette_scores']) if stats['silhouette_scores'] else 'N/A',
                    'Avg Explained Var': np.mean(stats['explained_variances']) if stats['explained_variances'] else 'N/A'
                }
                category_summary.append(row)
            
            category_df = pd.DataFrame(category_summary)
            st.dataframe(category_df, use_container_width=True)
    
    # Model visualizations
    st.subheader("🎨 Model Visualizations")
    
    # Show clustering results
    clustering_models = [r for r in successful_results if r['model_type'] == 'clustering']
    if clustering_models and X_train.shape[1] >= 2:
        
        # Prepare data for visualization
        if X_train.shape[1] == 2:
            X_vis = X_train
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_train)
        
        # Create subplots for clustering results
        n_models = len(clustering_models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[r['model_display_name'] for r in clustering_models],
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        for i, result in enumerate(clustering_models):
            row = i // cols + 1
            col = i % cols + 1
            
            try:
                model = result['model_instance']
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                elif hasattr(model, 'predict'):
                    labels = model.predict(X_train)
                else:
                    continue
                
                # Create scatter plot
                scatter = go.Scatter(
                    x=X_vis[:, 0], y=X_vis[:, 1],
                    mode='markers',
                    marker=dict(color=labels, colorscale='viridis', size=6),
                    name=result['model_display_name'],
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
            except Exception as e:
                st.warning(f"Could not visualize {result['model_display_name']}: {str(e)}")
        
        fig.update_layout(height=300*rows, title_text="Clustering Results Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Show dimensionality reduction results
    dr_models = [r for r in successful_results if r['model_type'] in ['dimensionality_reduction', 'manifold_learning']]
    if dr_models:
        
        for result in dr_models:
            try:
                model = result['model_instance']
                if hasattr(model, 'transform'):
                    X_transformed = model.transform(X_train)
                    
                    if X_transformed.shape[1] >= 2:
                        fig = px.scatter(
                            x=X_transformed[:, 0], y=X_transformed[:, 1],
                            color=y_true.astype(str) if y_true is not None else None,
                            title=f"{result['model_display_name']} - Transformed Data",
                            labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'True Label'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not visualize {result['model_display_name']}: {str(e)}")
    
    # Detailed results
    st.subheader("🔍 Detailed Results")
    
    for result in results:
        with st.expander(f"📋 {result['model_display_name']} Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model Type:** {result['model_type']}")
                st.write(f"**Category:** {result.get('model_category', 'N/A')}")
                st.write(f"**Training Status:** {'✅ Success' if result['training_successful'] else '❌ Failed'}")
                
                if result['training_successful']:
                    st.write(f"**Training Samples:** {result.get('training_samples', 'N/A')}")
                    
                    # Category-specific metrics
                    if result['model_type'] == 'clustering':
                        if result.get('silhouette_score') is not None:
                            st.write(f"**Silhouette Score:** {result['silhouette_score']:.4f}")
                        if result.get('n_clusters') is not None:
                            st.write(f"**Number of Clusters:** {result['n_clusters']}")
                        if result.get('n_noise') is not None and result['n_noise'] > 0:
                            st.write(f"**Noise Points:** {result['n_noise']}")
                    
                    elif result['model_type'] in ['dimensionality_reduction', 'manifold_learning']:
                        if result.get('output_dimensions') is not None:
                            st.write(f"**Output Dimensions:** {result['output_dimensions']}")
                        if result.get('explained_variance_ratio') is not None:
                            st.write(f"**Explained Variance:** {result['explained_variance_ratio']:.4f}")
                    
                    elif result['model_type'] == 'anomaly_detection':
                        if result.get('outlier_fraction') is not None:
                            st.write(f"**Outlier Fraction:** {result['outlier_fraction']:.4f}")
                        if result.get('n_outliers') is not None:
                            st.write(f"**Number of Outliers:** {result['n_outliers']}")
                else:
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")
            
            with col2:
                if result.get('hyperparameters'):
                    st.write("**Hyperparameters:**")
                    for param, value in result['hyperparameters'].items():
                        st.write(f"- {param}: {value}")

def show_algorithm_guide():
    """Show information about different unsupervised learning algorithms"""
    
    st.markdown('<h2 class="sub-header">📚 Algorithm Guide</h2>', unsafe_allow_html=True)
    
    algorithms = {
        "Clustering Algorithms": {
            "K-Means": "Partitions data into k clusters using centroids. Good for spherical clusters.",
            "DBSCAN": "Density-based clustering that can find arbitrary shaped clusters and handle noise.",
            "Gaussian Mixture Model": "Probabilistic clustering using Gaussian distributions. Soft clustering.",
            "Hierarchical Clustering": "Creates tree of clusters. Can be agglomerative (bottom-up) or divisive.",
            "Agglomerative Clustering": "Bottom-up hierarchical clustering that merges closest clusters.",
            "Spectral Clustering": "Uses eigenvalues of similarity matrix. Good for non-convex clusters.",
            "Mean Shift": "Finds dense areas of data points. Automatically determines number of clusters."
        },
        "Dimensionality Reduction": {
            "PCA": "Linear dimensionality reduction using principal components. Preserves variance.",
            "ICA": "Independent Component Analysis for blind source separation."
        },
        "Manifold Learning": {
            "t-SNE": "Non-linear dimensionality reduction for visualization. Preserves local structure."
        },
        "Anomaly Detection": {
            "Isolation Forest": "Ensemble method that isolates anomalies using random forests."
        }
    }
    
    for category, models in algorithms.items():
        st.subheader(category)
        for model, description in models.items():
            st.write(f"**{model}:** {description}")

# Sidebar navigation
with st.sidebar:
    st.markdown("---")
    if st.button("📚 Algorithm Guide"):
        show_algorithm_guide()

if __name__ == "__main__":
    main()