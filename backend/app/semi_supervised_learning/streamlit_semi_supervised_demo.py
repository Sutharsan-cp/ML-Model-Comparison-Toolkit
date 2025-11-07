"""
Streamlit Demo for Semi-Supervised Learning Models
Interactive dashboard for training and comparing semi-supervised algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

# Import our semi-supervised module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semi_supervised_learning import SemiSupervisedLearningRegistry, SemiSupervisedLearningTrainer

def create_synthetic_dataset(dataset_type, n_samples, n_features, n_classes, noise, random_state):
    """Create synthetic dataset based on user selection"""
    if dataset_type == "Classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=max(2, n_features//2),
            n_redundant=max(0, n_features//4),
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=random_state
        )
    elif dataset_type == "Blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            n_features=n_features,
            cluster_std=1.0 + noise * 2,
            random_state=random_state
        )
    else:
        # Default to classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=random_state
        )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def plot_data_distribution(X, y, title="Data Distribution"):
    """Plot 2D visualization of data distribution"""
    if X.shape[1] >= 2:
        fig = px.scatter(
            x=X[:, 0], y=X[:, 1], color=y.astype(str),
            title=title,
            labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'}
        )
        return fig
    else:
        # 1D case
        fig = px.histogram(
            x=X[:, 0], color=y.astype(str),
            title=title,
            labels={'x': 'Feature 1', 'color': 'Class'}
        )
        return fig

def plot_labeled_unlabeled_split(X_labeled, y_labeled, X_unlabeled, title="Labeled vs Unlabeled Data"):
    """Plot visualization showing labeled vs unlabeled data split"""
    if X_labeled.shape[1] >= 2 and X_unlabeled.shape[1] >= 2:
        # Create combined data for plotting
        X_combined = np.vstack([X_labeled, X_unlabeled])
        labels = ['Labeled'] * len(X_labeled) + ['Unlabeled'] * len(X_unlabeled)
        classes = np.hstack([y_labeled, np.full(len(X_unlabeled), -1)])
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot labeled data
        for class_val in np.unique(y_labeled):
            mask = (classes == class_val)
            fig.add_trace(go.Scatter(
                x=X_combined[mask, 0],
                y=X_combined[mask, 1],
                mode='markers',
                name=f'Labeled Class {class_val}',
                marker=dict(size=8, symbol='circle')
            ))
        
        # Plot unlabeled data
        unlabeled_mask = (classes == -1)
        fig.add_trace(go.Scatter(
            x=X_combined[unlabeled_mask, 0],
            y=X_combined[unlabeled_mask, 1],
            mode='markers',
            name='Unlabeled',
            marker=dict(size=6, symbol='x', color='gray')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2'
        )
        
        return fig
    return None

def plot_performance_comparison(results):
    """Create performance comparison charts"""
    if not results:
        return None, None
    
    # Filter successful results
    successful_results = [r for r in results if r['training_successful'] and r.get('test_accuracy') is not None]
    
    if not successful_results:
        return None, None
    
    # Create comparison dataframe
    comparison_data = []
    for result in successful_results:
        comparison_data.append({
            'Model': result['model_display_name'],
            'Type': result['model_type'],
            'Test_Accuracy': result.get('test_accuracy', 0),
            'Labeled_Samples': result.get('labeled_samples', 0),
            'Unlabeled_Samples': result.get('unlabeled_samples', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Bar chart for test accuracy
    fig1 = px.bar(df, x='Model', y='Test_Accuracy', color='Type',
                  title='Test Accuracy Comparison',
                  labels={'Test_Accuracy': 'Test Accuracy'})
    fig1.update_xaxes(tickangle=45)
    fig1.update_yaxes(range=[0, 1])
    
    # Scatter plot for accuracy vs data usage
    df['Total_Samples'] = df['Labeled_Samples'] + df['Unlabeled_Samples']
    df['Labeled_Ratio'] = df['Labeled_Samples'] / df['Total_Samples']
    
    fig2 = px.scatter(df, x='Labeled_Ratio', y='Test_Accuracy', 
                      color='Type', size='Total_Samples',
                      hover_data=['Model'],
                      title='Accuracy vs Labeled Data Ratio')
    fig2.update_xaxes(title='Labeled Data Ratio')
    fig2.update_yaxes(title='Test Accuracy', range=[0, 1])
    
    return fig1, fig2

def main():
    st.set_page_config(page_title="Semi-Supervised Learning Demo", layout="wide")
    
    st.title("🔄 Semi-Supervised Learning Models Comparison")
    st.markdown("Interactive dashboard for training and comparing semi-supervised learning algorithms")
    
    # Show model statistics
    if 'registry' in st.session_state:
        all_models = st.session_state.registry.get_all_models()
        total_models = sum(len(models) for models in all_models.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Categories", len(all_models))
        with col3:
            st.metric("Graph-Based", len(all_models.get('graph_based', {})))
        with col4:
            st.metric("Deep Learning", len(all_models.get('deep_learning', {})))
    
    # Initialize session state
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'registry' not in st.session_state:
        st.session_state.registry = SemiSupervisedLearningRegistry()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = SemiSupervisedLearningTrainer()
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    
    # Sidebar for configuration
    st.sidebar.header("Dataset Configuration")
    
    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Dataset Source",
        ["Synthetic Dataset", "Upload Custom Dataset"],
        help="Choose between generating synthetic data or uploading your own dataset"
    )
    
    if dataset_source == "Synthetic Dataset":
        # Dataset parameters
        dataset_type = st.sidebar.selectbox(
            "Dataset Type",
            ["Classification", "Blobs"],
            help="Choose the type of synthetic dataset"
        )
    
        n_samples = st.sidebar.slider("Number of Samples", 200, 2000, 500)
        n_features = st.sidebar.slider("Number of Features", 2, 20, 8)
        n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2)  # Default to binary for better results
        noise = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.1)
        random_state = st.sidebar.number_input("Random State", 0, 1000, 42)
    
    else:  # Upload Custom Dataset
        st.sidebar.subheader("Upload Your Dataset")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with features and target column"
        )
        
        if uploaded_file is not None:
            # Load the dataset
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show dataset preview
                with st.sidebar.expander("Dataset Preview"):
                    st.dataframe(df.head())
                
                # Target column selection
                target_column = st.sidebar.selectbox(
                    "Select Target Column",
                    df.columns.tolist(),
                    index=len(df.columns)-1,  # Default to last column
                    help="Choose the column containing the class labels"
                )
                
                # Feature columns (all except target)
                feature_columns = [col for col in df.columns if col != target_column]
                
                # Data preprocessing options
                st.sidebar.subheader("Preprocessing Options")
                handle_missing = st.sidebar.selectbox(
                    "Handle Missing Values",
                    ["Drop rows with missing values", "Fill with mean", "Fill with median"],
                    help="How to handle missing values in the dataset"
                )
                
                normalize_features = st.sidebar.checkbox(
                    "Normalize Features",
                    value=True,
                    help="Standardize features to have mean=0 and std=1"
                )
                
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
                uploaded_file = None
        else:
            st.sidebar.info("Please upload a CSV file to proceed")
    
    # Semi-supervised parameters
    st.sidebar.header("Semi-Supervised Configuration")
    labeled_ratio = st.sidebar.slider("Labeled Data Ratio", 0.05, 0.5, 0.15)
    test_ratio = st.sidebar.slider("Test Data Ratio", 0.2, 0.5, 0.3)
    
    # Generate/Process dataset button
    if dataset_source == "Synthetic Dataset":
        button_text = "🎲 Generate Dataset"
        button_disabled = False
    else:
        button_text = "📊 Process Dataset"
        button_disabled = uploaded_file is None
    
    if st.sidebar.button(button_text, disabled=button_disabled):
        if dataset_source == "Synthetic Dataset":
            # Generate synthetic dataset
            X, y = create_synthetic_dataset(dataset_type, n_samples, n_features, n_classes, noise, random_state)
            
        else:
            # Process uploaded dataset
            try:
                # Extract features and target
                X = df[feature_columns].values
                y = df[target_column].values
                
                # Handle missing values
                if handle_missing == "Drop rows with missing values":
                    # Find rows with no missing values
                    mask = ~np.isnan(X).any(axis=1) & ~pd.isna(y)
                    X = X[mask]
                    y = y[mask]
                elif handle_missing == "Fill with mean":
                    X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
                    y = pd.Series(y).fillna(pd.Series(y).mode()[0] if len(pd.Series(y).mode()) > 0 else 0).values
                elif handle_missing == "Fill with median":
                    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
                    y = pd.Series(y).fillna(pd.Series(y).mode()[0] if len(pd.Series(y).mode()) > 0 else 0).values
                
                # Encode categorical target if needed
                if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.sidebar.info(f"Target encoded: {len(le.classes_)} classes")
                
                # Normalize features if requested
                if normalize_features:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                
                st.sidebar.success(f"Dataset processed: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
                
            except Exception as e:
                st.sidebar.error(f"Error processing dataset: {str(e)}")
                return
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
        
        # Prepare semi-supervised data
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = st.session_state.trainer.prepare_semi_supervised_data(
            X_train, y_train, labeled_ratio=labeled_ratio, random_state=random_state
        )
        
        st.session_state.dataset = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_labeled': X_labeled,
            'X_unlabeled': X_unlabeled,
            'y_labeled': y_labeled,
            'y_unlabeled': y_unlabeled,
            'source': dataset_source
        }
        
        if dataset_source == "Synthetic Dataset":
            st.sidebar.success("Dataset generated!")
        else:
            st.sidebar.success("Dataset processed and ready!")
    
    # Model category selection
    category = st.sidebar.selectbox(
        "Model Category",
        ["graph_based", "self_training", "consistency", "contrastive", "deep_learning"],
        help="Choose the category of semi-supervised models to train"
    )
    
    # Get available models for selected category
    if category == "graph_based":
        available_models = st.session_state.registry.get_graph_based_models()
    elif category == "self_training":
        available_models = st.session_state.registry.get_self_training_models()
    elif category == "consistency":
        available_models = st.session_state.registry.get_consistency_models()
    elif category == "contrastive":
        available_models = st.session_state.registry.get_contrastive_models()
    elif category == "deep_learning":
        available_models = st.session_state.registry.get_deep_learning_models()
    else:
        available_models = {}
    
    # Model selection
    if available_models:
        available_model_keys = list(available_models.keys())
        selected_models = st.sidebar.multiselect(
            f"Select {category.replace('_', ' ').title()} Models",
            available_model_keys,
            default=available_model_keys[:2] if len(available_model_keys) >= 2 else available_model_keys,
            help=f"Choose which {category} models to train and compare"
        )
    else:
        st.sidebar.warning(f"No {category} models available")
        selected_models = []
    
    # Main content area
    if st.session_state.dataset is None:
        if dataset_source == "Synthetic Dataset":
            st.info("👈 Please generate a dataset first using the sidebar controls")
        else:
            st.info("👈 Please upload and process a dataset first using the sidebar controls")
        return
    
    # Display dataset information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Dataset Overview")
        
        # Dataset statistics
        dataset = st.session_state.dataset
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Samples", len(dataset['X']))
        with col_b:
            st.metric("Features", dataset['X'].shape[1])
        with col_c:
            st.metric("Classes", len(np.unique(dataset['y'])))
        with col_d:
            st.metric("Labeled Samples", len(dataset['X_labeled']))
        
        # Additional info for custom datasets
        if dataset.get('source') == 'Upload Custom Dataset':
            st.info("📁 **Custom Dataset Loaded:** Your data has been preprocessed and split for semi-supervised learning")
            
            # Show class distribution
            unique_classes, class_counts = np.unique(dataset['y'], return_counts=True)
            class_dist_df = pd.DataFrame({
                'Class': unique_classes,
                'Count': class_counts,
                'Percentage': (class_counts / len(dataset['y']) * 100).round(1)
            })
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**Class Distribution:**")
                st.dataframe(class_dist_df, use_container_width=True)
            
            with col_right:
                # Class distribution pie chart
                fig_pie = px.pie(
                    values=class_counts, 
                    names=unique_classes,
                    title="Class Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Dataset source info
        dataset_source_info = dataset.get('source', 'Synthetic Dataset')
        st.info(f"📊 Dataset Source: {dataset_source_info}")
        
        # Data visualization
        if dataset['X'].shape[1] >= 2:
            title_suffix = f" ({dataset_source_info})"
            fig_data = plot_data_distribution(dataset['X'], dataset['y'], f"Complete Dataset Distribution{title_suffix}")
            st.plotly_chart(fig_data, use_container_width=True)
            
            fig_split = plot_labeled_unlabeled_split(
                dataset['X_labeled'], dataset['y_labeled'], dataset['X_unlabeled'],
                f"Labeled vs Unlabeled Data Split{title_suffix}"
            )
            if fig_split:
                st.plotly_chart(fig_split, use_container_width=True)
        else:
            st.info("Dataset has only 1 feature. Visualization requires at least 2 features.")
    
    with col2:
        st.subheader("Available Models")
        if available_models:
            for model_key, model_info in available_models.items():
                with st.expander(f"{model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Data Types:** {', '.join(model_info['data_types'])}")
        else:
            st.info("No models available for selected category")
    
    # Training section
    st.subheader("🚀 Model Training")
    
    # Training button
    if st.button("Train Selected Models", disabled=not selected_models):
        if selected_models:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_results = []
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((i) / len(selected_models))
                
                try:
                    # Set appropriate hyperparameters based on model
                    hyperparams = {}
                    
                    if category == "graph_based":
                        if model_name == 'label_propagation':
                            hyperparams = {'kernel': 'rbf', 'gamma': 20, 'max_iter': 100}
                        elif model_name == 'label_spreading':
                            hyperparams = {'kernel': 'rbf', 'gamma': 20, 'alpha': 0.2, 'max_iter': 100}
                    elif category == "self_training":
                        if model_name == 'self_training':
                            from sklearn.ensemble import RandomForestClassifier
                            hyperparams = {
                                'base_estimator': RandomForestClassifier(n_estimators=50, random_state=42),
                                'threshold': 0.75, 
                                'max_iter': 5
                            }
                        elif model_name == 'semi_supervised_svm':
                            hyperparams = {'C': 1.0, 'C_star': 0.1, 'kernel': 'rbf'}
                    elif category == "consistency":
                        if model_name == 'mean_teacher':
                            hyperparams = {'alpha': 0.99, 'consistency_weight': 1.0, 'epochs': 20}
                        elif model_name == 'fixmatch':
                            hyperparams = {'threshold': 0.95, 'lambda_u': 1.0, 'epochs': 20}
                        elif model_name == 'mixmatch':
                            hyperparams = {'T': 0.5, 'alpha': 0.75, 'lambda_u': 10, 'epochs': 20}
                    elif category == "contrastive":
                        if model_name == 'simclr':
                            hyperparams = {'projection_dim': 64, 'temperature': 0.5, 'epochs': 20}
                        elif model_name == 'byol':
                            hyperparams = {'moving_average_decay': 0.99, 'projection_dim': 64, 'epochs': 20}
                    elif category == "deep_learning":
                        if model_name == 'ladder_network':
                            hyperparams = {'noise_std': 0.3, 'supervised_weight': 1.0, 'unsupervised_weight': 1.0, 'epochs': 20}
                        elif model_name == 'masked_autoencoder':
                            hyperparams = {'mask_ratio': 0.75, 'epochs': 20}
                    
                    result = st.session_state.trainer.train_model(
                        model_name, 
                        dataset['X_labeled'], dataset['y_labeled'],
                        dataset['X_unlabeled'], dataset['y_unlabeled'],
                        dataset['X_test'], dataset['y_test'],
                        category, hyperparams, verbose=False
                    )
                    new_results.append(result)
                    
                except Exception as e:
                    st.error(f"Error training {model_name}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Add new results to session state
            st.session_state.training_results.extend(new_results)
            
            st.success(f"Trained {len(new_results)} models successfully!")
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        st.session_state.training_results = []
        st.session_state.trainer.clear_history()
        st.success("Results cleared!")
    
    # Display results
    if st.session_state.training_results:
        st.subheader("📈 Training Results")
        
        # Results summary table
        summary_data = []
        for result in st.session_state.training_results:
            summary_data.append({
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Status': '✅ Success' if result['training_successful'] else '❌ Failed',
                'Test Accuracy': f"{result.get('test_accuracy', 'N/A'):.3f}" if isinstance(result.get('test_accuracy'), (int, float)) else 'N/A',
                'Labeled Samples': result.get('labeled_samples', 'N/A'),
                'Unlabeled Samples': result.get('unlabeled_samples', 'N/A'),
                'Error': result.get('error', '')[:50] + '...' if result.get('error') and len(result.get('error', '')) > 50 else result.get('error', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        col1, col2 = st.columns(2)
        
        fig1, fig2 = plot_performance_comparison(st.session_state.training_results)
        
        if fig1:
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
        
        if fig2:
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Best model
        best_model = st.session_state.trainer.get_best_model('test_accuracy')
        if best_model:
            st.subheader("🥇 Best Performing Model")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", best_model['model_display_name'])
            with col2:
                st.metric("Test Accuracy", f"{best_model['test_accuracy']:.3f}")
            with col3:
                st.metric("Labeled Samples", best_model['labeled_samples'])
            with col4:
                st.metric("Unlabeled Samples", best_model['unlabeled_samples'])
            
            # Classification report
            if best_model.get('classification_report'):
                st.subheader("📋 Detailed Classification Report")
                report_df = pd.DataFrame(best_model['classification_report']).transpose()
                st.dataframe(report_df, use_container_width=True)
    
    else:
        st.info("👆 Select models and click 'Train Selected Models' to see results")
    
    # Model overview section
    if st.session_state.training_results:
        st.subheader("📋 Model Categories Overview")
        
        # Create tabs for each category
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Graph-Based", "Self-Training", "Consistency", "Contrastive", "Deep Learning"])
        
        with tab1:
            st.write("**Graph-Based Methods:**")
            st.write("- **Label Propagation:** Propagates labels through similarity graph")
            st.write("- **Label Spreading:** Similar to label propagation with clamping factor")
        
        with tab2:
            st.write("**Self-Training Methods:**")
            st.write("- **Self-Training:** Iteratively adds confident predictions to training set")
            st.write("- **Semi-Supervised SVM:** SVM extended for semi-supervised learning")
        
        with tab3:
            st.write("**Consistency Regularization:**")
            st.write("- **Mean Teacher:** Student-teacher framework with exponential moving average")
            st.write("- **FixMatch:** Combines consistency regularization with pseudo-labeling")
            st.write("- **MixMatch:** Unified approach combining consistency and entropy minimization")
        
        with tab4:
            st.write("**Contrastive Learning:**")
            st.write("- **SimCLR:** Simple framework for contrastive learning")
            st.write("- **BYOL:** Bootstrap your own latent representation learning")
        
        with tab5:
            st.write("**Deep Learning Methods:**")
            st.write("- **Ladder Network:** Deep generative model with lateral connections")
            st.write("- **Masked Autoencoder:** Self-supervised learning with masking")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This demo supports both synthetic datasets and custom CSV uploads. "
                "Semi-supervised learning is most effective when you have a small amount of labeled data "
                "and a large amount of unlabeled data from the same distribution.")
    
    st.markdown("**Features:**")
    st.markdown("- 📊 **Custom Dataset Upload:** Upload your own CSV files")
    st.markdown("- 🎲 **Synthetic Data Generation:** Create test datasets with configurable parameters")
    st.markdown("- 🤖 **11 Semi-Supervised Models:** All models from your `models/semi_supervised/` directory")
    st.markdown("- 📈 **Interactive Visualization:** Real-time training and performance analysis")

if __name__ == "__main__":
    main()