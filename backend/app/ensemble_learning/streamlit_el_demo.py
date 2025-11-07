"""
Streamlit Demo for Ensemble Learning Models
Interactive dashboard for training and comparing ensemble learning algorithms
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
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

# Import our ensemble learning module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_learning import EnsembleLearningRegistry, EnsembleLearningTrainer

def create_synthetic_dataset(task_type, dataset_type, n_samples, n_features, n_classes, noise, random_state):
    """Create synthetic dataset based on user selection"""
    if task_type == "Classification":
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
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
    else:  # Regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//2),
            noise=noise * 10,
            random_state=random_state
        )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def plot_data_distribution(X, y, task_type, title="Data Distribution"):
    """Plot 2D visualization of data distribution"""
    if X.shape[1] >= 2:
        if task_type == "Classification":
            fig = px.scatter(
                x=X[:, 0], y=X[:, 1], color=y.astype(str),
                title=title,
                labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'}
            )
        else:  # Regression
            fig = px.scatter(
                x=X[:, 0], y=X[:, 1], color=y,
                title=title,
                labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Target Value'},
                color_continuous_scale='viridis'
            )
        return fig
    else:
        # 1D case
        if task_type == "Classification":
            fig = px.histogram(
                x=X[:, 0], color=y.astype(str),
                title=title,
                labels={'x': 'Feature 1', 'color': 'Class'}
            )
        else:
            fig = px.scatter(
                x=X[:, 0], y=y,
                title=title,
                labels={'x': 'Feature 1', 'y': 'Target Value'}
            )
        return fig

def plot_performance_comparison(results, task_type):
    """Create performance comparison charts"""
    if not results:
        return None, None
    
    # Filter successful results
    successful_results = [r for r in results if r['training_successful']]
    
    if not successful_results:
        return None, None    
 
   # Create comparison dataframe
    comparison_data = []
    for result in successful_results:
        if task_type == "Classification":
            comparison_data.append({
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Test_Accuracy': result.get('test_accuracy', 0),
                'Test_F1': result.get('test_f1', 0),
                'Test_Precision': result.get('test_precision', 0),
                'Test_Recall': result.get('test_recall', 0),
                'Training_Samples': result.get('training_samples', 0)
            })
        else:  # Regression
            comparison_data.append({
                'Model': result['model_display_name'],
                'Type': result['model_type'],
                'Test_R2': result.get('test_r2', 0),
                'Test_RMSE': result.get('test_rmse', 0),
                'Test_MAE': result.get('test_mae', 0),
                'Training_Samples': result.get('training_samples', 0)
            })
    
    df = pd.DataFrame(comparison_data)
    
    if task_type == "Classification":
        # Bar chart for test accuracy
        fig1 = px.bar(df, x='Model', y='Test_Accuracy', color='Type',
                      title='Test Accuracy Comparison',
                      labels={'Test_Accuracy': 'Test Accuracy'})
        fig1.update_xaxes(tickangle=45)
        fig1.update_yaxes(range=[0, 1])
        
        # Grouped bar chart for all metrics
        metrics_df = df.melt(id_vars=['Model', 'Type'], 
                            value_vars=['Test_Accuracy', 'Test_F1', 'Test_Precision', 'Test_Recall'],
                            var_name='Metric', value_name='Score')
        fig2 = px.bar(metrics_df, x='Model', y='Score', color='Metric', barmode='group',
                      title='All Classification Metrics Comparison')
        fig2.update_xaxes(tickangle=45)
        fig2.update_yaxes(range=[0, 1])
    else:  # Regression
        # Bar chart for R² score
        fig1 = px.bar(df, x='Model', y='Test_R2', color='Type',
                      title='Test R² Score Comparison',
                      labels={'Test_R2': 'R² Score'})
        fig1.update_xaxes(tickangle=45)
        
        # Scatter plot for RMSE vs MAE
        fig2 = px.scatter(df, x='Test_RMSE', y='Test_MAE', 
                         color='Type', size='Training_Samples',
                         hover_data=['Model'],
                         title='RMSE vs MAE Comparison')
        fig2.update_xaxes(title='Test RMSE')
        fig2.update_yaxes(title='Test MAE')
    
    return fig1, fig2

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Class {i}" for i in range(cm.shape[1])],
                    y=[f"Class {i}" for i in range(cm.shape[0])],
                    title=title,
                    color_continuous_scale='Blues',
                    text_auto=True)
    return fig

def plot_feature_importance(feature_importance, n_features, title="Feature Importance"):
    """Plot feature importance"""
    if feature_importance is None or len(feature_importance) == 0:
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': [f'Feature {i+1}' for i in range(min(len(feature_importance), n_features))],
        'Importance': feature_importance[:n_features]
    })
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title=title)
    return fig

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """Plot predictions vs actual values for regression"""
    fig = px.scatter(x=y_true, y=y_pred,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title=title)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(color='red', dash='dash')))
    
    return fig

def main():
    st.set_page_config(page_title="Ensemble Learning Demo", layout="wide")
    
    st.title("🌳 Ensemble Learning Models Comparison")
    st.markdown("Interactive dashboard for training and comparing ensemble learning algorithms")
    
    # Show model statistics
    if 'registry' in st.session_state:
        all_models = st.session_state.registry.get_all_models()
        total_models = sum(len(models) for models in all_models.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Classification", len(all_models.get('classification', {})))
        with col3:
            st.metric("Regression", len(all_models.get('regression', {})))
        with col4:
            trained_count = len([r for r in st.session_state.get('training_results', []) if r['training_successful']])
            st.metric("Trained Models", trained_count)    

    # Initialize session state
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'registry' not in st.session_state:
        st.session_state.registry = EnsembleLearningRegistry()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = EnsembleLearningTrainer()
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Task type selection
    task_type = st.sidebar.radio(
        "Task Type",
        ["Classification", "Regression"],
        help="Choose between classification or regression task"
    )
    
    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Dataset Source",
        ["Synthetic Dataset", "Upload Custom Dataset"],
        help="Choose between generating synthetic data or uploading your own dataset"
    )
    
    if dataset_source == "Synthetic Dataset":
        st.sidebar.subheader("Dataset Parameters")
        
        if task_type == "Classification":
            dataset_type = st.sidebar.selectbox(
                "Dataset Type",
                ["Classification", "Blobs"],
                help="Choose the type of synthetic dataset"
            )
        else:
            dataset_type = "Regression"
        
        n_samples = st.sidebar.slider("Number of Samples", 200, 2000, 500)
        n_features = st.sidebar.slider("Number of Features", 2, 20, 10)
        
        if task_type == "Classification":
            n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2)
        else:
            n_classes = 2  # Not used for regression
        
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
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                with st.sidebar.expander("Dataset Preview"):
                    st.dataframe(df.head())
                
                target_column = st.sidebar.selectbox(
                    "Select Target Column",
                    df.columns.tolist(),
                    index=len(df.columns)-1,
                    help="Choose the column containing the target values"
                )
                
                feature_columns = [col for col in df.columns if col != target_column]
                
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
    
    # Training parameters
    st.sidebar.header("Training Configuration")
    test_ratio = st.sidebar.slider("Test Data Ratio", 0.1, 0.5, 0.2)
    
    # Generate/Process dataset button
    if dataset_source == "Synthetic Dataset":
        button_text = "🎲 Generate Dataset"
        button_disabled = False
    else:
        button_text = "📊 Process Dataset"
        button_disabled = uploaded_file is None
    
    if st.sidebar.button(button_text, disabled=button_disabled):
        if dataset_source == "Synthetic Dataset":
            X, y = create_synthetic_dataset(task_type, dataset_type, n_samples, n_features, n_classes, noise, random_state)
        else:
            try:
                X = df[feature_columns].values
                y = df[target_column].values
                
                # Handle missing values
                if handle_missing == "Drop rows with missing values":
                    mask = ~np.isnan(X).any(axis=1) & ~pd.isna(y)
                    X = X[mask]
                    y = y[mask]
                elif handle_missing == "Fill with mean":
                    X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
                    y = pd.Series(y).fillna(pd.Series(y).mean() if task_type == "Regression" else pd.Series(y).mode()[0]).values
                elif handle_missing == "Fill with median":
                    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
                    y = pd.Series(y).fillna(pd.Series(y).median() if task_type == "Regression" else pd.Series(y).mode()[0]).values
                
                # Encode categorical target for classification
                if task_type == "Classification" and (y.dtype == 'object' or not np.issubdtype(y.dtype, np.number)):
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.sidebar.info(f"Target encoded: {len(le.classes_)} classes")
                
                # Normalize features
                if normalize_features:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                
                st.sidebar.success(f"Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
                
            except Exception as e:
                st.sidebar.error(f"Error processing dataset: {str(e)}")
                return
        
        # Split data
        X_train, X_test, y_train, y_test = st.session_state.trainer.prepare_data(
            X, y, test_size=test_ratio, random_state=random_state if dataset_source == "Synthetic Dataset" else 42,
            stratify=False  # Let the trainer auto-detect
        )
        
        st.session_state.dataset = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'task_type': task_type,
            'source': dataset_source
        }
        
        st.sidebar.success("Dataset ready!" if dataset_source == "Synthetic Dataset" else "Dataset processed!")
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    if task_type == "Classification":
        available_models = st.session_state.registry.get_classification_models()
        model_category = "classification"
    else:
        available_models = st.session_state.registry.get_regression_models()
        model_category = "regression"
    
    if available_models:
        available_model_keys = list(available_models.keys())
        selected_models = st.sidebar.multiselect(
            f"Select {task_type} Models",
            available_model_keys,
            default=available_model_keys[:3] if len(available_model_keys) >= 3 else available_model_keys,
            help=f"Choose which {task_type.lower()} models to train and compare"
        )
    else:
        st.sidebar.warning(f"No {task_type.lower()} models available")
        selected_models = []    
    
# Main content area
    if st.session_state.dataset is None:
        if dataset_source == "Synthetic Dataset":
            st.info("👈 Please generate a dataset first using the sidebar controls")
        else:
            st.info("👈 Please upload and process a dataset first using the sidebar controls")
        
        # Show model information
        st.subheader("📚 Available Ensemble Learning Models")
        
        tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])
        
        with tab1:
            classification_models = st.session_state.registry.get_classification_models()
            for model_key, model_info in classification_models.items():
                with st.expander(f"🌲 {model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Data Types:** {', '.join(model_info['data_types'])}")
                    st.write(f"**Task Types:** {', '.join(model_info['task_types'])}")
        
        with tab2:
            regression_models = st.session_state.registry.get_regression_models()
            for model_key, model_info in regression_models.items():
                with st.expander(f"🌲 {model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description']}")
                    st.write(f"**Data Types:** {', '.join(model_info['data_types'])}")
                    st.write(f"**Task Types:** {', '.join(model_info['task_types'])}")
        
        return
    
    # Display dataset information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Dataset Overview")
        
        dataset = st.session_state.dataset
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Samples", len(dataset['X']))
        with col_b:
            st.metric("Features", dataset['X'].shape[1])
        with col_c:
            if task_type == "Classification":
                st.metric("Classes", len(np.unique(dataset['y'])))
            else:
                st.metric("Target Range", f"{dataset['y'].min():.2f} - {dataset['y'].max():.2f}")
        with col_d:
            st.metric("Training Samples", len(dataset['X_train']))
        
        # Dataset source info
        st.info(f"📊 Dataset Source: {dataset.get('source', 'Synthetic Dataset')}")
        
        # Data visualization
        if dataset['X'].shape[1] >= 2:
            fig_data = plot_data_distribution(dataset['X'], dataset['y'], task_type, 
                                             f"Complete Dataset Distribution ({dataset.get('source', 'Synthetic')})")
            st.plotly_chart(fig_data, use_container_width=True)
        else:
            st.info("Dataset has only 1 feature. Full visualization requires at least 2 features.")
    
    with col2:
        st.subheader("📋 Selected Models")
        if selected_models:
            for model_name in selected_models:
                model_info = available_models[model_name]
                with st.expander(f"{model_info['name']}", expanded=False):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Description:** {model_info['description'][:100]}...")
        else:
            st.info("No models selected. Please select models from the sidebar.")    

    # Training section
    st.subheader("🚀 Model Training")
    
    col_train, col_clear = st.columns([3, 1])
    
    with col_train:
        train_button = st.button("🎯 Train Selected Models", disabled=not selected_models, use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.training_results = []
            st.session_state.trainer.clear_history()
            st.success("Results cleared!")
            st.rerun()
    
    if train_button and selected_models:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        new_results = []
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}... ({i+1}/{len(selected_models)})")
            progress_bar.progress((i) / len(selected_models))
            
            try:
                # Set appropriate hyperparameters based on model
                hyperparams = {}
                
                if model_name == 'randomforest':
                    hyperparams = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
                elif model_name == 'xgboost':
                    hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
                elif model_name == 'gradientboosting':
                    hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                elif model_name == 'lightgbm':
                    hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
                elif model_name == 'catboost':
                    hyperparams = {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'random_state': 42}
                elif model_name in ['bagging', 'stacking']:
                    # These will be handled by the trainer's special logic
                    hyperparams = {'random_state': 42}
                
                result = st.session_state.trainer.train_model(
                    model_name,
                    dataset['X_train'], dataset['y_train'],
                    dataset['X_test'], dataset['y_test'],
                    task_type.lower(),
                    hyperparams,
                    verbose=False
                )
                new_results.append(result)
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                # Add failed result
                new_results.append({
                    'model_name': model_name,
                    'model_display_name': available_models[model_name]['name'],
                    'training_successful': False,
                    'error': str(e)
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")
        
        # Add new results to session state
        st.session_state.training_results.extend(new_results)
        
        successful_count = len([r for r in new_results if r['training_successful']])
        st.success(f"Successfully trained {successful_count} out of {len(new_results)} models!")
        
        # Auto-rerun to show results
        st.rerun() 
   
    # Display results
    if st.session_state.training_results:
        st.subheader("📈 Training Results")
        
        # Results summary table
        summary_data = []
        for result in st.session_state.training_results:
            row = {
                'Model': result['model_display_name'],
                'Type': result.get('model_type', 'N/A'),
                'Status': '✅ Success' if result['training_successful'] else '❌ Failed'
            }
            
            if task_type == "Classification":
                row.update({
                    'Test Accuracy': f"{result.get('test_accuracy', 0):.3f}" if result['training_successful'] else 'N/A',
                    'Test F1': f"{result.get('test_f1', 0):.3f}" if result['training_successful'] else 'N/A',
                    'Test Precision': f"{result.get('test_precision', 0):.3f}" if result['training_successful'] else 'N/A',
                    'Test Recall': f"{result.get('test_recall', 0):.3f}" if result['training_successful'] else 'N/A'
                })
            else:  # Regression
                row.update({
                    'Test R²': f"{result.get('test_r2', 0):.3f}" if result['training_successful'] else 'N/A',
                    'Test RMSE': f"{result.get('test_rmse', 0):.3f}" if result['training_successful'] else 'N/A',
                    'Test MAE': f"{result.get('test_mae', 0):.3f}" if result['training_successful'] else 'N/A'
                })
            
            if not result['training_successful']:
                row['Error'] = result.get('error', 'Unknown error')[:50] + '...' if len(result.get('error', '')) > 50 else result.get('error', '')
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        
        fig1, fig2 = plot_performance_comparison(st.session_state.training_results, task_type)
        
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Best model
        if task_type == "Classification":
            best_model = st.session_state.trainer.get_best_model('classification', 'test_accuracy')
        else:
            best_model = st.session_state.trainer.get_best_model('regression', 'test_r2')
        
        if best_model:
            st.subheader("🥇 Best Performing Model")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", best_model['model_display_name'])
            
            if task_type == "Classification":
                with col2:
                    st.metric("Test Accuracy", f"{best_model['test_accuracy']:.3f}")
                with col3:
                    st.metric("Test F1 Score", f"{best_model['test_f1']:.3f}")
                with col4:
                    st.metric("Training Samples", best_model['training_samples'])
            else:
                with col2:
                    st.metric("Test R² Score", f"{best_model['test_r2']:.3f}")
                with col3:
                    st.metric("Test RMSE", f"{best_model['test_rmse']:.3f}")
                with col4:
                    st.metric("Test MAE", f"{best_model['test_mae']:.3f}")          
  
            # Detailed visualizations for best model
            st.subheader("📊 Best Model Detailed Analysis")
            
            if task_type == "Classification":
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Confusion matrix
                    if best_model.get('y_test_pred') is not None:
                        fig_cm = plot_confusion_matrix(
                            dataset['y_test'], 
                            best_model['y_test_pred'],
                            f"Confusion Matrix - {best_model['model_display_name']}"
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                
                with col_right:
                    # Feature importance
                    if best_model.get('feature_importance') is not None:
                        fig_fi = plot_feature_importance(
                            best_model['feature_importance'],
                            min(10, dataset['X'].shape[1]),
                            f"Top Features - {best_model['model_display_name']}"
                        )
                        if fig_fi:
                            st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model")
            
            else:  # Regression
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Predictions vs Actual
                    if best_model.get('y_test_pred') is not None:
                        fig_pred = plot_predictions_vs_actual(
                            dataset['y_test'],
                            best_model['y_test_pred'],
                            f"Predictions vs Actual - {best_model['model_display_name']}"
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                
                with col_right:
                    # Residuals plot
                    if best_model.get('y_test_pred') is not None:
                        residuals = dataset['y_test'] - best_model['y_test_pred']
                        fig_residuals = px.scatter(
                            x=best_model['y_test_pred'], 
                            y=residuals,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title=f"Residuals Plot - {best_model['model_display_name']}"
                        )
                        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Model comparison table
        st.subheader("📊 Detailed Model Comparison")
        
        comparison_df = st.session_state.trainer.compare_models(
            task_type.lower(),
            'test_accuracy' if task_type == "Classification" else 'test_r2'
        )
        
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download button for results
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"ensemble_learning_{task_type.lower()}_results.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Select models and click 'Train Selected Models' to see results")    
 
   # Model information section
    with st.expander("📚 Ensemble Learning Methods Overview", expanded=False):
        st.markdown("""
        ### Ensemble Learning Techniques
        
        Ensemble learning combines multiple models to achieve better predictive performance than any single model.
        
        #### 🌲 Bagging Methods
        - **Random Forest**: Builds multiple decision trees on bootstrapped samples and averages predictions
        - **Bagging**: Generic bootstrap aggregating with customizable base estimators
        
        #### 🚀 Boosting Methods
        - **Gradient Boosting**: Sequential ensemble that builds trees to correct previous errors
        - **XGBoost**: Extreme Gradient Boosting with regularization and advanced features
        - **LightGBM**: Fast gradient boosting with histogram-based learning
        - **CatBoost**: Gradient boosting optimized for categorical features
        
        #### 🎯 Stacking Methods
        - **Stacking**: Meta-learning approach that combines predictions from multiple base models
        
        ### Key Advantages
        - **Reduced Overfitting**: Averaging multiple models reduces variance
        - **Improved Accuracy**: Often achieves better performance than single models
        - **Robustness**: Less sensitive to noise and outliers
        - **Feature Importance**: Most ensemble methods provide feature importance scores
        
        ### When to Use
        - **Random Forest**: Good baseline, handles non-linear relationships well
        - **XGBoost/LightGBM**: When you need state-of-the-art performance
        - **Gradient Boosting**: When you have time for careful tuning
        - **Stacking**: When you want to combine strengths of different model types
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Features:**
    - 🎲 **Synthetic Data Generation**: Create test datasets with configurable parameters
    - 📊 **Custom Dataset Upload**: Upload your own CSV files for analysis
    - 🤖 **13 Ensemble Models**: All models from your `models/ensemble/` directory
    - 📈 **Interactive Visualization**: Real-time training and performance analysis
    - 🏆 **Model Comparison**: Side-by-side comparison of all trained models
    - 📥 **Export Results**: Download comparison results as CSV
    
    **Supported Models:**
    - Classification: Random Forest, XGBoost, Gradient Boosting, Bagging, Stacking, LightGBM, CatBoost
    - Regression: Random Forest, XGBoost, Gradient Boosting, Bagging, Stacking, LightGBM
    """)

if __name__ == "__main__":
    main()
