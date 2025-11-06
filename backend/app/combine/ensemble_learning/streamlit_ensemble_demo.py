"""
Streamlit Demo for Ensemble Learning Models
Interactive dashboard for training and comparing ensemble algorithms
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
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Import our ensemble module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_learning import EnsembleLearningRegistry, EnsembleLearningTrainer

def create_synthetic_dataset(task, n_samples, n_features, n_classes, noise, random_state):
    """Create synthetic dataset based on task type"""
    if task == "classification":
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
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//2),
            noise=noise * 10,  # Scale noise for regression
            random_state=random_state
        )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def plot_data_distribution(X, y, task, title="Data Distribution"):
    """Plot 2D visualization of data distribution"""
    if X.shape[1] >= 2:
        if task == "classification":
            fig = px.scatter(
                x=X[:, 0], y=X[:, 1], color=y.astype(str),
                title=title,
                labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'}
            )
        else:
            fig = px.scatter(
                x=X[:, 0], y=X[:, 1], color=y,
                title=title,
                labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Target'},
                color_continuous_scale='viridis'
            )
        return fig
    else:
        # 1D case
        if task == "classification":
            fig = px.histogram(
                x=X[:, 0], color=y.astype(str),
                title=title,
                labels={'x': 'Feature 1', 'color': 'Class'}
            )
        else:
            fig = px.scatter(
                x=X[:, 0], y=y,
                title=title,
                labels={'x': 'Feature 1', 'y': 'Target'}
            )
        return fig

def plot_performance_comparison(results, task):
    """Create performance comparison charts"""
    if not results:
        return None, None
    
    # Filter successful results
    successful_results = [r for r in results if r['training_successful'] and r.get('test_score') is not None]
    
    if not successful_results:
        return None, None
    
    # Create comparison dataframe
    comparison_data = []
    for result in successful_results:
        comparison_data.append({
            'Model': result['model_display_name'],
            'Type': result['model_type'],
            'CV_Score': result.get('cv_mean_score', 0),
            'Test_Score': result.get('test_score', 0),
            'CV_Std': result.get('cv_std_score', 0),
            'Training_Samples': result.get('training_samples', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Bar chart for test scores
    score_label = 'Accuracy' if task == 'classification' else 'R² Score'
    fig1 = px.bar(df, x='Model', y='Test_Score', color='Type',
                  title=f'Test {score_label} Comparison',
                  labels={'Test_Score': f'Test {score_label}'})
    fig1.update_xaxes(tickangle=45)
    if task == 'classification':
        fig1.update_yaxes(range=[0, 1])
    
    # CV vs Test score comparison
    fig2 = px.scatter(df, x='CV_Score', y='Test_Score', 
                      color='Type', size='Training_Samples',
                      hover_data=['Model'],
                      title=f'Cross-Validation vs Test {score_label}')
    fig2.update_xaxes(title=f'CV {score_label}')
    fig2.update_yaxes(title=f'Test {score_label}')
    
    # Add diagonal line for perfect correlation
    min_score = min(df['CV_Score'].min(), df['Test_Score'].min())
    max_score = max(df['CV_Score'].max(), df['Test_Score'].max())
    fig2.add_shape(
        type="line",
        x0=min_score, y0=min_score,
        x1=max_score, y1=max_score,
        line=dict(dash="dash", color="gray")
    )
    
    return fig1, fig2

def plot_feature_importance(results):
    """Plot feature importance for models that support it"""
    importance_data = []
    
    for result in results:
        if result['training_successful'] and result.get('feature_importance') is not None:
            importance = result['feature_importance']
            for i, imp in enumerate(importance):
                importance_data.append({
                    'Model': result['model_display_name'],
                    'Feature': f'Feature_{i}',
                    'Importance': imp
                })
    
    if not importance_data:
        return None
    
    df = pd.DataFrame(importance_data)
    
    # Create grouped bar chart
    fig = px.bar(df, x='Feature', y='Importance', color='Model',
                 title='Feature Importance Comparison',
                 barmode='group')
    fig.update_xaxes(tickangle=45)
    
    return fig

def main():
    st.set_page_config(page_title="Ensemble Learning Demo", layout="wide")
    
    st.title("🌳 Ensemble Learning Models Comparison")
    st.markdown("Interactive dashboard for training and comparing ensemble learning algorithms")
    
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
    st.sidebar.header("Dataset Configuration")
    
    # Task selection
    task = st.sidebar.selectbox(
        "Task Type",
        ["classification", "regression"],
        help="Choose between classification or regression task"
    )
    
    # Dataset parameters
    n_samples = st.sidebar.slider("Number of Samples", 200, 2000, 500)
    n_features = st.sidebar.slider("Number of Features", 2, 20, 8)
    
    if task == "classification":
        n_classes = st.sidebar.slider("Number of Classes", 2, 5, 3)
    else:
        n_classes = 2  # Not used for regression
    
    noise = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.1)
    random_state = st.sidebar.number_input("Random State", 0, 1000, 42)
    test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.3)
    
    # Generate dataset button
    if st.sidebar.button("🎲 Generate Dataset"):
        X, y = create_synthetic_dataset(task, n_samples, n_features, n_classes, noise, random_state)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        st.session_state.dataset = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'task': task
        }
        
        st.sidebar.success("Dataset generated!")
    
    # Model selection
    st.sidebar.header("Model Configuration")
    
    # Get available models for selected task
    if task == "classification":
        available_models = st.session_state.registry.get_classification_models()
    else:
        available_models = st.session_state.registry.get_regression_models()
    
    # Model selection
    if available_models:
        selected_models = st.sidebar.multiselect(
            f"Select {task.title()} Models",
            list(available_models.keys()),
            default=list(available_models.keys())[:3],
            help=f"Choose which {task} models to train and compare"
        )
    else:
        st.sidebar.warning(f"No {task} models available")
        selected_models = []
    
    # Cross-validation folds
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Main content area
    if st.session_state.dataset is None:
        st.info("👈 Please generate a dataset first using the sidebar controls")
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
            if dataset['task'] == 'classification':
                st.metric("Classes", len(np.unique(dataset['y'])))
            else:
                st.metric("Target Range", f"{dataset['y'].min():.2f} - {dataset['y'].max():.2f}")
        with col_d:
            st.metric("Training Samples", len(dataset['X_train']))
        
        # Data visualization
        if dataset['X'].shape[1] >= 2:
            fig_data = plot_data_distribution(dataset['X'], dataset['y'], dataset['task'], "Dataset Distribution")
            st.plotly_chart(fig_data, use_container_width=True)
    
    with col2:
        st.subheader("Available Models")
        if available_models:
            for model_key, model_info in available_models.items():
                with st.expander(f"{model_info['name']}"):
                    st.write(f"**Type:** {model_info['type']}")
                    st.write(f"**Task:** {model_info['task']}")
                    st.write(f"**Description:** {model_info['description']}")
        else:
            st.info("No models available for selected task")
    
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
                    hyperparams = {'random_state': 42}
                    
                    if model_name == 'random_forest':
                        hyperparams.update({'n_estimators': 100, 'max_depth': 10})
                    elif model_name == 'gradient_boosting':
                        hyperparams.update({'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5})
                    elif model_name in ['xgboost', 'lightgbm']:
                        hyperparams.update({'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5})
                    elif model_name == 'catboost':
                        hyperparams.update({'iterations': 100, 'learning_rate': 0.1, 'depth': 5, 'verbose': False})
                    elif model_name == 'bagging':
                        hyperparams.update({'n_estimators': 50})
                    elif model_name == 'stacking':
                        hyperparams.update({'cv': cv_folds, 'n_jobs': 1})
                    
                    result = st.session_state.trainer.train_model(
                        model_name, 
                        dataset['X_train'], dataset['y_train'],
                        dataset['X_test'], dataset['y_test'],
                        task, hyperparams, cv_folds, verbose=False
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
                'Task': result['task'],
                'Status': '✅ Success' if result['training_successful'] else '❌ Failed',
                'CV Score': f"{result.get('cv_mean_score', 'N/A'):.3f} ± {result.get('cv_std_score', 0):.3f}" if isinstance(result.get('cv_mean_score'), (int, float)) else 'N/A',
                'Test Score': f"{result.get('test_score', 'N/A'):.3f}" if isinstance(result.get('test_score'), (int, float)) else 'N/A',
                'Training Samples': result.get('training_samples', 'N/A'),
                'Error': result.get('error', '')[:50] + '...' if result.get('error') and len(result.get('error', '')) > 50 else result.get('error', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        col1, col2 = st.columns(2)
        
        fig1, fig2 = plot_performance_comparison(st.session_state.training_results, task)
        
        if fig1:
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
        
        if fig2:
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Feature importance
        fig_importance = plot_feature_importance(st.session_state.training_results)
        if fig_importance:
            st.subheader("🎯 Feature Importance")
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Best model
        best_model = st.session_state.trainer.get_best_model('test_score')
        if best_model:
            st.subheader("🥇 Best Performing Model")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", best_model['model_display_name'])
            with col2:
                score_label = 'Accuracy' if task == 'classification' else 'R² Score'
                st.metric(f"Test {score_label}", f"{best_model['test_score']:.3f}")
            with col3:
                st.metric("CV Score", f"{best_model.get('cv_mean_score', 0):.3f}")
            with col4:
                st.metric("Training Samples", best_model['training_samples'])
            
            # Detailed metrics for best model
            if task == 'classification' and best_model.get('classification_report'):
                st.subheader("📋 Detailed Classification Report")
                report_df = pd.DataFrame(best_model['classification_report']).transpose()
                st.dataframe(report_df, use_container_width=True)
            elif task == 'regression' and best_model.get('regression_metrics'):
                st.subheader("📊 Detailed Regression Metrics")
                metrics = best_model['regression_metrics']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                with col2:
                    st.metric("MSE", f"{metrics['mse']:.3f}")
                with col3:
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
    
    else:
        st.info("👆 Select models and click 'Train Selected Models' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This demo uses synthetic datasets for demonstration. "
                "Ensemble methods typically perform better on real-world datasets with complex patterns. "
                "Try different ensemble types (bagging, boosting, stacking) to see their strengths.")

if __name__ == "__main__":
    main()