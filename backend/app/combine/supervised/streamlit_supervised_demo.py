"""
Streamlit Demo for Supervised Learning Models
ML Model Comparison Toolkit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

from combine.supervised.supervised import (
    SupervisedModelRegistry,
    SupervisedModelTrainer,
    create_model_comparison_report,
)

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Toolkit - Supervised Learning",
    page_icon="🤖",
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
    .leaderboard-gold {
        background-color: #fff9c4 !important;
        border-left: 4px solid #ffd700;
    }
    .leaderboard-silver {
        background-color: #f5f5f5 !important;
        border-left: 4px solid #c0c0c0;
    }
    .leaderboard-bronze {
        background-color: #ffecb3 !important;
        border-left: 4px solid #cd7f32;
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
    
    # Check raw datasets
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
    
    # Check processed datasets
    for task_type in ["classification", "regression"]:
        task_dir = data_dir / "processed" / task_type
        if task_dir.exists():
            for file_path in task_dir.glob("*.csv"):
                name = file_path.stem.replace("_processed", "")
                datasets[task_type.title()][f"{name} (processed)"] = {
                    "path": str(file_path),
                    "type": "processed"
                }
    
    return datasets

def load_dataset(dataset_info, target_column=None):
    """Load dataset from file"""
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
        
        if target_column and target_column in df.columns:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            feature_names = X.columns.tolist()
            
            # Convert categorical features to numeric
            categorical_columns = []
            for col in X.columns:
                if X[col].dtype == 'object':
                    categorical_columns.append(col)
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Convert target to numeric if needed
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            
            # Convert to numpy arrays
            X_array = X.values.astype(float) if hasattr(X, 'values') else X.astype(float)
            y_array = y.values if hasattr(y, 'values') else y
            
            # Add info about categorical columns
            info = {
                'categorical_columns': categorical_columns,
                'original_shape': df.shape,
                'processed_shape': (X_array.shape[0], X_array.shape[1])
            }
            
            return X_array, y_array, feature_names, df, info
        else:
            return None, None, df.columns.tolist(), df, {}
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, [], pd.DataFrame(), {}

def load_sample_data(dataset_name, task_type):
    """Load synthetic sample datasets"""
    if task_type == "Classification":
        if dataset_name == "Synthetic Binary":
            X, y = make_classification(n_samples=500, n_features=10, n_classes=2, 
                                     n_informative=5, random_state=42)
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        else:  # Synthetic Multiclass
            X, y = make_classification(n_samples=500, n_features=10, n_classes=3, 
                                     n_informative=7, random_state=42)
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    else:  # Regression
        if dataset_name == "Synthetic Linear":
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        else:  # Synthetic Nonlinear
            X, y = make_regression(n_samples=500, n_features=10, noise=0.2, random_state=42)
            # Add some nonlinearity
            y = y + 0.1 * X[:, 0] * X[:, 1] + 0.05 * X[:, 2] ** 2
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    
    return X, y, feature_names

def get_working_models():
    """Get only the models that are known to work based on test results"""
    working_models = {
        'classification': [
            'logistic_regression',
            'random_forest_classifier', 
            'svm_classifier',
            'knn_classifier',
            'xgboost',
            'lda',
            'qda',
            'ridge_classifier',
            'lasso_classifier',
            'elastic_net_classifier',
            'decision_tree_cart',
            'decision_tree_id3',
            'adaboost',
            'catboost_classifier',
            'decision_stump',
            'decision_tree_c45',
            'decision_tree_chaid',
            'gradient_boosting',
            'naive_bayes'
        ],
        'regression': [
            'linear_regression',
            'random_forest_regressor',
            'svm_regressor',
            'ridge_regression',
            'lasso_regression',
            'elastic_net_regression',
            'decision_tree_regressor',
            'xgboost_regressor',
            'bayesian_ridge',
            'adaboost_regressor',
            'catboost_regressor',
            'gradient_boosting_regressor',
            'polynomial_regression'
        ]
    }
    return working_models

def plot_model_comparison(results_df):
    """Plot model comparison results"""
    if results_df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training scores
    ax1.barh(results_df['Model'], results_df['Train_Score'])
    ax1.set_xlabel('Training Score')
    ax1.set_title('Training Performance')
    ax1.set_xlim(0, 1)
    
    # Test scores
    test_scores = pd.to_numeric(results_df['Test_Score'], errors='coerce')
    ax2.barh(results_df['Model'], test_scores)
    ax2.set_xlabel('Test Score')
    ax2.set_title('Test Performance')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig

def create_leaderboard(results):
    """Create a leaderboard for top-performing models"""
    if not results:
        return None
    
    # Create leaderboard data
    leaderboard_data = []
    for result in results:
        if result['training_successful']:
            model_data = {
                'Model': result['model_display_name'],
                'Type': result['model_type']
            }
            
            # Add appropriate metrics based on task type
            if result['model_type'] == 'classification':
                model_data['Accuracy'] = result.get('accuracy', 0)
                model_data['F1 Score'] = result.get('f1_score', 0)
                model_data['Precision'] = result.get('precision', 0)
                model_data['Recall'] = result.get('recall', 0)
                if result.get('roc_auc') is not None:
                    model_data['ROC AUC'] = result.get('roc_auc', 0)
                # Primary metric for ranking
                model_data['Primary Score'] = result.get('f1_score', result.get('accuracy', 0))
                
            else:  # regression
                model_data['R²'] = result.get('r2', 0)
                model_data['MAE'] = result.get('mae', 0)
                model_data['RMSE'] = result.get('rmse', 0)
                # Primary metric for ranking (higher R² is better)
                model_data['Primary Score'] = result.get('r2', 0)
            
            leaderboard_data.append(model_data)
    
    if not leaderboard_data:
        return None
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Sort by primary score (descending for both classification and regression)
    leaderboard_df = leaderboard_df.sort_values('Primary Score', ascending=False)
    
    return leaderboard_df

def plot_feature_importance(results, feature_names):
    """Plot feature importance for models that provide it"""
    if not results or not feature_names:
        return None
    
    # Collect models with feature importance
    models_with_importance = []
    for result in results:
        if (result['training_successful'] and 
            'feature_importance' in result and 
            result['feature_importance'] is not None):
            
            importance = result['feature_importance']
            if len(importance) == len(feature_names):
                models_with_importance.append({
                    'model': result['model_display_name'],
                    'importance': importance,
                    'type': result['model_type']
                })
    
    if not models_with_importance:
        return None
    
    # Create subplots
    n_models = len(models_with_importance)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for idx, model_info in enumerate(models_with_importance):
        ax = axes[idx]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model_info['importance']
        }).sort_values('importance', ascending=True)
        
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_title(f"{model_info['model']}\nFeature Importance")
        ax.set_xlabel('Importance Score')
    
    plt.tight_layout()
    return fig

def create_failed_models_table(failed_results):
    """Create a summary table for failed models"""
    if not failed_results:
        return None
    
    failed_data = []
    for result in failed_results:
        failed_data.append({
            'Model': result['model_display_name'],
            'Error Type': type(result['error']).__name__,
            'Error Message': str(result['error'])[:100] + '...' if len(str(result['error'])) > 100 else str(result['error'])
        })
    
    return pd.DataFrame(failed_data)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Comparison Toolkit</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Supervised Learning Models</h2>', unsafe_allow_html=True)
    
    # Initialize registry
    registry = SupervisedModelRegistry()
    working_models = get_working_models()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Task type selection
    task_type = st.sidebar.selectbox(
        "Select Task Type",
        ["Classification", "Regression"]
    )
    
    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Dataset Source",
        ["Your Datasets", "Synthetic Data"]
    )
    
    # Dataset selection
    if dataset_source == "Your Datasets":
        available_datasets = get_available_datasets()
        task_datasets = available_datasets.get(task_type, {})
        
        if task_datasets:
            dataset_name = st.sidebar.selectbox(
                "Select Dataset", 
                list(task_datasets.keys())
            )
            dataset_info = task_datasets[dataset_name]
        else:
            st.sidebar.warning(f"No {task_type.lower()} datasets found in data directory")
            dataset_source = "Synthetic Data"
    
    if dataset_source == "Synthetic Data":
        if task_type == "Classification":
            dataset_options = ["Synthetic Binary", "Synthetic Multiclass"]
        else:
            dataset_options = ["Synthetic Linear", "Synthetic Nonlinear"]
        
        dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    # Model selection - only show working models
    task_key = task_type.lower()
    if task_key == 'classification':
        available_models = registry.get_classification_models()
        working_model_keys = working_models['classification']
    else:
        available_models = registry.get_regression_models()
        working_model_keys = working_models['regression']
    
    # Filter to only working models
    filtered_models = {k: v for k, v in available_models.items() if k in working_model_keys}
    model_names = list(filtered_models.keys())
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_names,
        default=model_names[:5] if len(model_names) >= 5 else model_names
    )
    
    # Data preprocessing options
    st.sidebar.subheader("Preprocessing")
    scale_features = st.sidebar.checkbox("Scale Features", value=True)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    
    # Target column selection for custom datasets
    target_column = None
    if dataset_source == "Your Datasets" and 'dataset_info' in locals():
        # Load dataset to show column options
        _, _, columns, sample_df, _ = load_dataset(dataset_info)
        if not sample_df.empty:
            st.sidebar.subheader("Dataset Preview")
            st.sidebar.dataframe(sample_df.head(3))
            
            target_column = st.sidebar.selectbox(
                "Select Target Column",
                columns,
                index=len(columns)-1 if columns else 0
            )
    
    # Main content
    if st.sidebar.button("Run Comparison", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to compare.")
            return
        
        # Load data
        with st.spinner("Loading dataset..."):
            if dataset_source == "Your Datasets" and 'dataset_info' in locals():
                X, y, feature_names, df, info = load_dataset(dataset_info, target_column)
                if X is None:
                    st.error("Failed to load dataset. Please check the target column selection.")
                    return
                
                # Show preprocessing info
                if info.get('categorical_columns'):
                    st.info(f"Converted categorical columns to numeric: {', '.join(info['categorical_columns'])}")
            else:
                X, y, feature_names = load_sample_data(dataset_name, task_type)
                df = pd.DataFrame(X, columns=feature_names)
                df['target'] = y
                info = {}
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", X.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            if task_type == "Classification":
                st.metric("Classes", len(np.unique(y)))
            else:
                st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
        with col4:
            st.metric("Task Type", task_type)
        
        # Show dataset preview
        with st.expander("Dataset Preview"):
            preview_df = df.head(10)
            st.dataframe(preview_df, use_container_width=True)
            
            # Basic statistics
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Data preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train models
        st.subheader("Model Training Results")
        
        trainer = SupervisedModelTrainer()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(selected_models))
            
            try:
                result = trainer.train_model(
                    model_name, X_train, y_train, X_test, y_test, task_key
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
            
            # Plot comparison
            if len(successful_results) > 1:
                st.subheader("Performance Comparison")
                fig = plot_model_comparison(comparison_df)
                if fig:
                    st.pyplot(fig)
            
            # # 🏅 Leaderboard for top-performing models
            st.subheader("🏅 Model Leaderboard")
            leaderboard_df = create_leaderboard(successful_results)
            
            if leaderboard_df is not None:
            #     # Display leaderboard with styling
            #     styled_leaderboard = leaderboard_df.drop('Primary Score', axis=1).reset_index(drop=True)
                
            #     # Add ranking
            #     styled_leaderboard.insert(0, 'Rank', range(1, len(styled_leaderboard) + 1))
                
            #     # Apply styling for top 3 models
            #     def highlight_top3(row):
            #         styles = [''] * len(row)
            #         if row.name == 0:  # Gold
            #             styles = ['background-color: #fff9c4; border-left: 4px solid #ffd700'] * len(row)
            #         elif row.name == 1:  # Silver
            #             styles = ['background-color: #f5f5f5; border-left: 4px solid #c0c0c0'] * len(row)
            #         elif row.name == 2:  # Bronze
            #             styles = ['background-color: #ffecb3; border-left: 4px solid #cd7f32'] * len(row)
            #         return styles
                
            #     styled_df = styled_leaderboard.style.apply(highlight_top3, axis=1)
            #     st.dataframe(styled_df, use_container_width=True)
                
                # Show top model celebration
                top_model = leaderboard_df.iloc[0]
                st.success(f"🎉 **Top Performer**: {top_model['Model']}")
            
            # 📈 Feature importance visualization
            # st.subheader("📈 Feature Importance")
            # feature_importance_fig = plot_feature_importance(successful_results, feature_names)
            
            # if feature_importance_fig:
            #     st.pyplot(feature_importance_fig)
            #     st.caption("Feature importance scores for models that support this feature")
            # else:
            #     st.info("No feature importance data available for the trained models.")
            
            # 🏆 Best performing model highlight
            best_model = trainer.get_best_model(task_key)

            # 🎨 Visualizations for the best model
            if best_model:
                metric_name = best_model.get('selected_metric', 'test_score')
                metric_value = best_model.get(metric_name, 0)

                st.subheader("🏆 Best Performing Model")
                st.success(f"**{best_model['model_display_name']}** — {metric_name.upper()}: {metric_value:.4f}")

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### 🎨 Model Performance Visualizations")

                y_true = y_test
                y_pred = best_model.get('test_predictions', None)
                print("y_pred",y_pred)
                print("model_type",best_model['model_type'])

                # --- Classification Plots ---
                if task_key == 'classification' and y_pred is not None:
                    st.subheader("📊 Confusion Matrix")
                    try:
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                        ax_cm.set_xlabel("Predicted Labels")
                        ax_cm.set_ylabel("True Labels")
                        ax_cm.set_title(f"{best_model['model_display_name']} — Confusion Matrix")
                        st.pyplot(fig_cm)
                    except Exception as e:
                        st.warning(f"Confusion matrix could not be displayed: {e}")

                    # ROC curve (for binary classification only)
                    if len(np.unique(y_true)) == 2 and hasattr(best_model['model_instance'], 'predict_proba'):
                        try:
                            y_prob = best_model['model_instance'].predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_true, y_prob)
                            roc_auc = auc(fpr, tpr)
                            fig_roc, ax_roc = plt.subplots()
                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.4f}")
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.legend(loc="lower right")
                            ax_roc.set_title(f"{best_model['model_display_name']} — ROC Curve")
                            st.pyplot(fig_roc)
                        except Exception as e:
                            st.warning(f"ROC curve could not be displayed: {e}")

                # --- Regression Plots ---
                elif task_key == 'regression' and y_pred is not None:
                    try:
                        st.subheader("📉 Predicted vs Actual")
                        fig_pa, ax_pa = plt.subplots()
                        ax_pa.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
                        ax_pa.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                        ax_pa.set_xlabel("Actual Values")
                        ax_pa.set_ylabel("Predicted Values")
                        ax_pa.set_title(f"{best_model['model_display_name']} — Predicted vs Actual")
                        st.pyplot(fig_pa)
                    except Exception as e:
                        st.warning(f"Predicted vs Actual plot could not be displayed: {e}")

                    try:
                        st.subheader("📊 Residual Distribution")
                        residuals = y_true - y_pred
                        fig_res, ax_res = plt.subplots()
                        sns.histplot(residuals, kde=True, ax=ax_res, color='teal')
                        ax_res.set_xlabel("Residuals (y_true - y_pred)")
                        ax_res.set_title(f"{best_model['model_display_name']} — Residual Distribution")
                        st.pyplot(fig_res)
                    except Exception as e:
                        st.warning(f"Residual plot could not be displayed: {e}")

                # --- Feature Importance ---
                if 'feature_importance' in best_model and best_model['feature_importance'] is not None:
                    try:
                        st.subheader("🔍 Feature Importance")
                        importances = best_model['feature_importance']
                        if isinstance(importances, (list, np.ndarray)) and len(importances) == len(feature_names):
                            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                            fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
                            fig_fi, ax_fi = plt.subplots()
                            sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax_fi, palette='magma')
                            ax_fi.set_title("Top 10 Important Features")
                            st.pyplot(fig_fi)
                    except Exception as e:
                        st.warning(f"Feature importance plot could not be displayed: {e}")

                elif y_pred is None:
                    st.warning("⚠️ Visualization skipped — model did not return predictions.")

            else:
                st.warning("No best model could be determined.")
                
            # 🧾 Detailed results section
            with st.expander("📊 Detailed Results (All Models)"):
                if not successful_results:
                    st.info("No successful models to display yet.")
                else:
                    for result in successful_results:
                        st.markdown(f"### 🧠 {result['model_display_name']}")
                        st.caption(f"Model Type: `{result['model_type']}`")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Training Score", f"{result['train_score']:.4f}")
                            test_score = result.get('test_score')
                            st.metric("Test Score", f"{test_score:.4f}" if test_score is not None else "N/A")

                        with col2:
                            if result['model_type'] == 'classification':
                                st.metric("Accuracy", f"{result.get('accuracy', 0):.4f}")
                                st.metric("F1 Score", f"{result.get('f1_score', 0):.4f}")
                                st.metric("Precision", f"{result.get('precision', 0):.4f}")
                                st.metric("Recall", f"{result.get('recall', 0):.4f}")
                                if result.get('roc_auc') is not None:
                                    st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
                            elif result['model_type'] == 'regression':
                                st.metric("R²", f"{result.get('r2', 0):.4f}")
                                st.metric("MAE", f"{result.get('mae', 0):.4f}")
                                st.metric("RMSE", f"{result.get('rmse', 0):.4f}")

                        with col3:
                            st.write("**Feature Importance:**")
                            if 'feature_importance' in result and result['feature_importance'] is not None:
                                st.success("Available ✅")
                            else:
                                st.warning("Not Provided ❌")

                            if result.get('hyperparameters'):
                                st.markdown("<br><b>Hyperparameters:</b>", unsafe_allow_html=True)
                                st.json(result['hyperparameters'])
                            else:
                                st.caption("No hyperparameters specified.")

                        st.divider()

        # ⚠️ Failed models summary table
        if failed_results:
            st.subheader("⚠️ Failed Models Summary")
            failed_df = create_failed_models_table(failed_results)
            
            if failed_df is not None:
                st.dataframe(failed_df, use_container_width=True)
                
                # Show error analysis
                st.subheader("Error Analysis")
                error_counts = failed_df['Error Type'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Error Type Distribution:**")
                    for error_type, count in error_counts.items():
                        st.write(f"- {error_type}: {count} model(s)")
                
                with col2:
                    st.write("**Troubleshooting Tips:**")
                    if "ValueError" in error_counts:
                        st.write("• Check data types and feature scaling")
                        st.write("• Verify target variable format")
                    if "ConvergenceWarning" in str(failed_df['Error Message']):
                        st.write("• Try increasing max iterations")
                        st.write("• Consider feature scaling")
                    st.write("• Review model-specific requirements")
                    st.write("• Check for missing or infinite values")
        
        # Summary statistics
        st.subheader("📋 Training Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", len(selected_models))
        with col2:
            st.metric("Successful", len(successful_results))
        with col3:
            st.metric("Failed", len(failed_results))
    
    # Model Information
    st.sidebar.markdown("---")
    if st.sidebar.button("Show Model Information"):
        st.subheader("Available Models")
        
        if task_type == "Classification":
            models_info = {k: v for k, v in registry.get_classification_models().items() 
                          if k in working_models['classification']}
        else:
            models_info = {k: v for k, v in registry.get_regression_models().items() 
                          if k in working_models['regression']}
        
        for model_name, info in models_info.items():
            status = "✅ Working" if model_name in (working_models.get('classification', []) + 
                                                   working_models.get('regression', [])) else "❌ Issues"
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