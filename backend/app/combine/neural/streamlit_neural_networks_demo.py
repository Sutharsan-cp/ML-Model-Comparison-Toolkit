"""
Streamlit Demo for Neural Network Models
ML Model Comparison Toolkit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our neural networks module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neural_networks import NeuralNetworkRegistry, NeuralNetworkTrainer, create_model_comparison_report

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Toolkit - Neural Networks",
    page_icon="🧠",
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

def generate_synthetic_data(dataset_name, task_type):
    """Generate synthetic datasets for neural networks"""
    np.random.seed(42)
    
    if task_type == "Classification":
        if dataset_name == "Binary Classification":
            X, y = make_classification(n_samples=500, n_features=10, n_classes=2, 
                                     n_informative=5, random_state=42)
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        elif dataset_name == "Multi-class Classification":
            X, y = make_classification(n_samples=500, n_features=10, n_classes=3, 
                                     n_informative=7, random_state=42)
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        else:  # Complex Classification
            X, y = make_classification(n_samples=500, n_features=20, n_classes=4, 
                                     n_informative=15, n_redundant=2, random_state=42)
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    elif task_type == "Regression":
        if dataset_name == "Linear Regression":
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        elif dataset_name == "Nonlinear Regression":
            X, y = make_regression(n_samples=500, n_features=10, noise=0.2, random_state=42)
            # Add nonlinearity
            y = y + 0.1 * X[:, 0] * X[:, 1] + 0.05 * X[:, 2] ** 2
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        else:  # Complex Regression
            X, y = make_regression(n_samples=500, n_features=20, noise=0.1, 
                                 n_informative=15, random_state=42)
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    elif task_type == "Image Classification":
        # Generate simple image-like data
        X = np.random.randn(200, 28, 28)
        y = np.random.randint(0, 3, 200)
        feature_names = ['Image Data']
    
    else:  # Unsupervised
        X = np.random.randn(500, 20)
        y = None
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    return X, y, feature_names

def get_working_models():
    """Get only the models that are known to work based on test results"""
    working_models = {
        'feedforward': [
            'perceptron',
            'mlp',
            'ann',
            'backpropagation'
        ],
        'convolutional': [
            'cnn'
        ],
        'recurrent': [
            # RNN models have broadcasting issues, skip for now
        ],
        'generative': [
            'gan'  # Autoencoder has issues
        ],
        'specialized': [
            # Transformer might be too complex for simple demo
        ]
    }
    return working_models

def plot_training_history(results):
    """Plot training history for neural networks"""
    models_with_history = [r for r in results if r['training_successful'] and 
                          ('loss_history' in r or 'accuracy_history' in r)]
    
    if not models_with_history:
        return None
    
    n_models = len(models_with_history)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss history
    ax1 = axes[0]
    for result in models_with_history:
        if 'loss_history' in result and result['loss_history']:
            epochs = range(len(result['loss_history']))
            ax1.plot(epochs, result['loss_history'], 
                    label=result['model_display_name'], marker='o', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy history
    ax2 = axes[1]
    for result in models_with_history:
        if 'accuracy_history' in result and result['accuracy_history']:
            epochs = range(len(result['accuracy_history']))
            ax2.plot(epochs, result['accuracy_history'], 
                    label=result['model_display_name'], marker='o', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(comparison_df):
    """Plot model comparison results"""
    if comparison_df.empty:
        return None
    
    # Convert numeric columns
    numeric_cols = ['Final_Loss', 'Final_Accuracy', 'Training_Accuracy']
    for col in numeric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracies
    ax1 = axes[0]
    if 'Final_Accuracy' in comparison_df.columns:
        valid_acc = comparison_df.dropna(subset=['Final_Accuracy'])
        if not valid_acc.empty:
            ax1.barh(valid_acc['Model'], valid_acc['Final_Accuracy'])
            ax1.set_xlabel('Final Accuracy')
            ax1.set_title('Model Final Accuracy Comparison')
            ax1.set_xlim(0, 1)
    
    # Plot losses
    ax2 = axes[1]
    if 'Final_Loss' in comparison_df.columns:
        valid_loss = comparison_df.dropna(subset=['Final_Loss'])
        if not valid_loss.empty:
            ax2.barh(valid_loss['Model'], valid_loss['Final_Loss'])
            ax2.set_xlabel('Final Loss')
            ax2.set_title('Model Final Loss Comparison')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 ML Model Comparison Toolkit</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Neural Network Models</h2>', unsafe_allow_html=True)
    
    # Initialize registry
    registry = NeuralNetworkRegistry()
    working_models = get_working_models()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model category selection
    category = st.sidebar.selectbox(
        "Select Model Category",
        ["Feedforward", "Convolutional", "Generative"]
    )
    
    # Task type selection based on category
    if category == "Feedforward":
        task_options = ["Classification", "Regression"]
    elif category == "Convolutional":
        task_options = ["Image Classification"]
    else:  # Generative
        task_options = ["Unsupervised"]
    
    task_type = st.sidebar.selectbox("Select Task Type", task_options)
    
    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Dataset Source",
        ["Synthetic Data", "Your Datasets"]
    )
    
    # Dataset selection
    if dataset_source == "Your Datasets":
        available_datasets = get_available_datasets()
        all_datasets = {}
        if task_type in ["Classification", "Regression"]:
            task_key = task_type
            all_datasets.update(available_datasets.get(task_key, {}))
        
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
        if task_type == "Classification":
            dataset_options = ["Binary Classification", "Multi-class Classification", "Complex Classification"]
        elif task_type == "Regression":
            dataset_options = ["Linear Regression", "Nonlinear Regression", "Complex Regression"]
        elif task_type == "Image Classification":
            dataset_options = ["Synthetic Images"]
        else:  # Unsupervised
            dataset_options = ["High-Dimensional Data"]
        
        dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    # Model selection - only show working models
    category_key = category.lower()
    if category_key in working_models:
        available_models = getattr(registry, f'get_{category_key}_models')()
        working_model_keys = working_models[category_key]
        
        # Filter to only working models
        filtered_models = {k: v for k, v in available_models.items() if k in working_model_keys}
        model_names = list(filtered_models.keys())
        
        selected_models = st.sidebar.multiselect(
            "Select Models to Compare",
            model_names,
            default=model_names[:2] if len(model_names) >= 2 else model_names
        )
    else:
        selected_models = []
        st.sidebar.warning(f"No working models available for {category}")
    
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
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    max_epochs = st.sidebar.slider("Max Epochs", 5, 100, 20)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=1)
    
    # Data preprocessing options
    st.sidebar.subheader("Preprocessing")
    scale_features = st.sidebar.checkbox("Scale Features", value=True)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    
    # Main content
    if st.sidebar.button("Train Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
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
                X, y, feature_names = generate_synthetic_data(dataset_name, task_type)
                if y is not None:
                    df = pd.DataFrame(X.reshape(X.shape[0], -1) if X.ndim > 2 else X, 
                                    columns=feature_names if X.ndim <= 2 else [f'Pixel_{i}' for i in range(X.shape[1] * X.shape[2])])
                    df['target'] = y
                else:
                    df = pd.DataFrame(X, columns=feature_names)
                info = {}
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", X.shape[0])
        with col2:
            if X.ndim == 2:
                st.metric("Features", X.shape[1])
            else:
                st.metric("Shape", f"{X.shape[1]}x{X.shape[2]}")
        with col3:
            if y is not None and task_type != "Regression":
                st.metric("Classes", len(np.unique(y)))
            elif y is not None:
                st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
            else:
                st.metric("Task", "Unsupervised")
        with col4:
            st.metric("Category", category)
        
        # Show dataset preview
        with st.expander("Dataset Preview"):
            if X.ndim == 2:  # Regular tabular data
                preview_df = df.head(10)
                st.dataframe(preview_df, use_container_width=True)
                
                # Basic statistics
                st.subheader("Dataset Statistics")
                st.dataframe(df.describe(), use_container_width=True)
            else:  # Image data
                st.write(f"Image data shape: {X.shape}")
                # Show a few sample images
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                for i in range(5):
                    axes[i].imshow(X[i], cmap='gray')
                    axes[i].set_title(f"Sample {i+1}")
                    axes[i].axis('off')
                st.pyplot(fig)
        
        # Data preprocessing
        if X.ndim == 2 and scale_features:  # Only scale tabular data
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
        else:
            X_processed = X.copy()
        
        # Split data if supervised
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test = train_test_split(X_processed, test_size=test_size, random_state=42)
            y_train, y_test = None, None
        
        # Train models
        st.subheader("Model Training Results")
        
        trainer = NeuralNetworkTrainer()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(selected_models))
            
            try:
                # Set hyperparameters
                hyperparams = {
                    'maxEpochs': max_epochs,
                    'batchSize': batch_size,
                    'learningRate': learning_rate
                }
                
                # Model-specific hyperparameters
                if model_name == 'mlp':
                    hyperparams['hiddenLayers'] = [64, 32]
                elif model_name == 'ann':
                    hyperparams['layers'] = [64, 32]
                elif model_name == 'backpropagation':
                    hyperparams['layers'] = [64, 32]
                elif model_name == 'gan':
                    hyperparams['latentDim'] = 10
                
                result = trainer.train_model(
                    model_name, X_train, y_train, category_key, hyperparams, verbose=False
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
            
            # Plot training history
            st.subheader("Training History")
            fig_history = plot_training_history(successful_results)
            if fig_history:
                st.pyplot(fig_history)
            
            # Plot model comparison
            if len(successful_results) > 1:
                st.subheader("Performance Comparison")
                fig_comparison = plot_model_comparison(comparison_df)
                if fig_comparison:
                    st.pyplot(fig_comparison)
            
            # Best model
            best_model = trainer.get_best_model('final_accuracy')
            if best_model:
                st.subheader("🏆 Best Performing Model")
                best_acc = best_model.get('final_accuracy', 'N/A')
                if isinstance(best_acc, float):
                    best_acc = f"{best_acc:.4f}"
                st.success(f"**{best_model['model_display_name']}** - Final Accuracy: {best_acc}")
            
            # Detailed results
            with st.expander("Detailed Results"):
                for result in successful_results:
                    st.write(f"**{result['model_display_name']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        final_acc = result.get('final_accuracy', 'N/A')
                        final_loss = result.get('final_loss', 'N/A')
                        if isinstance(final_acc, float):
                            final_acc = f"{final_acc:.4f}"
                        if isinstance(final_loss, float):
                            final_loss = f"{final_loss:.4f}"
                        st.write(f"- Final Accuracy: {final_acc}")
                        st.write(f"- Final Loss: {final_loss}")
                    with col2:
                        st.write(f"- Model Type: {result['model_type']}")
                        st.write(f"- Category: {result['model_category']}")
                        if 'converged' in result:
                            st.write(f"- Converged: {result['converged']}")
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
        
        all_models = registry.get_all_models()
        working_model_keys = working_models
        
        for category_name, models_info in all_models.items():
            if models_info:
                st.write(f"**{category_name.title()} Models:**")
                for model_name, info in models_info.items():
                    status = "✅ Working" if model_name in working_model_keys.get(category_name, []) else "⚠️ May have issues"
                    with st.expander(f"{info['name']} ({info['type']}) - {status}"):
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Type:** {info['type']}")
                        st.write(f"**Task Types:** {', '.join(info.get('task_types', []))}")
                        st.write("**Hyperparameters:**")
                        for param, values in info['hyperparameters'].items():
                            st.write(f"- {param}: {values}")

if __name__ == "__main__":
    main()