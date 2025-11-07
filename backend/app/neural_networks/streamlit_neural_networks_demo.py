"""
Streamlit Demo for Neural Networks Module
Interactive web interface for testing and comparing neural network models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_digits, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the neural_networks module to path
sys.path.append(os.path.dirname(__file__))

try:
    from neural_networks import (
        NeuralNetworksRegistry, 
        NeuralNetworksTrainer,
        get_neural_networks_info,
        create_model_comparison_report
    )
except ImportError as e:
    st.error(f"Error importing neural networks module: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Neural Networks Demo",
    page_icon="🧠",
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
        background-color: #f0f2f6;
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
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title
    st.markdown('<h1 class="main-header">🧠 Neural Networks Comparison Toolkit</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'trainer' not in st.session_state:
        st.session_state.trainer = NeuralNetworksTrainer()
    if 'registry' not in st.session_state:
        st.session_state.registry = NeuralNetworksRegistry()
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
                ["Classification (Synthetic)", "Regression (Synthetic)", "Wine Dataset", "Digits Dataset"]
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
            default=['basic_nn']
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
        max_epochs = st.slider("Max epochs (for applicable models):", 5, 100, 20, 5)
        batch_size = st.selectbox("Batch size:", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning rate:", [0.001, 0.01, 0.1], index=1)
        
        # Action buttons
        st.subheader("🚀 Actions")
        train_button = st.button("🏋️ Train Selected Models", type="primary")
        
        # Quick test all models button
        if st.button("⚡ Quick Test All Models", help="Test all 17 models with minimal epochs"):
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
        
        if X is not None and y is not None:
            # Display dataset info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f'<div class="metric-card"><strong>Samples:</strong> {len(X)}</div>', unsafe_allow_html=True)
            with col_info2:
                st.markdown(f'<div class="metric-card"><strong>Features:</strong> {X.shape[1] if X.ndim > 1 else 1}</div>', unsafe_allow_html=True)
            with col_info3:
                if y is not None:
                    unique_labels = len(np.unique(y))
                    st.markdown(f'<div class="metric-card"><strong>Classes/Targets:</strong> {unique_labels}</div>', unsafe_allow_html=True)
            
            # Dataset preview
            if st.checkbox("Show dataset preview"):
                preview_df = pd.DataFrame(X[:10])
                if y is not None:
                    preview_df['Target'] = y[:10]
                st.dataframe(preview_df)
            
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
                for i, (model_name, category) in enumerate(all_model_list):
                    status_text.text(f"Quick testing {model_name} ({i+1}/{len(all_model_list)})...")
                    
                    # Prepare appropriate data
                    X_train_model, X_test_model, y_train_model, y_test_model = prepare_model_data(
                        X, y, category, 0.3, st.session_state.trainer
                    )
                    
                    # Minimal hyperparameters for quick testing
                    hyperparameters = {'maxEpochs': 2}
                    
                    if model_name in ['rnn', 'lstm', 'gru']:
                        hyperparameters['hiddenDim'] = 16
                    elif model_name == 'autoencoder':
                        hyperparameters['encodingDim'] = min(5, X.shape[1] // 2)
                    elif model_name == 'gan':
                        hyperparameters['latentDim'] = 8
                    elif model_name == 'transformer':
                        hyperparameters.update({
                            'vocabSize': 50,
                            'dModel': 32,
                            'nHeads': 2,
                            'nLayers': 1,
                            'batchSize': 8
                        })
                    
                    try:
                        result = st.session_state.trainer.train_model(
                            model_name, X_train_model, y_train_model, X_test_model, y_test_model,
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
                display_training_results(results)
            
            # Training section
            elif train_button and selected_models:
                st.markdown('<h2 class="sub-header">🏋️ Training Results</h2>', unsafe_allow_html=True)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, (model_name, category) in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    
                    # Prepare appropriate data based on model category
                    X_train_model, X_test_model, y_train_model, y_test_model = prepare_model_data(
                        X, y, category, test_size, st.session_state.trainer
                    )
                    
                    # Set hyperparameters
                    hyperparameters = {}
                    model_info = st.session_state.registry.get_model_by_name(model_name, category)
                    
                    if model_info and 'hyperparameters' in model_info:
                        if 'maxEpochs' in model_info['hyperparameters']:
                            hyperparameters['maxEpochs'] = max_epochs
                        if 'batchSize' in model_info['hyperparameters']:
                            hyperparameters['batchSize'] = batch_size
                        if 'learningRate' in model_info['hyperparameters']:
                            hyperparameters['learningRate'] = learning_rate
                    
                    # Add model-specific hyperparameters
                    if model_name in ['rnn', 'lstm', 'gru']:
                        hyperparameters['hiddenDim'] = 32
                    elif model_name == 'autoencoder':
                        hyperparameters['encodingDim'] = min(8, X.shape[1] // 2)
                    elif model_name == 'gan':
                        hyperparameters['latentDim'] = 16
                    elif model_name == 'transformer':
                        hyperparameters.update({
                            'vocabSize': 100,
                            'dModel': 64,
                            'nHeads': 2,
                            'nLayers': 1
                        })
                    
                    try:
                        result = st.session_state.trainer.train_model(
                            model_name, X_train_model, y_train_model, X_test_model, y_test_model,
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
                display_training_results(results)
        
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
                st.write(f"**{category.replace('_', ' ').title()}:** {len(models)} models")
        
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
                best_model = st.session_state.trainer.get_best_model('test_accuracy')
                if best_model:
                    accuracy = best_model.get("test_accuracy", "N/A")
                    if accuracy != "N/A" and accuracy is not None:
                        st.markdown(f'<div class="success-card"><strong>🏆 Best Model:</strong><br>{best_model["model_display_name"]}<br><strong>Accuracy:</strong> {accuracy:.4f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-card"><strong>🏆 Best Model:</strong><br>{best_model["model_display_name"]}<br><strong>Status:</strong> Trained Successfully</div>', unsafe_allow_html=True)

def prepare_model_data(X, y, category, test_size, trainer):
    """Prepare appropriate data based on model category"""
    
    if category == 'basic_nn':
        # Use tabular data as-is
        return trainer.prepare_data(X, y, test_size=test_size, random_state=42)
    
    elif category == 'deep_learning':
        # Convert to image-like data if needed
        if X.ndim == 2:
            # Reshape tabular data to image-like format
            # Try to make it square-ish
            n_features = X.shape[1]
            img_size = int(np.sqrt(n_features))
            if img_size * img_size < n_features:
                img_size += 1
            
            # Pad with zeros if needed
            padded_features = img_size * img_size
            if padded_features > n_features:
                X_padded = np.zeros((X.shape[0], padded_features))
                X_padded[:, :n_features] = X
                X = X_padded
            
            # Reshape to image format
            X = X.reshape(X.shape[0], img_size, img_size)
        
        return trainer.prepare_data(X, y, test_size=test_size, scale_features=False, random_state=42)
    
    elif category == 'recurrent':
        # Convert to sequence data
        if X.ndim == 2:
            # Reshape tabular data to sequence format
            seq_length = min(10, X.shape[1])
            n_features = X.shape[1] // seq_length
            if n_features == 0:
                n_features = 1
                seq_length = X.shape[1]
            
            # Reshape to (samples, seq_length, features)
            X_seq = X[:, :seq_length * n_features].reshape(X.shape[0], seq_length, n_features)
            
            # Create sequence labels (repeat labels for each timestep)
            y_seq = np.repeat(y[:, np.newaxis], seq_length, axis=1)
        else:
            X_seq = X
            y_seq = y
        
        # Split the data
        n_samples = X_seq.shape[0]
        n_test = int(n_samples * test_size)
        
        X_train = X_seq[:-n_test] if n_test > 0 else X_seq
        X_test = X_seq[-n_test:] if n_test > 0 else X_seq[:5]  # At least 5 samples for test
        y_train = y_seq[:-n_test] if n_test > 0 else y_seq
        y_test = y_seq[-n_test:] if n_test > 0 else y_seq[:5]
        
        return X_train, X_test, y_train, y_test
    
    elif category == 'generative':
        # For unsupervised models like autoencoder
        X_train, X_test, _, _ = trainer.prepare_data(X, y, test_size=test_size, random_state=42)
        return X_train, None, None, None
    
    elif category == 'specialized':
        # For transformer - convert to token sequences
        if X.ndim == 2:
            # Convert tabular data to token-like sequences
            # Normalize and convert to integers
            X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
            X_tokens = (X_norm * 99).astype(int) + 1  # Scale to 1-100 range
            
            # Limit sequence length
            seq_len = min(10, X_tokens.shape[1])
            X_tokens = X_tokens[:, :seq_len]
            
            # Create target sequences (shifted input for language modeling)
            y_tokens = np.roll(X_tokens, -1, axis=1)
        else:
            X_tokens = X
            y_tokens = y
        
        # Split the data
        n_samples = X_tokens.shape[0]
        n_test = int(n_samples * test_size)
        
        X_train = X_tokens[:-n_test] if n_test > 0 else X_tokens
        X_test = X_tokens[-n_test:] if n_test > 0 else X_tokens[:5]
        y_train = y_tokens[:-n_test] if n_test > 0 else y_tokens
        y_test = y_tokens[-n_test:] if n_test > 0 else y_tokens[:5]
        
        return X_train, X_test, y_train, y_test
    
    else:
        # Default to tabular data
        return trainer.prepare_data(X, y, test_size=test_size, random_state=42)

def prepare_dataset(dataset_type, dataset_selection):
    """Prepare dataset based on user selection"""
    
    if dataset_type == "Built-in Dataset":
        if dataset_selection == "Classification (Synthetic)":
            X, y = make_classification(
                n_samples=1000, n_features=20, n_classes=3, 
                n_redundant=0, n_informative=15, random_state=42
            )
            info = "Synthetic classification dataset with 3 classes"
            
        elif dataset_selection == "Regression (Synthetic)":
            X, y = make_regression(
                n_samples=1000, n_features=20, noise=0.1, random_state=42
            )
            info = "Synthetic regression dataset"
            
        elif dataset_selection == "Wine Dataset":
            from sklearn.datasets import load_wine
            data = load_wine()
            X, y = data.data, data.target
            info = "Wine recognition dataset (3 classes, 13 features)"
            
        elif dataset_selection == "Digits Dataset":
            from sklearn.datasets import load_digits
            data = load_digits()
            X, y = data.data, data.target
            info = "Handwritten digits dataset (10 classes, 64 features)"
        
        return X, y, info
    
    else:  # Custom dataset upload
        if dataset_selection is not None:
            try:
                df = pd.read_csv(dataset_selection)
                
                # Let user select target column
                st.write("**Dataset Preview:**")
                st.dataframe(df.head())
                
                target_column = st.selectbox("Select target column:", df.columns)
                
                if target_column:
                    X = df.drop(columns=[target_column]).values
                    y = df[target_column].values
                    
                    # Handle categorical targets
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    info = f"Custom dataset: {len(df)} samples, {len(df.columns)-1} features"
                    return X, y, info
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return None, None, None
        
        return None, None, None

def display_training_results(results):
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
                'Accuracy': result.get('test_accuracy', 'N/A'),
                'MSE': result.get('test_mse', 'N/A'),
                'Training Samples': result.get('training_samples', 'N/A'),
                'Status': '✅ Success'
            })
        else:
            summary_data.append({
                'Model': result['model_display_name'],
                'Category': result['model_type'],
                'Accuracy': 'N/A',
                'MSE': 'N/A',
                'Training Samples': result.get('training_samples', 'N/A'),
                'Status': '❌ Failed'
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance visualization
    successful_results = [r for r in results if r['training_successful'] and r.get('test_accuracy') is not None]
    
    if successful_results:
        st.subheader("📈 Performance Comparison")
        
        # Accuracy comparison
        model_names = [r['model_display_name'] for r in successful_results]
        accuracies = [r.get('test_accuracy', 0) for r in successful_results]
        categories = [r.get('model_type', 'unknown') for r in successful_results]
        
        # Create DataFrame for better visualization
        viz_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Category': categories
        })
        
        # Bar chart with category colors
        fig = px.bar(
            viz_df, x='Model', y='Accuracy', color='Category',
            title="Model Accuracy Comparison by Category",
            labels={'Accuracy': 'Test Accuracy'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category performance summary
        if len(set(categories)) > 1:
            category_stats = viz_df.groupby('Category')['Accuracy'].agg(['mean', 'max', 'count']).round(4)
            category_stats.columns = ['Average Accuracy', 'Best Accuracy', 'Models Tested']
            
            st.subheader("📊 Performance by Category")
            st.dataframe(category_stats, use_container_width=True)
        
        # Training samples vs accuracy scatter plot
        if len(successful_results) > 1:
            training_samples = [r.get('training_samples', 0) for r in successful_results]
            
            fig_scatter = px.scatter(
                x=training_samples, y=accuracies, color=categories,
                hover_name=model_names,
                title="Training Samples vs Accuracy",
                labels={'x': 'Training Samples', 'y': 'Accuracy', 'color': 'Category'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
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
                    if result.get('test_accuracy') is not None:
                        st.write(f"**Test Accuracy:** {result['test_accuracy']:.4f}")
                    if result.get('test_mse') is not None:
                        st.write(f"**Test MSE:** {result['test_mse']:.4f}")
                else:
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")
            
            with col2:
                if result.get('hyperparameters'):
                    st.write("**Hyperparameters:**")
                    for param, value in result['hyperparameters'].items():
                        st.write(f"- {param}: {value}")
                
                if result.get('classification_report'):
                    st.write("**Classification Report Available**")
                    if st.button(f"Show Report for {result['model_display_name']}", key=f"report_{result['model_name']}"):
                        st.json(result['classification_report'])

def show_model_architecture_info():
    """Show information about different neural network architectures"""
    
    st.markdown('<h2 class="sub-header">🏗️ Neural Network Architectures</h2>', unsafe_allow_html=True)
    
    architectures = {
        "Basic Neural Networks": {
            "Perceptron": "Single-layer linear classifier, the simplest form of neural network",
            "Multi-Layer Perceptron (MLP)": "Feedforward network with multiple hidden layers",
            "Artificial Neural Network (ANN)": "General feedforward network with customizable architecture",
            "Backpropagation Network": "Network trained using backpropagation algorithm"
        },
        "Deep Learning Networks": {
            "Convolutional Neural Network (CNN)": "Specialized for image processing with convolutional layers",
            "ResNet": "Deep network with residual connections to avoid vanishing gradients",
            "VGG": "Deep network with small convolutional filters",
            "Inception": "Network with multi-scale convolutions in parallel",
            "DenseNet": "Network where each layer connects to every other layer",
            "EfficientNet": "Optimized network balancing depth, width, and resolution",
            "MobileNet": "Lightweight network optimized for mobile devices"
        },
        "Recurrent Networks": {
            "RNN": "Basic recurrent network for sequence data",
            "LSTM": "Long Short-Term Memory network for long sequences",
            "GRU": "Gated Recurrent Unit, simpler alternative to LSTM"
        },
        "Generative Networks": {
            "Autoencoder": "Network for dimensionality reduction and reconstruction",
            "GAN": "Generative Adversarial Network for generating synthetic data"
        },
        "Specialized Networks": {
            "Transformer": "Attention-based network for sequence-to-sequence tasks"
        }
    }
    
    for category, models in architectures.items():
        st.subheader(category)
        for model, description in models.items():
            st.write(f"**{model}:** {description}")

# Sidebar navigation
with st.sidebar:
    st.markdown("---")
    if st.button("📚 Architecture Guide"):
        show_model_architecture_info()

if __name__ == "__main__":
    main()