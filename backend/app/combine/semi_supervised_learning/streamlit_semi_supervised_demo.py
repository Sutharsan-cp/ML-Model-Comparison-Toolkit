"""
Streamlit UI for Semi-Supervised Learning (production-ready)

- Synthetic dataset generator (classification/blobs) + scaling
- Labeled/Unlabeled split
- Train any available models from registry
- Rich metrics table; choose ranking metric
- Best-model visualizations:
    * Confusion matrix heatmap
    * ROC curve (binary / multiclass OvR)
    * Precision-Recall curve (binary / per-class macro for multiclass)
    * 2D decision landscape via PCA (if features > 2)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from semi_supervised_learning import (
        SemiSupervisedLearningRegistry,
        SemiSupervisedLearningTrainer,
    )
except ImportError as e:
    st.error(f"Failed to import semi_supervised_learning: {e}")
    st.stop()

# ------------------------- Data helpers --------------------------------------
def create_synthetic_dataset(kind: str, n_samples: int, n_features: int, n_classes: int, noise: float, random_state: int):
    if kind == "Classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=random_state,
        )
    else:
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            n_features=n_features,
            cluster_std=1.0 + noise * 2,
            random_state=random_state,
        )
    X = StandardScaler().fit_transform(X)
    return X, y

def pca_2d(X: np.ndarray):
    if X.shape[1] > 2:
        p = PCA(n_components=2, random_state=0)
        return p.fit_transform(X)
    return X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(len(X))])


# ------------------------- Plot helpers --------------------------------------
def plot_data_distribution(X, y, title):
    if X.shape[1] >= 2:
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str), title=title, 
                        labels={"x": "Feature 1", "y": "Feature 2"})
    else:
        # For 1D data, create a histogram
        df = pd.DataFrame({'feature': X[:, 0], 'class': y.astype(str)})
        fig = px.histogram(df, x='feature', color='class', title=title, 
                          barmode='overlay', opacity=0.7)
    return fig

def plot_labeled_unlabeled(X_lab, y_lab, X_unlab):
    if X_lab.shape[1] < 2:
        return None
    fig = go.Figure()
    for c in np.unique(y_lab):
        mask = y_lab == c
        fig.add_trace(go.Scatter(x=X_lab[mask, 0], y=X_lab[mask, 1], mode="markers", 
                               name=f"Labeled {c}", marker=dict(size=7)))
    if len(X_unlab) > 0:
        fig.add_trace(go.Scatter(x=X_unlab[:, 0], y=X_unlab[:, 1], mode="markers", 
                               name="Unlabeled", marker=dict(size=6, symbol="x")))
    fig.update_layout(title="Labeled vs Unlabeled (2D view)", xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names):
    fig = go.Figure(data=go.Heatmap(
        z=cm, 
        x=class_names, 
        y=class_names, 
        text=cm, 
        texttemplate="%{text}", 
        hovertemplate="Pred %{x}<br>True %{y}<br>Count %{z}<extra></extra>",
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig

def plot_roc(y_true, y_proba, n_classes):
    if y_proba is None:
        return None
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        if n_classes == 2:
            # binary ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", 
                                   name=f"ROC (AUC = {roc_auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", 
                                   name="Chance", line=dict(dash="dash")))
            fig.update_layout(title=f"ROC Curve (AUC = {roc_auc:.3f})", 
                            xaxis_title="False Positive Rate", 
                            yaxis_title="True Positive Rate")
            return fig
        else:
            # OvR macro average
            Y = label_binarize(y_true, classes=np.arange(n_classes))
            fig = go.Figure()
            for k in range(n_classes):
                fpr, tpr, _ = roc_curve(Y[:, k], y_proba[:, k])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", 
                                       name=f"Class {k} (AUC = {roc_auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", 
                                   name="Chance", line=dict(dash="dash")))
            fig.update_layout(title="ROC Curve (One-vs-Rest)", 
                            xaxis_title="False Positive Rate", 
                            yaxis_title="True Positive Rate")
            return fig
    except Exception as e:
        st.warning(f"Could not create ROC curve: {e}")
        return None

def plot_pr(y_true, y_proba, n_classes):
    if y_proba is None:
        return None
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if n_classes == 2:
            p, r, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines", 
                                   name=f"AP = {avg_precision:.3f}"))
            fig.update_layout(title=f"Precision-Recall (AP = {avg_precision:.3f})", 
                            xaxis_title="Recall", yaxis_title="Precision")
            return fig
        else:
            from sklearn.preprocessing import label_binarize
            Y = label_binarize(y_true, classes=np.arange(n_classes))
            fig = go.Figure()
            for k in range(n_classes):
                p, r, _ = precision_recall_curve(Y[:, k], y_proba[:, k])
                avg_precision = average_precision_score(Y[:, k], y_proba[:, k])
                fig.add_trace(go.Scatter(x=r, y=p, mode="lines", 
                                       name=f"Class {k} (AP = {avg_precision:.3f})"))
            fig.update_layout(title="Precision-Recall Curve (One-vs-Rest)", 
                            xaxis_title="Recall", yaxis_title="Precision")
            return fig
    except Exception as e:
        st.warning(f"Could not create PR curve: {e}")
        return None

def plot_decision_landscape(model, X_train, y_train, X_test, y_test):
    """PCA to 2D then decision surface over that 2D space (approximate)."""
    if X_train.shape[1] < 2 or model is None:
        return None
    
    try:
        # project both train+test for a cleaner boundary
        X_all = np.vstack([X_train, X_test])
        p = PCA(n_components=2, random_state=0)
        X2 = p.fit_transform(X_all)
        X2_train = X2[:len(X_train)]
        X2_test = X2[len(X_train):]

        # Create mesh grid
        x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
        y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), 
                            np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Use original model if it works with 2D data
        try:
            if hasattr(model, "predict"):
                Z = model.predict(grid)
            else:
                return None
        except Exception:
            # Model might not work with 2D data, use a simple classifier as proxy
            from sklearn.ensemble import RandomForestClassifier
            proxy_model = RandomForestClassifier(n_estimators=10, random_state=42)
            proxy_model.fit(X2_train, y_train)
            Z = proxy_model.predict(grid)
        
        Z = Z.reshape(xx.shape)

        fig = go.Figure()
        
        # Add contour
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 50), 
            y=np.linspace(y_min, y_max, 50), 
            z=Z, 
            showscale=False, 
            opacity=0.4, 
            colorscale='Viridis',
            contours_showlines=False
        ))
        
        # plot train points
        for c in np.unique(y_train):
            mask = y_train == c
            fig.add_trace(go.Scatter(
                x=X2_train[mask, 0], y=X2_train[mask, 1], 
                mode="markers", 
                name=f"Train {c}", 
                marker=dict(size=8, line=dict(width=1, color='black'))
            ))
        
        # plot test points  
        if len(X2_test) > 0:
            fig.add_trace(go.Scatter(
                x=X2_test[:, 0], y=X2_test[:, 1], 
                mode="markers", 
                name="Test", 
                marker=dict(size=8, symbol="x", line=dict(width=1, color='black'))
            ))
            
        fig.update_layout(
            title="Decision Landscape (PCA → 2D)", 
            xaxis_title="PC1", 
            yaxis_title="PC2"
        )
        return fig
    except Exception as e:
        st.warning(f"Could not create decision landscape: {e}")
        return None


# ------------------------- Streamlit App -------------------------------------
def main():
    st.set_page_config(page_title="Semi-Supervised Learning Demo", layout="wide")
    st.title("🔄 Semi-Supervised Learning: Model Comparison")
    st.caption("Generate data → split labeled/unlabeled → train → rank by metric → visualize the winner.")

    # Session state
    if "trainer" not in st.session_state:
        st.session_state.trainer = SemiSupervisedLearningTrainer()
    if "registry" not in st.session_state:
        st.session_state.registry = SemiSupervisedLearningRegistry()
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "results" not in st.session_state:
        st.session_state.results = []

    # Sidebar: dataset
    st.sidebar.header("Dataset")
    dataset_type = st.sidebar.selectbox("Kind", ["Classification", "Blobs"])
    n_samples = st.sidebar.slider("Samples", 200, 4000, 800, step=100)
    n_features = st.sidebar.slider("Features", 2, 50, 6)
    n_classes = st.sidebar.slider("Classes", 2, 8, 3)
    noise = st.sidebar.slider("Noise", 0.0, 0.4, 0.1, step=0.01)
    random_state = st.sidebar.number_input("Random state", 0, 10_000, 42)

    st.sidebar.header("Semi-Supervised Split")
    labeled_ratio = st.sidebar.slider("Labeled ratio", 0.02, 0.6, 0.1)
    test_ratio = st.sidebar.slider("Test ratio", 0.1, 0.6, 0.3)

    if st.sidebar.button("🎲 Generate dataset"):
        with st.spinner("Generating dataset..."):
            X, y = create_synthetic_dataset(dataset_type, n_samples, n_features, n_classes, noise, random_state)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=random_state)
            X_lab, X_unlab, y_lab, y_unlab = st.session_state.trainer.prepare_semi_supervised_data(
                X_tr, y_tr, labeled_ratio=labeled_ratio, random_state=random_state)
            st.session_state.dataset = dict(
                X=X, y=y, 
                X_train=X_tr, y_train=y_tr, 
                X_test=X_te, y_test=y_te, 
                X_labeled=X_lab, y_labeled=y_lab, 
                X_unlabeled=X_unlab, y_unlabeled=y_unlab
            )
            st.session_state.results = []
        st.success("Dataset generated ✔")

    # Sidebar: models
    st.sidebar.header("Models")
    
    # Get available models
    registry = st.session_state.registry
    available_categories = {}
    for category in ["graph_based", "self_training", "consistency", "contrastive", "deep_learning"]:
        models = getattr(registry, f"get_{category}_models")()
        if models:
            available_categories[category] = models
    
    if not available_categories:
        st.sidebar.error("No semi-supervised models available. Please check your imports.")
        return
        
    category = st.sidebar.selectbox("Category", list(available_categories.keys()))
    available_models = available_categories[category]
    
    if available_models:
        selected = st.sidebar.multiselect(
            "Select models", 
            list(available_models.keys()), 
            default=list(available_models.keys())[:min(2, len(available_models))]
        )
    else:
        selected = []
        st.sidebar.warning(f"No models available in {category} category")

    # Sidebar: ranking metric
    st.sidebar.header("Ranking")
    ranking_metric = st.sidebar.selectbox(
        "Primary metric",
        ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr", "mcc", "log_loss"],
        help="log_loss is 'lower is better'; others are 'higher is better'",
    )

    # Main: data overview
    if st.session_state.dataset is None:
        st.info("👈 Generate a dataset to begin.")
        return

    ds = st.session_state.dataset
    st.subheader("📊 Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(ds["X"]))
    c2.metric("Features", ds["X"].shape[1])
    c3.metric("Classes", len(np.unique(ds["y"])))
    c4.metric("Labeled", len(ds["X_labeled"]))

    # 2D preview
    col1, col2 = st.columns(2)
    with col1:
        X2_full = pca_2d(ds["X"])
        st.plotly_chart(plot_data_distribution(X2_full, ds["y"], 
                        "Dataset (PCA 2D view)" if ds["X"].shape[1] > 2 else "Dataset"), 
                        use_container_width=True)
    
    with col2:
        X2_lab = pca_2d(ds["X_labeled"])
        X2_unlab = pca_2d(ds["X_unlabeled"])
        split_fig = plot_labeled_unlabeled(X2_lab, ds["y_labeled"], X2_unlab)
        if split_fig:
            st.plotly_chart(split_fig, use_container_width=True)

    # Training
    st.subheader("🚀 Train")
    if st.button("Train selected models", disabled=not selected):
        if not selected:
            st.warning("Please select at least one model to train.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, name in enumerate(selected):
                status_text.text(f"Training {available_models[name]['name']} …")
                
                # Set appropriate hyperparameters for each category
                hyperparams = {}
                if category == "graph_based":
                    hyperparams = {"kernel": "rbf", "gamma": 1.0, "max_iter": 200}
                    if name == "label_spreading":
                        hyperparams.update({"alpha": 0.2})
                elif category == "self_training":
                    if name == "self_training":
                        hyperparams = {"threshold": 0.8, "max_iter": 10}
                    elif name == "semi_supervised_svm":
                        hyperparams = {"C": 1.0, "kernel": "rbf", "max_iter": 200}
                elif category in ["consistency", "contrastive", "deep_learning"]:
                    hyperparams = {"learning_rate": 0.01, "epochs": 5}  # Lightweight for demo
                
                # Train model
                try:
                    res = st.session_state.trainer.train_model(
                        name,
                        ds["X_labeled"],
                        ds["y_labeled"],
                        ds["X_unlabeled"],
                        ds["y_unlabeled"],
                        ds["X_test"],
                        ds["y_test"],
                        category=category,
                        hyperparameters=hyperparams,
                        verbose=False,
                    )
                    results.append(res)
                except Exception as e:
                    st.error(f"Failed to train {name}: {e}")
                    # Add failed result
                    results.append({
                        "model_name": name,
                        "model_display_name": available_models[name]["name"],
                        "model_type": available_models[name]["type"],
                        "model_category": category,
                        "model_instance": None,
                        "training_successful": False,
                        "error": str(e),
                        "metrics": {},
                    })
                
                progress_bar.progress((i + 1) / len(selected))
            
            status_text.text("Training complete ✔")
            st.session_state.results = results

    # Results table
    if st.session_state.results:
        st.subheader("📈 Results")
        
        # Create results dataframe
        rows = []
        for r in st.session_state.results:
            m = r.get("metrics") or {}
            rows.append({
                "Model": r["model_display_name"],
                "Type": r["model_type"],
                "Category": r.get("model_category") or "Unknown",
                "Status": "✅" if r.get("training_successful") else "❌",
                "accuracy": m.get("accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "f1_macro": m.get("f1_macro"),
                "roc_auc_ovr": m.get("roc_auc_ovr"),
                "mcc": m.get("mcc"),
                "log_loss": m.get("log_loss"),
                "Labeled": r.get("labeled_samples"),
                "Unlabeled": r.get("unlabeled_samples"),
                "Error": r.get("error"),
            })
        
        df = pd.DataFrame(rows)
        
        # Rank models
        def metric_val(x):
            return -x if ranking_metric == "log_loss" else x
        
        df["_rank_key"] = df[ranking_metric].apply(lambda v: -np.inf if pd.isna(v) else metric_val(v))
        df = df.sort_values("_rank_key", ascending=False).drop(columns=["_rank_key"])
        
        # Display results
        st.dataframe(df, use_container_width=True)

        # Best model visualization
        successful_results = [r for r in st.session_state.results if r.get("training_successful")]
        if successful_results:
            best_result = st.session_state.trainer.get_best_model(metric=ranking_metric)
            
            if best_result:
                st.subheader("🥇 Best Model")
                b_m = best_result.get("metrics") or {}
                
                # Display metrics
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Model", best_result["model_display_name"])
                c2.metric("Accuracy", f"{(b_m.get('accuracy') or 0):.3f}")
                c3.metric("F1 (macro)", f"{(b_m.get('f1_macro') or 0):.3f}")
                c4.metric("Bal. Acc", f"{(b_m.get('balanced_accuracy') or 0):.3f}")
                c5.metric("ROC AUC (OvR)", "-" if b_m.get("roc_auc_ovr") is None else f"{b_m['roc_auc_ovr']:.3f}")
                c6.metric("MCC", f"{(b_m.get('mcc') or 0):.3f}")

                # Visualizations
                st.markdown("#### Visualizations")
                
                # Confusion matrix
                cm = np.array(b_m.get("confusion_matrix", []))
                if len(cm) > 0:
                    fig_cm = plot_confusion_matrix(cm, class_names=[str(c) for c in sorted(np.unique(ds["y"]))])
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ROC and PR curves
                y_proba = np.array(best_result.get("y_proba")) if best_result.get("y_proba") is not None else None
                n_classes = len(np.unique(ds["y"]))
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_roc = plot_roc(ds["y_test"], y_proba, n_classes)
                    if fig_roc:
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                with col2:
                    fig_pr = plot_pr(ds["y_test"], y_proba, n_classes)
                    if fig_pr:
                        st.plotly_chart(fig_pr, use_container_width=True)

                # Decision landscape
                fig_dec = plot_decision_landscape(
                    best_result["model_instance"], 
                    ds["X_train"], 
                    ds["y_train"], 
                    ds["X_test"], 
                    ds["y_test"]
                )
                if fig_dec:
                    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Tip: try a very small labeled ratio (e.g., 5–10%) to see the benefit of semi-supervised learning."
    )

if __name__ == "__main__":
    main()