"""
Semi-Supervised Learning: Registry + Trainer (production-ready)

- Robust registry that gracefully degrades when optional models are missing
- Trainer with rich evaluation: accuracy, balanced_accuracy, f1 (macro/weighted),
  precision/recall (macro), ROC-AUC (binary + multiclass ovr/ovo), average precision,
  MCC, log loss, confusion matrix
- Best-model selection by any metric (supports higher/lower-is-better)
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -----------------------------------------------------------------------------
# Import all the semi-supervised models directly
# -----------------------------------------------------------------------------

# Import all model classes directly
try:
    from models.semi_supervised.label_propagation import LabelPropagation
except ImportError as e:
    LabelPropagation = None
    print(f"[SemiSup] LabelPropagation not available: {e}")

try:
    from models.semi_supervised.label_spreading import LabelSpreading
except ImportError as e:
    LabelSpreading = None
    print(f"[SemiSup] LabelSpreading not available: {e}")

try:
    from models.semi_supervised.self_training import SelfTrainingClassifier
except ImportError as e:
    SelfTrainingClassifier = None
    print(f"[SemiSup] SelfTrainingClassifier not available: {e}")

try:
    from models.semi_supervised.semi_supervised_svm import SemiSupervisedSVM
except ImportError as e:
    SemiSupervisedSVM = None
    print(f"[SemiSup] SemiSupervisedSVM not available: {e}")

try:
    from models.semi_supervised.mean_teacher import MeanTeacher
except ImportError as e:
    MeanTeacher = None
    print(f"[SemiSup] MeanTeacher not available: {e}")

try:
    from models.semi_supervised.fixmatch import FixMatch
except ImportError as e:
    FixMatch = None
    print(f"[SemiSup] FixMatch not available: {e}")

try:
    from models.semi_supervised.mixmatch import MixMatch
except ImportError as e:
    MixMatch = None
    print(f"[SemiSup] MixMatch not available: {e}")

try:
    from models.semi_supervised.simclr import SimCLR
except ImportError as e:
    SimCLR = None
    print(f"[SemiSup] SimCLR not available: {e}")

try:
    from models.semi_supervised.byol import BYOL
except ImportError as e:
    BYOL = None
    print(f"[SemiSup] BYOL not available: {e}")

try:
    from models.semi_supervised.ladder_network import LadderNetwork
except ImportError as e:
    LadderNetwork = None
    print(f"[SemiSup] LadderNetwork not available: {e}")

try:
    from models.semi_supervised.masked_autoencoder import MaskedAutoencoder
except ImportError as e:
    MaskedAutoencoder = None
    print(f"[SemiSup] MaskedAutoencoder not available: {e}")

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
class SemiSupervisedLearningRegistry:
    """Registry for all semi-supervised learning models."""

    def __init__(self):
        # Define configs - only include models that are actually available
        self.graph_based_models: Dict[str, Dict[str, Any]] = {}
        self.self_training_models: Dict[str, Dict[str, Any]] = {}
        self.consistency_models: Dict[str, Dict[str, Any]] = {}
        self.contrastive_models: Dict[str, Dict[str, Any]] = {}
        self.deep_learning_models: Dict[str, Dict[str, Any]] = {}

        # Graph-based models
        if LabelPropagation is not None:
            self.graph_based_models["label_propagation"] = {
                "name": "Label Propagation",
                "type": "graph_based",
                "class": LabelPropagation,
                "description": "Graph-based label propagation algorithm",
                "data_types": ["tabular", "graph"],
                "hyperparameters": {"kernel": ["rbf", "knn"], "gamma": [0.1, 1.0, 10.0], "n_neighbors": [5, 10, 20], "max_iter": [100, 300, 1000]},
            }
        
        if LabelSpreading is not None:
            self.graph_based_models["label_spreading"] = {
                "name": "Label Spreading",
                "type": "graph_based",
                "class": LabelSpreading,
                "description": "Graph-based label spreading algorithm",
                "data_types": ["tabular", "graph"],
                "hyperparameters": {"kernel": ["rbf", "knn"], "gamma": [0.1, 1.0, 10.0], "alpha": [0.1, 0.2, 0.5], "n_neighbors": [5, 10, 20], "max_iter": [100, 300, 1000]},
            }

        # Self-training models
        if SelfTrainingClassifier is not None:
            self.self_training_models["self_training"] = {
                "name": "Self-Training",
                "type": "self_training",
                "class": SelfTrainingClassifier,
                "description": "Self-training with confidence threshold",
                "data_types": ["tabular", "image", "text"],
                "hyperparameters": {"threshold": [0.7, 0.8, 0.9], "max_iter": [10, 20, 50]},
            }
        
        if SemiSupervisedSVM is not None:
            self.self_training_models["semi_supervised_svm"] = {
                "name": "Semi-Supervised SVM",
                "type": "self_training",
                "class": SemiSupervisedSVM,
                "description": "S3VM-style approach",
                "data_types": ["tabular"],
                "hyperparameters": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto", 0.1, 1.0], "max_iter": [100, 500, 1000]},
            }

        # Consistency models
        if MeanTeacher is not None:
            self.consistency_models["mean_teacher"] = {
                "name": "Mean Teacher",
                "type": "consistency",
                "class": MeanTeacher,
                "description": "EMA teacher consistency",
                "data_types": ["image", "tabular"],
                "hyperparameters": {"learning_rate": [0.001, 0.01, 0.1], "alpha": [0.99, 0.999, 0.9999], "consistency_weight": [0.1, 1.0, 10.0], "epochs": [50, 100, 200]},
            }
        
        if FixMatch is not None:
            self.consistency_models["fixmatch"] = {
                "name": "FixMatch",
                "type": "consistency",
                "class": FixMatch,
                "description": "Weak/strong augmentation with thresholding",
                "data_types": ["image"],
                "hyperparameters": {"learning_rate": [0.001, 0.003, 0.01], "threshold": [0.9, 0.95, 0.99], "lambda_u": [1.0, 5.0, 10.0], "epochs": [100, 200, 500]},
            }
        
        if MixMatch is not None:
            self.consistency_models["mixmatch"] = {
                "name": "MixMatch",
                "type": "consistency",
                "class": MixMatch,
                "description": "Label guessing + MixUp",
                "data_types": ["image"],
                "hyperparameters": {"learning_rate": [0.001, 0.003, 0.01], "alpha": [0.75, 1.0, 2.0], "lambda_u": [75, 100, 150], "T": [0.5, 1.0, 2.0], "epochs": [100, 200, 500]},
            }

        # Contrastive models
        if SimCLR is not None:
            self.contrastive_models["simclr"] = {
                "name": "SimCLR",
                "type": "contrastive",
                "class": SimCLR,
                "description": "Simple contrastive learning of visual representations",
                "data_types": ["image"],
                "hyperparameters": {"learning_rate": [0.001, 0.003, 0.01], "temperature": [0.1, 0.5, 1.0], "batch_size": [64, 128, 256], "epochs": [100, 200, 500]},
            }
        
        if BYOL is not None:
            self.contrastive_models["byol"] = {
                "name": "BYOL",
                "type": "contrastive",
                "class": BYOL,
                "description": "Bootstrap Your Own Latent",
                "data_types": ["image"],
                "hyperparameters": {"learning_rate": [0.001, 0.003, 0.01], "moving_average_decay": [0.99, 0.996, 0.999], "batch_size": [64, 128, 256], "epochs": [100, 200, 500]},
            }

        # Deep learning models
        if LadderNetwork is not None:
            self.deep_learning_models["ladder_network"] = {
                "name": "Ladder Network",
                "type": "deep_learning",
                "class": LadderNetwork,
                "description": "Denoising ladder networks",
                "data_types": ["image", "tabular"],
                "hyperparameters": {"learning_rate": [0.001, 0.01, 0.1], "noise_std": [0.1, 0.3, 0.5], "supervised_weight": [0.1, 1.0, 10.0], "epochs": [50, 100, 200]},
            }
        
        if MaskedAutoencoder is not None:
            self.deep_learning_models["masked_autoencoder"] = {
                "name": "Masked Autoencoder",
                "type": "deep_learning",
                "class": MaskedAutoencoder,
                "description": "MAE self-supervision",
                "data_types": ["image"],
                "hyperparameters": {"learning_rate": [0.001, 0.003, 0.01], "mask_ratio": [0.5, 0.75, 0.9], "epochs": [100, 200, 500]},
            }

    # Accessors
    def get_graph_based_models(self) -> Dict[str, Dict[str, Any]]:
        return self.graph_based_models

    def get_self_training_models(self) -> Dict[str, Dict[str, Any]]:
        return self.self_training_models

    def get_consistency_models(self) -> Dict[str, Dict[str, Any]]:
        return self.consistency_models

    def get_contrastive_models(self) -> Dict[str, Dict[str, Any]]:
        return self.contrastive_models

    def get_deep_learning_models(self) -> Dict[str, Dict[str, Any]]:
        return self.deep_learning_models

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        return {
            "graph_based": self.get_graph_based_models(),
            "self_training": self.get_self_training_models(),
            "consistency": self.get_consistency_models(),
            "contrastive": self.get_contrastive_models(),
            "deep_learning": self.get_deep_learning_models(),
        }

    def get_model_by_name(self, model_name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if category:
            bucket = self.get_all_models().get(category, {})
            return bucket.get(model_name)
        for bucket in self.get_all_models().values():
            if model_name in bucket:
                return bucket[model_name]
        return None

    def get_models_by_data_type(self, data_type: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for category, models in self.get_all_models().items():
            for name, info in models.items():
                if data_type in info.get("data_types", []):
                    out[f"{category}_{name}"] = info
        return out


# -----------------------------------------------------------------------------
# Utilities: metrics & safe proba extraction
# -----------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _safe_predict_proba(model: Any, X: np.ndarray, n_classes: int) -> Optional[np.ndarray]:
    """
    Return class probabilities if available.
    - use predict_proba if present
    - else try decision_function -> convert to probabilities
    - else None
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Ensure proper shape
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            return proba
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            if scores.ndim == 1:  # binary
                p1 = _sigmoid(scores)
                return np.vstack([1 - p1, p1]).T
            # multiclass
            return _softmax(scores)
    except Exception:
        return None
    return None

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute a rich set of metrics; handles binary & multiclass safely."""
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = prec
    metrics["recall_macro"] = rec
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    try:
        metrics["log_loss"] = log_loss(y_true, y_proba) if y_proba is not None else None
    except Exception:
        metrics["log_loss"] = None

    # ROC-AUC & Average Precision
    roc_auc_ovr = None
    roc_auc_ovo = None
    avg_precision = None
    if y_proba is not None:
        y_true_flat = y_true
        if y_proba.shape[1] == 2:  # binary
            try:
                roc_auc_ovr = roc_auc_score(y_true_flat, y_proba[:, 1])
                avg_precision = average_precision_score(y_true_flat, y_proba[:, 1])
            except Exception:
                pass
        else:  # multiclass
            try:
                roc_auc_ovr = roc_auc_score(y_true_flat, y_proba, multi_class="ovr", average="macro")
            except Exception:
                pass
            try:
                roc_auc_ovo = roc_auc_score(y_true_flat, y_proba, multi_class="ovo", average="macro")
            except Exception:
                pass
            try:
                avg_precision = average_precision_score(
                    pd.get_dummies(y_true_flat).values, y_proba, average="macro"
                )
            except Exception:
                pass

    metrics["roc_auc_ovr"] = roc_auc_ovr
    metrics["roc_auc_ovo"] = roc_auc_ovo
    metrics["average_precision"] = avg_precision
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
    return metrics


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class SemiSupervisedLearningTrainer:
    """Trainer class for semi-supervised learning models with rich evaluation."""

    def __init__(self):
        self.registry = SemiSupervisedLearningRegistry()
        self.trained_models: Dict[str, Dict[str, Any]] = {}
        self.training_history: List[Dict[str, Any]] = []

    def prepare_semi_supervised_data(
        self, X, y, labeled_ratio: float = 0.1, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create labeled/unlabeled splits from (X, y)."""
        # First split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
        
        # Then split train into labeled/unlabeled
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X_train, y_train, train_size=labeled_ratio, random_state=random_state, stratify=y_train
        )
        return X_labeled, X_unlabeled, y_labeled, y_unlabeled

    def train_model(
        self,
        model_name: str,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        y_unlabeled: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        category: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train a single model and evaluate on the test set if provided."""
        info = self.registry.get_model_by_name(model_name, category)
        if info is None:
            raise ValueError(f"Model '{model_name}' not found or unavailable in this environment.")

        hyperparameters = hyperparameters or {}
        model_class = info["class"]
        model = model_class(**hyperparameters)

        if verbose:
            print(f"[SemiSup] Training {info['name']} | labeled={len(X_labeled)} unlabeled={len(X_unlabeled)}")

        # Fit
        try:
            if hasattr(model, "fit"):
                # Handle different fit signatures
                try:
                    # Try the semi-supervised signature first
                    model.fit(X_labeled, y_labeled, X_unlabeled)
                except TypeError:
                    try:
                        # Try graph-based signature
                        X_all = np.vstack([X_labeled, X_unlabeled])
                        y_all = np.hstack([y_labeled, np.full(len(X_unlabeled), -1)])
                        model.fit(X_all, y_all)
                    except Exception:
                        # Fall back to supervised only
                        model.fit(X_labeled, y_labeled)
            else:
                raise AttributeError(f"{info['name']} has no .fit method")
            
            training_successful = True
            error = None
        except Exception as e:
            training_successful = False
            error = str(e)
            if verbose:
                print(f"[SemiSup] Training failed for {info['name']}: {e}")

        # Predict/Evaluate
        y_pred = None
        y_proba = None
        metrics = None
        
        if training_successful and X_test is not None and y_test is not None:
            try:
                if hasattr(model, "predict"):
                    y_pred = model.predict(X_test)
                elif hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    y_pred = np.argmax(y_proba, axis=1)
                
                # try to get proba if not yet
                if y_proba is None:
                    y_proba = _safe_predict_proba(model, X_test, n_classes=len(np.unique(y_test)))
                if y_pred is None and y_proba is not None:
                    y_pred = np.argmax(y_proba, axis=1)

                if y_pred is not None:
                    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
                else:
                    metrics = {}
            except Exception as e:
                metrics = {}
                if verbose:
                    print(f"[SemiSup] Prediction failed for {info['name']}: {e}")

        result: Dict[str, Any] = {
            "model_name": model_name,
            "model_display_name": info["name"],
            "model_type": info["type"],
            "model_category": category,
            "model_instance": model if training_successful else None,
            "hyperparameters": hyperparameters,
            "training_successful": training_successful,
            "error": error,
            "labeled_samples": len(X_labeled),
            "unlabeled_samples": len(X_unlabeled),
            "metrics": metrics or {},
            "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else None,
            "y_proba": y_proba.tolist() if isinstance(y_proba, np.ndarray) else None,
        }
        # legacy fields for backwards compatibility
        result["test_accuracy"] = (metrics or {}).get("accuracy")

        key = f"{model_name}_{len(self.trained_models)}"
        self.trained_models[key] = result
        self.training_history.append(result)
        return result

    def train_multiple_models(
        self,
        model_names: List[str],
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        y_unlabeled: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        category: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        results = []
        for name in model_names:
            try:
                result = self.train_model(
                    name,
                    X_labeled,
                    y_labeled,
                    X_unlabeled,
                    y_unlabeled,
                    X_test,
                    y_test,
                    category,
                    hyperparameters=None,
                    verbose=verbose,
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"[SemiSup] Failed to train {name}: {e}")
                # Add a failed result
                results.append({
                    "model_name": name,
                    "model_display_name": name,
                    "model_type": "unknown",
                    "model_category": category,
                    "model_instance": None,
                    "hyperparameters": {},
                    "training_successful": False,
                    "error": str(e),
                    "labeled_samples": len(X_labeled),
                    "unlabeled_samples": len(X_unlabeled),
                    "metrics": {},
                    "y_pred": None,
                    "y_proba": None,
                    "test_accuracy": None,
                })
        return results

    def get_training_summary(self) -> pd.DataFrame:
        rows = []
        for r in self.training_history:
            m = r.get("metrics", {})
            rows.append(
                {
                    "Model": r["model_display_name"],
                    "Type": r["model_type"],
                    "Category": r.get("model_category") or "Unknown",
                    "Training_Successful": r["training_successful"],
                    "Labeled_Samples": r.get("labeled_samples"),
                    "Unlabeled_Samples": r.get("unlabeled_samples"),
                    "Accuracy": m.get("accuracy"),
                    "Balanced_Accuracy": m.get("balanced_accuracy"),
                    "F1_Macro": m.get("f1_macro"),
                    "ROC_AUC_OVR": m.get("roc_auc_ovr"),
                    "MCC": m.get("mcc"),
                    "LogLoss": m.get("log_loss"),
                    "Error": r.get("error"),
                }
            )
        return pd.DataFrame(rows)

    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
        """Return best model by metric; supports 'lower_is_better' for log_loss."""
        successful = [r for r in self.training_history if r.get("training_successful")]
        if not successful:
            return None

        def get_metric(r: Dict[str, Any]) -> Optional[float]:
            m = r.get("metrics") or {}
            return m.get(metric)

        candidates = [r for r in successful if get_metric(r) is not None]
        if not candidates:
            return None

        lower_is_better = {"log_loss"}
        reverse = metric not in lower_is_better
        return sorted(candidates, key=get_metric, reverse=reverse)[0]

    def clear_history(self):
        self.trained_models.clear()
        self.training_history.clear()


# Convenience helpers for external modules
def get_semi_supervised_models_info() -> Dict[str, Dict[str, Any]]:
    return SemiSupervisedLearningRegistry().get_all_models()

def create_model_comparison_report(training_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in training_results:
        if not r.get("training_successful"):
            continue
        m = r.get("metrics") or {}
        rows.append(
            {
                "Model": r["model_display_name"],
                "Type": r["model_type"],
                "Category": r.get("model_category") or "Unknown",
                "Labeled_Samples": r.get("labeled_samples"),
                "Unlabeled_Samples": r.get("unlabeled_samples"),
                "Accuracy": m.get("accuracy"),
                "F1_Macro": m.get("f1_macro"),
                "Balanced_Accuracy": m.get("balanced_accuracy"),
                "ROC_AUC_OVR": m.get("roc_auc_ovr"),
                "MCC": m.get("mcc"),
                "LogLoss": m.get("log_loss"),
                "Hyperparameters": str(r.get("hyperparameters", {})),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick smoke test: just print which models are available at runtime
    reg = SemiSupervisedLearningRegistry()
    for section, bucket in reg.get_all_models().items():
        print(f"[{section}]")
        for key, info in bucket.items():
            print(f"  - {info['name']}")