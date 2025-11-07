"""
Unsupervised Learning Module for ML Comparison Toolkit

This module provides a comprehensive collection of unsupervised learning algorithms
organized into categories: clustering, dimensionality reduction, manifold learning,
and anomaly detection.

Available Models:
- Clustering: K-Means, DBSCAN, Gaussian Mixture, Hierarchical, Agglomerative, Spectral, Mean Shift (7 models)
- Dimensionality Reduction: PCA, ICA (2 models)
- Manifold Learning: t-SNE (1 model)
- Anomaly Detection: Isolation Forest (1 model)

Total: 11 unsupervised learning models
"""

from .unsupervised_learning import (
    UnsupervisedLearningRegistry,
    UnsupervisedLearningTrainer,
    get_unsupervised_learning_info,
    create_model_comparison_report
)

__version__ = "1.0.0"
__author__ = "ML Comparison Toolkit Team"

__all__ = [
    'UnsupervisedLearningRegistry',
    'UnsupervisedLearningTrainer', 
    'get_unsupervised_learning_info',
    'create_model_comparison_report'
]