# Semi-Supervised Learning Module

This directory contains the comprehensive semi-supervised learning module for the ML Model Comparison Toolkit.

## Structure

- `semi_supervised_learning.py` - Main semi-supervised module with registry and trainer
- `test_semi_supervised_learning.py` - Comprehensive test suite
- `streamlit_semi_supervised_demo.py` - Interactive Streamlit dashboard
- `examples/` - Example usage and tutorials

## Quick Start

```bash
cd backend/app/semi_supervised_learning
python test_semi_supervised_learning.py
```

## Available Algorithms

### Graph-Based Methods
- Label Propagation
- Label Spreading

### Self-Training Methods
- Self-Training
- Semi-Supervised SVM

### Consistency Regularization
- Mean Teacher
- Virtual Adversarial Training
- FixMatch
- MixMatch

### Contrastive Learning
- SimCLR
- BYOL
- VICReg
- DINO

### Deep Learning Methods
- Ladder Network
- Masked Autoencoder
- SWAY

## Features

- 15 different semi-supervised algorithms
- Multiple data types support (tabular, image, text)
- Interactive training visualization
- Performance comparison tools
- Custom dataset support
- Labeled/unlabeled data handling

## Data Requirements

Semi-supervised learning requires both labeled and unlabeled data:
- **Labeled data**: Small amount of data with ground truth labels
- **Unlabeled data**: Large amount of data without labels
- **Typical ratio**: 10-20% labeled, 80-90% unlabeled