# Ensemble Learning Streamlit Demo

An interactive web-based dashboard for training and comparing ensemble learning models.

## Features

### 🎯 Core Functionality
- **Dual Task Support**: Both classification and regression tasks
- **Synthetic Data Generation**: Create test datasets with configurable parameters
- **Custom Dataset Upload**: Upload and process your own CSV files
- **13 Ensemble Models**: All models from the ensemble learning module
- **Real-time Training**: Interactive model training with progress tracking
- **Performance Visualization**: Comprehensive charts and metrics

### 📊 Supported Models

#### Classification (7 models)
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- Bagging Classifier
- Stacking Classifier
- LightGBM Classifier
- CatBoost Classifier

#### Regression (6 models)
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Bagging Regressor
- Stacking Regressor
- LightGBM Regressor

## Installation

### Prerequisites
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
```

### Running the Demo

From the project root directory:
```bash
streamlit run backend/app/ensemble_learning/streamlit_el_demo.py
```

Or from the ensemble_learning directory:
```bash
cd backend/app/ensemble_learning
streamlit run streamlit_el_demo.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage Guide

### 1. Configure Dataset

#### Option A: Synthetic Dataset
1. Select **Task Type** (Classification or Regression)
2. Choose **Dataset Source** → "Synthetic Dataset"
3. Configure parameters:
   - Number of Samples (200-2000)
   - Number of Features (2-20)
   - Number of Classes (2-5, for classification)
   - Noise Level (0.0-0.3)
   - Random State (for reproducibility)
4. Click **🎲 Generate Dataset**

#### Option B: Custom Dataset
1. Select **Task Type** (Classification or Regression)
2. Choose **Dataset Source** → "Upload Custom Dataset"
3. Upload a CSV file
4. Select the target column
5. Configure preprocessing:
   - Handle missing values (drop, fill with mean/median)
   - Normalize features (recommended)
6. Click **📊 Process Dataset**

### 2. Select Models

In the sidebar under "Model Selection":
- Choose one or more models to train
- Default: Top 3 models are pre-selected
- You can select all models for comprehensive comparison

### 3. Train Models

1. Click **🎯 Train Selected Models**
2. Watch the progress bar as models train
3. View results automatically when training completes

### 4. Analyze Results

The dashboard provides multiple views:

#### Results Summary Table
- Model names and types
- Training status (success/failure)
- Key performance metrics
- Error messages (if any)

#### Performance Comparison Charts
- **Classification**: Accuracy, F1, Precision, Recall
- **Regression**: R², RMSE, MAE

#### Best Model Analysis
- Top performing model highlighted
- Detailed metrics
- Confusion matrix (classification)
- Predictions vs Actual plot (regression)
- Residuals plot (regression)
- Feature importance (if available)

#### Model Comparison Table
- Side-by-side comparison of all models
- Sortable by any metric
- Downloadable as CSV

## Features in Detail

### Dataset Visualization
- 2D scatter plots for datasets with 2+ features
- Color-coded by class (classification) or target value (regression)
- Interactive Plotly charts with zoom and pan

### Training Configuration
- Adjustable test/train split ratio
- Automatic stratification for classification
- Progress tracking during training
- Error handling and reporting

### Performance Metrics

#### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

#### Regression Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Residuals Plot**: Error distribution analysis

### Model-Specific Features

#### Random Forest
- Fast training
- Feature importance available
- Good baseline performance

#### XGBoost
- State-of-the-art performance
- Regularization support
- Feature importance available

#### Gradient Boosting
- Sequential learning
- Often high accuracy
- Feature importance available

#### Bagging
- Variance reduction
- Uses decision trees as base estimators
- Parallel training

#### Stacking
- Meta-learning approach
- Combines multiple base models
- Often best performance

#### LightGBM
- Very fast training
- Memory efficient
- Handles large datasets well

#### CatBoost (Classification only)
- Automatic categorical feature handling
- Robust to overfitting
- Good default parameters

## Tips for Best Results

### Dataset Preparation
1. **Clean your data**: Handle missing values appropriately
2. **Normalize features**: Enable feature normalization for better results
3. **Balanced classes**: For classification, try to have balanced class distributions
4. **Sufficient samples**: Use at least 200 samples for reliable results

### Model Selection
1. **Start with Random Forest**: Good baseline model
2. **Try XGBoost**: Often achieves best performance
3. **Use Stacking**: For maximum performance (slower training)
4. **Compare multiple models**: Different models work better for different datasets

### Hyperparameter Tuning
The demo uses sensible defaults:
- Random Forest: 100 trees, max depth 10
- XGBoost: 100 estimators, learning rate 0.1
- Gradient Boosting: 100 estimators, learning rate 0.1

For production use, consider:
- Grid search or random search for optimal parameters
- Cross-validation for robust evaluation
- Feature engineering for better performance

## Troubleshooting

### Common Issues

**Issue**: "No models available"
- **Solution**: Ensure the ensemble learning module is properly installed

**Issue**: "Error loading dataset"
- **Solution**: Check CSV format, ensure numeric features, verify target column

**Issue**: "Training failed"
- **Solution**: Check error message, try different hyperparameters, verify data quality

**Issue**: "Visualization not showing"
- **Solution**: Ensure dataset has at least 2 features for 2D plots

### Performance Issues

**Slow training**:
- Reduce number of samples
- Use fewer estimators (e.g., 50 instead of 100)
- Select fewer models to train
- Use LightGBM for faster training

**Memory issues**:
- Reduce dataset size
- Use fewer features
- Train models one at a time

## Advanced Usage

### Custom Hyperparameters

To modify hyperparameters, edit the training section in `streamlit_el_demo.py`:

```python
if model_name == 'randomforest':
    hyperparams = {
        'n_estimators': 200,  # Increase trees
        'max_depth': 15,      # Deeper trees
        'random_state': 42
    }
```

### Adding New Models

1. Implement model in `backend/app/models/ensemble/`
2. Add import in `ensemble_learning.py`
3. Add configuration in registry
4. Model will automatically appear in UI

### Exporting Results

Click **📥 Download Results as CSV** to export:
- Model names and types
- All performance metrics
- Training configuration
- Comparison data

## Keyboard Shortcuts

- **R**: Rerun the app
- **C**: Clear cache
- **Ctrl+C**: Stop the server (in terminal)

## Browser Compatibility

Tested and working on:
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Performance Benchmarks

Typical training times (on standard laptop):
- Random Forest: 1-3 seconds
- XGBoost: 2-5 seconds
- Gradient Boosting: 2-5 seconds
- LightGBM: 1-2 seconds
- Stacking: 5-10 seconds (trains multiple models)

## Screenshots

### Main Dashboard
- Dataset overview with statistics
- Model selection panel
- Training controls

### Results View
- Performance comparison charts
- Best model analysis
- Detailed metrics table

### Visualizations
- Confusion matrix (classification)
- Predictions vs Actual (regression)
- Feature importance charts
- Residuals plots

## Integration with ML Comparison Toolkit

This Streamlit demo is part of the larger ML Comparison Toolkit:
- Uses the same ensemble learning module
- Consistent API and model interface
- Can be integrated with other toolkit components

## Future Enhancements

Planned features:
- [ ] Hyperparameter tuning interface
- [ ] Cross-validation support
- [ ] Model persistence (save/load)
- [ ] Batch prediction interface
- [ ] Advanced feature engineering
- [ ] Model explainability (SHAP values)
- [ ] A/B testing framework

## Support

For issues or questions:
1. Check this README
2. Review the main ensemble learning module documentation
3. Check the ML Comparison Toolkit documentation

## License

Part of the ML Comparison Toolkit project.

## Credits

Built using:
- Streamlit for the web interface
- Plotly for interactive visualizations
- Scikit-learn for data processing
- Custom ensemble learning module

---

**Happy Model Training! 🚀**