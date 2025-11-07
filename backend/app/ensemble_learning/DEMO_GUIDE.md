# Ensemble Learning Streamlit Demo - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Run the Demo
**Windows:**
```bash
run_demo.bat
```

**Linux/Mac:**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

**Or manually:**
```bash
streamlit run streamlit_el_demo.py
```

### Step 3: Open Your Browser
The app will automatically open at `http://localhost:8501`

## 📊 Quick Demo Walkthrough

### Example 1: Classification with Synthetic Data

1. **Configure Dataset** (Left Sidebar)
   - Task Type: `Classification`
   - Dataset Source: `Synthetic Dataset`
   - Number of Samples: `500`
   - Number of Features: `10`
   - Number of Classes: `2`
   - Click `🎲 Generate Dataset`

2. **Select Models**
   - Choose: `Random Forest`, `XGBoost`, `Gradient Boosting`

3. **Train**
   - Click `🎯 Train Selected Models`
   - Wait for training to complete (~5-10 seconds)

4. **Analyze Results**
   - View accuracy comparison chart
   - Check best model metrics
   - Examine confusion matrix
   - Download results as CSV

### Example 2: Regression with Custom Data

1. **Prepare Your CSV**
   - Ensure it has numeric features
   - Include a target column
   - Example format:
     ```
     feature1,feature2,feature3,target
     1.2,3.4,5.6,10.5
     2.3,4.5,6.7,15.2
     ...
     ```

2. **Upload Dataset**
   - Task Type: `Regression`
   - Dataset Source: `Upload Custom Dataset`
   - Upload your CSV file
   - Select target column
   - Enable "Normalize Features"
   - Click `📊 Process Dataset`

3. **Select Models**
   - Choose: `Random Forest`, `XGBoost`, `LightGBM`

4. **Train and Compare**
   - Click `🎯 Train Selected Models`
   - Compare R² scores
   - View predictions vs actual plot
   - Analyze residuals

## 🎯 Key Features

### Dataset Options
- ✅ Synthetic data generation with configurable parameters
- ✅ CSV file upload with automatic preprocessing
- ✅ Missing value handling (drop, mean, median)
- ✅ Automatic feature normalization
- ✅ Support for both classification and regression

### Model Training
- ✅ 13 ensemble models (7 classification, 6 regression)
- ✅ Parallel training with progress tracking
- ✅ Automatic hyperparameter configuration
- ✅ Error handling and reporting

### Visualization
- ✅ Dataset distribution plots
- ✅ Performance comparison charts
- ✅ Confusion matrices (classification)
- ✅ Predictions vs actual plots (regression)
- ✅ Residuals analysis (regression)
- ✅ Feature importance charts

### Results Analysis
- ✅ Best model identification
- ✅ Detailed metrics table
- ✅ Side-by-side model comparison
- ✅ CSV export functionality

## 📈 Performance Metrics

### Classification
- **Accuracy**: Overall correctness (0-1, higher is better)
- **Precision**: Positive prediction accuracy (0-1, higher is better)
- **Recall**: True positive rate (0-1, higher is better)
- **F1 Score**: Balance of precision and recall (0-1, higher is better)

### Regression
- **R² Score**: Variance explained (-∞ to 1, higher is better, 1 is perfect)
- **RMSE**: Root Mean Squared Error (0 to ∞, lower is better)
- **MAE**: Mean Absolute Error (0 to ∞, lower is better)

## 🎨 UI Components

### Sidebar (Left)
- **Configuration**: Task type, dataset source
- **Dataset Parameters**: Samples, features, classes, noise
- **Training Configuration**: Test ratio
- **Model Selection**: Choose models to train

### Main Area
- **Dataset Overview**: Statistics and visualizations
- **Selected Models**: Model information cards
- **Training Controls**: Train and clear buttons
- **Results**: Tables, charts, and detailed analysis

## 💡 Tips for Best Results

### Dataset Size
- **Small** (200-500 samples): Fast training, good for testing
- **Medium** (500-1000 samples): Balanced performance
- **Large** (1000+ samples): Better model performance, slower training

### Model Selection
- **Quick Test**: Random Forest only (~2 seconds)
- **Comparison**: Random Forest + XGBoost + Gradient Boosting (~10 seconds)
- **Comprehensive**: All models (~30-60 seconds)

### Hyperparameter Tuning
Default parameters are optimized for:
- Fast training
- Good baseline performance
- Stability across different datasets

For production use, consider:
- Increasing `n_estimators` (e.g., 200-500)
- Tuning `learning_rate` (0.01-0.3)
- Adjusting `max_depth` (3-15)

## 🔧 Troubleshooting

### Issue: App won't start
**Solution**: 
```bash
pip install --upgrade streamlit
streamlit --version  # Should be 1.28.0 or higher
```

### Issue: Import errors
**Solution**:
```bash
pip install -r requirements_streamlit.txt
```

### Issue: Models not appearing
**Solution**: Ensure you're in the correct directory
```bash
cd backend/app/ensemble_learning
python -c "from ensemble_learning import EnsembleLearningRegistry; print(len(EnsembleLearningRegistry().get_all_models()))"
```

### Issue: Slow performance
**Solutions**:
- Reduce number of samples
- Select fewer models
- Use LightGBM (fastest)
- Close other applications

### Issue: CSV upload fails
**Solutions**:
- Ensure CSV has headers
- Check for non-numeric features
- Verify target column exists
- Try "Fill with mean" for missing values

## 📚 Model Descriptions

### Random Forest
- **Best for**: Baseline, general-purpose
- **Speed**: Fast ⚡⚡⚡
- **Accuracy**: Good ⭐⭐⭐
- **Pros**: Robust, handles non-linear data, feature importance
- **Cons**: Can be memory intensive

### XGBoost
- **Best for**: Maximum accuracy
- **Speed**: Medium ⚡⚡
- **Accuracy**: Excellent ⭐⭐⭐⭐⭐
- **Pros**: State-of-the-art, regularization, handles missing values
- **Cons**: More complex, requires tuning

### Gradient Boosting
- **Best for**: High accuracy with careful tuning
- **Speed**: Medium ⚡⚡
- **Accuracy**: Excellent ⭐⭐⭐⭐
- **Pros**: Sequential learning, often high accuracy
- **Cons**: Slower than Random Forest, prone to overfitting

### LightGBM
- **Best for**: Large datasets, speed
- **Speed**: Very Fast ⚡⚡⚡⚡
- **Accuracy**: Excellent ⭐⭐⭐⭐
- **Pros**: Fast, memory efficient, handles large data
- **Cons**: Requires careful tuning for small datasets

### Bagging
- **Best for**: Variance reduction
- **Speed**: Fast ⚡⚡⚡
- **Accuracy**: Good ⭐⭐⭐
- **Pros**: Reduces overfitting, parallel training
- **Cons**: May not improve bias

### Stacking
- **Best for**: Maximum performance
- **Speed**: Slow ⚡
- **Accuracy**: Excellent ⭐⭐⭐⭐⭐
- **Pros**: Combines model strengths, often best performance
- **Cons**: Slowest, most complex, requires more data

### CatBoost (Classification only)
- **Best for**: Categorical features
- **Speed**: Medium ⚡⚡
- **Accuracy**: Excellent ⭐⭐⭐⭐
- **Pros**: Handles categories automatically, robust
- **Cons**: Classification only in this demo

## 🎓 Learning Resources

### Understanding Ensemble Methods
1. **Bagging**: Reduces variance by averaging multiple models
2. **Boosting**: Reduces bias by sequentially correcting errors
3. **Stacking**: Combines different model types for best results

### When to Use Each
- **Random Forest**: Start here, good baseline
- **XGBoost**: When you need best performance
- **LightGBM**: When you have large datasets
- **Stacking**: When you want to squeeze out maximum accuracy

### Interpreting Results
- **High training, low test accuracy**: Overfitting
- **Low training, low test accuracy**: Underfitting
- **Similar train/test accuracy**: Good generalization

## 🔄 Workflow Examples

### Workflow 1: Quick Model Comparison
```
1. Generate synthetic data (500 samples, 10 features)
2. Select Random Forest, XGBoost, Gradient Boosting
3. Train models
4. Compare accuracy/R² scores
5. Identify best model
```

### Workflow 2: Custom Data Analysis
```
1. Upload your CSV file
2. Select target column
3. Enable normalization
4. Process dataset
5. Select all available models
6. Train and compare
7. Download results
```

### Workflow 3: Hyperparameter Testing
```
1. Generate dataset
2. Train with default parameters
3. Note best model
4. Modify hyperparameters in code
5. Retrain
6. Compare results
```

## 📞 Support

For help:
1. Check `STREAMLIT_README.md` for detailed documentation
2. Review `README.md` for module documentation
3. Check error messages in the UI
4. Verify all dependencies are installed

## 🎉 Have Fun!

Experiment with different:
- Dataset sizes and complexities
- Model combinations
- Task types (classification vs regression)
- Custom datasets

The best way to learn is by trying different configurations and observing the results!

---

**Made with ❤️ for the ML Comparison Toolkit**