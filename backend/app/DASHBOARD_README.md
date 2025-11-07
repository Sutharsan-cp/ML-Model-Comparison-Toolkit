# ML Model Comparison Toolkit - Dashboard

## 🎯 Overview

The ML Dashboard is a centralized hub for accessing all machine learning model comparison tools in the toolkit. It provides a clean, intuitive interface for navigating between different ML categories.

## 📁 File Structure

```
backend/app/
├── ML_Dashboard.py              # Main dashboard (entry point)
├── pages/                       # Streamlit pages directory
│   ├── Supervised.py           # Supervised learning page
│   ├── Unsupervised.py         # Unsupervised learning page
│   ├── Ensemble.py             # Ensemble learning page
│   ├── Reinforcement.py        # Reinforcement learning page
│   ├── Semi_Supervised.py      # Semi-supervised learning page
│   └── Neural_Network.py       # Neural networks page
├── supervised/
│   └── streamlit_supervised_demo.py
├── unsupervised_learning/
│   └── streamlit_unsupervised_demo.py
├── ensemble_learning/
│   └── streamlit_el_demo.py
├── reinforcement_learning/
│   └── streamlit_rl_demo.py
└── semi_supervised_learning/
    └── streamlit_semi_supervised_demo.py
```

## 🚀 How to Run

### Quick Start

```bash
# Navigate to the app directory
cd backend/app

# Run the dashboard
streamlit run ML_Dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Alternative: Run from Project Root

```bash
streamlit run backend/app/ML_Dashboard.py
```

## 🎨 Features

### Main Dashboard
- **Clean Interface**: Centered layout with easy navigation
- **Category Cards**: Six main ML categories with descriptions
- **Responsive Design**: Works on different screen sizes
- **Custom Styling**: Professional look with custom CSS

### Navigation
- **One-Click Access**: Click any category to access its demo
- **Multi-Page App**: Uses Streamlit's native multi-page functionality
- **Back Navigation**: Each page can navigate back to dashboard

## 📊 Available Categories

### 1. 🎯 Supervised Learning
- **Models**: 32 models (19 classification, 13 regression)
- **Features**: 
  - Classification & Regression
  - Confusion Matrix
  - ROC Curves
  - Feature Importance
  - CSV Upload

### 2. 🔍 Unsupervised Learning
- **Models**: Clustering, Dimensionality Reduction, Anomaly Detection
- **Features**:
  - K-Means, DBSCAN, Hierarchical Clustering
  - PCA, t-SNE
  - Isolation Forest, LOF
  - Visualization

### 3. 🌳 Ensemble Learning
- **Models**: 13 models (7 classification, 6 regression)
- **Features**:
  - Random Forest, XGBoost, Gradient Boosting
  - LightGBM, CatBoost
  - Feature Importance
  - Model Comparison

### 4. 🤖 Reinforcement Learning
- **Models**: 8 working models
- **Features**:
  - Q-Learning, SARSA, DQN
  - Policy Gradient, REINFORCE
  - Grid World Environment
  - Gym Integration
  - Learning Curves

### 5. 🔄 Semi-Supervised Learning
- **Models**: 11 models
- **Features**:
  - Label Propagation, Self-Training
  - Graph-based methods
  - Consistency regularization
  - Labeled vs Unlabeled visualization

### 6. 🧠 Neural Networks
- **Status**: Coming soon / Placeholder
- **Planned**: CNN, RNN, LSTM, Transformers

## 🔧 Technical Details

### Multi-Page App Structure

Streamlit automatically detects the `pages/` directory and creates a multi-page app. Each page file:
1. Sets up the Python path
2. Imports the corresponding demo module
3. Executes the demo

### Page Loading

Each page uses `importlib` to dynamically load and execute the demo:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("demo_name", demo_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

### Navigation

Navigation between pages uses Streamlit's `st.switch_page()`:

```python
if st.button("Supervised Learning"):
    st.switch_page("pages/Supervised.py")
```

## 🎯 Usage Examples

### Example 1: Compare Classification Models

1. Run the dashboard: `streamlit run ML_Dashboard.py`
2. Click "🎯 Supervised Learning"
3. Select "Classification" task
4. Generate synthetic data or upload CSV
5. Select models to compare
6. Click "Train Selected Models"
7. View results and download CSV

### Example 2: Train RL Agent

1. Run the dashboard
2. Click "🤖 Reinforcement Learning"
3. Create Grid World environment
4. Select RL algorithms (Q-Learning, DQN, etc.)
5. Set training parameters
6. Train and compare agents
7. View learning curves

### Example 3: Cluster Data

1. Run the dashboard
2. Click "🔍 Unsupervised Learning"
3. Generate or upload data
4. Select clustering algorithms
5. Train and visualize clusters
6. Compare silhouette scores

## 🛠️ Customization

### Modify Dashboard Appearance

Edit `ML_Dashboard.py` to customize:
- Colors and styling (CSS section)
- Button layout
- Category descriptions
- Footer content

### Add New Categories

1. Create new demo in appropriate directory
2. Add button to `ML_Dashboard.py`
3. Create page file in `pages/` directory
4. Update this README

## 📝 Notes

- All demos run independently
- Each demo has its own session state
- Navigation preserves demo state within session
- Refreshing returns to dashboard

## 🐛 Troubleshooting

### Issue: Page not found
**Solution**: Ensure the `pages/` directory is in the same location as `ML_Dashboard.py`

### Issue: Import errors
**Solution**: Check that all demo files exist in their respective directories

### Issue: Module not found
**Solution**: Verify sys.path is set correctly in page files

## 🎉 Quick Test

Test the dashboard:

```bash
# Run the dashboard
streamlit run backend/app/ML_Dashboard.py

# You should see:
# - Main dashboard with 6 category buttons
# - Clicking any button navigates to that demo
# - Each demo runs independently
```

## 📚 Additional Resources

- [Streamlit Multi-Page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [Streamlit Navigation](https://docs.streamlit.io/library/api-reference/navigation)
- Individual demo READMEs in their respective directories

---

**Created**: November 7, 2025  
**Status**: ✅ Fully Functional  
**Version**: 1.0
