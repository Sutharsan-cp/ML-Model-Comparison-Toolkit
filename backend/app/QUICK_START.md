# ML Model Comparison Toolkit - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn tensorflow gym
```

Or use the requirements file:
```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Run the Dashboard

**Option A: Using the launcher (Windows)**
```bash
cd backend/app
run_dashboard.bat
```

**Option B: Direct command**
```bash
cd backend/app
streamlit run ML_Dashboard.py
```

**Option C: From project root**
```bash
streamlit run backend/app/ML_Dashboard.py
```

### Step 3: Explore!

The dashboard will open at `http://localhost:8501`. Click any category to start comparing models!

---

## 📊 What You Can Do

### 🎯 Supervised Learning
1. Click "Supervised Learning"
2. Choose Classification or Regression
3. Generate data or upload CSV
4. Select models (Logistic Regression, Random Forest, SVM, etc.)
5. Train and compare
6. View confusion matrix, ROC curves
7. Download results

### 🌳 Ensemble Learning
1. Click "Ensemble Learning"
2. Choose Classification or Regression
3. Generate data or upload CSV
4. Select ensemble models (Random Forest, XGBoost, etc.)
5. Train and compare
6. View feature importance
7. Download results

### 🤖 Reinforcement Learning
1. Click "Reinforcement Learning"
2. Create Grid World or load Gym environment
3. Select RL algorithms (Q-Learning, DQN, etc.)
4. Set training parameters
5. Train agents
6. View learning curves
7. Compare performance

### 🔄 Semi-Supervised Learning
1. Click "Semi-Supervised Learning"
2. Generate data or upload CSV
3. Set labeled data ratio
4. Select models (Label Propagation, Self-Training, etc.)
5. Train and compare
6. View labeled vs unlabeled visualization

### 🔍 Unsupervised Learning
1. Click "Unsupervised Learning"
2. Generate data or upload CSV
3. Select clustering/DR algorithms
4. Train and visualize
5. Compare silhouette scores

---

## 🎯 Quick Examples

### Example 1: Compare 3 Classification Models (2 minutes)

```bash
# 1. Start dashboard
streamlit run backend/app/ML_Dashboard.py

# 2. In browser:
#    - Click "Supervised Learning"
#    - Task Type: Classification
#    - Click "Generate Dataset"
#    - Select: Logistic Regression, Random Forest, SVM
#    - Click "Train Selected Models"
#    - View results!
```

### Example 2: Train RL Agent (3 minutes)

```bash
# 1. Start dashboard
streamlit run backend/app/ML_Dashboard.py

# 2. In browser:
#    - Click "Reinforcement Learning"
#    - Grid Size: 5
#    - Click "Create Environment"
#    - Select: Q-Learning, SARSA, DQN
#    - Episodes: 100
#    - Click "Train Selected Models"
#    - View learning curves!
```

### Example 3: Cluster Your Data (2 minutes)

```bash
# 1. Start dashboard
streamlit run backend/app/ML_Dashboard.py

# 2. In browser:
#    - Click "Unsupervised Learning"
#    - Upload your CSV or generate data
#    - Select: K-Means, DBSCAN, Hierarchical
#    - Click "Train Selected Models"
#    - View clusters!
```

---

## 📁 Project Structure

```
backend/app/
├── ML_Dashboard.py          ← START HERE
├── pages/                   ← Navigation pages
├── supervised/              ← Supervised learning demo
├── ensemble_learning/       ← Ensemble learning demo
├── reinforcement_learning/  ← RL demo
├── semi_supervised_learning/← Semi-supervised demo
└── unsupervised_learning/   ← Unsupervised demo
```

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Import errors
```bash
# Install all dependencies
pip install -r backend/app/requirements_streamlit.txt
```

### Port already in use
```bash
# Use a different port
streamlit run ML_Dashboard.py --server.port 8502
```

### Page navigation not working
```bash
# Ensure pages/ directory exists
ls backend/app/pages/

# Should show: Supervised.py, Ensemble.py, etc.
```

---

## 💡 Tips

1. **Start with Supervised Learning**: Easiest to understand
2. **Use Synthetic Data**: Quick testing without uploading files
3. **Compare 2-3 Models**: Faster than comparing all models
4. **Download Results**: Save comparisons as CSV for later analysis
5. **Experiment**: Try different parameters and datasets

---

## 🎓 Learning Path

### Beginner
1. Start with **Supervised Learning** (Classification)
2. Try **Ensemble Learning** (Random Forest)
3. Explore **Unsupervised Learning** (K-Means)

### Intermediate
1. **Semi-Supervised Learning** (Label Propagation)
2. **Ensemble Learning** (XGBoost, Gradient Boosting)
3. **Reinforcement Learning** (Q-Learning)

### Advanced
1. **Reinforcement Learning** (DQN, Policy Gradient)
2. **Neural Networks** (when available)
3. Custom datasets and hyperparameter tuning

---

## 📞 Support

For issues:
1. Check individual demo READMEs
2. Review `DASHBOARD_README.md`
3. Check `STREAMLIT_STATUS.md` for status

---

## 🎉 You're Ready!

Run this command and start exploring:

```bash
streamlit run backend/app/ML_Dashboard.py
```

**Happy Model Comparing! 🚀**
