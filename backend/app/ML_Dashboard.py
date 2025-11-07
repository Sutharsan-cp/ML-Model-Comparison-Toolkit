"""
ML Model Comparison Toolkit - Main Dashboard
Central hub for accessing all machine learning model comparison tools
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Toolkit",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .stButton > button {
        width: 100%;
        height: 80px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">🤖 ML Model Comparison Toolkit</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Compare and analyze machine learning models across different categories</p>', unsafe_allow_html=True)

# Introduction
st.info("👋 Welcome! Select a machine learning category below to start comparing models.")

# Center the buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📊 Select a Category")
    
    if st.button("🎯 Supervised Learning", use_container_width=True, help="Classification & Regression models"):
        st.switch_page("pages/Supervised.py")
    
    if st.button("🔍 Unsupervised Learning", use_container_width=True, help="Clustering & Dimensionality Reduction"):
        st.switch_page("pages/Unsupervised.py")
    
    if st.button("🌳 Ensemble Learning", use_container_width=True, help="Random Forest, XGBoost, Gradient Boosting"):
        st.switch_page("pages/Ensemble.py")
    
    if st.button("🤖 Reinforcement Learning", use_container_width=True, help="Q-Learning, DQN, Policy Gradient"):
        st.switch_page("pages/Reinforcement.py")
    
    if st.button("🔄 Semi-Supervised Learning", use_container_width=True, help="Label Propagation, Self-Training"):
        st.switch_page("pages/Semi_Supervised.py")
    
    if st.button("🧠 Neural Networks", use_container_width=True, help="Deep Learning models"):
        st.switch_page("pages/Neural_Network.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Features:</strong> Model Training • Performance Comparison • Visualization • Export Results</p>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
