"""
Neural Networks Page
Redirects to the neural networks demo
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if neural networks demo exists
demo_path = os.path.join(os.path.dirname(__file__), '..', 'neural_networks', 'streamlit_neural_networks_demo.py')

if os.path.exists(demo_path):
    # Import and run the neural networks demo
    import importlib.util
    spec = importlib.util.spec_from_file_location("neural_networks_demo", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    # Show placeholder if demo doesn't exist
    import streamlit as st
    st.set_page_config(page_title="Neural Networks", page_icon="🧠", layout="wide")
    st.title("🧠 Neural Networks")
    st.info("Neural Networks demo is coming soon!")
    st.markdown("""
    ### Available Soon:
    - Convolutional Neural Networks (CNN)
    - Recurrent Neural Networks (RNN)
    - Long Short-Term Memory (LSTM)
    - Transformers
    - Autoencoders
    - And more...
    """)
    if st.button("← Back to Dashboard"):
        st.switch_page("ML_Dashboard.py")
