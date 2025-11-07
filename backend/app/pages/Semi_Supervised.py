"""
Semi-Supervised Learning Page
Redirects to the semi-supervised learning demo
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and run the semi-supervised learning demo
import importlib.util
spec = importlib.util.spec_from_file_location("semi_supervised_demo", 
    os.path.join(os.path.dirname(__file__), '..', 'semi_supervised_learning', 'streamlit_semi_supervised_demo.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
