"""
Ensemble Learning Page
Redirects to the ensemble learning demo
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and run the ensemble learning demo
import importlib.util
spec = importlib.util.spec_from_file_location("ensemble_demo", 
    os.path.join(os.path.dirname(__file__), '..', 'ensemble_learning', 'streamlit_el_demo.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
