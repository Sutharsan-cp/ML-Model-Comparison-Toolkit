"""
Reinforcement Learning Page
Redirects to the reinforcement learning demo
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and run the reinforcement learning demo
import importlib.util
spec = importlib.util.spec_from_file_location("rl_demo", 
    os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning', 'streamlit_rl_demo.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
