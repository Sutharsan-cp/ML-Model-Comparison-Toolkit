"""
Quick runner for Ensemble Learning Streamlit demo
"""

import subprocess
import sys
import os

def run_demo():
    """Run the Ensemble Learning Streamlit demo"""
    demo_path = os.path.join(os.path.dirname(__file__), 'streamlit_ensemble_demo.py')
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', demo_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
    except FileNotFoundError:
        print("Streamlit not found. Please install it with: pip install streamlit")

if __name__ == "__main__":
    run_demo()