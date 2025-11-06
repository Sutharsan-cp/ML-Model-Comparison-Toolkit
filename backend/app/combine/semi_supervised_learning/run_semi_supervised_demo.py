"""
Quick runner for Semi-Supervised Learning Streamlit demo
"""

import subprocess
import sys
import os

def run_demo():
    """Run the Semi-Supervised Learning Streamlit demo"""
    demo_path = os.path.join(os.path.dirname(__file__), 'streamlit_semi_supervised_demo.py')
    
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