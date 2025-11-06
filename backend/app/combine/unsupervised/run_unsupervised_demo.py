"""
Script to run the Streamlit demo for unsupervised learning models
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main function to run the Streamlit demo"""
    print("🔍 ML Model Comparison Toolkit - Unsupervised Learning Demo")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    app_dir = Path(__file__).parent
    
    print(f"Current directory: {current_dir}")
    print(f"App directory: {app_dir}")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if unsupervised.py exists
    unsupervised_file = app_dir / "unsupervised.py"
    if not unsupervised_file.exists():
        print("❌ unsupervised.py not found!")
        return
    
    # Check if streamlit demo exists
    demo_file = app_dir / "streamlit_unsupervised_demo.py"
    if not demo_file.exists():
        print("❌ streamlit_unsupervised_demo.py not found!")
        return
    
    print("✅ All files found")
    print("✅ Requirements satisfied")
    
    # Change to app directory
    os.chdir(app_dir)
    
    print("\n🚀 Starting Streamlit demo...")
    print("The demo will open in your default web browser")
    print("Press Ctrl+C to stop the demo")
    print("-" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_unsupervised_demo.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()