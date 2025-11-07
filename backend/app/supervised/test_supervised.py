"""
Test script for supervised learning models
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import our supervised module
from supervised import SupervisedModelRegistry, SupervisedModelTrainer

def test_classification_models():
    """Test classification models"""
    print("Testing Classification Models")
    print("=" * 50)
    
    # Generate sample classification data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SupervisedModelRegistry()
    trainer = SupervisedModelTrainer()
    
    # Get available classification models
    classification_models = registry.get_classification_models()
    print(f"Available classification models: {len(classification_models)}")
    
    # Test a few models
    test_models = list(classification_models.keys())
    available_test_models = [m for m in test_models if m in classification_models]
    
    if available_test_models:
        print(f"Testing models: {available_test_models}")
        results = trainer.train_multiple_models(
            available_test_models, X_train, y_train, X_test, y_test, 'classification'
        )
        
        for result in results:
            if result['training_successful']:
                print(f"✅ {result['model_display_name']}: Train={result['train_score']:.3f}, Test={result.get('test_score', 'N/A')}")
            else:
                print(f"❌ {result['model_display_name']}: {result['error']}")
    else:
        print("No test models available for classification")
    
    print()

def test_regression_models():
    """Test regression models"""
    print("Testing Regression Models")
    print("=" * 50)
    
    # Generate sample regression data
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize registry and trainer
    registry = SupervisedModelRegistry()
    trainer = SupervisedModelTrainer()
    
    # Get available regression models
    regression_models = registry.get_regression_models()
    print(f"Available regression models: {len(regression_models)}")
    
    # Test a few models
    test_models = list(regression_models.keys())
    available_test_models = [m for m in test_models if m in regression_models]
    
    if available_test_models:
        print(f"Testing models: {available_test_models}")
        results = trainer.train_multiple_models(
            available_test_models, X_train, y_train, X_test, y_test, 'regression'
        )
        
        for result in results:
            if result['training_successful']:
                print(f"✅ {result['model_display_name']}: Train={result['train_score']:.3f}, Test={result.get('test_score', 'N/A')}")
            else:
                print(f"❌ {result['model_display_name']}: {result['error']}")
    else:
        print("No test models available for regression")
    
    print()

def main():
    """Main test function"""
    print("Supervised Learning Models Test")
    print("=" * 60)
    
    try:
        test_classification_models()
        test_regression_models()
        
        # Show summary
        registry = SupervisedModelRegistry()
        all_models = registry.get_all_models()
        
        print("Summary:")
        print(f"Total Classification Models: {len(all_models['classification'])}")
        print(f"Total Regression Models: {len(all_models['regression'])}")
        
        print("\nClassification Models:")
        for name, info in all_models['classification'].items():
            print(f"  - {info['name']} ({info['type']})")
        
        print("\nRegression Models:")
        for name, info in all_models['regression'].items():
            print(f"  - {info['name']} ({info['type']})")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()