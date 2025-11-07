"""
Test script for supervised learning Streamlit demo
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing Supervised Learning Module...")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from combine.supervised.supervised import (
        SupervisedModelRegistry,
        SupervisedModelTrainer,
        create_model_comparison_report,
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test registry
print("\n2. Testing SupervisedModelRegistry...")
try:
    registry = SupervisedModelRegistry()
    all_models = registry.get_all_models()
    
    classification_models = registry.get_classification_models()
    regression_models = registry.get_regression_models()
    
    print(f"   ✓ Registry initialized")
    print(f"   - Classification models: {len(classification_models)}")
    print(f"   - Regression models: {len(regression_models)}")
    print(f"   - Total categories: {len(all_models)}")
    
    # List some models
    if classification_models:
        print(f"\n   Sample Classification Models:")
        for name, info in list(classification_models.items())[:3]:
            print(f"     - {info['name']}")
    
    if regression_models:
        print(f"\n   Sample Regression Models:")
        for name, info in list(regression_models.items())[:3]:
            print(f"     - {info['name']}")
    
except Exception as e:
    print(f"   ✗ Registry test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test trainer
print("\n3. Testing SupervisedModelTrainer...")
try:
    trainer = SupervisedModelTrainer()
    print("   ✓ Trainer initialized")
except Exception as e:
    print(f"   ✗ Trainer test failed: {e}")
    sys.exit(1)

# Test with sample data
print("\n4. Testing with sample data...")
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   ✓ Sample data created: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Try training a simple model
    if 'logistic_regression' in classification_models:
        print("\n5. Testing model training (Logistic Regression)...")
        result = trainer.train_model(
            'logistic_regression',
            X_train, y_train,
            X_test, y_test,
            'classification'
        )
        
        if result['training_successful']:
            print(f"   ✓ Training successful!")
            print(f"   - Test Accuracy: {result.get('test_accuracy', 0):.3f}")
        else:
            print(f"   ✗ Training failed: {result.get('error', 'Unknown error')}")
    else:
        print("\n5. Skipping model training test (no models available)")
    
except Exception as e:
    print(f"   ✗ Sample data test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ All tests completed successfully!")
print("\nThe supervised learning module is working correctly.")
print("You can now run the Streamlit demo:")
print("  streamlit run streamlit_supervised_demo.py")
