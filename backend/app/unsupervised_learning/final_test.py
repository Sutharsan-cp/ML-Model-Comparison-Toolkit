"""
Final comprehensive test with all 11 unsupervised learning models
"""

import numpy as np
from sklearn.datasets import make_blobs, make_classification
import sys
import os

sys.path.append(os.path.dirname(__file__))

from unsupervised_learning import UnsupervisedLearningRegistry, UnsupervisedLearningTrainer

def test_all_models():
    """Test all 11 unsupervised learning models"""
    
    print("=" * 80)
    print("FINAL COMPREHENSIVE TEST - ALL 11 UNSUPERVISED LEARNING MODELS")
    print("=" * 80)
    
    registry = UnsupervisedLearningRegistry()
    trainer = UnsupervisedLearningTrainer()
    
    # Prepare different types of data
    print("\n📊 Preparing test datasets...")
    
    # Clustering data
    X_blobs, y_blobs = make_blobs(n_samples=200, centers=4, n_features=6, 
                                  random_state=42, cluster_std=1.5)
    X_train_cluster, X_test_cluster = trainer.prepare_data(X_blobs, test_size=0.3)
    
    # High-dimensional data for dimensionality reduction
    X_classification, y_classification = make_classification(
        n_samples=150, n_features=20, n_classes=3, 
        n_informative=15, n_redundant=0, random_state=42
    )
    X_train_dr, X_test_dr = trainer.prepare_data(X_classification, test_size=0.3)
    
    # Smaller dataset for t-SNE (computationally expensive)
    X_small = X_classification[:100]
    X_train_small, X_test_small = trainer.prepare_data(X_small, test_size=0.3)
    
    all_models = registry.get_all_models()
    
    total_models = 0
    successful_models = 0
    failed_models = 0
    
    print("\n🧪 Testing all models...")
    print("-" * 80)
    
    for category, models in all_models.items():
        print(f"\n📁 Category: {category.upper()}")
        
        for model_name, model_info in models.items():
            total_models += 1
            print(f"   {total_models}. {model_info['name']} ({model_name})...", end=" ")
            
            try:
                # Choose appropriate data based on model type
                if category == 'clustering':
                    X_train, X_test = X_train_cluster, X_test_cluster
                elif category == 'dimensionality_reduction':
                    X_train, X_test = X_train_dr, X_test_dr
                elif category == 'manifold_learning':
                    X_train, X_test = X_train_small, X_test_small  # Smaller for t-SNE
                elif category == 'anomaly_detection':
                    X_train, X_test = X_train_dr, X_test_dr
                
                # Set appropriate hyperparameters
                hyperparameters = {}
                
                if category == 'clustering':
                    if 'n_clusters' in model_info['hyperparameters']:
                        hyperparameters['n_clusters'] = 4
                    if model_name == 'dbscan':
                        hyperparameters = {'eps': 1.0, 'min_samples': 5}
                    elif model_name == 'mean_shift':
                        hyperparameters = {'bandwidth': 2.0}
                
                elif category in ['dimensionality_reduction', 'manifold_learning']:
                    hyperparameters['n_components'] = 3
                    if model_name == 'tsne':
                        hyperparameters.update({
                            'n_iter': 250,  # Reduced for speed
                            'perplexity': min(20, len(X_train)//4),
                            'learning_rate': 200.0
                        })
                
                elif category == 'anomaly_detection':
                    hyperparameters = {
                        'n_estimators': 50,  # Reduced for speed
                        'contamination': 0.1,
                        'random_state': 42
                    }
                
                # Train the model
                result = trainer.train_model(
                    model_name, X_train, X_test,
                    category=category, hyperparameters=hyperparameters,
                    verbose=False
                )
                
                if result['training_successful']:
                    successful_models += 1
                    
                    # Show relevant metrics
                    if category == 'clustering' and result.get('silhouette_score') is not None:
                        print(f"✅ SUCCESS (Silhouette: {result['silhouette_score']:.3f})")
                    elif category in ['dimensionality_reduction', 'manifold_learning'] and result.get('explained_variance_ratio') is not None:
                        print(f"✅ SUCCESS (Explained Var: {result['explained_variance_ratio']:.3f})")
                    elif category == 'anomaly_detection' and result.get('outlier_fraction') is not None:
                        print(f"✅ SUCCESS (Outliers: {result['outlier_fraction']:.3f})")
                    else:
                        print(f"✅ SUCCESS")
                else:
                    failed_models += 1
                    error = result.get('error', 'Unknown error')
                    print(f"❌ FAILED ({error[:50]}...)")
                
            except Exception as e:
                failed_models += 1
                print(f"💥 EXCEPTION ({type(e).__name__}: {str(e)[:50]}...)")
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Total models tested: {total_models}")
    print(f"✅ Successful: {successful_models}")
    print(f"❌ Failed: {failed_models}")
    print(f"📊 Success rate: {successful_models/total_models*100:.1f}%")
    print("=" * 80)
    
    # Print summary by category
    print("\n📋 Summary by Category:")
    for category, models in all_models.items():
        print(f"   {category}: {len(models)} models")
    
    if successful_models == total_models:
        print("\n🎉 ALL MODELS WORKING! 🎉")
    elif successful_models >= total_models * 0.8:
        print("\n✨ Most models working! Great job!")
    else:
        print("\n⚠️  Some models need attention")
    
    return successful_models, total_models

if __name__ == "__main__":
    successful, total = test_all_models()