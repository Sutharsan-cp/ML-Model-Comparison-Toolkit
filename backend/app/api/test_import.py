import sys
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.append(str(app_dir))

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTesting imports...")

# Test preprocessing imports
print("\n1. Testing Preprocessing Imports:")
try:
    from preprocessing.data_cleaning import DataCleaner
    print("✓ DataCleaner imported successfully")
except ImportError as e:
    print(f"✗ DataCleaner import error: {e}")

try:
    from preprocessing.data_loader import DataLoader
    print("✓ DataLoader imported successfully")
except ImportError as e:
    print(f"✗ DataLoader import error: {e}")

try:
    from preprocessing.encoding import Encoder
    print("✓ Encoder imported successfully")
except ImportError as e:
    print(f"✗ Encoder import error: {e}")

try:
    from preprocessing.scaling import Scaler
    print("✓ Scaler imported successfully")
except ImportError as e:
    print(f"✗ Scaler import error: {e}")

try:
    from preprocessing.feature_engineering import FeatureEngineer
    print("✓ FeatureEngineer imported successfully")
except ImportError as e:
    print(f"✗ FeatureEngineer import error: {e}")

try:
    from preprocessing.dimensionality_reduction import DimensionalityReducer
    print("✓ DimensionalityReducer imported successfully")
except ImportError as e:
    print(f"✗ DimensionalityReducer import error: {e}")

try:
    from preprocessing.target_processing import TargetProcessor
    print("✓ TargetProcessor imported successfully")
except ImportError as e:
    print(f"✗ TargetProcessor import error: {e}")

try:
    from preprocessing.type_conversion import TypeConverter
    print("✓ TypeConverter imported successfully")
except ImportError as e:
    print(f"✗ TypeConverter import error: {e}")

try:
    from preprocessing.preprocessing_pipeline import PreprocessingPipeline
    print("✓ PreprocessingPipeline imported successfully")
except ImportError as e:
    print(f"✗ PreprocessingPipeline import error: {e}")

# Test model imports
print("\n2. Testing Model Imports:")
try:
    from models.supervised.regression.linear_regression.linear_regression import LinearRegression
    print("✓ LinearRegression imported successfully")
except ImportError as e:
    print(f"✗ LinearRegression import error: {e}")

try:
    from models.supervised.regression.linear_regression.ridge_linear import RidgeLinearRegression
    print("✓ RidgeLinearRegression imported successfully")
except ImportError as e:
    print(f"✗ RidgeLinearRegression import error: {e}")

try:
    from models.supervised.regression.linear_regression.lasso_linear import LassoLinearRegression
    print("✓ LassoLinearRegression imported successfully")
except ImportError as e:
    print(f"✗ LassoLinearRegression import error: {e}")

try:
    from models.supervised.regression.linear_regression.elastic_linear import ElasticNetLinearRegression
    print("✓ ElasticNetLinearRegression imported successfully")
except ImportError as e:
    print(f"✗ ElasticNetLinearRegression import error: {e}")

try:
    from models.supervised.classification.logistic_regression.logistic_regression import LogisticRegression
    print("✓ LogisticRegression imported successfully")
except ImportError as e:
    print(f"✗ LogisticRegression import error: {e}")

try:
    from models.supervised.classification.logistic_regression.ridge_logistic import RidgeLogisticRegression
    print("✓ RidgeLogisticRegression imported successfully")
except ImportError as e:
    print(f"✗ RidgeLogisticRegression import error: {e}")

try:
    from models.supervised.classification.logistic_regression.lasso_logistic import LassoLogisticRegression
    print("✓ LassoLogisticRegression imported successfully")
except ImportError as e:
    print(f"✗ LassoLogisticRegression import error: {e}")

try:
    from models.supervised.classification.logistic_regression.elastic_logistic import ElasticNetLogisticRegression
    print("✓ ElasticNetLogisticRegression imported successfully")
except ImportError as e:
    print(f"✗ ElasticNetLogisticRegression import error: {e}")

# Test training imports
print("\n3. Testing Training Imports:")
try:
    from api.train import ModelTrainer, run_training_pipeline, TrainingInterface
    print("✓ ModelTrainer imported successfully")
    print("✓ run_training_pipeline imported successfully")
    print("✓ TrainingInterface imported successfully")
    
    # Test model loading
    trainer = ModelTrainer()
    reg_models = trainer.get_available_models('regression')
    cls_models = trainer.get_available_models('classification')
    
    print(f"✓ Regression models found: {reg_models}")
    print(f"✓ Classification models found: {cls_models}")
    
except ImportError as e:
    print(f"✗ Training imports error: {e}")

# Test utility imports
print("\n4. Testing Utility Imports:")
try:
    from utils.visualization_regression import RegressionVisualizer
    print("✓ RegressionVisualizer imported successfully")
except ImportError as e:
    print(f"✗ RegressionVisualizer import error: {e}")

try:
    from utils.visualization_classification import ClassificationVisualizer
    print("✓ ClassificationVisualizer imported successfully")
except ImportError as e:
    print(f"✗ ClassificationVisualizer import error: {e}")

print("\n" + "="*50)
print("IMPORT TEST COMPLETED")
print("="*50)