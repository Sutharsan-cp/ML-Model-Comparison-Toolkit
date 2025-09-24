import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.append(str(app_dir))

try:
    from utils.visualization_regression import RegressionVisualizer
    from utils.visualization_classification import ClassificationVisualizer
except ImportError as e:
    print(f"Visualization import warning: {e}")

class ModelTrainer:
    def __init__(self):
        self.regression_models = {}
        self.classification_models = {}
        self._load_all_models()
        
    def _load_all_models(self):
        """Dynamically load all available models from the models directory"""
        # Regression Models
        try:
            from models.supervised.regression.linear_regression.linear_regression import LinearRegression
            from models.supervised.regression.linear_regression.ridge_linear import RidgeLinearRegression
            from models.supervised.regression.linear_regression.lasso_linear import LassoLinearRegression
            from models.supervised.regression.linear_regression.elastic_linear import ElasticNetLinearRegression
            
            self.regression_models = {
                'linear_regression': LinearRegression,
                'ridge_regression': RidgeLinearRegression,
                'lasso_regression': LassoLinearRegression,
                'elastic_net_regression': ElasticNetLinearRegression,
            }
            print("✓ Regression models loaded successfully")
        except ImportError as e:
            print(f"⚠ Regression models import error: {e}")
        
        # Classification Models  
        try:
            from models.supervised.classification.logistic_regression.logistic_regression import LogisticRegression
            from models.supervised.classification.logistic_regression.ridge_logistic import RidgeLogisticRegression
            from models.supervised.classification.logistic_regression.lasso_logistic import LassoLogisticRegression
            from models.supervised.classification.logistic_regression.elastic_logistic import ElasticNetLogisticRegression
            
            self.classification_models = {
                'logistic_regression': LogisticRegression,
                'ridge_logistic': RidgeLogisticRegression,
                'lasso_logistic': LassoLogisticRegression,
                'elastic_net_logistic': ElasticNetLogisticRegression,
            }
            print("✓ Classification models loaded successfully")
        except ImportError as e:
            print(f"⚠ Classification models import error: {e}")
    
    def get_available_models(self, task_type):
        """Get available models for a specific task type"""
        if task_type == 'regression':
            return list(self.regression_models.keys())
        elif task_type == 'classification':
            return list(self.classification_models.keys())
        else:
            return []
    
    def load_dataset(self, dataset_path, target_column, test_size=0.2, random_state=42):
        """Load and split dataset for training"""
        try:
            df = pd.read_csv(dataset_path)
            X = df.drop(columns=[target_column]).values
            y = df[target_column].values
            
            # Reshape y for models that expect 2D array
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"✓ Dataset loaded: {X_train.shape} train, {X_test.shape} test")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None, None
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test, task_type, **model_params):
        """Train a specific model and return results"""
        try:
            if task_type == 'regression':
                if model_name not in self.regression_models:
                    raise ValueError(f"Model {model_name} not available for regression")
                
                model_class = self.regression_models[model_name]
                model = model_class(**model_params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {'mse': mse, 'r2': r2}
                
            elif task_type == 'classification':
                if model_name not in self.classification_models:
                    raise ValueError(f"Model {model_name} not available for classification")
                
                model_class = self.classification_models[model_name]
                model = model_class(**model_params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'accuracy': accuracy}
                
            else:
                raise ValueError("Task type must be 'regression' or 'classification'")
            
            return model, y_pred, metrics
            
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            return None, None, None
    
    def create_visualizations(self, y_true, y_pred, task_type, model_name, dataset_name):
        """Create appropriate visualizations for the model"""
        try:
            if task_type == 'regression':
                visualizer = RegressionVisualizer()
                visualizer.plot_predictions_comparison(
                    [y_true], [y_pred], [dataset_name], [model_name]
                )
                visualizer.plot_residuals(
                    [y_true], [y_pred], [dataset_name], [model_name]
                )
            elif task_type == 'classification':
                visualizer = ClassificationVisualizer()
                visualizer.plot_confusion_matrix(
                    y_true, y_pred, model_name=model_name
                )
            print("✓ Visualizations created successfully")
        except Exception as e:
            print(f"⚠ Could not create visualizations: {e}")

def run_training_pipeline(dataset_path, target_column, model_name, task_type, **model_params):
    """Complete training pipeline"""
    print(f"\n🚀 Starting training pipeline for {model_name}...")
    print(f"Dataset: {dataset_path}")
    print(f"Target: {target_column}")
    print(f"Task: {task_type}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_dataset(dataset_path, target_column)
    if X_train is None:
        return None, None, None
    
    # Train model
    model, y_pred, metrics = trainer.train_model(
        model_name, X_train, y_train, X_test, y_test, task_type, **model_params
    )
    
    if model is None:
        return None, None, None
    
    # Print results
    print(f"\n📊 Model Results:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Create visualizations
    dataset_name = Path(dataset_path).stem
    trainer.create_visualizations(y_test, y_pred, task_type, model_name, dataset_name)
    
    return model, y_pred, metrics

# Menu-driven training interface
class TrainingInterface:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
    
    def list_datasets(self, task_type):
        """List available datasets for a specific task type"""
        processed_dir = self.data_dir / "processed" / task_type
        raw_dir = self.data_dir / "raw" / task_type
        
        datasets = []
        
        # Check processed directory first
        if processed_dir.exists():
            for file in processed_dir.glob("*.csv"):
                datasets.append(("processed", file))
        
        # Fallback to raw directory
        if not datasets and raw_dir.exists():
            for file in raw_dir.glob("*.csv"):
                datasets.append(("raw", file))
                
        return datasets
    
    def get_dataset_info(self, dataset_path):
        """Get basic info about a dataset"""
        try:
            df = pd.read_csv(dataset_path)
            return {
                'shape': df.shape,
                'columns': list(df.columns),
                'sample_data': df.head(3)
            }
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return None
    
    def run_training_menu(self):
        """Main training menu interface"""
        while True:
            print("\n" + "="*60)
            print(" MODEL TRAINING SYSTEM")
            print("="*60)
            print("\n1. Train Regression Model")
            print("2. Train Classification Model") 
            print("3. View Available Models")
            print("4. Back to Main Menu")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self._train_regression()
            elif choice == "2":
                self._train_classification()
            elif choice == "3":
                self._view_models()
            elif choice == "4":
                break
            else:
                print("Invalid choice, try again.")
    
    def _train_regression(self):
        """Handle regression training"""
        datasets = self.list_datasets('regression')
        if not datasets:
            print("No regression datasets found!")
            return
        
        # Dataset selection
        print("\nAvailable regression datasets:")
        for i, (source, filepath) in enumerate(datasets, 1):
            print(f"{i}. [{source.upper()}] {filepath.name}")
        
        try:
            choice = int(input("\nSelect dataset: ")) - 1
            if 0 <= choice < len(datasets):
                source, dataset_path = datasets[choice]
                
                # Target column selection
                info = self.get_dataset_info(dataset_path)
                if info:
                    print(f"\nDataset columns: {info['columns']}")
                    target_col = input("Enter target column name: ")
                    
                    if target_col in info['columns']:
                        # Model selection
                        models = self.trainer.get_available_models('regression')
                        print("\nAvailable regression models:")
                        for i, model in enumerate(models, 1):
                            print(f"{i}. {model}")
                        
                        model_choice = int(input("\nSelect model: ")) - 1
                        if 0 <= model_choice < len(models):
                            model_name = models[model_choice]
                            
                            # Get model parameters
                            params = self._get_model_parameters('regression', model_name)
                            
                            # Train model
                            run_training_pipeline(
                                dataset_path, target_col, model_name, 'regression', **params
                            )
            else:
                print("Invalid selection!")
                
        except (ValueError, IndexError):
            print("Invalid input!")
    
    def _train_classification(self):
        """Handle classification training"""
        datasets = self.list_datasets('classification')
        if not datasets:
            print("No classification datasets found!")
            return
        
        print("\nAvailable classification datasets:")
        for i, (source, filepath) in enumerate(datasets, 1):
            print(f"{i}. [{source.upper()}] {filepath.name}")
        
        try:
            choice = int(input("\nSelect dataset: ")) - 1
            if 0 <= choice < len(datasets):
                source, dataset_path = datasets[choice]
                
                info = self.get_dataset_info(dataset_path)
                if info:
                    print(f"\nDataset columns: {info['columns']}")
                    target_col = input("Enter target column name: ")
                    
                    if target_col in info['columns']:
                        models = self.trainer.get_available_models('classification')
                        print("\nAvailable classification models:")
                        for i, model in enumerate(models, 1):
                            print(f"{i}. {model}")
                        
                        model_choice = int(input("\nSelect model: ")) - 1
                        if 0 <= model_choice < len(models):
                            model_name = models[model_choice]
                            
                            params = self._get_model_parameters('classification', model_name)
                            
                            run_training_pipeline(
                                dataset_path, target_col, model_name, 'classification', **params
                            )
            else:
                print("Invalid selection!")
                
        except (ValueError, IndexError):
            print("Invalid input!")
    
    def _get_model_parameters(self, task_type, model_name):
        """Get model parameters from user"""
        params = {}
        
        print(f"\nEnter parameters for {model_name} (press Enter for defaults):")
        
        # Common parameters
        try:
            lr = input("Learning rate (default: 0.01): ")
            if lr: params['lr'] = float(lr)
            
            epochs = input("Epochs (default: 1000): ")
            if epochs: params['epochs'] = int(epochs)
            
            # Regularization parameters
            if 'ridge' in model_name or 'lasso' in model_name or 'elastic' in model_name:
                lamda = input("Lambda/Alpha (default: 0.1): ")
                if lamda: params['lamda'] = float(lamda)
                
            if 'elastic' in model_name:
                alpha = input("ElasticNet Alpha (default: 0.5): ")
                if alpha: params['alpha'] = float(alpha)
                
        except ValueError:
            print("Using default parameters due to invalid input")
        
        return params
    
    def _view_models(self):
        """Display available models"""
        reg_models = self.trainer.get_available_models('regression')
        cls_models = self.trainer.get_available_models('classification')
        
        print("\nAvailable Regression Models:")
        for model in reg_models:
            print(f"  ✓ {model}")
        
        print("\nAvailable Classification Models:")
        for model in cls_models:
            print(f"  ✓ {model}")

# For direct execution
if __name__ == "__main__":
    interface = TrainingInterface()
    interface.run_training_menu()