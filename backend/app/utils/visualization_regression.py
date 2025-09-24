import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

class RegressionVisualizer:
    def __init__(self):
        self.fig_size = (15, 10)
        
    def plot_predictions_comparison(self, y_true_list, y_pred_list, dataset_names, 
                                  model_names=None, figsize=None):
        """
        Generalized function to compare predictions vs actual values for multiple datasets/models
        
        Parameters:
        - y_true_list: List of true y values for each dataset
        - y_pred_list: List of predicted y values for each dataset  
        - dataset_names: List of dataset names
        - model_names: List of model names (optional)
        - figsize: Figure size (optional)
        """
        if figsize is None:
            figsize = self.fig_size
            
        n_plots = len(y_true_list)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
            
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            ax = axes[i]
            
            # Scatter plot of predictions vs actual
            ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            title = dataset_names[i]
            if model_names and i < len(model_names):
                title = f"{model_names[i]} - {title}"
                
            ax.set_title(f'{title}\nMSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def plot_residuals(self, y_true_list, y_pred_list, dataset_names, 
                      model_names=None, figsize=None):
        """
        Plot residuals for regression models
        """
        if figsize is None:
            figsize = (15, 5)
            
        n_plots = len(y_true_list)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
            
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            ax = axes[i]
            residuals = y_true - y_pred
            
            ax.scatter(y_pred, residuals, alpha=0.6, color='green')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            title = f"Residuals - {dataset_names[i]}"
            if model_names and i < len(model_names):
                title = f"{model_names[i]} - {dataset_names[i]}"
                
            ax.set_title(title)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves(self, train_scores, val_scores, model_name="Model"):
        """
        Plot learning curves for training and validation
        """
        plt.figure(figsize=(10, 6))
        
        if train_scores is not None:
            plt.plot(train_scores, label='Training Score', marker='o')
        if val_scores is not None:
            plt.plot(val_scores, label='Validation Score', marker='s')
            
        plt.title(f'{model_name} - Learning Curve')
        plt.xlabel('Epochs/Iterations')
        plt.ylabel('Score (MSE/R²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_feature_importance(self, feature_names, importance_scores, 
                              model_name="Model", top_n=10):
        """
        Plot feature importance (works for models that provide feature importance)
        """
        if len(feature_names) != len(importance_scores):
            raise ValueError("Feature names and importance scores must have same length")
            
        # Create DataFrame for easy sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort and select top N features
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def comparative_metrics_barplot(self, metrics_dict, title="Model Comparison"):
        """
        Create bar plot comparing metrics across different models
        """
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]
            axes[i].bar(models, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Example usage function
def create_regression_visualization(y_true, y_pred, dataset_name="Dataset", 
                                  model_name="Model", show_residuals=True):
    """
    Helper function to create comprehensive regression visualization
    """
    visualizer = RegressionVisualizer()
    
    # Plot predictions vs actual
    visualizer.plot_predictions_comparison(
        [y_true], [y_pred], [dataset_name], [model_name]
    )
    
    # Plot residuals if requested
    if show_residuals:
        visualizer.plot_residuals(
            [y_true], [y_pred], [dataset_name], [model_name]
        )
    
    return visualizer