import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
from itertools import cycle

class ClassificationVisualizer:
    def __init__(self):
        self.fig_size = (15, 10)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            normalize=False, model_name="Model", figsize=(8, 6)):
        """
        Plot confusion matrix for classification results
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'{model_name} - Confusion Matrix{title_suffix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, y_true, y_scores, class_names=None, 
                      model_name="Model", figsize=(10, 8)):
        """
        Plot ROC curve for binary or multiclass classification
        """
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
            
        else:
            # Multiclass classification
            n_classes = len(np.unique(y_true))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Convert y_true to one-hot encoding
            y_true_bin = np.eye(n_classes)[y_true]
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            # Plot all ROC curves
            plt.figure(figsize=figsize)
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                          'purple', 'brown', 'pink', 'gray', 'olive'])
            
            for i, color in zip(range(n_classes), colors):
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - Multiclass ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
            
    def plot_precision_recall_curve(self, y_true, y_scores, model_name="Model", figsize=(10, 8)):
        """
        Plot Precision-Recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = np.mean(precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, lw=2, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve (AP = {avg_precision:.2f})')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_classification_report(self, y_true, y_pred, class_names=None, 
                                 model_name="Model", figsize=(10, 6)):
        """
        Visualize classification report as heatmap
        """
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Remove accuracy row for heatmap
        report_df = report_df.drop('accuracy', errors='ignore')
        
        plt.figure(figsize=figsize)
        sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', 
                   fmt='.2f', cbar_kws={'label': 'Score'})
        plt.title(f'{model_name} - Classification Report')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance_classification(self, feature_names, importance_scores, 
                                             model_name="Model", top_n=15):
        """
        Plot feature importance for classification models
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves_classification(self, train_acc, val_acc, train_loss, val_loss, 
                                          model_name="Model"):
        """
        Plot learning curves for classification models
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        if train_acc is not None:
            ax1.plot(train_acc, label='Training Accuracy', marker='o')
        if val_acc is not None:
            ax1.plot(val_acc, label='Validation Accuracy', marker='s')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        if train_loss is not None:
            ax2.plot(train_loss, label='Training Loss', marker='o')
        if val_loss is not None:
            ax2.plot(val_loss, label='Validation Loss', marker='s')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def comparative_metrics_classification(self, metrics_dict, title="Model Comparison"):
        """
        Compare multiple classification models using metrics
        """
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']  # Common metrics
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            axes[i].bar(models, values, color='lightcoral', alpha=0.7)
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Example usage function
def create_classification_visualization(y_true, y_pred, y_scores=None, class_names=None,
                                      dataset_name="Dataset", model_name="Model"):
    """
    Helper function to create comprehensive classification visualization
    """
    visualizer = ClassificationVisualizer()
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(y_true, y_pred, class_names, model_name=model_name)
    
    # Plot ROC curve if scores are provided
    if y_scores is not None:
        visualizer.plot_roc_curve(y_true, y_scores, class_names, model_name=model_name)
        
    # Plot classification report
    visualizer.plot_classification_report(y_true, y_pred, class_names, model_name=model_name)
    
    return visualizer