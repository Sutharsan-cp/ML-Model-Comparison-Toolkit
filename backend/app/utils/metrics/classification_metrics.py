import numpy as np
import scipy.stats as stats
from scipy.integrate import trapezoid
from itertools import combinations
import warnings

class ClassificationMetrics:
    """Comprehensive classification metrics without sklearn"""
    
    @staticmethod
    def _validate_inputs(y_true, y_pred, y_score=None):
        """Validate and convert inputs to numpy arrays"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_score is not None:
            y_score = np.asarray(y_score)
        return y_true, y_pred, y_score
    
    @staticmethod
    def _get_labels(y_true, y_pred):
        """Get unique labels from true and predicted values"""
        return np.unique(np.concatenate([y_true, y_pred]))
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
        """Compute confusion matrix"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        
        if labels is None:
            labels = ClassificationMetrics._get_labels(y_true, y_pred)
        
        n_labels = len(labels)
        cm = np.zeros((n_labels, n_labels), dtype=np.float64)
        
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        for true, pred in zip(y_true, y_pred):
            cm[label_to_idx[true], label_to_idx[pred]] += 1
        
        if normalize in ['true', 'pred', 'all']:
            if normalize == 'true':
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == 'pred':
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == 'all':
                cm = cm / cm.sum()
            cm = np.nan_to_num(cm)
        
        return cm, labels
    
    # =========================================================================
    # BINARY CLASSIFICATION METRICS
    # =========================================================================
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy score"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision_binary(y_true, y_pred, pos_label=1):
        """Binary precision score"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if len(labels) != 2:
            raise ValueError("Binary precision requires exactly 2 classes")
        
        pos_idx = np.where(labels == pos_label)[0][0]
        neg_idx = 1 - pos_idx
        
        tp = cm[pos_idx, pos_idx]
        fp = cm[neg_idx, pos_idx]
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall_binary(y_true, y_pred, pos_label=1):
        """Binary recall score"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if len(labels) != 2:
            raise ValueError("Binary recall requires exactly 2 classes")
        
        pos_idx = np.where(labels == pos_label)[0][0]
        neg_idx = 1 - pos_idx
        
        tp = cm[pos_idx, pos_idx]
        fn = cm[pos_idx, neg_idx]
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def specificity_binary(y_true, y_pred, pos_label=1):
        """Binary specificity (True Negative Rate)"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if len(labels) != 2:
            raise ValueError("Binary specificity requires exactly 2 classes")
        
        pos_idx = np.where(labels == pos_label)[0][0]
        neg_idx = 1 - pos_idx
        
        tn = cm[neg_idx, neg_idx]
        fp = cm[neg_idx, pos_idx]
        
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def f1_binary(y_true, y_pred, pos_label=1):
        """Binary F1 score"""
        precision = ClassificationMetrics.precision_binary(y_true, y_pred, pos_label)
        recall = ClassificationMetrics.recall_binary(y_true, y_pred, pos_label)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    @staticmethod
    def f_beta_binary(y_true, y_pred, beta=1.0, pos_label=1):
        """Binary F-beta score"""
        precision = ClassificationMetrics.precision_binary(y_true, y_pred, pos_label)
        recall = ClassificationMetrics.recall_binary(y_true, y_pred, pos_label)
        beta2 = beta ** 2
        
        return (1 + beta2) * (precision * recall) / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0
    
    @staticmethod
    def roc_auc_binary(y_true, y_score, pos_label=1):
        """Binary ROC AUC score"""
        y_true, _, y_score = ClassificationMetrics._validate_inputs(y_true, None, y_score)
        
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC AUC requires exactly 2 classes")
        
        # Convert to binary (0, 1)
        y_true_binary = (y_true == pos_label).astype(int)
        
        # Sort by predicted score
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        y_score_sorted = y_score[sorted_indices]
        
        # Calculate cumulative sums
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        
        # Calculate TPR and FPR
        total_tp = tps[-1] if len(tps) > 0 else 0
        total_fp = fps[-1] if len(fps) > 0 else 0
        
        tpr = tps / total_tp if total_tp > 0 else np.zeros_like(tps, dtype=float)
        fpr = fps / total_fp if total_fp > 0 else np.zeros_like(fps, dtype=float)
        
        # Add (0,0) and (1,1) points
        tpr = np.concatenate([[0.0], tpr, [1.0]])
        fpr = np.concatenate([[0.0], fpr, [1.0]])
        
        # Calculate AUC using trapezoidal rule
        auc = trapezoid(tpr, fpr)
        return auc
    
    @staticmethod
    def precision_recall_auc_binary(y_true, y_score, pos_label=1):
        """Binary Precision-Recall AUC"""
        y_true, _, y_score = ClassificationMetrics._validate_inputs(y_true, None, y_score)
        
        if len(np.unique(y_true)) != 2:
            raise ValueError("Precision-Recall AUC requires exactly 2 classes")
        
        # Convert to binary
        y_true_binary = (y_true == pos_label).astype(int)
        
        # Sort by predicted score
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        y_score_sorted = y_score[sorted_indices]
        
        # Calculate precision and recall at each threshold
        precisions = []
        recalls = []
        tp = 0
        fp = 0
        total_positives = np.sum(y_true_sorted)
        
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / total_positives if total_positives > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Add endpoints
        precisions = [1.0] + precisions + [0.0]
        recalls = [0.0] + recalls + [1.0]
        
        # Calculate AUC
        auc = trapezoid(precisions, recalls)
        return auc
    
    @staticmethod
    def cohens_kappa(y_true, y_pred):
        """Cohen's Kappa"""
        cm, _ = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        n = np.sum(cm)
        po = np.sum(np.diag(cm)) / n
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    
    @staticmethod
    def matthews_corrcoef(y_true, y_pred):
        """Matthews Correlation Coefficient"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if cm.shape[0] != 2:
            raise ValueError("MCC requires exactly 2 classes")
        
        tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        """Balanced accuracy"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        recalls = []
        for i in range(len(labels)):
            tp = cm[i, i]
            actual = np.sum(cm[i, :])
            recalls.append(tp / actual if actual > 0 else 0.0)
        return np.mean(recalls)
    
    @staticmethod
    def g_mean_binary(y_true, y_pred, pos_label=1):
        """Geometric mean for binary classification"""
        sensitivity = ClassificationMetrics.recall_binary(y_true, y_pred, pos_label)
        specificity = ClassificationMetrics.specificity_binary(y_true, y_pred, pos_label)
        return np.sqrt(sensitivity * specificity)
    
    @staticmethod
    def youdens_j(y_true, y_pred, pos_label=1):
        """Youden's J statistic"""
        sensitivity = ClassificationMetrics.recall_binary(y_true, y_pred, pos_label)
        specificity = ClassificationMetrics.specificity_binary(y_true, y_pred, pos_label)
        return sensitivity + specificity - 1
    
    @staticmethod
    def informedness(y_true, y_pred, pos_label=1):
        """Informedness (Youden's J statistic)"""
        return ClassificationMetrics.youdens_j(y_true, y_pred, pos_label)
    
    @staticmethod
    def hamming_loss(y_true, y_pred):
        """Hamming loss"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        return np.mean(y_true != y_pred)
    
    @staticmethod
    def jaccard_score_binary(y_true, y_pred, pos_label=1):
        """Jaccard similarity coefficient for binary classification"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if len(labels) != 2:
            raise ValueError("Binary Jaccard requires exactly 2 classes")
        
        pos_idx = np.where(labels == pos_label)[0][0]
        tp = cm[pos_idx, pos_idx]
        fp = cm[1-pos_idx, pos_idx]
        fn = cm[pos_idx, 1-pos_idx]
        
        return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    @staticmethod
    def lift_curve(y_true, y_score, pos_label=1, n_bins=10):
        """Lift curve metrics"""
        y_true, _, y_score = ClassificationMetrics._validate_inputs(y_true, None, y_score)
        y_true_binary = (y_true == pos_label).astype(int)
        
        # Sort by score
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        
        # Calculate cumulative metrics
        population_percentage = np.arange(1, len(y_true) + 1) / len(y_true)
        cumulative_capture_rate = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        lift = cumulative_capture_rate / population_percentage
        
        return {
            'population_percentage': population_percentage,
            'cumulative_capture_rate': cumulative_capture_rate,
            'lift': lift
        }
    
    @staticmethod
    def gain_chart(y_true, y_score, pos_label=1, n_bins=10):
        """Gain chart metrics"""
        lift_data = ClassificationMetrics.lift_curve(y_true, y_score, pos_label, n_bins)
        return {
            'population_percentage': lift_data['population_percentage'],
            'gain': lift_data['cumulative_capture_rate']
        }
    
    # =========================================================================
    # MULTI-CLASS CLASSIFICATION METRICS
    # =========================================================================
    
    @staticmethod
    def precision_multiclass(y_true, y_pred, average='macro'):
        """Multi-class precision"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        n_classes = len(labels)
        
        precisions = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            tp_total = np.sum(np.diag(cm))
            fp_total = np.sum(cm) - tp_total
            return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        elif average == 'weighted':
            support = np.sum(cm, axis=1)
            return np.average(precisions, weights=support)
        else:
            return precisions
    
    @staticmethod
    def recall_multiclass(y_true, y_pred, average='macro'):
        """Multi-class recall"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        n_classes = len(labels)
        
        recalls = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            tp_total = np.sum(np.diag(cm))
            fn_total = np.sum(cm) - tp_total
            return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        elif average == 'weighted':
            support = np.sum(cm, axis=1)
            return np.average(recalls, weights=support)
        else:
            return recalls
    
    @staticmethod
    def f1_multiclass(y_true, y_pred, average='macro'):
        """Multi-class F1 score"""
        precisions = ClassificationMetrics.precision_multiclass(y_true, y_pred, average=None)
        recalls = ClassificationMetrics.recall_multiclass(y_true, y_pred, average=None)
        
        f1_scores = []
        for p, r in zip(precisions, recalls):
            f1_scores.append(2 * (p * r) / (p + r) if (p + r) > 0 else 0.0)
        
        if average == 'macro':
            return np.mean(f1_scores)
        elif average == 'micro':
            # Micro F1 is same as accuracy for multi-class
            return ClassificationMetrics.accuracy(y_true, y_pred)
        elif average == 'weighted':
            cm, _ = ClassificationMetrics.confusion_matrix(y_true, y_pred)
            support = np.sum(cm, axis=1)
            return np.average(f1_scores, weights=support)
        else:
            return f1_scores
    
    @staticmethod
    def f_beta_multiclass(y_true, y_pred, beta=1.0, average='macro'):
        """Multi-class F-beta score"""
        precisions = ClassificationMetrics.precision_multiclass(y_true, y_pred, average=None)
        recalls = ClassificationMetrics.recall_multiclass(y_true, y_pred, average=None)
        beta2 = beta ** 2
        
        f_beta_scores = []
        for p, r in zip(precisions, recalls):
            f_beta_scores.append((1 + beta2) * (p * r) / (beta2 * p + r) if (beta2 * p + r) > 0 else 0.0)
        
        if average == 'macro':
            return np.mean(f_beta_scores)
        elif average == 'micro':
            # Micro F-beta calculation
            cm, _ = ClassificationMetrics.confusion_matrix(y_true, y_pred)
            tp_total = np.sum(np.diag(cm))
            fp_total = np.sum(cm) - tp_total
            fn_total = np.sum(cm) - tp_total
            
            p_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            r_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            
            return (1 + beta2) * (p_micro * r_micro) / (beta2 * p_micro + r_micro) if (beta2 * p_micro + r_micro) > 0 else 0.0
        elif average == 'weighted':
            cm, _ = ClassificationMetrics.confusion_matrix(y_true, y_pred)
            support = np.sum(cm, axis=1)
            return np.average(f_beta_scores, weights=support)
        else:
            return f_beta_scores
    
    @staticmethod
    def roc_auc_ovr(y_true, y_score, labels=None):
        """One-vs-Rest ROC AUC for multi-class"""
        y_true, _, y_score = ClassificationMetrics._validate_inputs(y_true, None, y_score)
        
        if labels is None:
            labels = np.unique(y_true)
        
        auc_scores = []
        for i, label in enumerate(labels):
            y_true_binary = (y_true == label).astype(int)
            y_score_binary = y_score[:, i] if y_score.ndim > 1 else y_score
            auc_scores.append(ClassificationMetrics.roc_auc_binary(y_true_binary, y_score_binary))
        
        return {
            'per_class': auc_scores,
            'macro': np.mean(auc_scores),
            'weighted': np.average(auc_scores, weights=np.bincount(y_true))
        }
    
    @staticmethod
    def roc_auc_ovo(y_true, y_score, labels=None):
        """One-vs-One ROC AUC for multi-class"""
        y_true, _, y_score = ClassificationMetrics._validate_inputs(y_true, None, y_score)
        
        if labels is None:
            labels = np.unique(y_true)
        
        n_classes = len(labels)
        pairwise_auc = np.zeros((n_classes, n_classes))
        
        for i, j in combinations(range(n_classes), 2):
            # Get samples for these two classes
            mask = (y_true == labels[i]) | (y_true == labels[j])
            y_true_pair = y_true[mask]
            y_score_pair_i = y_score[mask, i] if y_score.ndim > 1 else y_score[mask]
            y_score_pair_j = y_score[mask, j] if y_score.ndim > 1 else y_score[mask]
            
            # Convert to binary
            y_true_binary = (y_true_pair == labels[i]).astype(int)
            
            # Use appropriate scores
            if y_score.ndim > 1:
                y_score_binary = y_score_pair_i - y_score_pair_j
            else:
                y_score_binary = y_score_pair
            
            auc = ClassificationMetrics.roc_auc_binary(y_true_binary, y_score_binary)
            pairwise_auc[i, j] = auc
            pairwise_auc[j, i] = 1 - auc
        
        # Calculate overall AUC
        total_pairs = n_classes * (n_classes - 1)
        macro_auc = np.sum(pairwise_auc) / total_pairs if total_pairs > 0 else 0.0
        
        return {
            'pairwise': pairwise_auc,
            'macro': macro_auc
        }
    
    @staticmethod
    def classification_report(y_true, y_pred, labels=None, target_names=None):
        """Comprehensive classification report"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred, labels=labels)
        
        if target_names is None:
            target_names = [f"Class {label}" for label in labels]
        
        report = {}
        n_classes = len(labels)
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = np.sum(cm[i, :])
            
            report[target_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
        
        # Add averages
        precisions = [report[name]['precision'] for name in target_names]
        recalls = [report[name]['recall'] for name in target_names]
        f1_scores = [report[name]['f1-score'] for name in target_names]
        supports = [report[name]['support'] for name in target_names]
        
        report['macro avg'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1-score': np.mean(f1_scores),
            'support': np.sum(supports)
        }
        
        report['weighted avg'] = {
            'precision': np.average(precisions, weights=supports),
            'recall': np.average(recalls, weights=supports),
            'f1-score': np.average(f1_scores, weights=supports),
            'support': np.sum(supports)
        }
        
        return report
    
    @staticmethod
    def hamming_loss_multiclass(y_true, y_pred):
        """Hamming loss for multi-class"""
        return ClassificationMetrics.hamming_loss(y_true, y_pred)
    
    @staticmethod
    def jaccard_score_multiclass(y_true, y_pred, average='macro'):
        """Jaccard similarity coefficient for multi-class"""
        cm, labels = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        n_classes = len(labels)
        
        jaccard_scores = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            jaccard_scores.append(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0)
        
        if average == 'macro':
            return np.mean(jaccard_scores)
        elif average == 'micro':
            tp_total = np.sum(np.diag(cm))
            fp_fn_total = np.sum(cm) - tp_total
            return tp_total / (tp_total + fp_fn_total) if (tp_total + fp_fn_total) > 0 else 0.0
        elif average == 'weighted':
            support = np.sum(cm, axis=1)
            return np.average(jaccard_scores, weights=support)
        else:
            return jaccard_scores
    
    # =========================================================================
    # MULTI-LABEL CLASSIFICATION METRICS
    # =========================================================================
    
    @staticmethod
    def exact_match_ratio(y_true, y_pred):
        """Exact Match Ratio (Subset Accuracy)"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        return np.mean(np.all(y_true == y_pred, axis=1))
    
    @staticmethod
    def hamming_loss_multilabel(y_true, y_pred):
        """Hamming loss for multi-label"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        return np.mean(y_true != y_pred)
    
    @staticmethod
    def jaccard_score_multilabel(y_true, y_pred, average='macro'):
        """Jaccard similarity coefficient for multi-label"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        
        intersection = np.sum(y_true & y_pred, axis=1)
        union = np.sum(y_true | y_pred, axis=1)
        jaccard_per_sample = intersection / union
        
        if average == 'macro':
            return np.mean(jaccard_per_sample)
        elif average == 'micro':
            intersection_total = np.sum(y_true & y_pred)
            union_total = np.sum(y_true | y_pred)
            return intersection_total / union_total if union_total > 0 else 0.0
        elif average == 'samples':
            return np.mean(jaccard_per_sample)
        else:
            return jaccard_per_sample
    
    @staticmethod
    def precision_multilabel(y_true, y_pred, average='macro'):
        """Precision for multi-label classification"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        
        tp = np.sum(y_true & y_pred, axis=0)
        fp = np.sum(~y_true & y_pred, axis=0)
        precision_per_label = tp / (tp + fp)
        
        if average == 'macro':
            return np.mean(precision_per_label)
        elif average == 'micro':
            tp_total = np.sum(y_true & y_pred)
            fp_total = np.sum(~y_true & y_pred)
            return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        elif average == 'samples':
            tp_per_sample = np.sum(y_true & y_pred, axis=1)
            fp_per_sample = np.sum(~y_true & y_pred, axis=1)
            precision_per_sample = tp_per_sample / (tp_per_sample + fp_per_sample)
            return np.mean(precision_per_sample)
        else:
            return precision_per_label
    
    @staticmethod
    def recall_multilabel(y_true, y_pred, average='macro'):
        """Recall for multi-label classification"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        
        tp = np.sum(y_true & y_pred, axis=0)
        fn = np.sum(y_true & ~y_pred, axis=0)
        recall_per_label = tp / (tp + fn)
        
        if average == 'macro':
            return np.mean(recall_per_label)
        elif average == 'micro':
            tp_total = np.sum(y_true & y_pred)
            fn_total = np.sum(y_true & ~y_pred)
            return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        elif average == 'samples':
            tp_per_sample = np.sum(y_true & y_pred, axis=1)
            fn_per_sample = np.sum(y_true & ~y_pred, axis=1)
            recall_per_sample = tp_per_sample / (tp_per_sample + fn_per_sample)
            return np.mean(recall_per_sample)
        else:
            return recall_per_label
    
    @staticmethod
    def f1_multilabel(y_true, y_pred, average='macro'):
        """F1 score for multi-label classification"""
        precision = ClassificationMetrics.precision_multilabel(y_true, y_pred, average=None)
        recall = ClassificationMetrics.recall_multilabel(y_true, y_pred, average=None)
        
        f1_per_label = 2 * (precision * recall) / (precision + recall)
        
        if average == 'macro':
            return np.mean(f1_per_label)
        elif average == 'micro':
            p_micro = ClassificationMetrics.precision_multilabel(y_true, y_pred, average='micro')
            r_micro = ClassificationMetrics.recall_multilabel(y_true, y_pred, average='micro')
            return 2 * (p_micro * r_micro) / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
        elif average == 'samples':
            p_samples = ClassificationMetrics.precision_multilabel(y_true, y_pred, average='samples')
            r_samples = ClassificationMetrics.recall_multilabel(y_true, y_pred, average='samples')
            return 2 * (p_samples * r_samples) / (p_samples + r_samples) if (p_samples + r_samples) > 0 else 0.0
        else:
            return f1_per_label
    
    @staticmethod
    def example_based_metrics(y_true, y_pred):
        """Example-based metrics for multi-label classification"""
        y_true, y_pred, _ = ClassificationMetrics._validate_inputs(y_true, y_pred)
        
        # Example-based precision
        tp_per_example = np.sum(y_true & y_pred, axis=1)
        pred_per_example = np.sum(y_pred, axis=1)
        example_precision = tp_per_example / pred_per_example
        
        # Example-based recall
        actual_per_example = np.sum(y_true, axis=1)
        example_recall = tp_per_example / actual_per_example
        
        # Example-based F1
        example_f1 = 2 * (example_precision * example_recall) / (example_precision + example_recall)
        
        # Example-based accuracy
        example_accuracy = np.sum(y_true == y_pred, axis=1) / y_true.shape[1]
        
        return {
            'precision': np.mean(example_precision),
            'recall': np.mean(example_recall),
            'f1': np.mean(example_f1),
            'accuracy': np.mean(example_accuracy)
        }
    
    # =========================================================================
    # PROBABILITY CALIBRATION METRICS
    # =========================================================================
    
    @staticmethod
    def brier_score(y_true, y_prob):
        """Brier score"""
        y_true, _, y_prob = ClassificationMetrics._validate_inputs(y_true, None, y_prob)
        return np.mean((y_true - y_prob) ** 2)
    
    @staticmethod
    def log_loss(y_true, y_prob, eps=1e-15):
        """Log loss (cross-entropy loss)"""
        y_true, _, y_prob = ClassificationMetrics._validate_inputs(y_true, None, y_prob)
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    @staticmethod
    def calibration_curve(y_true, y_prob, n_bins=10):
        """Calibration curve"""
        y_true, _, y_prob = ClassificationMetrics._validate_inputs(y_true, None, y_prob)
        
        # Sort by predicted probability
        sorted_indices = np.argsort(y_prob)
        y_true_sorted = y_true[sorted_indices]
        y_prob_sorted = y_prob[sorted_indices]
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_sorted, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate mean predicted and actual probabilities per bin
        mean_predicted = np.zeros(n_bins)
        mean_actual = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.any(mask):
                mean_predicted[bin_idx] = np.mean(y_prob_sorted[mask])
                mean_actual[bin_idx] = np.mean(y_true_sorted[mask])
                bin_counts[bin_idx] = np.sum(mask)
        
        return {
            'mean_predicted': mean_predicted,
            'mean_actual': mean_actual,
            'bin_counts': bin_counts
        }
    
    @staticmethod
    def reliability_diagram(y_true, y_prob, n_bins=10):
        """Reliability diagram data"""
        calibration_data = ClassificationMetrics.calibration_curve(y_true, y_prob, n_bins)
        
        return {
            'predicted': calibration_data['mean_predicted'],
            'actual': calibration_data['mean_actual'],
            'counts': calibration_data['bin_counts']
        }
    
    # =========================================================================
    # COMPREHENSIVE METRICS SUMMARY
    # =========================================================================
    
    @staticmethod
    def comprehensive_classification_report(y_true, y_pred, y_score=None, labels=None, problem_type='binary'):
        """
        Comprehensive classification report for all problem types
        
        Parameters:
        - problem_type: 'binary', 'multiclass', 'multilabel'
        """
        report = {}
        
        if problem_type == 'binary':
            report['binary_metrics'] = {
                'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
                'precision': ClassificationMetrics.precision_binary(y_true, y_pred),
                'recall': ClassificationMetrics.recall_binary(y_true, y_pred),
                'f1': ClassificationMetrics.f1_binary(y_true, y_pred),
                'f_beta_2': ClassificationMetrics.f_beta_binary(y_true, y_pred, beta=2.0),
                'cohens_kappa': ClassificationMetrics.cohens_kappa(y_true, y_pred),
                'matthews_corrcoef': ClassificationMetrics.matthews_corrcoef(y_true, y_pred),
                'balanced_accuracy': ClassificationMetrics.balanced_accuracy(y_true, y_pred),
                'g_mean': ClassificationMetrics.g_mean_binary(y_true, y_pred),
                'youdens_j': ClassificationMetrics.youdens_j(y_true, y_pred),
                'informedness': ClassificationMetrics.informedness(y_true, y_pred),
                'hamming_loss': ClassificationMetrics.hamming_loss(y_true, y_pred),
                'jaccard_score': ClassificationMetrics.jaccard_score_binary(y_true, y_pred)
            }
            
            if y_score is not None:
                report['binary_metrics'].update({
                    'roc_auc': ClassificationMetrics.roc_auc_binary(y_true, y_score),
                    'pr_auc': ClassificationMetrics.precision_recall_auc_binary(y_true, y_score),
                    'brier_score': ClassificationMetrics.brier_score(y_true, y_score),
                    'log_loss': ClassificationMetrics.log_loss(y_true, y_score)
                })
        
        elif problem_type == 'multiclass':
            report['multiclass_metrics'] = {
                'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
                'precision_macro': ClassificationMetrics.precision_multiclass(y_true, y_pred, 'macro'),
                'precision_micro': ClassificationMetrics.precision_multiclass(y_true, y_pred, 'micro'),
                'precision_weighted': ClassificationMetrics.precision_multiclass(y_true, y_pred, 'weighted'),
                'recall_macro': ClassificationMetrics.recall_multiclass(y_true, y_pred, 'macro'),
                'recall_micro': ClassificationMetrics.recall_multiclass(y_true, y_pred, 'micro'),
                'recall_weighted': ClassificationMetrics.recall_multiclass(y_true, y_pred, 'weighted'),
                'f1_macro': ClassificationMetrics.f1_multiclass(y_true, y_pred, 'macro'),
                'f1_micro': ClassificationMetrics.f1_multiclass(y_true, y_pred, 'micro'),
                'f1_weighted': ClassificationMetrics.f1_multiclass(y_true, y_pred, 'weighted'),
                'cohens_kappa': ClassificationMetrics.cohens_kappa(y_true, y_pred),
                'hamming_loss': ClassificationMetrics.hamming_loss_multiclass(y_true, y_pred),
                'jaccard_macro': ClassificationMetrics.jaccard_score_multiclass(y_true, y_pred, 'macro'),
                'jaccard_micro': ClassificationMetrics.jaccard_score_multiclass(y_true, y_pred, 'micro'),
                'jaccard_weighted': ClassificationMetrics.jaccard_score_multiclass(y_true, y_pred, 'weighted')
            }
            
            if y_score is not None:
                roc_ovr = ClassificationMetrics.roc_auc_ovr(y_true, y_score, labels)
                roc_ovo = ClassificationMetrics.roc_auc_ovo(y_true, y_score, labels)
                
                report['multiclass_metrics'].update({
                    'roc_auc_ovr_macro': roc_ovr['macro'],
                    'roc_auc_ovr_weighted': roc_ovr['weighted'],
                    'roc_auc_ovo_macro': roc_ovo['macro']
                })
        
        elif problem_type == 'multilabel':
            report['multilabel_metrics'] = {
                'exact_match_ratio': ClassificationMetrics.exact_match_ratio(y_true, y_pred),
                'hamming_loss': ClassificationMetrics.hamming_loss_multilabel(y_true, y_pred),
                'jaccard_macro': ClassificationMetrics.jaccard_score_multilabel(y_true, y_pred, 'macro'),
                'jaccard_micro': ClassificationMetrics.jaccard_score_multilabel(y_true, y_pred, 'micro'),
                'jaccard_samples': ClassificationMetrics.jaccard_score_multilabel(y_true, y_pred, 'samples'),
                'precision_macro': ClassificationMetrics.precision_multilabel(y_true, y_pred, 'macro'),
                'precision_micro': ClassificationMetrics.precision_multilabel(y_true, y_pred, 'micro'),
                'precision_samples': ClassificationMetrics.precision_multilabel(y_true, y_pred, 'samples'),
                'recall_macro': ClassificationMetrics.recall_multilabel(y_true, y_pred, 'macro'),
                'recall_micro': ClassificationMetrics.recall_multilabel(y_true, y_pred, 'micro'),
                'recall_samples': ClassificationMetrics.recall_multilabel(y_true, y_pred, 'samples'),
                'f1_macro': ClassificationMetrics.f1_multilabel(y_true, y_pred, 'macro'),
                'f1_micro': ClassificationMetrics.f1_multilabel(y_true, y_pred, 'micro'),
                'f1_samples': ClassificationMetrics.f1_multilabel(y_true, y_pred, 'samples')
            }
            
            example_metrics = ClassificationMetrics.example_based_metrics(y_true, y_pred)
            report['multilabel_metrics'].update({
                'example_based_precision': example_metrics['precision'],
                'example_based_recall': example_metrics['recall'],
                'example_based_f1': example_metrics['f1'],
                'example_based_accuracy': example_metrics['accuracy']
            })
        
        # Add confusion matrix
        report['confusion_matrix'] = ClassificationMetrics.confusion_matrix(y_true, y_pred, labels)[0]
        
        return report