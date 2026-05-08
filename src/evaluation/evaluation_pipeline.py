"""Comprehensive evaluation pipeline."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, cohen_kappa_score
)
import warnings

from .metrics import calculate_metrics, get_classification_report
from .confusion_matrix import get_confusion_matrix
from .statistical_tests import (
    mcnemar_test, paired_t_test, compare_classifiers_comprehensive,
    effect_size_cohens_d
)


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for classification models.
    
    Provides unified interface for:
    - Performance metrics (accuracy, precision, recall, F1, etc.)
    - Statistical tests (McNemar, paired t-test)
    - Per-class analysis
    - Misclassification analysis
    - Model comparison
    """
    
    def __init__(self, target_names: Optional[list] = None):
        """
        Initialize evaluation pipeline.
        
        Args:
            target_names: Names of target classes
        """
        self.target_names = target_names
        self.results = {}
    
    def evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None,
                             model_name: str = 'model') -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Probability estimates (optional)
            model_name: Name of the model
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {'model': model_name}
        
        # Overall metrics
        results['overall'] = calculate_metrics(y_true, y_pred, y_proba)
        results['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        results['per_class'] = self._get_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = get_classification_report(
            y_true, y_pred, self.target_names
        )
        
        # ROC-AUC for multi-class (one-vs-rest)
        if len(np.unique(y_true)) > 2 and y_proba is not None:
            try:
                results['roc_auc_macro'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
                results['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
            except Exception as e:
                results['roc_auc_macro'] = None
                results['roc_auc_weighted'] = None
                results['roc_auc_error'] = str(e)
        
        # Error analysis
        results['errors'] = self._analyze_errors(y_true, y_pred)
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models_dict: Dict[str, Any], X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple models on test set.
        
        Args:
            models_dict: Dictionary mapping model names to fitted models
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Comprehensive comparison results
        """
        predictions = {}
        probabilities = {}
        
        for model_name, model in models_dict.items():
            predictions[model_name] = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities[model_name] = model.predict_proba(X_test)
                except:
                    pass
        
        # Evaluate each model
        comparison = {'individual_evaluations': {}}
        for model_name, preds in predictions.items():
            comparison['individual_evaluations'][model_name] = self.evaluate_single_model(
                y_test, preds,
                probabilities.get(model_name, None),
                model_name
            )
        
        # Pairwise comparisons
        comparison['pairwise_comparisons'] = {}
        model_names = list(predictions.keys())
        for i, name_1 in enumerate(model_names):
            for name_2 in model_names[i + 1:]:
                comparison['pairwise_comparisons'][f"{name_1} vs {name_2}"] = \
                    mcnemar_test(predictions[name_1], predictions[name_2], y_test)
        
        # Best model
        accuracies = {name: compare_classifiers_comprehensive(
            y_test, {name: preds})[name]['accuracy']
            for name, preds in predictions.items()
        }
        best_model = max(accuracies, key=accuracies.get)
        comparison['best_model'] = best_model
        comparison['best_accuracy'] = accuracies[best_model]
        comparison['all_accuracies'] = accuracies
        
        return comparison
    
    def cross_validation_summary(self, fold_results: Dict[str, list]) -> Dict[str, Any]:
        """
        Summarize cross-validation results across folds.
        
        Args:
            fold_results: Dictionary mapping metric names to lists of fold values
        
        Returns:
            Summary statistics for all metrics
        """
        from .statistical_tests import compare_metrics_across_folds
        
        return compare_metrics_across_folds(fold_results)
    
    def _get_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) \
            -> Dict[str, Dict[str, float]]:
        """Get precision, recall, F1 for each class."""
        n_classes = len(np.unique(y_true))
        
        per_class = {}
        for class_idx in range(n_classes):
            class_name = self.target_names[class_idx] if self.target_names else f'class_{class_idx}'
            
            # Binary classification for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            per_class[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true == class_idx),
            }
        
        return per_class
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze misclassification patterns."""
        errors = y_true != y_pred
        n_errors = errors.sum()
        
        analysis = {
            'n_errors': int(n_errors),
            'error_rate': float(n_errors / len(y_true)),
            'n_correct': int((~errors).sum()),
        }
        
        if n_errors > 0:
            # Most common error classes
            error_indices = np.where(errors)[0]
            error_true = y_true[error_indices]
            error_pred = y_pred[error_indices]
            
            # Error confusion (true -> predicted)
            error_confusion = {}
            for true_label, pred_label in zip(error_true, error_pred):
                key = f"true_{true_label}_pred_{pred_label}"
                error_confusion[key] = error_confusion.get(key, 0) + 1
            
            analysis['error_confusion'] = error_confusion
        
        return analysis
    
    def get_summary_report(self) -> str:
        """Generate text summary of all evaluation results."""
        report = "=== EVALUATION SUMMARY ===\n"
        
        for model_name, results in self.results.items():
            report += f"\n{model_name}:\n"
            report += f"  Overall Accuracy: {results['overall']['accuracy']:.3f}\n"
            report += f"  Precision: {results['overall']['precision']:.3f}\n"
            report += f"  Recall: {results['overall']['recall']:.3f}\n"
            report += f"  F1-Score: {results['overall']['f1']:.3f}\n"
            report += f"  Cohen's Kappa: {results['kappa']:.3f}\n"
        
        return report


# Convenience functions
def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: Optional[np.ndarray] = None,
                   target_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Quick comprehensive evaluation of predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability estimates (optional)
        target_names: Class names (optional)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    pipeline = EvaluationPipeline(target_names=target_names)
    return pipeline.evaluate_single_model(y_true, y_pred, y_proba, 'model')


def compare_two_models(y_true: np.ndarray, y_pred_1: np.ndarray,
                      y_pred_2: np.ndarray, model_1_name: str = 'Model 1',
                      model_2_name: str = 'Model 2') -> Dict[str, Any]:
    """
    Compare two models using McNemar's test and other metrics.
    
    Args:
        y_true: True labels
        y_pred_1: Predictions from model 1
        y_pred_2: Predictions from model 2
        model_1_name: Name of first model
        model_2_name: Name of second model
    
    Returns:
        Comparison results
    """
    pipeline = EvaluationPipeline()
    
    eval_1 = pipeline.evaluate_single_model(y_true, y_pred_1, model_name=model_1_name)
    eval_2 = pipeline.evaluate_single_model(y_true, y_pred_2, model_name=model_2_name)
    
    mcnemar_result = mcnemar_test(y_pred_1, y_pred_2, y_true)
    
    return {
        'model_1': eval_1,
        'model_2': eval_2,
        'mcnemar_test': mcnemar_result,
        'better_model': model_1_name if eval_1['overall']['accuracy'] > eval_2['overall']['accuracy'] else model_2_name,
    }
