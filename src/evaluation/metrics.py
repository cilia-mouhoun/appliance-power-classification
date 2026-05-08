"""Evaluation metrics for classification models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: np.ndarray = None, average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            target_names: list = None) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
