"""Confusion matrix utilities."""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Tuple


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_confusion_matrix_metrics(cm: np.ndarray) -> dict:
    """
    Extract metrics from confusion matrix.
    
    Args:
        cm: Confusion matrix
        
    Returns:
        Dictionary of metrics (TP, TN, FP, FN)
    """
    if cm.shape[0] != 2:
        raise ValueError("This function is designed for binary classification")
    
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    
    return {
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }
