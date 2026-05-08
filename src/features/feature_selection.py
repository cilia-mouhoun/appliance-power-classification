"""Feature selection utilities."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from typing import List


def select_top_features(X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 10) -> List[str]:
    """
    Select top k features based on scoring method.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: 'mutual_info' or 'f_score'
        k: Number of features to select
        
    Returns:
        List of selected feature names
    """
    if method == 'mutual_info':
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == 'f_score':
        scores = f_classif(X, y)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    top_indices = np.argsort(scores)[-k:]
    selected_features = [X.columns[i] for i in sorted(top_indices, reverse=True)]
    
    return selected_features


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
    """
    Remove features with low variance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
        
    Returns:
        List of features to keep
    """
    variances = X.var()
    features_to_keep = variances[variances > threshold].index.tolist()
    
    return features_to_keep


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove highly correlated features.
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold
        
    Returns:
        List of features to keep
    """
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    features_to_drop = set()
    for column in upper_triangle.columns:
        for idx, value in upper_triangle[column].items():
            if value > threshold:
                features_to_drop.add(column)
    
    features_to_keep = [f for f in X.columns if f not in features_to_drop]
    
    return features_to_keep
