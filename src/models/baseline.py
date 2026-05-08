"""Baseline models for time-series classification."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any


class BaselineClassifier:
    """Simple baseline classifier using statistical features."""
    
    def __init__(self, model_type: str = 'knn'):
        """
        Initialize baseline classifier.
        
        Args:
            model_type: 'knn', 'dt', or 'lr'
        """
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'dt':
            self.model = DecisionTreeClassifier(random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineClassifier':
        """Fit the baseline model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return self.model.score(X, y)
