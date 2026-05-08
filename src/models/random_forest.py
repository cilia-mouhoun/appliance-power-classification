"""Random Forest model for time-series classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any


class RandomForestModel:
    """Random Forest classifier for time-series data."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Fit the Random Forest model."""
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
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
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.feature_importance_
