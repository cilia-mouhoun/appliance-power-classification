"""LightGBM model for time-series classification."""

import numpy as np
import lightgbm as lgb
from typing import Dict, Any


class LightGBMModel:
    """LightGBM classifier for time-series data."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = -1,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize LightGBM model.
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMModel':
        """Fit the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return self.model.score(X, y)
