"""CatBoost model for time-series classification."""

import numpy as np
from catboost import CatBoostClassifier
from typing import Dict, Any


class CatBoostModel:
    """CatBoost classifier for time-series data."""
    
    def __init__(self, iterations: int = 100, depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize CatBoost model.
        """
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostModel':
        """Fit the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
