"""XGBoost model for time-series classification."""

import numpy as np
import xgboost as xgb
from typing import Dict, Any


class XGBoostModel:
    """XGBoost classifier for time-series data."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            random_state: Random seed
        """
        self.params = {
            'objective': 'multi:softmax',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
        }
        self.model = None
        self.n_estimators = n_estimators
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        """Fit the XGBoost model."""
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores."""
        return self.model.get_score()
