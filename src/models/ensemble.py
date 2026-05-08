"""Ensemble models for time-series classification."""

import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression


class VotingEnsemble:
    """Soft voting ensemble of multiple models."""
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained model instances (must implement predict_proba)
            weights: Weights for each model
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        all_probas = []
        for model, weight in zip(self.models, self.weights):
            p = model.predict_proba(X)
            all_probas.append(p * weight)
            
        avg_probas = np.sum(all_probas, axis=0) / np.sum(self.weights)
        return avg_probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class StackingEnsemble:
    """Stacking ensemble that learns how to best combine base models."""
    
    def __init__(self, base_models: List[Any], meta_model: Any = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of trained base models
            meta_model: Model to combine predictions (defaults to LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit the meta-model on base model predictions."""
        meta_features = []
        for model in self.base_models:
            meta_features.append(model.predict_proba(X))
            
        X_meta = np.hstack(meta_features)
        self.meta_model.fit(X_meta, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        meta_features = []
        for model in self.base_models:
            meta_features.append(model.predict_proba(X))
            
        X_meta = np.hstack(meta_features)
        return self.meta_model.predict(X_meta)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
