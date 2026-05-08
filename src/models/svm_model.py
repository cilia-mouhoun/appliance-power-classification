"""Support Vector Machine model for time-series classification."""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Optional


class SVMModel:
    """
    Support Vector Machine classifier for time-series data.
    
    SVMs are particularly effective for high-dimensional feature spaces
    and work well with both temporal and spectral feature representations.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 random_state: int = 42, probability: bool = True):
        """
        Initialize SVM model.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter (lower C = stronger regularization)
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            random_state: Random seed for reproducibility
            probability: Whether to enable probability estimates
            
        Mathematical Intuition:
            - Finds optimal hyperplane maximizing margin between classes
            - Kernel trick enables non-linear classification in high dimensions
            - C parameter controls trade-off between margin and training errors
            - gamma controls influence range of single training samples
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=probability
        )
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        """
        Fit the SVM model with feature scaling.
        
        Note: SVMs are sensitive to feature scaling, so standardization is applied.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make hard predictions.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using calibrated SVM.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Class probability estimates (n_samples, n_classes)
            
        Note:
            Probabilities are obtained from Platt scaling.
            Requires probability=True during initialization.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function for samples.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Decision function values (n_samples,) or (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy (0-1)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Get the support vectors (training samples at the decision boundary).
        
        Returns:
            Support vectors (n_support_vectors, n_features)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.support_vectors_
    
    def get_n_support_vectors(self) -> int:
        """
        Get the number of support vectors.
        
        Returns:
            Number of support vectors
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        return len(self.model.support_vectors_)
