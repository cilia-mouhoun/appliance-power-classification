"""k-Nearest Neighbors model for time-series classification."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class KNNModel:
    """
    k-Nearest Neighbors classifier for time-series data.
    
    kNN is a non-parametric, lazy learning algorithm that classifies samples
    based on the labels of their k nearest neighbors in feature space.
    
    Advantages for time-series:
    - No training phase (lazy learner)
    - Interpretable decisions via neighbor inspection
    - Flexible for multi-modal distributions
    
    Disadvantages:
    - Slow predictions on large datasets
    - Sensitive to feature scaling and irrelevant features
    - Curse of dimensionality in high dimensions
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform', n_jobs: int = -1):
        """
        Initialize kNN model.
        
        Args:
            n_neighbors: Number of neighbors to consider (k)
            metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
            weights: Weight function ('uniform' or 'distance')
            n_jobs: Number of parallel jobs (-1 for all processors)
            
        Mathematical Intuition:
            - Euclidean: sqrt(sum((x_i - y_i)^2)) - standard L2 distance
            - Manhattan: sum(|x_i - y_i|) - L1 distance, robust to outliers
            - Cosine: 1 - (x·y)/(||x|| * ||y||) - angle-based similarity
            - uniform weights: all neighbors equally important
            - distance weights: closer neighbors more important (1/d)
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=n_jobs
        )
        self.scaler = StandardScaler()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNModel':
        """
        Store training data (lazy learning).
        
        kNN stores training data but doesn't build a model during fit.
        Scaling is applied for fair distance computation across features.
        
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
        Make hard predictions based on majority vote of k neighbors.
        
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
        Estimate class probabilities.
        
        Probability for class i = (count of neighbors with class i) / k
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Class probability estimates (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for query points.
        
        Useful for interpretability and analysis.
        
        Args:
            X: Query features (n_samples, n_features)
            n_neighbors: Number of neighbors to return (default: self.n_neighbors)
            
        Returns:
            distances: Distances to neighbors (n_samples, n_neighbors)
            indices: Indices of neighbors (n_samples, n_neighbors)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        X_scaled = self.scaler.transform(X)
        return self.model.kneighbors(X_scaled, n_neighbors=n_neighbors)
    
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
    
    def get_nearest_neighbors_info(self, X: np.ndarray, y_true: np.ndarray = None) \
            -> dict:
        """
        Get interpretable information about nearest neighbors.
        
        Useful for understanding model decisions and analyzing misclassifications.
        
        Args:
            X: Query features (n_samples, n_features)
            y_true: True labels (optional, for identifying errors)
            
        Returns:
            Dictionary with neighbor information
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        distances, indices = self.kneighbors(X)
        predictions = self.predict(X)
        
        info = {
            'n_queries': len(X),
            'distances': distances,
            'indices': indices,
            'predictions': predictions,
            'avg_neighbor_distance': distances.mean(),
            'max_neighbor_distance': distances.max(),
        }
        
        if y_true is not None:
            correct = (predictions == y_true)
            info['accuracy'] = correct.mean()
            info['n_correct'] = correct.sum()
            info['n_errors'] = (~correct).sum()
            info['avg_error_distance'] = distances[~correct].mean() if (~correct).any() else 0
        
        return info
