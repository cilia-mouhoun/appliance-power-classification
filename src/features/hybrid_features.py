"""Hybrid features combining temporal and spectral domains."""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from .temporal_features import TemporalFeatureExtractor
from .spectral_features import SpectralFeatureExtractor


class HybridFeatureExtractor:
    """
    Combines temporal and spectral features for enhanced classification.
    
    Mathematical Intuition:
    - Temporal features capture signal dynamics and statistical properties
    - Spectral features capture frequency content and harmonic characteristics
    - Hybrid approach leverages complementary information from both domains
    - Potential benefits: improved generalization, robustness to different appliance behaviors
    
    Challenges:
    - Curse of dimensionality (temporal + spectral can exceed 50+ features)
    - Feature redundancy (temporal energy ≈ spectral power)
    - Computational cost of feature extraction and classification
    
    Solutions:
    - Mutual information feature selection
    - Optional PCA for dimensionality reduction
    - Careful feature engineering to minimize redundancy
    """
    
    def __init__(self, use_pca: bool = False, pca_components: int = 30,
                 use_mutual_info_selection: bool = True, n_features_to_select: int = None):
        """
        Initialize hybrid feature extractor.
        
        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components if use_pca=True
            use_mutual_info_selection: Whether to use mutual information for feature selection
            n_features_to_select: Number of features to select (None = no selection)
        """
        self.temporal_extractor = TemporalFeatureExtractor()
        self.spectral_extractor = SpectralFeatureExtractor()
        
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None if not use_pca else PCA(n_components=pca_components)
        
        self.use_mutual_info_selection = use_mutual_info_selection
        self.n_features_to_select = n_features_to_select
        self.selected_feature_indices = None
        self.selected_feature_names = None
        
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_features(self, signal: np.ndarray, sampling_rate: float = 1.0,
                        return_feature_names: bool = False) -> \
            Tuple[np.ndarray, Optional[List[str]]]:
        """
        Extract hybrid features from a signal.
        
        Args:
            signal: Input time-series signal (1D array)
            sampling_rate: Sampling rate in Hz
            return_feature_names: Whether to return feature names
            
        Returns:
            features: Combined temporal + spectral features (1D array)
            feature_names: Names of features (if return_feature_names=True)
        """
        # Extract features from both domains
        temporal_feats, temporal_names = self.temporal_extractor.extract_features(
            signal, return_feature_names=True
        )
        spectral_feats, spectral_names = self.spectral_extractor.extract_features(
            signal, sampling_rate=sampling_rate, return_feature_names=True
        )
        
        # Concatenate features
        hybrid_features = np.concatenate([temporal_feats, spectral_feats])
        hybrid_names = temporal_names + spectral_names
        
        if return_feature_names:
            return hybrid_features, hybrid_names
        return hybrid_features
    
    def extract_batch_features(self, signals: np.ndarray, sampling_rate: float = 1.0,
                              return_feature_names: bool = False) -> \
            Tuple[np.ndarray, Optional[List[str]]]:
        """
        Extract hybrid features from multiple signals.
        
        Args:
            signals: Array of signals (n_samples, signal_length)
            sampling_rate: Sampling rate in Hz
            return_feature_names: Whether to return feature names
            
        Returns:
            features: Feature matrix (n_samples, n_features)
            feature_names: Names of features (if return_feature_names=True)
        """
        features_list = []
        feature_names = None
        
        for i, signal in enumerate(signals):
            if i == 0 and return_feature_names:
                feats, feature_names = self.extract_features(
                    signal, sampling_rate, return_feature_names=True
                )
            else:
                feats = self.extract_features(signal, sampling_rate)
            features_list.append(feats)
        
        features = np.array(features_list)
        
        if return_feature_names:
            return features, feature_names
        return features
    
    def fit_feature_selection(self, X: np.ndarray, y: np.ndarray) -> 'HybridFeatureExtractor':
        """
        Fit mutual information feature selection on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        if self.use_mutual_info_selection and self.n_features_to_select is not None:
            # Compute mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Select top features
            selected_indices = np.argsort(mi_scores)[-self.n_features_to_select:]
            self.selected_feature_indices = sorted(selected_indices)
        
        return self
    
    def fit_scaler(self, X: np.ndarray) -> 'HybridFeatureExtractor':
        """
        Fit feature standardization scaler.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.selected_feature_indices is not None:
            X_fit = X[:, self.selected_feature_indices]
        else:
            X_fit = X
            
        self.scaler.fit(X_fit)
        return self
    
    def fit_pca(self, X: np.ndarray) -> 'HybridFeatureExtractor':
        """
        Fit PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.use_pca:
            if self.selected_feature_indices is not None:
                X_selected = X[:, self.selected_feature_indices]
            else:
                X_selected = X
            
            X_scaled = self.scaler.transform(X_selected)
            self.pca.fit(X_scaled)
        
        self._is_fitted = True
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridFeatureExtractor':
        """
        Fit all components (feature selection, scaling, PCA).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self.fit_feature_selection(X, y)
        self.fit_scaler(X)
        self.fit_pca(X)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted components.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Transformed features
        """
        if not self._is_fitted:
            raise ValueError("Extractor must be fitted before transform")
        
        # Apply feature selection
        if self.selected_feature_indices is not None:
            X_selected = X[:, self.selected_feature_indices]
        else:
            X_selected = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Apply PCA if enabled
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute mutual information scores for all features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Dictionary mapping feature names to MI scores
        """
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Get feature names
        _, feature_names = self.extract_batch_features(
            np.zeros((1, 100)), return_feature_names=True  # Dummy signal to get names
        )
        
        feature_importance = {name: score for name, score in zip(feature_names, mi_scores)}
        return feature_importance
    
    def get_pca_variance_explained(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio from PCA.
        
        Returns:
            Explained variance ratio for each component (if PCA enabled)
        """
        if self.use_pca and self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None
    
    def get_n_features_after_reduction(self) -> int:
        """
        Get number of features after all reductions.
        
        Returns:
            Final feature dimensionality
        """
        n_features = self.temporal_extractor.n_features + \
                     self.spectral_extractor.n_features
        
        if self.selected_feature_indices is not None:
            n_features = len(self.selected_feature_indices)
        
        if self.use_pca:
            n_features = self.pca_components
        
        return n_features
    
    def get_feature_statistics(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Dictionary with feature statistics
        """
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'n_features_temporal': self.temporal_extractor.n_features,
            'n_features_spectral': self.spectral_extractor.n_features,
        }
