"""Tests for feature extraction module."""

import pytest
import numpy as np
from src.features import temporal_features, spectral_features


class TestTemporalFeatures:
    """Test cases for temporal feature extraction."""
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        data = np.array([1, 2, 3, 4, 5])
        features = temporal_features.extract_statistical_features(data)
        
        assert 'mean' in features
        assert 'std' in features
        assert features['mean'] == 3.0
    
    def test_extract_autocorrelation_features(self):
        """Test autocorrelation feature extraction."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        features = temporal_features.extract_autocorrelation_features(data, max_lag=3)
        
        assert 'acf_lag_1' in features
        assert len(features) == 3


class TestSpectralFeatures:
    """Test cases for spectral feature extraction."""
    
    def test_extract_fft_features(self):
        """Test FFT feature extraction."""
        data = np.sin(np.linspace(0, 4*np.pi, 100))
        features = spectral_features.extract_fft_features(data)
        
        assert 'fft_max_power' in features
        assert 'spectral_centroid' in features
    
    def test_extract_welch_features(self):
        """Test Welch PSD feature extraction."""
        data = np.random.randn(256)
        features = spectral_features.extract_welch_features(data)
        
        assert 'welch_max_psd' in features
        assert features['welch_max_psd'] > 0


if __name__ == '__main__':
    pytest.main([__file__])
