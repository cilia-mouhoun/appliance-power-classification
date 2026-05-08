"""Tests for preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import cleaning, normalization, segmentation


class TestCleaning:
    """Test cases for data cleaning functions."""
    
    def test_remove_outliers_iqr(self):
        """Test IQR-based outlier removal."""
        data = pd.Series([1, 2, 3, 4, 5, 100])
        cleaned = cleaning.remove_outliers(data, method='iqr')
        assert np.isnan(cleaned.iloc[-1])
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        filled = cleaning.handle_missing_values(data, method='interpolate')
        assert not filled.isna().any()


class TestNormalization:
    """Test cases for normalization functions."""
    
    def test_standardize(self):
        """Test standardization."""
        data = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        scaled, _ = normalization.standardize(data)
        assert np.isclose(scaled.mean().values[0], 0, atol=1e-10)
        assert np.isclose(scaled.std().values[0], 1, atol=1e-10)
    
    def test_normalize(self):
        """Test min-max normalization."""
        data = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        scaled, _ = normalization.normalize(data, feature_range=(0, 1))
        assert scaled.min().values[0] >= 0
        assert scaled.max().values[0] <= 1


class TestSegmentation:
    """Test cases for segmentation functions."""
    
    def test_sliding_window(self):
        """Test sliding window creation."""
        data = np.arange(10)
        windows = segmentation.sliding_window(data, window_size=3, step=1)
        assert windows.shape[0] == 8
        assert windows.shape[1] == 3
    
    def test_uniform_segmentation(self):
        """Test uniform segmentation."""
        data = np.arange(100)
        segments = segmentation.uniform_segmentation(data, n_segments=5)
        assert len(segments) == 5


if __name__ == '__main__':
    pytest.main([__file__])
