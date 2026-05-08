"""Temporal feature extraction for time-series data."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


def extract_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from time-series data.
    
    Args:
        data: Input time-series data
        
    Returns:
        Dictionary of statistical features
    """
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'range': np.max(data) - np.min(data),
        'rms': np.sqrt(np.mean(data**2)),
    }
    return features


def extract_autocorrelation_features(data: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
    """
    Extract autocorrelation features.
    
    Args:
        data: Input time-series data
        max_lag: Maximum lag for autocorrelation
        
    Returns:
        Dictionary of autocorrelation features
    """
    features = {}
    acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    
    for i in range(1, min(max_lag + 1, len(acf))):
        features[f'acf_lag_{i}'] = acf[i]
    
    return features


def extract_trend_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract trend-related features.
    
    Args:
        data: Input time-series data
        
    Returns:
        Dictionary of trend features
    """
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, 1)
    trend = np.polyval(coeffs, x)
    residuals = data - trend
    
    features = {
        'trend_slope': coeffs[0],
        'trend_intercept': coeffs[1],
        'trend_explained_var': 1 - np.var(residuals) / np.var(data),
    }
    return features


def extract_difference_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract features from first-order differences.
    
    Args:
        data: Input time-series data
        
    Returns:
        Dictionary of difference features
    """
    diff = np.diff(data)
    features = {
        'diff_mean': np.mean(diff),
        'diff_std': np.std(diff),
        'diff_min': np.min(diff),
        'diff_max': np.max(diff),
    }
    return features
