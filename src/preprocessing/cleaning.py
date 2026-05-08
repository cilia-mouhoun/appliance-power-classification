"""Data cleaning utilities for time-series data."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.signal import savgol_filter


def remove_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Remove outliers from time-series data.
    
    Args:
        data: Input time-series data
        method: 'iqr' or 'zscore'
        threshold: Threshold for IQR method
        
    Returns:
        Cleaned data with outliers replaced by NaN or interpolation
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        data_clean = data.copy()
        data_clean[(data < lower_bound) | (data > upper_bound)] = np.nan
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        data_clean = data.copy()
        data_clean[z_scores > threshold] = np.nan
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return data_clean


def smooth_signal(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth the signal.
    
    Args:
        data: Input signal array
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial to fit
        
    Returns:
        Smoothed signal
    """
    # Ensure window_length is odd and smaller than data length
    if window_length >= len(data):
        window_length = len(data) if len(data) % 2 != 0 else len(data) - 1
    if window_length < 3:
        return data
        
    return savgol_filter(data, window_length, polyorder)


def handle_missing_values(data: pd.Series, method: str = 'interpolate') -> pd.Series:
    """
    Handle missing values in time-series data.
    
    Args:
        data: Input time-series data
        method: 'interpolate', 'forward_fill', or 'drop'
        
    Returns:
        Data with missing values handled
    """
    if method == 'interpolate':
        return data.interpolate(method='linear', limit_direction='both')
    elif method == 'forward_fill':
        return data.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def remove_trend(data: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Remove linear trend from time-series data.
    
    Args:
        data: Input time-series data
        
    Returns:
        Detrended data and trend component
    """
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data.values, 1)
    trend = np.polyval(coeffs, x)
    detrended = data.values - trend
    
    return pd.Series(detrended, index=data.index), pd.Series(trend, index=data.index)
