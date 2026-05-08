"""Normalization utilities for time-series data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standardize(data: pd.DataFrame, scaler=None):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: Input data
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        Standardized data and fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)
    
    return pd.DataFrame(data_scaled, columns=data.columns, index=data.index), scaler


def normalize(data: pd.DataFrame, scaler=None, feature_range: tuple = (0, 1)):
    """
    Normalize data to a specific range.
    
    Args:
        data: Input data
        scaler: Pre-fitted scaler (optional)
        feature_range: Target range for normalization
        
    Returns:
        Normalized data and fitted scaler
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)
    
    return pd.DataFrame(data_scaled, columns=data.columns, index=data.index), scaler


def robust_scale(data: pd.DataFrame):
    """
    Robust scaling using median and IQR (less sensitive to outliers).
    
    Args:
        data: Input data
        
    Returns:
        Robustly scaled data
    """
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    data_scaled = (data - median) / iqr
    return data_scaled
