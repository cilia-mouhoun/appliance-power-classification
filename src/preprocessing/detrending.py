"""Detrending utilities for time-series data."""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple


def linear_detrend(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove linear trend using least squares fitting.
    
    Mathematical intuition:
    - Fits a line y = mx + b to the data using least squares
    - Removes the linear component, preserving fluctuations around the trend
    - Important for appliance signals that may have slowly varying trends
    
    Args:
        data: Input time-series array
        
    Returns:
        Detrended data and trend component
    """
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, 1)
    trend = np.polyval(coeffs, x)
    detrended = data - trend
    
    return detrended, trend


def polynomial_detrend(data: np.ndarray, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove polynomial trend from time-series.
    
    Mathematical intuition:
    - Fits a polynomial of specified order to the data
    - Higher orders capture more complex trend patterns
    - Useful for signals with non-linear drift
    
    Args:
        data: Input time-series array
        order: Polynomial order (default: 2 for quadratic)
        
    Returns:
        Detrended data and trend component
    """
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, order)
    trend = np.polyval(coeffs, x)
    detrended = data - trend
    
    return detrended, trend


def seasonal_detrend(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove seasonal component using seasonal decomposition.
    
    Mathematical intuition:
    - Decomposes signal into trend + seasonal + residual
    - Removes both trend and seasonal patterns
    - Useful for appliances with cyclical power consumption
    
    Args:
        data: Input time-series array
        period: Seasonal period (e.g., 24 for hourly data with daily patterns)
        
    Returns:
        Detrended (residual) data and removed component (trend + seasonal)
    """
    from scipy.signal import detrend as scipy_detrend
    
    # Simple moving average for trend
    if len(data) > period:
        trend = pd.Series(data).rolling(window=period, center=True).mean().values
        trend = np.nan_to_num(trend, nan=np.nanmean(data))
    else:
        trend = np.ones_like(data) * np.mean(data)
    
    residual = data - trend
    return residual, trend


def hp_filter_detrend(data: np.ndarray, lamb: float = 1600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott filter for detrending.
    
    Mathematical intuition:
    - Solves optimization problem: minimize sum of squared deviations + lambda * sum of squared second differences
    - Lambda controls smoothness: higher lambda = smoother trend
    - Particularly useful for economic/power data with cyclical patterns
    
    Args:
        data: Input time-series array
        lamb: HP filter parameter (default 1600 for annual data)
        
    Returns:
        Detrended data and trend component
    """
    T = len(data)
    
    # Build second difference matrix
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    # HP filter formula: trend = (I + lambda * D'D)^-1 * data
    I = np.eye(T)
    trend = np.linalg.solve(I + lamb * (D.T @ D), data)
    detrended = data - trend
    
    return detrended, trend
