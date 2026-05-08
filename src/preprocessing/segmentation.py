"""Time-series segmentation utilities."""

import numpy as np
import pandas as pd
from typing import List, Tuple


def sliding_window(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Create sliding windows from time-series data.
    
    Args:
        data: Input time-series data
        window_size: Size of the window
        step: Step size for sliding window
        
    Returns:
        Array of shape (n_windows, window_size)
    """
    n_windows = (len(data) - window_size) // step + 1
    windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        windows[i] = data[i*step:i*step+window_size]
    
    return windows


def segment_by_changepoint(data: np.ndarray, threshold: float = 1.0) -> List[Tuple[int, int]]:
    """
    Segment time-series based on changepoints.
    
    Args:
        data: Input time-series data
        threshold: Threshold for detecting changepoints
        
    Returns:
        List of (start, end) tuples for each segment
    """
    differences = np.abs(np.diff(data))
    changepoints = np.where(differences > threshold)[0] + 1
    
    segments = []
    start = 0
    for cp in changepoints:
        segments.append((start, cp))
        start = cp
    segments.append((start, len(data)))
    
    return segments


def uniform_segmentation(data: np.ndarray, n_segments: int) -> List[Tuple[int, int]]:
    """
    Divide time-series into uniform segments.
    
    Args:
        data: Input time-series data
        n_segments: Number of segments
        
    Returns:
        List of (start, end) tuples for each segment
    """
    segment_size = len(data) // n_segments
    segments = []
    
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(data)
        segments.append((start, end))
    
    return segments
