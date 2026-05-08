"""Helper functions for common tasks."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Dict, List


def save_pickle(obj: Any, filepath: Path) -> None:
    """Save object to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_model(model: Any, filepath: Path) -> None:
    """Save trained model."""
    save_pickle(model, filepath)


def load_model(filepath: Path) -> Any:
    """Load trained model."""
    return load_pickle(filepath)


def train_test_split_time_series(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Split time-series data into train and test sets maintaining temporal order.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """Print formatted metrics summary."""
    print("\n" + "="*50)
    print("MODEL METRICS SUMMARY")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper():<20}: {metric_value:.4f}")
    print("="*50 + "\n")
