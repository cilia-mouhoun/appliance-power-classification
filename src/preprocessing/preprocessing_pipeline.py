"""Unified preprocessing pipeline for time-series data."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from . import cleaning, normalization, detrending, segmentation
from ..utils.logger import logger


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for time-series classification.
    
    This pipeline handles:
    1. Data loading and validation
    2. Missing value handling
    3. Outlier detection and removal
    4. Detrending (optional)
    5. Normalization/standardization
    6. Train/test/validation splitting
    7. Optional segmentation
    """
    
    def __init__(
        self,
        normalization_method: str = 'zscore',
        detrending_method: Optional[str] = 'linear',
        remove_outliers: bool = True,
        outlier_method: str = 'iqr',
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            normalization_method: 'zscore' or 'minmax'
            detrending_method: None, 'linear', 'polynomial', 'hp', or 'seasonal'
            remove_outliers: Whether to remove outliers
            outlier_method: 'iqr' or 'zscore'
            test_size: Proportion of data for testing
            validation_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
        """
        self.normalization_method = normalization_method
        self.detrending_method = detrending_method
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        self.scaler = None
        logger.info(f"Initialized preprocessing pipeline with {normalization_method} normalization")
    
    def process(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            X: Input features (time-series samples)
            y: Target labels (optional)
            fit: Whether to fit scalers (True for training data)
            
        Returns:
            Processed features and labels
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Handle missing values
        logger.debug("Handling missing values...")
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                X_processed[col] = cleaning.handle_missing_values(X_processed[col])
        
        # Step 2: Remove outliers
        if self.remove_outliers:
            logger.debug(f"Removing outliers using {self.outlier_method}...")
            for col in X_processed.columns:
                X_processed[col] = cleaning.remove_outliers(
                    X_processed[col],
                    method=self.outlier_method,
                    threshold=1.5
                )
        
        # Step 3: Detrending
        if self.detrending_method:
            logger.debug(f"Detrending using {self.detrending_method}...")
            X_processed = self._apply_detrending(X_processed)
        
        # Step 4: Normalization
        logger.debug(f"Normalizing using {self.normalization_method}...")
        X_processed, self.scaler = normalization.standardize(X_processed) if fit else \
                                   normalization.standardize(X_processed, scaler=self.scaler)
        
        logger.info(f"Preprocessing complete. Shape: {X_processed.shape}")
        
        return X_processed, y
    
    def _apply_detrending(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply detrending to all time-series."""
        X_detrended = X.copy()
        
        for col in X_detrended.columns:
            if self.detrending_method == 'linear':
                detrended, _ = detrending.linear_detrend(X_detrended[col].values)
            elif self.detrending_method == 'polynomial':
                detrended, _ = detrending.polynomial_detrend(X_detrended[col].values, order=2)
            elif self.detrending_method == 'hp':
                detrended, _ = detrending.hp_filter_detrend(X_detrended[col].values)
            elif self.detrending_method == 'seasonal':
                # Estimate period as 1/4 of length
                period = max(2, len(X_detrended[col]) // 4)
                detrended, _ = detrending.seasonal_detrend(X_detrended[col].values, period)
            else:
                logger.warning(f"Unknown detrending method: {self.detrending_method}")
                detrended = X_detrended[col].values
            
            X_detrended[col] = detrended
        
        return X_detrended
    
    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            X: Features
            y: Labels
            stratify: Whether to stratify splits by class
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Splitting data: test_size={self.test_size}, val_size={self.validation_size}")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
        
        # Second split: train vs val
        val_size = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp if stratify else None
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
