"""Spectral domain classification pipeline."""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
import time

from ..features.spectral_features import SpectralFeatureExtractor
from ..models.model_factory import ModelFactory
from ..evaluation.evaluation_pipeline import EvaluationPipeline
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from ..utils.logger import setup_logger
from ..utils.constants import DEFAULT_SAMPLING_RATE


logger = setup_logger(__name__)


class SpectralPipeline:
    """
    Spectral domain classification pipeline.
    
    Workflow:
    1. Preprocessing: Cleaning, normalization, detrending
    2. Feature extraction: Spectral features (FFT, PSD, harmonics, etc.)
    3. Model training: Fit classifier
    4. Evaluation: Metrics, confusion matrix, analysis
    
    Spectral features capture frequency content and harmonic characteristics
    of appliance signals, complementary to temporal domain analysis.
    """
    
    def __init__(self, model_type: str = 'random_forest', model_config: Dict = None,
                 preprocessing_config: Dict = None, sampling_rate: float = None):
        """
        Initialize spectral pipeline.
        
        Args:
            model_type: Type of model to use
            model_config: Custom model hyperparameters
            preprocessing_config: Custom preprocessing parameters
            sampling_rate: Sampling rate in Hz (default from constants)
        """
        self.model_type = model_type
        self.model = ModelFactory.create_model(model_type, model_config)
        
        self.preprocessor = PreprocessingPipeline(preprocessing_config or {})
        self.feature_extractor = SpectralFeatureExtractor()
        self.sampling_rate = sampling_rate or DEFAULT_SAMPLING_RATE
        self.evaluator = EvaluationPipeline()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray,
                       test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Preprocess and split data.
        
        Args:
            X: Input signals (n_samples, signal_length)
            y: Target labels
            test_size: Fraction of data for testing
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing...")
        
        # Preprocess all signals
        X_processed = np.array([self.preprocessor.preprocess(signal) for signal in X])
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Preprocessing complete. Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spectral features from preprocessed signals.
        
        Returns:
            Tuple of (X_train_features, X_test_features)
        """
        if self.X_train is None:
            raise ValueError("Must call preprocess_data first")
        
        logger.info("Extracting spectral features...")
        start_time = time.time()
        
        X_train_feat = self.feature_extractor.extract_batch_features(
            self.X_train, sampling_rate=self.sampling_rate
        )
        X_test_feat = self.feature_extractor.extract_batch_features(
            self.X_test, sampling_rate=self.sampling_rate
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Feature extraction complete in {elapsed:.2f}s. "
                   f"Features per sample: {X_train_feat.shape[1]}")
        
        return X_train_feat, X_test_feat
    
    def train(self, X_train_feat: np.ndarray) -> 'SpectralPipeline':
        """
        Train the classifier.
        
        Args:
            X_train_feat: Training features
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()
        
        self.model.fit(X_train_feat, self.y_train)
        
        elapsed = time.time() - start_time
        logger.info(f"Model training complete in {elapsed:.2f}s")
        
        return self
    
    def evaluate(self, X_test_feat: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate on test set.
        
        Args:
            X_test_feat: Test features
        
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test_feat)
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(X_test_feat)
            except:
                pass
        
        results = self.evaluator.evaluate_single_model(
            self.y_test, y_pred, y_proba, self.model_type
        )
        
        self.results = results
        logger.info(f"Accuracy: {results['overall']['accuracy']:.3f}")
        
        return results
    
    def run(self, X: np.ndarray, y: np.ndarray,
           test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            X: Input signals
            y: Target labels
            test_size: Test set fraction
        
        Returns:
            Pipeline results
        """
        logger.info("=" * 60)
        logger.info("SPECTRAL PIPELINE")
        logger.info("=" * 60)
        
        # Preprocess
        self.preprocess_data(X, y, test_size=test_size)
        
        # Extract features
        X_train_feat, X_test_feat = self.extract_features()
        
        # Train
        self.train(X_train_feat)
        
        # Evaluate
        results = self.evaluate(X_test_feat)
        
        results['pipeline'] = 'spectral'
        results['n_features'] = X_train_feat.shape[1]
        results['feature_extractor'] = 'SpectralFeatureExtractor'
        results['sampling_rate'] = self.sampling_rate
        
        logger.info("=" * 60)
        
        return results
