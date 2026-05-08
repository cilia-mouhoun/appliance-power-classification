"""Hybrid domain classification pipeline."""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
import time

from ..features.hybrid_features import HybridFeatureExtractor
from ..models.model_factory import ModelFactory
from ..evaluation.evaluation_pipeline import EvaluationPipeline
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from ..utils.logger import setup_logger
from ..utils.constants import DEFAULT_SAMPLING_RATE


logger = setup_logger(__name__)


class HybridPipeline:
    """
    Hybrid domain classification pipeline.
    
    Combines temporal and spectral features for enhanced classification.
    
    Workflow:
    1. Preprocessing: Cleaning, normalization, detrending
    2. Feature extraction: Temporal + Spectral features
    3. Feature selection: Mutual information selection
    4. Dimensionality reduction: Optional PCA
    5. Model training: Fit classifier
    6. Evaluation: Metrics, comparison with single-domain pipelines
    
    Hybrid Approach Advantages:
    - Leverages complementary information from both domains
    - Can capture temporal dynamics AND frequency content
    - More robust to domain-specific artifacts
    
    Challenges:
    - Curse of dimensionality (>50 features)
    - Feature redundancy (temporal energy ≈ spectral power)
    - Increased computational cost
    """
    
    def __init__(self, model_type: str = 'random_forest', model_config: Dict = None,
                 preprocessing_config: Dict = None, use_pca: bool = True,
                 pca_components: int = 30, use_feature_selection: bool = True,
                 sampling_rate: float = None):
        """
        Initialize hybrid pipeline.
        
        Args:
            model_type: Type of model to use
            model_config: Custom model hyperparameters
            preprocessing_config: Custom preprocessing parameters
            use_pca: Whether to apply PCA
            pca_components: Number of PCA components
            use_feature_selection: Whether to use mutual information selection
            sampling_rate: Sampling rate in Hz
        """
        self.model_type = model_type
        self.model = ModelFactory.create_model(model_type, model_config)
        
        self.preprocessor = PreprocessingPipeline(preprocessing_config or {})
        self.feature_extractor = HybridFeatureExtractor(
            use_pca=use_pca,
            pca_components=pca_components,
            use_mutual_info_selection=use_feature_selection,
            n_features_to_select=min(50, pca_components) if use_feature_selection else None
        )
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
        Extract hybrid features from preprocessed signals.
        
        Returns:
            Tuple of (X_train_features, X_test_features)
        """
        if self.X_train is None:
            raise ValueError("Must call preprocess_data first")
        
        logger.info("Extracting hybrid features (temporal + spectral)...")
        start_time = time.time()
        
        # Extract raw features
        X_train_raw = self.feature_extractor.extract_batch_features(
            self.X_train, sampling_rate=self.sampling_rate
        )
        X_test_raw = self.feature_extractor.extract_batch_features(
            self.X_test, sampling_rate=self.sampling_rate
        )
        
        logger.info(f"Raw hybrid features: {X_train_raw.shape[1]} features")
        
        # Fit feature selection and PCA on training data
        self.feature_extractor.fit(X_train_raw, self.y_train)
        
        # Transform features
        X_train_feat = self.feature_extractor.transform(X_train_raw)
        X_test_feat = self.feature_extractor.transform(X_test_raw)
        
        elapsed = time.time() - start_time
        logger.info(f"Feature extraction complete in {elapsed:.2f}s. "
                   f"Final features per sample: {X_train_feat.shape[1]}")
        logger.info(f"Dimensionality reduction: {X_train_raw.shape[1]} -> {X_train_feat.shape[1]}")
        
        return X_train_feat, X_test_feat
    
    def train(self, X_train_feat: np.ndarray) -> 'HybridPipeline':
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
        logger.info("HYBRID PIPELINE (Temporal + Spectral)")
        logger.info("=" * 60)
        
        # Preprocess
        self.preprocess_data(X, y, test_size=test_size)
        
        # Extract features
        X_train_feat, X_test_feat = self.extract_features()
        
        # Train
        self.train(X_train_feat)
        
        # Evaluate
        results = self.evaluate(X_test_feat)
        
        results['pipeline'] = 'hybrid'
        results['n_features'] = X_train_feat.shape[1]
        results['feature_extractor'] = 'HybridFeatureExtractor'
        results['sampling_rate'] = self.sampling_rate
        
        # Add feature reduction info
        if hasattr(self.feature_extractor, 'get_n_features_after_reduction'):
            results['final_n_features'] = self.feature_extractor.get_n_features_after_reduction()
        
        logger.info("=" * 60)
        
        return results
