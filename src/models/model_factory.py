"""Model factory for instantiating and configuring classification models."""

from typing import Dict, Any, Literal
import numpy as np

from .baseline import BaselineClassifier
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .svm_model import SVMModel
from .knn_model import KNNModel
from .rocket import RocketModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel


ModelType = Literal['baseline', 'random_forest', 'xgboost', 'svm', 'knn', 'rocket', 'lightgbm', 'catboost']

# Default hyperparameter configurations for each model
DEFAULT_CONFIG = {
    'baseline': {
        'model_type': 'lr',  # 'knn', 'dt', 'lr'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42,
        'n_jobs': -1,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'random_state': 42,
        'probability': True,
    },
    'knn': {
        'n_neighbors': 5,
        'metric': 'euclidean',
        'weights': 'uniform',
        'n_jobs': -1,
    },
    'rocket': {
        'n_kernels': 10000,
        'kernel_size': None,
        'random_state': 42,
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42,
    },
    'catboost': {
        'iterations': 100,
        'learning_rate': 0.1,
        'random_state': 42,
    },
}

# Model descriptions and interpretability notes
MODEL_DESCRIPTIONS = {
    'baseline': {
        'name': 'Logistic Regression (Baseline)',
        'interpretability': 'High',
        'speed': 'Very Fast',
        'memory': 'Low',
        'pros': ['Linear interpretability', 'Probabilistic', 'Fast'],
        'cons': ['Limited for complex patterns', 'Requires feature scaling'],
    },
    'random_forest': {
        'name': 'Random Forest',
        'interpretability': 'Medium-High',
        'speed': 'Fast',
        'memory': 'Medium',
        'pros': ['Feature importance', 'Handles non-linearity', 'Robust to outliers'],
        'cons': ['Black box for individual trees', 'High memory for large forests'],
    },
    'xgboost': {
        'name': 'XGBoost',
        'interpretability': 'Medium',
        'speed': 'Medium',
        'memory': 'Medium-High',
        'pros': ['State-of-the-art performance', 'Feature importance', 'Handles class imbalance'],
        'cons': ['Complex hyperparameter tuning', 'Prone to overfitting'],
    },
    'svm': {
        'name': 'Support Vector Machine',
        'interpretability': 'Low',
        'speed': 'Medium (prediction slow)',
        'memory': 'Low-Medium',
        'pros': ['Effective in high dimensions', 'Kernel methods for non-linearity', 'Well-founded theory'],
        'cons': ['Hard to interpret', 'Slow predictions on large data', 'Sensitive to scaling'],
    },
    'knn': {
        'name': 'k-Nearest Neighbors',
        'interpretability': 'High',
        'speed': 'Slow (lazy learner)',
        'memory': 'High',
        'pros': ['Highly interpretable', 'No training phase', 'Adaptive neighborhoods'],
        'cons': ['Slow predictions', 'Curse of dimensionality', 'Memory intensive'],
    },
    'rocket': {
        'name': 'ROCKET',
        'interpretability': 'Medium',
        'speed': 'Medium-Fast',
        'memory': 'Medium',
        'pros': ['Specialized for time series', 'Fast training', 'Captures temporal patterns'],
        'cons': ['Less interpretable', 'Requires kernel tuning', 'May miss spectral patterns'],
    },
    'lightgbm': {
        'name': 'LightGBM',
        'interpretability': 'Medium',
        'speed': 'Very Fast',
        'memory': 'Low',
        'pros': ['Extremely fast', 'Handles large data', 'Great performance'],
        'cons': ['Can overfit on small data'],
    },
    'catboost': {
        'name': 'CatBoost',
        'interpretability': 'Medium',
        'speed': 'Medium',
        'memory': 'Medium',
        'pros': ['Excellent defaults', 'Robust to overfitting', 'Handles categorical (not used here)'],
        'cons': ['Slower than LightGBM'],
    },
}


class ModelFactory:
    """
    Factory class for creating and configuring classification models.
    
    Provides a unified interface for model instantiation with consistent
    hyperparameters and configurations.
    
    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create_model('random_forest')
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
    """
    
    @staticmethod
    def create_model(model_type: ModelType, config: Dict[str, Any] = None) -> Any:
        """
        Create a classification model instance.
        
        Args:
            model_type: Type of model to create
                - 'baseline': Logistic Regression baseline
                - 'random_forest': Random Forest ensemble
                - 'xgboost': XGBoost gradient boosting
                - 'svm': Support Vector Machine
                - 'knn': k-Nearest Neighbors
                - 'rocket': Random Convolutional Kernel Transform
            config: Optional hyperparameter configuration. If not provided,
                    uses default configuration.
        
        Returns:
            Instantiated model object with fit, predict, predict_proba methods
            
        Raises:
            ValueError: If model_type is not recognized
            
        Example:
            >>> # Create with defaults
            >>> rf_model = ModelFactory.create_model('random_forest')
            >>> 
            >>> # Create with custom config
            >>> custom_config = {'n_estimators': 200, 'max_depth': 20}
            >>> rf_custom = ModelFactory.create_model('random_forest', custom_config)
        """
        if model_type not in DEFAULT_CONFIG:
            available = list(DEFAULT_CONFIG.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available models: {available}"
            )
        
        # Merge provided config with defaults
        final_config = DEFAULT_CONFIG[model_type].copy()
        if config is not None:
            final_config.update(config)
        
        # Create model based on type
        if model_type == 'baseline':
            return BaselineClassifier(**final_config)
        elif model_type == 'random_forest':
            return RandomForestModel(**final_config)
        elif model_type == 'xgboost':
            return XGBoostModel(**final_config)
        elif model_type == 'svm':
            return SVMModel(**final_config)
        elif model_type == 'knn':
            return KNNModel(**final_config)
        elif model_type == 'rocket':
            return RocketModel(**final_config)
        elif model_type == 'lightgbm':
            return LightGBMModel(**final_config)
        elif model_type == 'catboost':
            return CatBoostModel(**final_config)
    
    @staticmethod
    def get_model_info(model_type: ModelType) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model description, interpretability, speed, etc.
        """
        if model_type not in MODEL_DESCRIPTIONS:
            raise ValueError(f"Unknown model type '{model_type}'")
        return MODEL_DESCRIPTIONS[model_type]
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available model types.
        
        Returns:
            List of supported model type strings
        """
        return list(DEFAULT_CONFIG.keys())
    
    @staticmethod
    def get_all_models_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information for all available models.
        
        Returns:
            Dictionary mapping model types to their descriptions
        """
        return MODEL_DESCRIPTIONS.copy()
    
    @staticmethod
    def create_all_models(configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create instances of all available models.
        
        Useful for running comprehensive model comparisons.
        
        Args:
            configs: Optional dictionary mapping model types to their custom configs.
                     Example: {'random_forest': {'n_estimators': 200}, 'svm': {'C': 10}}
        
        Returns:
            Dictionary mapping model names to model instances
            
        Example:
            >>> models = ModelFactory.create_all_models()
            >>> for name, model in models.items():
            >>>     model.fit(X_train, y_train)
            >>>     score = model.score(X_test, y_test)
            >>>     print(f"{name}: {score:.3f}")
        """
        if configs is None:
            configs = {}
        
        models = {}
        for model_type in ModelFactory.get_available_models():
            config = configs.get(model_type, None)
            models[model_type] = ModelFactory.create_model(model_type, config)
        
        return models
    
    @staticmethod
    def get_default_config(model_type: ModelType) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with default hyperparameters
        """
        if model_type not in DEFAULT_CONFIG:
            raise ValueError(f"Unknown model type '{model_type}'")
        return DEFAULT_CONFIG[model_type].copy()
