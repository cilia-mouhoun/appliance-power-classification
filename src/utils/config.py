"""Configuration utilities and constants."""

import os
from pathlib import Path


class Config:
    """Configuration class for the project."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    EXTERNAL_DATA_DIR = DATA_DIR / 'external'
    
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'
    FIGURES_DIR = OUTPUT_DIR / 'figures'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
    REPORTS_DIR = OUTPUT_DIR / 'reports'
    
    # Parameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Feature extraction parameters
    WINDOW_SIZE = 256
    STEP_SIZE = 128
    
    # Model parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    LEARNING_RATE = 0.1
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR,
                        cls.FIGURES_DIR, cls.MODELS_DIR, cls.PREDICTIONS_DIR, cls.REPORTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
