"""Enhanced configuration management."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    normalization_method: str = 'zscore'  # 'zscore' or 'minmax'
    detrending_method: str = 'linear'     # 'linear' or 'polynomial'
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    segmentation_window: Optional[int] = None
    segmentation_step: Optional[int] = None


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    # Temporal
    extract_temporal: bool = True
    temporal_lags: int = 5
    
    # Spectral
    extract_spectral: bool = True
    fft_nperseg: int = 256
    spectral_bands: list = None
    
    # Selection
    feature_selection_method: str = 'mutual_info'  # 'mutual_info' or 'correlation'
    n_features_to_keep: Optional[int] = None
    correlation_threshold: float = 0.95


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str
    hyperparameters: dict = None
    random_state: int = 42


class Config:
    """Main configuration class."""
    
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
    LOGS_DIR = OUTPUT_DIR / 'logs'
    
    EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
    TEMPORAL_EXPERIMENT_DIR = EXPERIMENTS_DIR / 'temporal_only'
    SPECTRAL_EXPERIMENT_DIR = EXPERIMENTS_DIR / 'spectral_only'
    HYBRID_EXPERIMENT_DIR = EXPERIMENTS_DIR / 'hybrid'
    
    CONFIGS_DIR = PROJECT_ROOT / 'configs'
    
    # Default configs
    preprocessing = PreprocessingConfig()
    features = FeatureConfig(spectral_bands=[0, 10, 50, 100, 200])
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [
            cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR,
            cls.FIGURES_DIR, cls.MODELS_DIR, cls.PREDICTIONS_DIR, cls.REPORTS_DIR, cls.LOGS_DIR,
            cls.TEMPORAL_EXPERIMENT_DIR, cls.SPECTRAL_EXPERIMENT_DIR, cls.HYBRID_EXPERIMENT_DIR,
            cls.CONFIGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_yaml(cls, config_file: Path) -> dict:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def save_yaml(cls, config_dict: dict, output_file: Path):
        """Save configuration to YAML file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Initialize directories
Config.create_directories()
