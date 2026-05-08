"""Project constants."""

# Class labels
APPLIANCE_CLASSES = {
    0: 'Refrigerator',
    1: 'Microwave',
    2: 'Dishwasher',
    3: 'Washer',
    4: 'Dryer',
}

# Feature types
TEMPORAL_FEATURES = [
    'mean', 'std', 'min', 'max', 'median',
    'skewness', 'kurtosis', 'rms'
]

SPECTRAL_FEATURES = [
    'fft_max_power', 'fft_mean_power', 'spectral_centroid',
    'welch_max_psd', 'welch_total_power'
]

# Color palette for visualizations
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9896',
}

# Model names
MODELS = {
    'baseline_knn': 'KNN Baseline',
    'baseline_dt': 'Decision Tree Baseline',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'rocket': 'ROCKET',
}
