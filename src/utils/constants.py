"""Project constants."""

# 10 appliance classes as specified
APPLIANCE_CLASSES = {
    0: 'Mobile Phone Chargers',
    1: 'Coffee Machines',
    2: 'Computer Stations',
    3: 'Fridges/Freezers',
    4: 'Hi-Fi Systems',
    5: 'Lamps (CFL)',
    6: 'Laptops',
    7: 'Microwave Ovens',
    8: 'Printers',
    9: 'Televisions',
}

NUM_CLASSES = len(APPLIANCE_CLASSES)

# Temporal features list
TEMPORAL_FEATURES = [
    'rms', 'variance', 'skewness', 'kurtosis', 'entropy',
    'zero_crossing_rate', 'energy', 'peak_value',
    'peak_count', 'mean', 'std', 'min', 'max', 'median'
]

# Spectral features list
SPECTRAL_FEATURES = [
    'fft_max', 'fft_mean', 'spectral_centroid', 'spectral_entropy',
    'spectral_rolloff', 'spectral_spread', 'dominant_frequency'
]

# Feature domain names
DOMAINS = ['temporal', 'spectral', 'hybrid']

# Random seeds
RANDOM_SEED = 42

# Color palette for visualizations
COLORS = {
    'temporal': '#1f77b4',      # Blue
    'spectral': '#ff7f0e',      # Orange
    'hybrid': '#2ca02c',        # Green
    'success': '#2ca02c',
    'warning': '#ff9896',
    'danger': '#d62728',
}

# Model names and display names
MODEL_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'svm': 'Support Vector Machine',
    'knn': 'k-Nearest Neighbors',
    'rocket': 'ROCKET',
}

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
