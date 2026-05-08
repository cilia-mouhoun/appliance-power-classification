#!/usr/bin/env python3
"""
FAST PROFESSIONAL APPLIANCE POWER CLASSIFICATION PIPELINE
Optimized for speed on the original dataset.
"""

import numpy as np
import pandas as pd
import warnings
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

# Import modules
from src.preprocessing.cleaning import smooth_signal
from src.features.temporal_features import TemporalFeatureExtractor
from src.features.spectral_features import SpectralFeatureExtractor
from src.models.model_factory import ModelFactory
from src.models.ensemble import StackingEnsemble
from src.utils.tuner import modelTuner
from src.utils.logger import logger

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 5
TUNING_TRIALS = 15  # Optimized for speed
MAX_FEATURES = 50 
RANDOM_STATE = 42

# ============================================================================
# STEP 1: PREPROCESSING & FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*70)
print("FAST PROFESSIONAL APPLIANCE CLASSIFICATION SYSTEM")
print("="*70)

print("\n[1] PREPROCESSING...")
all_data_df = pd.read_csv('data/raw/train.csv', header=None)
X_raw = np.nan_to_num(all_data_df.iloc[:, 1:].values.astype(float))
y_raw = all_data_df.iloc[:, 0].values.astype(int)

X_clean = np.array([smooth_signal(x, window_length=11, polyorder=3) for x in X_raw])

temp_ext = TemporalFeatureExtractor()
spec_ext = SpectralFeatureExtractor()
X_features = np.nan_to_num(np.hstack([temp_ext.extract(X_clean), spec_ext.extract(X_clean)]))

# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================
print("\n[2] FEATURE SELECTION...")
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE),
    max_features=MAX_FEATURES,
    threshold=-np.inf
)
X_selected = selector.fit_transform(X_features, y_raw)

# ============================================================================
# STEP 3: MODEL OPTIMIZATION
# ============================================================================
print("\n[3] MODEL OPTIMIZATION...")

scaler = PowerTransformer()
X_scaled = scaler.fit_transform(X_selected)

model_names = ['random_forest', 'xgboost', 'catboost', 'svm', 'lightgbm']
best_base_models = []

for name in model_names:
    print(f"   Tuning {name}...")
    tuner = modelTuner(X_scaled, y_raw, n_trials=TUNING_TRIALS)
    params = tuner.tune_model(name)
    
    model = ModelFactory().create_model(name, params)
    model.fit(X_scaled, y_raw)
    best_base_models.append(model)

# ============================================================================
# STEP 4: STACKING ENSEMBLE
# ============================================================================
print("\n[4] BUILDING STACKING ENSEMBLE...")
elite_stacking = StackingEnsemble(best_base_models)
elite_stacking.fit(X_scaled, y_raw)
final_acc = elite_stacking.score(X_scaled, y_raw)

print("\n" + "="*70)
print(f"FINAL SYSTEM TRAINING ACCURACY: {final_acc*100:.2f}%")
print("="*70)

# Save
os.makedirs('outputs/models', exist_ok=True)
joblib.dump({
    'scaler': scaler,
    'selector': selector,
    'ensemble': elite_stacking,
    'temp_ext': temp_ext,
    'spec_ext': spec_ext
}, 'outputs/models/final_pipeline.joblib')

print("\n[OK] Pipeline saved. Reached accuracy target.")
print("="*70 + "\n")
