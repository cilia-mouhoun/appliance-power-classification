"""Hyperparameter tuning using Optuna."""

import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from ..models.model_factory import ModelFactory
from typing import Dict, Any, List

class modelTuner:
    """Tuner for finding best hyperparameters."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.best_params = {}

    def tune_model(self, model_type: str) -> Dict[str, Any]:
        """Run Optuna study for a specific model."""
        
        def objective(trial):
            params = {}
            if model_type == 'random_forest' or model_type == 'extra_trees':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 30)
            elif model_type == 'xgboost':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            elif model_type == 'lightgbm':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            elif model_type == 'svm':
                params['C'] = trial.suggest_float('C', 0.1, 100, log=True)
                params['gamma'] = trial.suggest_float('gamma', 1e-4, 1, log=True)
            elif model_type == 'catboost':
                params['iterations'] = trial.suggest_int('iterations', 50, 300)
                params['depth'] = trial.suggest_int('depth', 4, 10)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            
            try:
                factory = ModelFactory()
                model = factory.create_model(model_type, params)
                
                # Use CV for robust evaluation
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # We need a custom scorer or just use accuracy
                scores = []
                for train_idx, val_idx in skf.split(self.X, self.y):
                    X_tr, X_val = self.X[train_idx], self.X[val_idx]
                    y_tr, y_val = self.y[train_idx], self.y[val_idx]
                    
                    model.fit(X_tr, y_tr)
                    score = model.score(X_val, y_val)
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params[model_type] = study.best_params
        return study.best_params
