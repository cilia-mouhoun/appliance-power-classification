"""Tests for model module."""

import pytest
import numpy as np
from src.models import baseline, random_forest, xgboost_model


class TestBaselineModel:
    """Test cases for baseline models."""
    
    def test_baseline_knn_fit_predict(self):
        """Test KNN baseline fit and predict."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 0, 1, 1])
        
        model = baseline.BaselineClassifier(model_type='knn')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)
        assert model.score(X_train, y_train) > 0


class TestRandomForestModel:
    """Test cases for Random Forest model."""
    
    def test_rf_fit_predict(self):
        """Test Random Forest fit and predict."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 0, 1, 1])
        
        model = random_forest.RandomForestModel(n_estimators=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)
        assert model.feature_importance_ is not None


class TestXGBoostModel:
    """Test cases for XGBoost model."""
    
    def test_xgb_fit_predict(self):
        """Test XGBoost fit and predict."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
        y_train = np.array([0, 0, 1, 1])
        
        model = xgboost_model.XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)


if __name__ == '__main__':
    pytest.main([__file__])
