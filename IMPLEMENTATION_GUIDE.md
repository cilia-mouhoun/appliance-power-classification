# Appliance Power Classification - Implementation Guide

## 🎯 Project Overview

This is an academic-grade time-series classification project for appliance power consumption analysis. The project compares three representation domains:

1. **Temporal Domain**: Statistical and dynamic features (RMS, entropy, autocorrelation, etc.)
2. **Spectral Domain**: Frequency-based features (FFT, PSD, harmonics, spectral centroid, etc.)
3. **Hybrid Domain**: Combined temporal + spectral features with intelligent feature selection

## 📊 Recent Implementations

### New Models

- **SVMModel** (`src/models/svm_model.py`): Support Vector Machine with kernel selection
- **KNNModel** (`src/models/knn_model.py`): k-Nearest Neighbors with interpretable neighbor analysis
- **ModelFactory** (`src/models/model_factory.py`): Unified interface for all 6 model types

### Enhanced Features

- **HybridFeatureExtractor** (`src/features/hybrid_features.py`): Combines temporal + spectral features with:
  - Mutual information feature selection
  - Optional PCA for dimensionality reduction
  - Statistics and feature importance analysis

### Advanced Evaluation

- **statistical_tests.py**: McNemar's test, paired t-tests, Kruskal-Wallis test, Cohen's d
- **evaluation_pipeline.py**: Comprehensive evaluation framework with:
  - Per-class metrics
  - Confusion matrix analysis
  - Error analysis
  - Model comparison utilities

### Pipeline Orchestration

- **TemporalPipeline**: Temporal-only domain classification
- **SpectralPipeline**: Spectral-only domain classification
- **HybridPipeline**: Hybrid temporal+spectral classification
- **UnifiedPipeline**: Orchestrates all three domains + generates comparative analysis

## 🚀 Quick Start

### Basic Usage: Run Complete Analysis

```python
from src.pipelines.unified_pipeline import UnifiedPipeline
import numpy as np

# Load your data
X = np.load('data.npy')  # (n_samples, signal_length)
y = np.load('labels.npy')  # (n_samples,)

# Create and run unified pipeline
pipeline = UnifiedPipeline(
    models=['random_forest', 'xgboost'],
    output_dir='outputs/analysis'
)

# Run complete analysis
results = pipeline.run_complete_analysis(X, y, test_size=0.2)

# View results
print(results['report'])
```

### Running Individual Pipelines

#### Temporal Domain

```python
from src.pipelines.temporal_pipeline import TemporalPipeline

temporal = TemporalPipeline(model_type='random_forest')
results = temporal.run(X, y, test_size=0.2)

print(f"Accuracy: {results['overall']['accuracy']:.3f}")
print(f"F1-Score: {results['overall']['f1']:.3f}")
```

#### Spectral Domain

```python
from src.pipelines.spectral_pipeline import SpectralPipeline

spectral = SpectralPipeline(model_type='random_forest')
results = spectral.run(X, y, test_size=0.2)
```

#### Hybrid Domain

```python
from src.pipelines.hybrid_pipeline import HybridPipeline

hybrid = HybridPipeline(
    model_type='random_forest',
    use_pca=True,
    pca_components=30
)
results = hybrid.run(X, y, test_size=0.2)
```

## 🔧 Feature Extraction

### Temporal Features (30+ features)

- **Statistical**: RMS, variance, std_dev, skewness, kurtosis, entropy
- **Autocorrelation**: lag-1, lag-5, lag-10
- **Peaks**: peak value, peak count, zero-crossing rate
- **Energy**: signal energy, mean energy

```python
from src.features.temporal_features import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor()
features, feature_names = extractor.extract_features(
    signal, return_feature_names=True
)
```

### Spectral Features (22+ features)

- **FFT**: Power spectrum statistics
- **Frequency Domain**: Dominant frequency, spectral centroid, spectral entropy
- **Harmonics**: Fundamental frequency, harmonic amplitudes
- **Band Powers**: DC, low/mid/high frequency power

```python
from src.features.spectral_features import SpectralFeatureExtractor

extractor = SpectralFeatureExtractor()
features, feature_names = extractor.extract_features(
    signal, sampling_rate=1.0, return_feature_names=True
)
```

### Hybrid Features (50+ features → dimensionality reduction)

Combines both domains with feature selection and PCA.

```python
from src.features.hybrid_features import HybridFeatureExtractor

extractor = HybridFeatureExtractor(
    use_pca=True,
    pca_components=30,
    use_mutual_info_selection=True,
    n_features_to_select=50
)

# Fit on training data
X_train_feat = extractor.fit_transform(X_train_raw, y_train)

# Transform test data
X_test_feat = extractor.transform(X_test_raw)
```

## 🤖 Model Usage

### Model Factory

```python
from src.models.model_factory import ModelFactory

# Create model with default config
model = ModelFactory.create_model('random_forest')

# Create with custom hyperparameters
config = {'n_estimators': 200, 'max_depth': 20}
model = ModelFactory.create_model('random_forest', config)

# Create all models
all_models = ModelFactory.create_all_models()

# Get model info
info = ModelFactory.get_model_info('svm')
print(info)  # Interpretability, speed, pros/cons, etc.
```

### Available Models

- `baseline`: Logistic Regression
- `random_forest`: Random Forest Ensemble
- `xgboost`: XGBoost Gradient Boosting
- `svm`: Support Vector Machine
- `knn`: k-Nearest Neighbors
- `rocket`: Random Convolutional Kernel Transform

## 📈 Evaluation & Comparison

### Comprehensive Evaluation

```python
from src.evaluation.evaluation_pipeline import EvaluationPipeline

evaluator = EvaluationPipeline(target_names=class_names)

# Single model evaluation
results = evaluator.evaluate_single_model(
    y_true, y_pred, y_proba, model_name='RandomForest'
)

# Multiple model comparison
comparison = evaluator.compare_models(
    models_dict={'RF': rf_model, 'XGB': xgb_model},
    X_test=X_test, y_test=y_test
)
```

### Statistical Tests

```python
from src.evaluation.statistical_tests import (
    mcnemar_test, paired_t_test, effect_size_cohens_d
)

# Compare two models
result = mcnemar_test(y_pred_1, y_pred_2, y_true)
print(f"Significant difference: {result['significant']}")

# Effect size
d = effect_size_cohens_d(scores_1, scores_2)
print(f"Cohen's d: {d:.3f}")
```

## ⚙️ Configuration Files

Configuration files in `configs/` directory allow easy setup without modifying code:

### temporal_config.yaml

- Preprocessing: normalization, detrending, windowing
- Features: RMS, entropy, autocorrelation, etc.
- Model: Random Forest with hyperparameters
- Evaluation: metrics to compute

### spectral_config.yaml

- FFT options: n_fft, overlap_ratio
- Spectral features: centroid, entropy, harmonics
- Band power configuration
- Model training options

### hybrid_config.yaml

- Both temporal and spectral features
- Feature selection method
- Dimensionality reduction (PCA)
- Combined training configuration

## 📚 Notebook Workflow

### 06_unified_analysis.ipynb

Complete end-to-end workflow demonstrating:

1. **Data Loading**: Import preprocessed signals and labels
2. **Feature Extraction**: Temporal, spectral, and hybrid examples
3. **Unified Analysis**: Run all three pipelines
4. **Results Summary**: Accuracy rankings and domain comparison
5. **Visualization**: Plots and comparative analysis
6. **Conclusions**: Final insights and recommendations

**Usage:**

```bash
jupyter notebook notebooks/06_unified_analysis.ipynb
```

## 🔍 Architecture Overview

```
Raw Signals
    ↓
┌───────────────────────────┐
│   Preprocessing Pipeline  │
│ (normalize, detrend, etc) │
└─────────────┬─────────────┘
              ↓
    ┌─────────┴──────────┬──────────────┐
    ↓                    ↓              ↓
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│   Temporal  │  │   Spectral   │  │   Hybrid     │
│  Features   │  │   Features   │  │  Features    │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘
       ↓                ↓                 ↓
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│   Temporal  │  │   Spectral   │  │   Hybrid     │
│  Classifier │  │  Classifier  │  │  Classifier  │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘
       ↓                ↓                 ↓
       └────────────────┴─────────────────┘
                       ↓
       ┌─────────────────────────────┐
       │  Statistical Comparison     │
       │  (McNemar, effect sizes)    │
       └─────────────────────────────┘
                       ↓
       "Which domain performs best?"
```

## 📊 Expected Results

### Performance Metrics

Each pipeline provides:

- Overall accuracy, precision, recall, F1-score
- Per-class metrics for each appliance type
- Confusion matrix analysis
- Cohen's Kappa coefficient
- ROC-AUC scores (for multi-class)

### Domain Comparison

- Accuracy rankings across domains
- Mean ± std across models
- Feature dimensionality analysis
- Computational cost comparison
- Interpretability assessment

## 🎓 Research Insights

### Temporal Domain

**Advantages:**

- Fast computation (no FFT)
- Highly interpretable features
- Effective for appliances with distinct amplitude patterns

**Disadvantages:**

- May miss frequency-specific characteristics
- Sensitive to signal shifts
- Limited to statistical properties

### Spectral Domain

**Advantages:**

- Captures frequency signatures
- Robust to amplitude scaling
- Good for harmonic-rich appliances

**Disadvantages:**

- More computationally expensive (FFT required)
- Loss of temporal dynamics
- May miss short-duration transients

### Hybrid Domain

**Advantages:**

- Leverages complementary information
- Potentially superior performance
- More robust across appliance types

**Disadvantages:**

- Higher dimensionality
- Risk of feature redundancy
- Requires careful feature selection

## 🛠️ Development Roadmap

### Completed (✅)

- ✅ Temporal features (30+ features)
- ✅ Spectral features (22+ features)
- ✅ Hybrid features with PCA
- ✅ 6 classification models
- ✅ Model factory
- ✅ Statistical tests
- ✅ Pipeline orchestration
- ✅ Configuration system
- ✅ Comprehensive evaluation

### Future Enhancements

- [ ] SHAP analysis for model explainability
- [ ] Advanced visualization (t-SNE, PCA projections)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation framework
- [ ] Deep learning models (LSTM, CNN-1D)
- [ ] Ensemble methods
- [ ] Real-time classification pipeline
- [ ] Model compression and quantization

## 📖 Documentation

Each module includes:

- **Docstrings**: Comprehensive function/class documentation
- **Type Hints**: Full type annotations
- **Mathematical Intuition**: Explanation of algorithms
- **Engineering Notes**: Design decisions and trade-offs

## 🤝 Contributing

When adding new features or models:

1. Follow existing code patterns
2. Add comprehensive docstrings
3. Include type hints
4. Add appropriate logging
5. Update configuration if needed
6. Test with notebook workflow

## 📝 License

Academic project for educational and research purposes.

---

**Project Status**: Core implementation complete. Ready for extended analysis and optimization.
