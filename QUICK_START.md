# 🎯 Quick Start Guide - Run Your First Domain Comparison

## ⚡ 30-Second Summary

This project implements **three parallel machine learning pipelines** for appliance power classification:

1. **Temporal Pipeline** → Statistical signal features (RMS, entropy, etc.)
2. **Spectral Pipeline** → Frequency-domain features (FFT, harmonics, etc.)
3. **Hybrid Pipeline** → Combined temporal + spectral features

**Goal**: Answer "Which domain performs best for appliance classification?"

---

## 🚀 How to Run (Step-by-Step)

### Step 1: Load Your Data

```python
import numpy as np
import pandas as pd

# Load your CSV files
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

# Extract signals and labels
# (Adjust column names if needed)
X_train = train_df.iloc[:, :-1].values  # All columns except last
y_train = train_df.iloc[:, -1].values   # Last column is label

print(f"Loaded {X_train.shape[0]} training samples")
print(f"Signal length: {X_train.shape[1]}")
print(f"Classes: {np.unique(y_train)}")
```

### Step 2: Run Unified Pipeline

```python
from src.pipelines.unified_pipeline import UnifiedPipeline

# Create pipeline
pipeline = UnifiedPipeline(
    models=['random_forest', 'xgboost'],
    output_dir='outputs/analysis'
)

# Run complete analysis
results = pipeline.run_complete_analysis(
    X_train, y_train,
    test_size=0.2,
    save_results=True
)
```

### Step 3: View Results

```python
# Print report
print(results['report'])

# Access detailed results
comparison = results['comparison']
print(f"Best pipeline: {comparison['best_pipeline']}")
print(f"Best accuracy: {comparison['best_accuracy']:.3f}")
```

---

## 📊 What You'll Get

### Console Output:

```
================================================================================
APPLIANCE POWER CLASSIFICATION - DOMAIN COMPARISON REPORT
================================================================================

EXECUTIVE SUMMARY
────────────────────────────────────────────────────────────────────────────
Best Pipeline: hybrid_random_forest (0.920)

DOMAIN PERFORMANCE ANALYSIS
────────────────────────────────────────────────────────────────────────────

TEMPORAL:
  Mean Accuracy: 0.880
  Std Dev: 0.012
  Range: [0.865, 0.892]

SPECTRAL:
  Mean Accuracy: 0.895
  Std Dev: 0.015
  Range: [0.878, 0.910]

HYBRID:
  Mean Accuracy: 0.920
  Std Dev: 0.008
  Range: [0.910, 0.928]

...
```

### Output Files (in `outputs/analysis/`):

- `results.json` - Complete numerical results
- `report.txt` - Human-readable report
- `domain_comparison.png` - Visualization

---

## 🔍 Understanding the Results

### Three Domains Explained

#### 1️⃣ Temporal Domain

**What it captures**: How the signal changes over time

- Features: RMS, variance, entropy, zero-crossings
- Why it works: Appliances have distinct power draw patterns
- Speed: ⚡ Fast
- Interpretability: 🟢 High

#### 2️⃣ Spectral Domain

**What it captures**: Frequency content of the signal

- Features: FFT, harmonics, spectral centroid
- Why it works: Different appliances have different electrical frequencies
- Speed: ⚡⚡ Medium
- Interpretability: 🟡 Medium

#### 3️⃣ Hybrid Domain

**What it captures**: Both dynamics AND frequencies

- Features: All temporal + all spectral features
- Why it works: Complementary information from both domains
- Speed: ⚡⚡⚡ Slower
- Interpretability: 🔴 Lower (but highest accuracy)

---

## 💡 Expected Patterns

### If Temporal Wins 🥇

```
Appliances have very different power draw magnitudes
Example: Microwave (2000W) vs LED lamp (5W)
→ Use simple, fast temporal features
```

### If Spectral Wins 🥇

```
Appliances have distinctive frequency signatures
Example: Motors have 50-60 Hz harmonics
→ Use frequency-domain features
```

### If Hybrid Wins 🥇 (Most Common)

```
Both amplitude AND frequency matter
Example: Coffee machine has high power + 50 Hz signature
→ Use combined features (with feature selection)
```

---

## 🎓 Example Interpretation

**Output:**

```
Best Pipeline: hybrid_random_forest (0.920)

RANKING:
1. hybrid_random_forest: 0.920
2. spectral_random_forest: 0.895
3. temporal_random_forest: 0.880
```

**What this means:**

- ✅ Hybrid domain provides 3.2% accuracy improvement over temporal
- ✅ Spectral alone is competitive (0.895)
- ✅ Combining both domains captures complementary information
- 💭 Interpretation: Appliance signals have both distinctive amplitudes AND frequencies

---

## 🔧 Customization

### Use Different Models

```python
# Try different model combinations
pipeline = UnifiedPipeline(
    models=['random_forest', 'xgboost', 'svm', 'knn'],
    output_dir='outputs/analysis'
)
```

### Adjust Test Size

```python
results = pipeline.run_complete_analysis(
    X_train, y_train,
    test_size=0.3,  # Use 30% for testing
)
```

### Hybrid Configuration

```python
from src.pipelines.hybrid_pipeline import HybridPipeline

hybrid = HybridPipeline(
    model_type='random_forest',
    use_pca=True,
    pca_components=50,  # More components = more info
    use_feature_selection=True,
    sampling_rate=1.0  # Adjust if needed
)

results = hybrid.run(X_train, y_train, test_size=0.2)
```

---

## 📈 Performance Benchmarks

### Typical Results

| Domain   | Accuracy  | Speed  | Memory |
| -------- | --------- | ------ | ------ |
| Temporal | 0.85-0.90 | ⚡     | 💾     |
| Spectral | 0.88-0.93 | ⚡⚡   | 💾💾   |
| Hybrid   | 0.90-0.95 | ⚡⚡⚡ | 💾💾💾 |

_Note: Results depend on appliance types and signal quality_

---

## ⚠️ Troubleshooting

### "Module not found" error

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path().cwd().parent / 'src'))
```

### Data shape mismatch

```python
# Expected: (n_samples, signal_length)
print(X_train.shape)  # Should be (100, 1000) or similar
print(y_train.shape)  # Should be (100,)
```

### Feature extraction slow

```python
# Use subset for testing
results = pipeline.run_complete_analysis(
    X_train[:50],  # First 50 samples
    y_train[:50],
)
```

---

## 📚 Next Steps

### 1. View Detailed Results

Open `IMPLEMENTATION_GUIDE.md` for:

- Feature extraction details
- Model descriptions
- Statistical tests
- Architecture overview

### 2. Run Notebook

```bash
jupyter notebook notebooks/06_unified_analysis.ipynb
```

### 3. Try Individual Pipelines

```python
from src.pipelines.temporal_pipeline import TemporalPipeline

temporal = TemporalPipeline(model_type='random_forest')
results = temporal.run(X_train, y_train, test_size=0.2)
print(f"Temporal accuracy: {results['overall']['accuracy']:.3f}")
```

### 4. Implement Advanced Analysis

```python
from src.evaluation.statistical_tests import mcnemar_test

# Compare two models statistically
result = mcnemar_test(y_pred_1, y_pred_2, y_true)
print(f"Significant difference: {result['significant']}")
```

---

## 🎯 Research Question Template

After running the analysis, you can answer:

> **"For appliance power classification, the **{DOMAIN}** domain achieves the highest accuracy (**{ACC:.1%}**) when using **{MODEL}**. This is because:**
>
> - **Domain Insight**: {TEMPORAL/SPECTRAL/HYBRID_ADVANTAGE}
> - **Feature Interpretation**: {KEY_FEATURES}
> - **Practical Implication**: {USE_CASE}

---

## 📞 Getting Help

### Check Documentation

1. `IMPLEMENTATION_GUIDE.md` - Complete feature/model reference
2. `IMPLEMENTATION_SUMMARY.md` - What was built and why

### Review Existing Code

- Look at `src/pipelines/unified_pipeline.py` for orchestration
- Check `src/features/` for feature extraction details
- Review `src/models/model_factory.py` for model usage

### Run Examples

- Execute `notebooks/06_unified_analysis.ipynb` for full workflow
- Review cell-by-cell for usage patterns

---

## ✅ Success Criteria

You'll know everything is working when:

1. ✅ Data loads without errors
2. ✅ Features extract in <1 minute (per domain)
3. ✅ Models train in <5 minutes
4. ✅ Results show accuracy rankings
5. ✅ Report explains domain performance
6. ✅ Output files saved to `outputs/analysis/`

---

## 🚀 You're Ready!

Start with:

```python
from src.pipelines.unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline(models=['random_forest'])
results = pipeline.run_complete_analysis(X, y)
print(results['report'])
```

**Then explore, analyze, and discover which domain wins! 🏆**
