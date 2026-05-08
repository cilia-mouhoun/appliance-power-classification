"""Statistical tests for model comparison."""

import numpy as np
from scipy import stats
from scipy.stats import binomtest
from typing import Dict, Tuple, Optional


def mcnemar_test(y_pred_1: np.ndarray, y_pred_2: np.ndarray, y_true: np.ndarray,
                alternative: str = 'two-sided') -> Dict[str, float]:
    """
    McNemar's test for comparing two classifiers.
    
    Tests if two classifiers have significantly different error rates.
    
    Mathematical Background:
    - For binary outcome: χ² = (b - c)² / (b + c)
    - Tests symmetry of disagreements between classifiers
    - Null hypothesis: classifiers have same error rate
    
    Args:
        y_pred_1: Predictions from classifier 1 (n_samples,)
        y_pred_2: Predictions from classifier 2 (n_samples,)
        y_true: True labels (n_samples,)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dictionary with:
        - 'statistic': Test statistic
        - 'p_value': p-value for the test
        - 'significant': Boolean (True if p < 0.05)
        - 'agreement': Percentage of samples where both classifiers agree
        - 'disagreements_1_correct': Samples where classifier 1 correct, 2 wrong
        - 'disagreements_2_correct': Samples where classifier 2 correct, 1 wrong
    """
    errors_1 = y_pred_1 != y_true
    errors_2 = y_pred_2 != y_true
    
    # Contingency table
    both_correct = (~errors_1) & (~errors_2)
    both_wrong = errors_1 & errors_2
    disagreements_1_correct = (~errors_1) & errors_2
    disagreements_2_correct = errors_1 & (~errors_2)
    
    # McNemar test: counts disagreements
    b = disagreements_1_correct.sum()
    c = disagreements_2_correct.sum()
    
    if b + c == 0:
        # Perfect agreement
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'agreement': 1.0,
            'disagreements_1_correct': 0,
            'disagreements_2_correct': 0,
            'note': 'Perfect agreement between classifiers'
        }
    
    # Use binomial test for exact p-value
    p_value = binomtest(b, b + c, 0.5, alternative=alternative).pvalue
    
    # Chi-square approximation (for larger samples)
    chi_square_stat = (b - c) ** 2 / (b + c) if b + c > 0 else 0
    
    total_samples = len(y_true)
    agreement_rate = (both_correct.sum() + both_wrong.sum()) / total_samples
    
    return {
        'statistic': chi_square_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'agreement': agreement_rate,
        'disagreements_1_correct': int(b),
        'disagreements_2_correct': int(c),
        'both_correct': int(both_correct.sum()),
        'both_wrong': int(both_wrong.sum()),
    }


def paired_t_test(scores_1: np.ndarray, scores_2: np.ndarray, paired: bool = True,
                 alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Paired t-test for comparing model performance metrics across folds.
    
    Mathematical Background:
    - Test if mean difference = 0
    - t = mean(diff) / (std(diff) / sqrt(n))
    - Assumes normally distributed differences
    
    Args:
        scores_1: Performance scores from model 1 (n_folds,)
        scores_2: Performance scores from model 2 (n_folds,)
        paired: Whether to use paired t-test (recommended for cross-validation)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dictionary with:
        - 'statistic': t-statistic
        - 'p_value': p-value
        - 'significant': Boolean (True if p < 0.05)
        - 'mean_diff': Mean difference in scores
        - 'std_diff': Std of differences
    """
    if paired:
        statistic, p_value = stats.ttest_rel(scores_1, scores_2, alternative=alternative)
    else:
        statistic, p_value = stats.ttest_ind(scores_1, scores_2, alternative=alternative)
    
    diff = scores_1 - scores_2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Compute 95% confidence interval for difference
    n = len(diff)
    t_crit = stats.t.ppf(0.975, n - 1)
    se = std_diff / np.sqrt(n)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': n,
    }


def kruskal_wallis_test(groups: list, alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Kruskal-Wallis test for comparing multiple independent samples.
    
    Non-parametric alternative to one-way ANOVA.
    Useful for comparing metrics across multiple models.
    
    Args:
        groups: List of score arrays from different models/groups
        alternative: 'two-sided', 'less', 'greater'
    
    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.kruskal(*groups, alternative=alternative)
    
    means = [np.mean(g) for g in groups]
    stds = [np.std(g) for g in groups]
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_groups': len(groups),
        'means': means,
        'stds': stds,
    }


def effect_size_cohens_d(scores_1: np.ndarray, scores_2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    
    Args:
        scores_1: First group scores
        scores_2: Second group scores
    
    Returns:
        Cohen's d value
    """
    mean1, mean2 = np.mean(scores_1), np.mean(scores_2)
    std1, std2 = np.std(scores_1, ddof=1), np.std(scores_2, ddof=1)
    n1, n2 = len(scores_1), len(scores_2)
    
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def compare_metrics_across_folds(metric_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Comprehensive analysis of metrics across cross-validation folds.
    
    Args:
        metric_dict: Dictionary mapping metric names to fold values
                    Example: {'accuracy': [0.8, 0.82, 0.81], 'f1': [...]}
    
    Returns:
        Dictionary with comprehensive statistics
    """
    results = {}
    
    for metric_name, values in metric_dict.items():
        values = np.array(values)
        results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'min': np.min(values),
            'max': np.max(values),
            'n_folds': len(values),
            'ci_95': (
                np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)),
                np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            ),
        }
    
    return results


def compare_classifiers_comprehensive(y_true: np.ndarray, 
                                     predictions_dict: Dict[str, np.ndarray],
                                     probabilities_dict: Dict[str, np.ndarray] = None) \
        -> Dict[str, Dict]:
    """
    Comprehensive comparison of multiple classifiers.
    
    Args:
        y_true: True labels
        predictions_dict: Dict mapping classifier names to predictions
        probabilities_dict: Dict mapping classifier names to probability estimates (optional)
    
    Returns:
        Comparison results including pairwise McNemar tests
    """
    from sklearn.metrics import accuracy_score
    
    results = {}
    classifier_names = list(predictions_dict.keys())
    
    # Individual classifier performance
    for name, preds in predictions_dict.items():
        results[name] = {'accuracy': accuracy_score(y_true, preds)}
    
    # Pairwise comparisons
    results['pairwise_comparisons'] = {}
    for i, name_1 in enumerate(classifier_names):
        for name_2 in classifier_names[i + 1:]:
            comp_key = f"{name_1} vs {name_2}"
            results['pairwise_comparisons'][comp_key] = mcnemar_test(
                predictions_dict[name_1],
                predictions_dict[name_2],
                y_true
            )
    
    return results
