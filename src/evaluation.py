import numpy as np
from scipy.stats import shapiro, ttest_rel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED

def create_subject_folds(subject_id, n_folds=5, seed=42):
    """
    DEPRECATED: Use get_stratified_group_folds instead.
    Kept for backward compatibility if needed, but strongly discouraged.
    """
    unique_subjects = np.unique(subject_id)
    np.random.seed(seed)
    np.random.shuffle(unique_subjects)
    
    # Split subjects into folds
    fold_assignments = {}
    for i, subj in enumerate(unique_subjects):
        fold_assignments[subj] = i % n_folds
        
    # Map back to samples
    fold_indices = np.array([fold_assignments[s] for s in subject_id])
    return fold_indices

def get_stratified_group_folds(X, y, groups, n_folds=5, seed=42):
    """
    Generate fold indices using StratifiedGroupKFold.
    This ensures that:
    1. Samples from the same subject (group) are in the same fold.
    2. The ratio of classes is preserved across folds.
    
    Returns:
        fold_indices (np.array): Array of shape (n_samples,) where each value is the fold index (0 to n_folds-1).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Initialize fold indices with -1
    fold_indices = np.full(len(y), -1, dtype=int)
    
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_indices[test_idx] = fold_idx
        
    return fold_indices

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if np.any(y_pred != y_pred[0]) else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    auprc = average_precision_score(y_true, y_prob) if y_prob is not None else np.nan

    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc,
        "AUC": auc,
        "AUPRC": auprc,
    }


def bootstrap_metric_diff(metric_a, metric_b, n_boot=1000, seed=SEED):
    """Non-parametric bootstrap on paired metric samples."""
    metric_a = np.asarray(metric_a)
    metric_b = np.asarray(metric_b)
    if metric_a.shape != metric_b.shape:
        raise ValueError("Metric arrays must share the same shape for paired bootstrap.")

    rng = np.random.default_rng(seed)
    diffs = []
    n = metric_a.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(np.mean(metric_a[idx] - metric_b[idx]))

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_val = 2 * min(
        (diffs <= 0).mean(),
        (diffs >= 0).mean(),
    )

    return {
        "diff_mean": float(np.mean(diffs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_val),
    }


def parametric_metric_test(metric_a, metric_b, alpha=0.05):
    """Paired t-test after optional Shapiro-Wilk check on differences."""
    metric_a = np.asarray(metric_a, dtype=float)
    metric_b = np.asarray(metric_b, dtype=float)
    if metric_a.shape != metric_b.shape:
        raise ValueError("Metric arrays must share the same shape for paired tests.")
    diffs = metric_a - metric_b

    shapiro_p = np.nan
    if diffs.size >= 3:
        try:
            shapiro_p = shapiro(diffs).pvalue
        except ValueError:
            shapiro_p = np.nan

    t_stat, p_val = ttest_rel(metric_a, metric_b)
    assumption_ok = bool(shapiro_p >= alpha) if not np.isnan(shapiro_p) else False

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "assumption_ok": assumption_ok,
    }


def compare_metric_distributions(metric_a, metric_b, metric_name, n_boot=1000, alpha=0.05, seed=SEED):
    """Convenience wrapper returning both bootstrap and parametric comparisons."""
    boot = bootstrap_metric_diff(metric_a, metric_b, n_boot=n_boot, seed=seed)
    param = parametric_metric_test(metric_a, metric_b, alpha=alpha)
    return {
        "metric": metric_name,
        "bootstrap": boot,
        "parametric": param,
    }
