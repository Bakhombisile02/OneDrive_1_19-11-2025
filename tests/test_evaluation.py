import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.evaluation import (
    compute_metrics,
    bootstrap_metric_diff,
    parametric_metric_test,
    compare_metric_distributions,
)

def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0]) # 1 FN
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    assert metrics['Accuracy'] == 0.75
    assert metrics['Sensitivity'] == 0.5 # TP=1, FN=1 -> 1/2
    assert metrics['Specificity'] == 1.0 # TN=2, FP=0 -> 2/2
    assert metrics['MCC'] == pytest.approx(0.57735, rel=1e-3)
    expected_auprc = average_precision_score(y_true, y_prob)
    assert metrics['AUPRC'] == pytest.approx(expected_auprc, rel=1e-6)


def test_bootstrap_metric_diff_shapes():
    a = np.array([0.8, 0.7, 0.9])
    b = np.array([0.6, 0.65, 0.7])
    stats = bootstrap_metric_diff(a, b, n_boot=200, seed=0)
    assert 'p_value' in stats
    assert stats['ci_high'] >= stats['ci_low']


def test_parametric_metric_test_checks_normality():
    a = np.array([0.8, 0.81, 0.79, 0.8])
    b = np.array([0.7, 0.69, 0.71, 0.7])
    stats = parametric_metric_test(a, b)
    assert 'p_value' in stats
    assert 'shapiro_p' in stats


def test_compare_metric_distributions_combines_methods():
    a = np.array([0.8, 0.82, 0.78, 0.81])
    b = np.array([0.75, 0.74, 0.76, 0.73])
    report = compare_metric_distributions(a, b, metric_name="MCC", n_boot=100)
    assert report['metric'] == 'MCC'
    assert 'bootstrap' in report and 'parametric' in report
