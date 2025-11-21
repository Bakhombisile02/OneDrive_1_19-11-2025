import numpy as np
from src.evaluation import compute_metrics

def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0]) # 1 FN
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    assert metrics['Accuracy'] == 0.75
    assert metrics['Sensitivity'] == 0.5 # TP=1, FN=1 -> 1/2
    assert metrics['Specificity'] == 1.0 # TN=2, FP=0 -> 2/2
    assert 'AUC' in metrics
