import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

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
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    
    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "AUC": auc
    }
