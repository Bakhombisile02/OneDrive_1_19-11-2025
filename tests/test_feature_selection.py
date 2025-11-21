import numpy as np
import pytest
from src.feature_selection import mrmr_rank, discretize_matrix

def test_discretize_matrix():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])
    X_disc = discretize_matrix(X, qbins=2)
    assert X_disc.shape == X.shape
    # Check if values are 0 and 1
    assert np.all(np.isin(X_disc, [0, 1]))

def test_mrmr_rank():
    # Create a synthetic dataset where feature 0 is highly correlated with y
    # and feature 1 is redundant with feature 0
    np.random.seed(42)
    n_samples = 100
    y = np.random.randint(0, 2, n_samples)
    
    # Feature 0: correlated with y
    f0 = y.copy()
    # Feature 1: copy of f0 (redundant)
    f1 = f0.copy()
    # Feature 2: random noise
    f2 = np.random.rand(n_samples)
    
    X = np.column_stack([f0, f1, f2])
    
    selected = mrmr_rank(X, y, k=2, qbins=2)
    
    assert len(selected) == 2
    assert selected[0] == 0  # Should pick f0 first
    assert all(isinstance(i, (int, np.integer)) for i in selected)
    assert len(set(selected)) == 2
