import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score


def discretize_matrix(X, qbins=5):
    X_disc = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        col = X[:, i]
        n_unique = len(np.unique(col))
        
        if n_unique <= qbins:
            # If few unique values, just encode them as 0..n_unique-1
            # This preserves the information for MI calculation
            X_disc[:, i] = pd.factorize(col)[0]
        else:
            try:
                X_disc[:, i] = pd.qcut(
                    col, qbins, labels=False, duplicates='drop'
                )
            except ValueError:
                # Handle constant columns or too few unique values
                X_disc[:, i] = 0
    return X_disc


def mi_discrete(x, y):
    return mutual_info_score(x, y)


def mrmr_rank(X, y, k=50, qbins=5):
    n_samples, n_features = X.shape

    # Discretize
    X_disc = discretize_matrix(X, qbins)

    # Calculate MI between each feature and target (Parallelize this)
    mi_xy = Parallel(n_jobs=-1)(
        delayed(mi_discrete)(X_disc[:, i], y) for i in range(n_features)
    )
    mi_xy = np.array(mi_xy)

    selected = []
    remaining = list(range(n_features))

    # Select first feature (max MI with target)
    first_feat = np.argmax(mi_xy)
    selected.append(first_feat)
    remaining.remove(first_feat)

    # Initialize cumulative redundancy
    cum_redundancy = np.zeros(n_features)
    pairwise_cache = {}

    while len(selected) < k and remaining:
        last_sel = selected[-1]

        # Compute MI between last_selected and remaining features
        # only once per pair.
        cache_keys = [
            (min(last_sel, j), max(last_sel, j)) for j in remaining
        ]
        missing = [
            j for j, key in zip(remaining, cache_keys)
            if key not in pairwise_cache
        ]

        if missing:
            new_values = Parallel(n_jobs=-1)(
                delayed(mi_discrete)(X_disc[:, j], X_disc[:, last_sel])
                for j in missing
            )
            for j, val in zip(missing, new_values):
                pairwise_cache[(min(last_sel, j), max(last_sel, j))] = val

        new_mis_subset = [
            pairwise_cache[(min(last_sel, j), max(last_sel, j))]
            for j in remaining
        ]

        # Update cumulative redundancy for remaining features
        best_j = -1
        best_score = -np.inf

        for idx, j in enumerate(remaining):
            cum_redundancy[j] += new_mis_subset[idx]
            redundancy = cum_redundancy[j] / len(selected)
            score = mi_xy[j] - redundancy

            if score > best_score:
                best_score = score
                best_j = j

        selected.append(best_j)
        remaining.remove(best_j)

    return selected
