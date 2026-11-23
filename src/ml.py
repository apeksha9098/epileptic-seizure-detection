"""
ml.py  -  the classical machine-learning models of the project:
          Random Forest, k-Nearest Neighbours (k-NN) and Logistic Regression.

(The draft report calls the third one "Linear Regression", but its code - and this project -
 use LOGISTIC regression, which is the correct algorithm for a yes/no classification.)

How the best settings are chosen
--------------------------------
For every model we try several hyper-parameter combinations (the ranges come from the
report / slides). Every combination is trained on the TRAINING data and scored on the
VALIDATION data. The combination with the best validation F1 wins and is evaluated ONE
time on the TEST data. (The test data is never used for choosing - that would be cheating.)
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from . import config
from .evaluate import binary_metrics

# --- candidate settings --------------------------------------------------------------
# Random Forest: (n_estimators, max_depth, max_features, min_samples_leaf) -> report Table 5
RF_GRID = [
    (150, 40, 2, 1), (100, 40, 2, 1), (100, 40, 8, 3), (150, 30, 8, 1), (150, 40, 6, 3),
    (200, None, "sqrt", 1),
]
# k-NN: (n_neighbors, p, leaf_size) -> report Table 6.  p=1 Manhattan, p=2 Euclidean distance
KNN_GRID = [(10, 2, 10), (4, 2, 10), (3, 2, 5), (1, 2, 4), (1, 1, 6), (1, 1, 20), (5, 2, 30), (5, 1, 30)]
# Logistic regression: C = inverse regularisation strength
LR_GRID = [0.01, 0.1, 1.0, 10.0]


def _candidates(quick: bool):
    rf = [dict(n_estimators=n, max_depth=d, max_features=f, min_samples_leaf=m,
               random_state=config.SEED, n_jobs=-1) for n, d, f, m in RF_GRID]
    knn = [dict(n_neighbors=k, p=p, leaf_size=ls) for k, p, ls in KNN_GRID]
    lr = [dict(C=c, max_iter=3000) for c in LR_GRID]
    out = {"Random Forest": (RandomForestClassifier, rf),
           "k-NN": (KNeighborsClassifier, knn),
           "Logistic Regression": (LogisticRegression, lr)}
    if quick:                                   # fast smoke test: only first setting
        out = {k: (cls, grid[:1]) for k, (cls, grid) in out.items()}
    return out


def run_ml_models(splits, quick: bool = False, verbose: bool = True):
    """
    Train the three ML models.

    Returns
    -------
    results : dict   model name -> {"prob_val", "prob_test", "params", "seconds", "grid_table"}
    """
    results = {}
    for name, (cls, grid) in _candidates(quick).items():
        t0 = time.time()
        rows, best = [], None
        for params in grid:
            model = cls(**params).fit(splits.X_train, splits.y_train)
            p_val = model.predict_proba(splits.X_val)[:, 1]
            m = binary_metrics(splits.y_val, p_val)
            shown = {k: v for k, v in params.items() if k not in ("random_state", "n_jobs", "max_iter")}
            rows.append({"Settings tried": ", ".join(f"{k}={v}" for k, v in shown.items()),
                         "val_accuracy": m["accuracy"], "val_precision": m["precision"],
                         "val_recall": m["recall_sensitivity"], "val_f1": m["f1"]})
            if best is None or m["f1"] > best[0]:
                best = (m["f1"], params, model, p_val)
        _, params, model, p_val = best
        p_test = model.predict_proba(splits.X_test)[:, 1]
        results[name] = {
            "prob_val": p_val, "prob_test": p_test,
            "params": {k: v for k, v in params.items() if k not in ("random_state", "n_jobs", "max_iter")},
            "seconds": time.time() - t0,
            "grid_table": pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True),
            "model": model,
        }
        if verbose:
            mt = binary_metrics(splits.y_test, p_test)
            print(f"[ML] {name:20s} best settings {results[name]['params']}  ->  "
                  f"test accuracy {100 * mt['accuracy']:.2f}%  F1 {mt['f1']:.4f}")
    return results
