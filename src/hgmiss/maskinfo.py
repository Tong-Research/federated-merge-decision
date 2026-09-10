"""Mask informativeness: how much the missingness pattern alone predicts y."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def mask_informativeness(
    mask: np.ndarray, y: np.ndarray, *, seed: int, n_splits: int = 5
) -> float:
    """``AUROC(mask -> y) - 0.5``, cross-validated.

    Zero means the missingness pattern carries no information about the target.
    The classifier sees the mask only; ``X`` is discarded by construction.
    """
    y = np.asarray(y).ravel()
    if len(np.unique(y)) < 2:
        raise ValueError("mask_informativeness needs two classes in y")

    clf = HistGradientBoostingClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        clf, mask.astype(float), y, cv=cv, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, proba) - 0.5)
