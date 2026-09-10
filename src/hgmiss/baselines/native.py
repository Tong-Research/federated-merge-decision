"""Learners that consume NaN directly, with no imputation step."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def histgb_native(X_tr, y_tr, X_te, *, seed: int = 0):
    # sklearn 1.9's HistGradientBoostingClassifier binning step
    # (_find_binning_thresholds -> sliding_window_view(distinct_values, 2))
    # raises ValueError on a column with fewer than two distinct observed
    # (non-NaN) training values -- an all-NaN column, or a constant one, which
    # the full PLCO feature set contains. Neither carries signal for a tree, so
    # such columns are dropped for this call only (X_tr/X_te are untouched
    # elsewhere); this is a workaround for the degenerate case, not a
    # weakening of the model's use of partially-observed columns.
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    observed = np.array([len(np.unique(col[~np.isnan(col)])) >= 2 for col in X_tr.T])
    if not observed.any():
        prior = float(np.mean(y_tr))
        return np.full(X_te.shape[0], prior)
    clf = HistGradientBoostingClassifier(random_state=seed)
    try:
        clf.fit(X_tr[:, observed], y_tr)
    except ValueError as exc:
        if "window shape" not in str(exc):
            raise
        # Above 10,000 rows sklearn enables early stopping and fits its binner on a 90% split of
        # the training rows, so a column whose second value is very rare can be constant there
        # and the same binning error fires although the guard above passed. Retry with such
        # columns dropped. This path is reached only when the plain fit has already failed, so
        # every call that succeeded before is unchanged.
        counts = []
        for col in X_tr.T:
            v = col[~np.isnan(col)]
            _, c = np.unique(v, return_counts=True)
            counts.append(int(np.sort(c)[-2]) if len(c) >= 2 else 0)
        observed = np.array(counts) >= 20
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X_tr[:, observed], y_tr)
    return clf.predict_proba(X_te[:, observed])[:, 1]


def xgboost_native(X_tr, y_tr, X_te, *, seed: int = 0):
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        random_state=seed, eval_metric="logloss", tree_method="hist"
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def lightgbm_native(X_tr, y_tr, X_te, *, seed: int = 0):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(random_state=seed, verbosity=-1)
    clf.fit(X_tr, y_tr)
    return np.asarray(clf.predict_proba(X_te))[:, 1]
