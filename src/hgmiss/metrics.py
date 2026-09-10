"""Discrimination, calibration, and intervals appropriate to rare outcomes."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

_EPS = 1e-6


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Metrics for one (cohort, method, fold).

    ``auprc`` is average precision, never trapezoidal interpolation, which is
    optimistically biased. ``n_positive`` is reported because it -- not the row
    count -- is the effective sample size for AUPRC.

    Degenerate folds (single class) return NaN for discrimination metrics
    (auroc, auprc) to prevent silent contamination by misleading 0.0 values.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.clip(np.asarray(y_prob).ravel().astype(float), _EPS, 1 - _EPS)

    slope, intercept = _calibration(y_true, y_prob)

    # Guard single-class case for discrimination metrics
    has_both_classes = len(np.unique(y_true)) >= 2
    auroc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_prob >= 0.5)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "n_positive": int(y_true.sum()),
    }


def _calibration(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Weak calibration: regress outcome on predicted log-odds.

    Slope 1 / intercept 0 is perfect. A flexible calibration curve is NOT fitted
    here -- it needs large samples, and events per cohort are few.
    """
    logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def logit_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-theory CI computed on the logit scale, back-transformed.

    Used for bounded metrics (AUPRC, AUROC) where a symmetric interval on the
    raw scale can cross 0 or 1 -- which happens easily at low prevalence.
    """
    v = np.clip(np.asarray(values, dtype=float).ravel(), _EPS, 1 - _EPS)
    if v.size < 2:
        raise ValueError("need at least two values for an interval")
    z = np.log(v / (1 - v))
    m = z.mean()
    se = z.std(ddof=1) / np.sqrt(z.size)
    crit = stats.t.ppf(1 - alpha / 2, df=z.size - 1)
    lo, hi = m - crit * se, m + crit * se
    return float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))
