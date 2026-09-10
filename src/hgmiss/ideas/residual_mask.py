"""A4: condition on the part of the mask the observed VALUES cannot predict.

Paper 1's own explanation of its negative result is that the pattern's information "is already
recoverable from the observed values, so the pooled model loses nothing by ignoring the pattern".
If that is right, then the mask decomposes into a part the values already carry and a residual,
and only the residual could ever add anything. This conditions on the residual alone.

It is a test of the paper's stated mechanism as much as a candidate method, which is why it is
first: a null result here CONFIRMS the explanation rather than merely failing to beat a baseline.

THE LEAKAGE THAT WOULD INVALIDATE IT. Predicting mask column j from the mean-imputed design must
exclude column j. After mean imputation x_ij equals the column mean exactly when j is missing, so
column j nearly determines m_ij and the "residual" would collapse to noise for a reason that has
nothing to do with the science. Each column is therefore residualised against the other d-1
columns only.

Done efficiently rather than with d separate fits: the Gram matrices X'X and X'M are computed once
on the training fold, and column j's coefficients come from deleting row and column j. Ridge
regression, not logistic -- the object wanted is the linear part of the mask that the values
explain, and the closed form makes d solves cheap.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE = 1e-3


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def _mean_impute(X_tr):
    mu = np.nanmean(np.asarray(X_tr, float), axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)          # a wholly-missing column fills with zero

    def apply(X):
        X = np.asarray(X, float).copy()
        idx = np.isnan(X)
        X[idx] = np.take(mu, np.where(idx)[1])
        return X
    return apply


def residual_mask(X_tr, X_te, M_tr=None, M_te=None):
    """(R_tr, R_te): the mask minus its leave-one-column-out linear prediction from the values."""
    imp = _mean_impute(X_tr)
    A_tr, A_te = imp(X_tr), imp(X_te)
    if M_tr is None:
        M_tr = (~np.isnan(np.asarray(X_tr, float))).astype(float)
    if M_te is None:
        M_te = (~np.isnan(np.asarray(X_te, float))).astype(float)
    M_tr = np.asarray(M_tr, float)
    M_te = np.asarray(M_te, float)

    # Standardise on train so the ridge penalty means the same thing for every column.
    mu, sd = A_tr.mean(0), A_tr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z_tr, Z_te = (A_tr - mu) / sd, (A_te - mu) / sd

    d = Z_tr.shape[1]
    G = Z_tr.T @ Z_tr + RIDGE * len(Z_tr) * np.eye(d)
    B = Z_tr.T @ M_tr
    R_tr, R_te = M_tr.copy(), M_te.copy()
    keep = np.arange(d)
    for j in range(d):
        if M_tr[:, j].std() < 1e-12:
            R_tr[:, j] = 0.0                          # no variation to explain or to residualise
            R_te[:, j] = 0.0
            continue
        o = keep[keep != j]                           # leave column j out: see the docstring
        try:
            w = np.linalg.solve(G[np.ix_(o, o)], B[o, j])
        except np.linalg.LinAlgError:
            continue                                  # singular: leave the raw mask for this column
        R_tr[:, j] = M_tr[:, j] - Z_tr[:, o] @ w
        R_te[:, j] = M_te[:, j] - Z_te[:, o] @ w
    return R_tr, R_te


def mean_residual_indicator_lr(X_tr, y_tr, X_te, *, seed: int = 0):
    """[mean-imputed values, RESIDUAL mask] -> logistic regression.

    The arm to compare against is `mean_indicator_lr`, which is the same design with the raw mask.
    """
    imp = _mean_impute(X_tr)
    R_tr, R_te = residual_mask(X_tr, X_te)
    tr = np.column_stack([imp(X_tr), R_tr])
    te = np.column_stack([imp(X_te), R_te])
    return _lr().fit(tr, y_tr).predict_proba(te)[:, 1]
