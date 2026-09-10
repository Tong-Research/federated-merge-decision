"""The proposed estimator: per-pattern logistic models, shrunk to ancestors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .patterns import extract_patterns, immediate_ancestors, supports_for
from .predict import combine, cover
from .shrinkage import INTERCEPT, shrink


class HypergraphShrinkage:
    """Pattern-conditional linear models with hierarchical shrinkage.

    ``kappa = 0`` leaves patterns independent (Mercaldo & Blume). Large
    ``kappa`` pulls each pattern to its ancestors. ``min_support`` drops
    patterns too small to fit at all.
    """

    def __init__(self, kappa: float = 10.0, min_support: int = 30,
                 coordinatewise: bool = False):
        self.kappa = kappa
        self.min_support = min_support
        # C4: redistribute shrinkage across coordinates by the ancestor support behind
        # each one. Reduces exactly to the scalar rule when that support is equal, so
        # `coordinatewise=False` is bit-identical to the estimator as published.
        self.coordinatewise = coordinatewise
        self.no_match_rate = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HypergraphShrinkage":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.prior_ = float(y.mean())
        self.n_features_ = X.shape[1]

        # One scaler, fit once on the whole training set, reused for every
        # pattern's fit and for prediction. Fitting a separate scaler per
        # pattern would express a shared variable's coefficient in different
        # units for different patterns, which would silently corrupt the
        # shrinkage step (it mixes a descendant's coefficients with its
        # ancestors' for the variables they share). `StandardScaler` ignores
        # NaN when computing the mean/std and leaves NaN untouched on
        # `transform`, so it composes cleanly with the per-pattern column
        # subsetting below.
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        mask = ~np.isnan(X)
        realised = [e for e in extract_patterns(mask) if e]
        # Batched over `supports_for` rather than calling `support` per pattern
        # in a loop -- realised cohorts have ~5,000 patterns, and the batched
        # matmul form is materially faster than the equivalent per-pattern
        # calls (see patterns.supports_for docstring).
        self.supports_ = supports_for(realised, mask)
        self.patterns_ = [
            e for e in realised if self.supports_[e] >= self.min_support
        ]
        if not self.patterns_:
            self.raw_theta_, self.theta_, self.ancestors_ = {}, {}, {}
            return self

        self.ancestors_ = immediate_ancestors(self.patterns_)
        self.raw_theta_ = {
            e: self._fit_one(X_scaled, y, mask, e) for e in self.patterns_
        }
        self.theta_ = shrink(
            self.raw_theta_, self.supports_, self.ancestors_, self.kappa,
            coordinatewise=self.coordinatewise,
        )
        return self

    def _fit_one(self, X, y, mask, e: frozenset[int]) -> dict[int, float]:
        """Logistic model for one pattern, keyed by variable index."""
        cols = sorted(e)
        rows = mask[:, cols].all(axis=1)
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            return {INTERCEPT: 0.0, **{v: 0.0 for v in cols}}
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X[np.ix_(rows, cols)], yy)
        coef = lr.coef_.ravel()
        return {INTERCEPT: float(lr.intercept_[0]),
                **{v: float(coef[j]) for j, v in enumerate(cols)}}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mask = ~np.isnan(X)
        X_scaled = self.scaler_.transform(X)
        preds, misses = [], 0
        for i in range(X.shape[0]):
            obs = frozenset(np.flatnonzero(mask[i]).tolist())
            applicable = cover(obs, self.patterns_) if self.theta_ else []
            local = {}
            for e in applicable:
                th = self.theta_[e]
                z = th[INTERCEPT] + sum(th[v] * X_scaled[i, v] for v in sorted(e))
                local[e] = expit(z)
            if not local:
                misses += 1
            preds.append(combine(local, self.supports_, fallback=self.prior_))
        self.no_match_rate = misses / max(X.shape[0], 1)
        return np.clip(np.asarray(preds), 1e-6, 1 - 1e-6)

    def ancestor_discrepancy(self) -> pd.DataFrame:
        """H2 diagnostic: |theta_e - theta_ancestor| on shared variables.

        Large values mean the ancestors estimate materially different
        quantities -- the omitted-variable bias discussed in the paper. This is
        a property of the data, reported whichever way H2 resolves.
        """
        rows = []
        for e, parents in self.ancestors_.items():
            if e not in self.raw_theta_:
                continue
            for p in parents:
                if p not in self.raw_theta_:
                    continue
                shared = sorted(set(p) & set(e))
                if not shared:
                    continue
                diffs = [
                    abs(self.raw_theta_[e][v] - self.raw_theta_[p][v])
                    for v in shared
                ]
                rows.append({
                    "pattern_size": len(e),
                    "ancestor_size": len(p),
                    "n_shared": len(shared),
                    "mean_abs_diff": float(np.mean(diffs)),
                })
        return pd.DataFrame(rows)


def fit_predict(
    kappa: float | Callable[[dict[frozenset[int], int]], float] = 10.0,
    min_support: int = 30,
):
    """Adapter producing a ``protocol.FitPredict`` callable.

    ``kappa`` is either a fixed float, or a callable ``kappa(supports) ->
    float`` that derives kappa from the TRAINING fold alone -- e.g. the
    pre-specified main-table operating point in ``run_h1_h4.py``
    (``kappa = median(|S(e)|)`` over the fitted patterns, putting median
    lambda at exactly 0.5). ``supports`` is ``est.supports_`` restricted to
    the patterns that were actually fitted (``est.patterns_``).

    ``raw_theta_`` (the per-pattern logistic fits) does not depend on kappa
    at all -- only the shrinkage step (``shrinkage.shrink``) does. So when
    ``kappa`` is callable we fit once with a kappa=0.0 placeholder to obtain
    ``raw_theta_``/``supports_``/``ancestors_``, derive the real kappa from
    those training-fold supports, and re-run only the cheap shrink step at
    the derived value -- no second round of per-pattern logistic fits.
    Because ``fit_predict`` is called by ``protocol.run_cv`` with the
    training rows for the current fold only, the derived kappa never sees
    the held-out fold: the leakage boundary is the CV fold boundary itself.
    """
    def _fp(X_tr, y_tr, X_te):
        if callable(kappa):
            est = HypergraphShrinkage(kappa=0.0, min_support=min_support).fit(X_tr, y_tr)
            if est.raw_theta_:
                fitted_supports = {e: est.supports_[e] for e in est.patterns_}
                k = kappa(fitted_supports)
                est.kappa = k
                est.theta_ = shrink(est.raw_theta_, est.supports_, est.ancestors_, k)
            return est.predict_proba(X_te)
        return (
            HypergraphShrinkage(kappa, min_support)
            .fit(X_tr, y_tr)
            .predict_proba(X_te)
        )
    return _fp
