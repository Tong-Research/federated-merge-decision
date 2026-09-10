"""Hypergraph-fused pattern regression.

The estimator in ``estimator.py`` fits one model per missingness pattern
independently and then shrinks and averages the resulting predictions. That is a
two-stage plug-in procedure, and an oracle analysis over its shrinkage path shows
almost no headroom on real data. This module implements the one-stage alternative:
all pattern-specific coefficients are estimated *jointly* in a single convex
problem, with a penalty that ties each pattern to its neighbours in the containment
order.

Writing :math:`S(e)` for the records whose observed set is exactly the pattern
:math:`e`, and :math:`\\theta_e` for that pattern's coefficients over the variables
in :math:`e`:

.. math::

    \\min_{\\{\\theta_e\\}} \\sum_e \\sum_{i \\in S(e)}
        \\ell\\bigl(y_i, x_i[e]^\\top \\theta_e\\bigr)
      + \\lambda \\sum_{(e,e') \\in \\mathcal{H}} w_{ee'}
        \\bigl\\| \\theta_e|_{e \\cap e'} - \\theta_{e'}|_{e \\cap e'} \\bigr\\|_2^2
      + \\rho \\sum_e \\|\\theta_e\\|_2^2

Three properties distinguish this from the fused/joint-lasso family, in which the
subgroups come from side information and share one feature space:

* the subgroups are the missingness patterns themselves, so no grouping variable is
  needed and the grouping is determined by the data;
* the parameter spaces are *nested and unequal* -- pattern :math:`e` has
  coefficients only for the variables it observes -- so fusion applies to the shared
  coordinates :math:`e \\cap e'` only;
* the fusion graph :math:`\\mathcal{H}` is the Hasse diagram of the containment
  order rather than an all-pairs graph, so strength is borrowed along the nesting.

**Nothing is imputed.** Each record contributes to the likelihood only through the
variables it actually has, which is the point of comparison against a pooled model
that must impute every missing cell before it can fit. As :math:`\\lambda \\to
\\infty` the shared coordinates are forced to agree and the fit approaches a pooled
model estimated without imputation; at :math:`\\lambda = 0` it separates into
independent pattern submodels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .patterns import extract_patterns, immediate_ancestors, supports_for


class FusedPatternRegression:
    """Jointly-fitted pattern-conditional logistic regression with poset fusion.

    Parameters
    ----------
    lam : float
        Fusion strength. ``0`` gives independent submodels; large values force the
        shared coordinates of neighbouring patterns to agree.
    ridge : float
        Ridge penalty, which also keeps the problem strictly convex when a pattern
        is separable.
    min_support : int
        Patterns with fewer records are not given their own parameters; their
        records are assigned to the most specific fitted pattern contained in their
        observed set.
    """

    def __init__(self, lam: float = 1.0, ridge: float = 1e-2, min_support: int = 15,
                 max_iter: int = 500):
        self.lam = float(lam)
        self.ridge = float(ridge)
        self.min_support = int(min_support)
        self.max_iter = int(max_iter)

    # -- structure -------------------------------------------------------------
    def _assign(self, mask):
        """Map each record to the most specific fitted pattern it can use."""
        idx = np.full(len(mask), -1)
        obs_sets = [frozenset(np.flatnonzero(m).tolist()) for m in mask]
        for i, o in enumerate(obs_sets):
            best, best_len = -1, -1
            for j, e in enumerate(self.patterns_):
                if len(e) > best_len and e <= o:
                    best, best_len = j, len(e)
            idx[i] = best
        return idx

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.prior_ = float(y.mean())
        mask = ~np.isnan(X)
        # scale on observed entries only; nothing is imputed
        self.mu_ = np.nanmean(np.where(mask, X, np.nan), axis=0)
        sd = np.nanstd(np.where(mask, X, np.nan), axis=0)
        self.mu_ = np.where(np.isfinite(self.mu_), self.mu_, 0.0)
        self.sd_ = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Z = (X - self.mu_) / self.sd_

        realised = [e for e in extract_patterns(mask) if e]
        supports = supports_for(realised, mask)
        self.patterns_ = sorted((e for e in realised if supports[e] >= self.min_support),
                                key=lambda e: (-len(e), sorted(e)))
        if not self.patterns_:
            self.theta_ = {}
            return self
        self.supports_ = supports
        self.cols_ = [sorted(e) for e in self.patterns_]
        self.offs_, off = [], 0
        for c in self.cols_:
            self.offs_.append(off)
            off += len(c) + 1                      # +1 intercept
        self.n_par_ = off

        assign = self._assign(mask)
        self.blocks_ = []
        for j, c in enumerate(self.cols_):
            rows = np.flatnonzero(assign == j)
            if rows.size:
                Zj = np.nan_to_num(Z[np.ix_(rows, c)], nan=0.0)
                self.blocks_.append((j, np.column_stack([np.ones(len(rows)), Zj]), y[rows]))

        # fusion edges from the Hasse diagram of the containment order
        anc = immediate_ancestors(self.patterns_)
        pos = {e: j for j, e in enumerate(self.patterns_)}
        self.edges_ = []
        for e, parents in anc.items():
            if e not in pos:
                continue
            for p in parents:
                if p not in pos:
                    continue
                shared = sorted(set(e) & set(p))
                if not shared:
                    continue
                je, jp = pos[e], pos[p]
                ie = [self.cols_[je].index(v) + 1 for v in shared]
                ip = [self.cols_[jp].index(v) + 1 for v in shared]
                # Constant edge weight, deliberately NOT scaled by support. The loss
                # of a pattern already grows with its record count, so a fixed
                # penalty makes the effective pull toward a neighbour scale like
                # lambda / n_e: patterns with little data of their own are dominated
                # by the penalty and are drawn onto their neighbours, which is the
                # behaviour of the shrinkage weight n_e / (n_e + kappa). Weighting
                # the edge by the supports instead inverts this and leaves the
                # smallest patterns almost untied, which is where borrowing matters
                # most (measured: it collapsed horse colic to 0.57 against 0.86).
                self.edges_.append((je, jp, np.asarray(ie), np.asarray(ip), 1.0))

        theta = np.zeros(self.n_par_)
        res = minimize(self._objective, theta, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.theta_ = res.x
        self.converged_ = bool(res.success)
        return self

    # -- objective -------------------------------------------------------------
    def _objective(self, theta):
        f = 0.0
        g = np.zeros_like(theta)
        for j, A, yy in self.blocks_:
            o = self.offs_[j]
            t = theta[o:o + A.shape[1]]
            z = A @ t
            p = expit(z)
            f += float(-(yy * np.log(p + 1e-12) + (1 - yy) * np.log(1 - p + 1e-12)).sum())
            g[o:o + A.shape[1]] += A.T @ (p - yy)
        # ridge (intercepts unpenalised)
        for j, c in enumerate(self.cols_):
            o = self.offs_[j]
            sl = slice(o + 1, o + 1 + len(c))
            f += self.ridge * float(theta[sl] @ theta[sl])
            g[sl] += 2 * self.ridge * theta[sl]
        # fusion along the containment order, on shared coordinates only
        for je, jp, ie, ip, w in self.edges_:
            a = theta[self.offs_[je] + ie]
            b = theta[self.offs_[jp] + ip]
            d = a - b
            f += self.lam * w * float(d @ d)
            g[self.offs_[je] + ie] += 2 * self.lam * w * d
            g[self.offs_[jp] + ip] -= 2 * self.lam * w * d
        return f, g

    # -- prediction ------------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if not len(getattr(self, "patterns_", [])):
            return np.full(len(X), self.prior_)
        Z = (X - self.mu_) / self.sd_
        mask = ~np.isnan(X)
        assign = self._assign(mask)
        out = np.full(len(X), self.prior_)
        for i in range(len(X)):
            j = assign[i]
            if j < 0:
                continue
            c = self.cols_[j]
            o = self.offs_[j]
            t = self.theta_[o:o + len(c) + 1]
            z = t[0] + float(np.nan_to_num(Z[i, c], nan=0.0) @ t[1:])
            out[i] = expit(z)
        return np.clip(out, 1e-6, 1 - 1e-6)


def fit_predict(lam=1.0, ridge=1e-2, min_support=15):
    """``protocol.FitPredict`` adapter."""
    def _fp(X_tr, y_tr, X_te):
        return (FusedPatternRegression(lam=lam, ridge=ridge, min_support=min_support)
                .fit(X_tr, y_tr).predict_proba(X_te))
    return _fp
