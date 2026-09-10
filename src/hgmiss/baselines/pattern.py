"""Pattern-based baselines expressed as variants of the proposed estimator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..estimator import HypergraphShrinkage


def pattern_submodels(X_tr, y_tr, X_te, *, min_support: int = 30):
    """kappa = 0: independent per-pattern models (Mercaldo & Blume)."""
    est = HypergraphShrinkage(kappa=0.0, min_support=min_support)
    return est.fit(X_tr, y_tr).predict_proba(X_te)


def spsm_global(
    X_tr, y_tr, X_te, *,
    kappa: float | Callable[[dict[frozenset[int], int]], float] = 10.0,
    min_support: int = 30,
):
    """Shrink every pattern toward ONE global model (Stempfle et al.).

    This is H2's comparator: identical shrinkage strength, different target.
    The global model is the smallest realised pattern -- the best-supported one
    -- and every pattern is pulled toward it directly rather than toward its
    own ancestors.

    ``kappa`` is either a fixed float, or a callable ``kappa(supports) ->
    float`` deriving kappa from this TRAINING fold's supports (restricted to
    the fitted patterns) -- see ``estimator.fit_predict`` for the matching
    ``ours`` path and the leakage-boundary rationale. ``raw_theta_`` does not
    depend on kappa, so the initial fit always uses a kappa=0.0 placeholder;
    only the (cheap) global-shrinkage rebuild below uses the real value.
    """
    est = HypergraphShrinkage(kappa=0.0, min_support=min_support)
    est.fit(X_tr, y_tr)
    if not est.raw_theta_:
        return np.full(len(X_te), est.prior_)

    if callable(kappa):
        fitted_supports = {e: est.supports_[e] for e in est.patterns_}
        k = kappa(fitted_supports)
    else:
        k = kappa
    est.kappa = k

    root = min(est.patterns_, key=len)
    global_theta = est.raw_theta_[root]

    rebuilt = {}
    for e, own in est.raw_theta_.items():
        n_e = est.supports_[e]
        lam = n_e / (n_e + k) if (n_e + k) > 0 else 0.0
        rebuilt[e] = {
            key: (lam * v + (1 - lam) * global_theta[key])
            if key in global_theta else v
            for key, v in own.items()
        }
    est.theta_ = rebuilt
    return est.predict_proba(X_te)
