"""R1 -- the published hierarchy, re-rooted at the imputed model.

The frozen estimator's kappa -> infinity limit collapses every pattern onto its SMALLEST
ancestor: a model on few variables fitted on many rows. That is not mean imputation, and it is why
"shrink toward global" lost in H1 while "shrink toward ancestors" held. The hierarchy is rooted at
the wrong node.

This adds a virtual root e0 = all variables, whose parameters are the coefficients of a logistic
regression on ALL training rows under column-mean imputation, and makes it the ultimate ancestor
of every pattern. Two things follow by construction, and the selftest asserts both:

  * at kappa = inf the method IS `mean_impute_lr` -- same imputer, same scaler, same estimator,
    predictions identical to the last bit -- so it can never do worse than the baseline at the
    corner inner CV is always allowed to pick;
  * the root applies to every test row, so the marginal-value fallback is never taken.

Three departures from the frozen estimator, each forced by the design and stated here rather
than discovered in a diff:

  1. The root is the largest pattern, but `patterns.immediate_ancestors` defines ancestors as
     STRICTLY SMALLER patterns, and `shrinkage.shrink` processes smallest-first and consumes only
     parents already shrunk. Injecting e0 as a pattern would place it as everyone's DESCENDANT and
     process it last. So the root step of Eq. shrink is applied here by hand to each minimal
     pattern, and the frozen `shrink` then runs the rest of the recursion unchanged -- which is
     exactly the recursion with e0 at the bottom, because a pattern with no parents is kept as
     given.
  2. Prediction evaluates every pattern on the FULL imputed vector, with variables outside the
     pattern carrying the root's coefficient. The frozen estimator restricts each pattern to its
     own variables, which at kappa = inf would give a support-weighted mix of restricted root
     models, not the root model. Exactness at the corner requires the full vector.
  3. The scaler is the one `mean_impute_lr` uses -- fitted on IMPUTED columns -- not the frozen
     estimator's NaN-ignoring scaler. The two differ in standard deviation and therefore in
     L2-regularised coefficients; using the frozen estimator's scaler for the root would make the
     corner only approximately the baseline. Within a pattern all variables are observed, so the
     choice of affine map changes the fit only through the penalty. The frozen estimator is left
     untouched and is run alongside as its own arm.

Frozen modules are imported, never modified.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ..baselines.simple import _fit_impute, _lr, mean_impute_lr
from ..patterns import extract_patterns, immediate_ancestors, supports_for
from ..predict import combine, cover
from ..shrinkage import INTERCEPT, shrink

# The published grid (run_h1_h4.FULL_KAPPAS) plus the exact corner. inf, not 1e9: with
# lam = n/(n + kappa), 1e9 leaves lam ~ 3e-5 and the corner only approximates the baseline.
KAPPA_GRID = (0.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 50_000.0, 1e6, np.inf)
MIN_SUPPORT = 30
INNER_FOLDS = 3


class RootedShrinkage:
    """Fit once; re-shrink at any kappa without refitting a single regression."""

    def __init__(self, min_support: int = MIN_SUPPORT):
        self.min_support = min_support

    # ---- fitting: everything that does not depend on kappa ---------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).ravel()
        self.d_ = X.shape[1]

        # (3) the baseline's own imputer and scaler, so the corner is the baseline exactly
        self.imp_ = _fit_impute(X)
        pipe = _lr().fit(self.imp_.transform(X), y)
        self.scaler_ = pipe.named_steps["standardscaler"]
        root_lr: LogisticRegression = pipe.named_steps["logisticregression"]
        self.root_ = {INTERCEPT: float(root_lr.intercept_[0]),
                      **{j: float(c) for j, c in enumerate(root_lr.coef_.ravel())}}
        Z = self.scaler_.transform(self.imp_.transform(X))   # imputed, scaled; observed values intact

        mask = ~np.isnan(X)
        realised = [e for e in extract_patterns(mask) if e]
        self.supports_ = supports_for(realised, mask)
        self.patterns_ = [e for e in realised if self.supports_[e] >= self.min_support]
        self.ancestors_ = immediate_ancestors(self.patterns_) if self.patterns_ else {}
        self.raw_ = {}
        for e in self.patterns_:
            cols = sorted(e)
            rows = mask[:, cols].all(axis=1)
            yy = y[rows]
            if len(np.unique(yy)) < 2:
                self.raw_[e] = {INTERCEPT: 0.0, **{v: 0.0 for v in cols}}
                continue
            lr = LogisticRegression(max_iter=2000).fit(Z[np.ix_(rows, cols)], yy)
            self.raw_[e] = {INTERCEPT: float(lr.intercept_[0]),
                            **{v: float(c) for v, c in zip(cols, lr.coef_.ravel())}}
        return self

    # ---- shrinking: the only kappa-dependent step -------------------------------------------
    def theta_at(self, kappa: float) -> dict:
        """Full-length parameter dict per pattern, shrunk toward the imputed root."""
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        pre = {}
        for e, own in self.raw_.items():
            parents = self.ancestors_.get(e, [])
            # (1) the root step of Eq. shrink, by hand. The frozen `shrink` keeps a pattern's OWN
            # estimate for any variable its realised parents do not observe, and for every
            # variable when it has no parents. The root observes every variable, so each of those
            # coefficients must be shrunk toward the root here, or at kappa = inf they survive
            # unshrunk and the corner is not the baseline (the selftest caught this at 5.2e-3).
            # Variables some parent DOES cover are left to `shrink`, whose parents are themselves
            # already root-shrunk by this same rule -- so the recursion is Eq. shrink with the
            # root at the bottom.
            covered = set().union(*parents) if parents else set()
            if parents:
                covered.add(INTERCEPT)          # the intercept is carried by every ancestor
            n_e = self.supports_[e]
            lam = n_e / (n_e + kappa) if np.isfinite(kappa) else 0.0
            pre[e] = {k: (v if k in covered else lam * v + (1.0 - lam) * self.root_[k])
                      for k, v in own.items()}
        shrunk = shrink(pre, self.supports_, self.ancestors_, kappa) if pre else {}
        # (2) every pattern carries the root's coefficient for variables it does not observe
        return {e: {**self.root_, **th} for e, th in shrunk.items()}

    def predict_proba_at(self, X, kappa: float) -> np.ndarray:
        X = np.asarray(X, float)
        theta = self.theta_at(kappa)
        Z = self.scaler_.transform(self.imp_.transform(X))
        mask = ~np.isnan(X)
        cols = np.arange(self.d_)
        root_z = self.root_[INTERCEPT] + Z @ np.array([self.root_[j] for j in cols])
        root_p = expit(root_z)
        out = np.empty(len(X))
        self.no_match_ = 0
        for i in range(len(X)):
            obs = frozenset(np.flatnonzero(mask[i]).tolist())
            applicable = cover(obs, self.patterns_) if theta else []
            local = {}
            for e in applicable:
                th = theta[e]
                local[e] = expit(th[INTERCEPT] + Z[i] @ np.array([th[j] for j in cols]))
            if not local:
                self.no_match_ += 1
            out[i] = combine(local, self.supports_, fallback=float(root_p[i]))
        self.no_match_rate = self.no_match_ / max(len(X), 1)
        return np.clip(out, 1e-6, 1 - 1e-6)


# ---- the four arms, all fn(X_tr, y_tr, X_te) -> probabilities ---------------------------------
def rooted_kinf(X_tr, y_tr, X_te):
    return RootedShrinkage().fit(X_tr, y_tr).predict_proba_at(X_te, np.inf)


def rooted_cv(X_tr, y_tr, X_te, *, seed: int = 0, record: dict | None = None):
    """kappa chosen by inner CV on the TRAINING fold only, over KAPPA_GRID, by mean AUPRC."""
    X_tr, y_tr = np.asarray(X_tr, float), np.asarray(y_tr).ravel()
    scores = {k: [] for k in KAPPA_GRID}
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for fit_i, val_i in inner.split(X_tr, y_tr):
        m = RootedShrinkage().fit(X_tr[fit_i], y_tr[fit_i])     # fit once ...
        for k in KAPPA_GRID:                                      # ... shrink ten times
            scores[k].append(average_precision_score(y_tr[val_i],
                                                     m.predict_proba_at(X_tr[val_i], k)))
    best = max(KAPPA_GRID, key=lambda k: (float(np.mean(scores[k])), -k if np.isfinite(k) else 1))
    if record is not None:
        record["kappa_chosen"] = best
    return RootedShrinkage().fit(X_tr, y_tr).predict_proba_at(X_te, best)


def selftest() -> int:
    """Each assertion was checked to fail when the behaviour it guards is removed."""
    rng = np.random.default_rng(0)
    n, d = 3000, 10
    X = rng.normal(size=(n, d))
    for j in range(3):
        X[rng.random(n) < 0.3, j] = np.nan
    y = (rng.random(n) < expit(0.6 * X[:, 5] + 0.9 * np.nan_to_num(X[:, 0]))).astype(int)
    tr, te = np.arange(2000), np.arange(2000, n)

    m = RootedShrinkage().fit(X[tr], y[tr])
    assert m.patterns_, "synthetic data produced no fittable pattern -- the test is vacuous"

    # the corner IS the baseline, to the last bit
    a = m.predict_proba_at(X[te], np.inf)
    b = mean_impute_lr(X[tr], y[tr], X[te])
    # 1e-12, not array_equal: at kappa=inf every applicable pattern predicts the root value and
    # `combine` returns sum(w*p)/sum(w) over identical p, which rounds to one ULP (2.2e-16) away
    # from p. The design bug this caught before the fix was 5.2e-3, ten orders larger.
    gap = float(np.abs(a - np.clip(b, 1e-6, 1 - 1e-6)).max())
    assert gap < 1e-12, f"kappa=inf is not mean_impute_lr (max gap {gap:.3e})"
    print("    kappa=inf reproduces mean_impute_lr exactly")

    # the root always applies
    assert m.no_match_rate == 0.0, f"no-match rate {m.no_match_rate} -- the root did not apply"
    print("    no-match rate is exactly zero")

    # kappa matters: the corner and the pattern-submodel end must differ
    c = m.predict_proba_at(X[te], 0.0)
    assert np.abs(a - c).max() > 1e-6, "kappa=0 equals kappa=inf -- the patterns do nothing"
    print("    kappa=0 and kappa=inf are distinct predictions")

    # variables outside a pattern carry the root's coefficient, at every kappa
    th = m.theta_at(1000.0)
    for e, t in th.items():
        for j in range(d):
            if j not in e:
                assert t[j] == m.root_[j], f"pattern {sorted(e)} var {j} is not the root's"
    print("    off-pattern variables carry the root's coefficient")

    # arms are pairwise distinct on the same data (Result 142)
    rec = {}
    preds = {"rooted_kinf": rooted_kinf(X[tr], y[tr], X[te]),
             "rooted_cv": rooted_cv(X[tr], y[tr], X[te], record=rec),
             "rooted_k0": c}
    print(f"    inner CV chose kappa = {rec['kappa_chosen']}")
    ks = sorted(preds)
    for i, p in enumerate(ks):
        for q in ks[i + 1:]:
            if rec["kappa_chosen"] == np.inf and {p, q} == {"rooted_kinf", "rooted_cv"}:
                continue                        # identical by design when CV picks the corner
            assert np.abs(preds[p] - preds[q]).max() > 1e-9, f"{p} and {q} are the same function"
    print("    arms are distinct (except where inner CV legitimately picks the corner)")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
