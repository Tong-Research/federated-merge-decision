"""Baselines named in the paper's §Baselines that had never been run (Result 115).

Three imputers and two pattern-family methods. The imputers all end in the same logistic
regression the rest of the paper uses, so a difference between them is a difference in the
imputation and nothing else -- the §Baselines text claims each imputer crossed with four
classifiers, but the paper's model class is logistic regression and the extra three would
triple the runs without addressing any claim the paper makes.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def _impute_lr(imputer, X_tr, y_tr, X_te):
    """Fit imputer on TRAIN only, then a logistic regression on the imputed design."""
    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float)
    # A column with no observed training value carries nothing; sklearn's imputers drop such
    # columns by default, which silently changes the design between train and test. Keeping
    # them (as zeros after the fill) preserves the column count.
    imputer.fit(X_tr)
    return _lr().fit(imputer.transform(X_tr), y_tr).predict_proba(imputer.transform(X_te))[:, 1]


def knn_impute_lr(X_tr, y_tr, X_te, *, seed: int = 0, k: int = 5):
    return _impute_lr(KNNImputer(n_neighbors=k, keep_empty_features=True), X_tr, y_tr, X_te)


def mice_impute_lr(X_tr, y_tr, X_te, *, seed: int = 0):
    """Chained equations. sklearn's IterativeImputer is the MICE construction with a single
    imputation rather than the multiple-imputation pooling; the paper's claim is about
    predictive accuracy from one design matrix, which is what this measures."""
    return _impute_lr(
        IterativeImputer(random_state=seed, max_iter=10, keep_empty_features=True),
        X_tr, y_tr, X_te)


def mice_impute_lr_conv(X_tr, y_tr, X_te, *, seed: int = 0):
    """MICE at max_iter=50 rather than sklearn's default 10.

    Exists to answer one question and then be deleted or kept as a footnote. On colorectal
    every fold logged `Early stopping criterion not reached`, so the reported MICE numbers
    are a truncated imputation, not a converged one. Reporting a comparator at a setting it
    never reached is the same defect as the fixed-lambda pFedMe handicap: the method loses,
    and it is unclear whether it lost or was not allowed to finish.

    Deliberately NOT in run_extra_baselines.METHODS -- it is reachable only through --only,
    so a daemon relaunch cannot silently start slow MICE on all four cohorts.
    """
    return _impute_lr(
        IterativeImputer(random_state=seed, max_iter=50, keep_empty_features=True),
        X_tr, y_tr, X_te)


def missforest_lr(X_tr, y_tr, X_te, *, seed: int = 0):
    """MissForest is iterative imputation with a random forest as the per-column estimator
    (Stekhoven & Buehlmann 2012), which is exactly IterativeImputer under that estimator.
    Trees are shallow and few here because this runs on 40,000 x 188 inside a 5-fold loop."""
    est = RandomForestRegressor(n_estimators=20, max_depth=8, n_jobs=1, random_state=seed)
    return _impute_lr(
        IterativeImputer(estimator=est, random_state=seed, max_iter=4,
                         keep_empty_features=True),
        X_tr, y_tr, X_te)


def refe(X_tr, y_tr, X_te, *, seed: int = 0):
    """Reduced-Feature Ensemble, Saar-Tsechansky & Provost (JMLR 2007) §4.2.

    Verified against the source 2026-08-24: a set of models 'each induced by excluding a
    single attribute'; for a test row, the members applied are those excluding an attribute
    that row is missing; 'when multiple values are missing, ReFE ensemble members rely on
    imputation for the additional missing values'; predictions are averaged, unweighted.

    Two deviations from the paper, both deliberate and both stated. Their members are
    classification trees and these are logistic regressions, so that the comparison isolates
    the ensemble construction rather than the learner. And a member is fitted only for
    attributes that are ever missing in training -- a model excluding an always-observed
    attribute can never be selected at prediction time, so fitting it would cost d models to
    use a handful.
    """
    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float)
    prior = float(np.mean(y_tr))

    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X_tr)
    Z_tr = imp.transform(X_tr)
    Z_te = imp.transform(X_te)

    full = _lr().fit(Z_tr, y_tr)
    ever_missing = np.flatnonzero(np.isnan(X_tr).any(axis=0))
    if ever_missing.size == 0:
        return full.predict_proba(Z_te)[:, 1]

    members = {}
    for j in ever_missing:
        keep = np.ones(X_tr.shape[1], bool)
        keep[j] = False
        try:
            members[int(j)] = (_lr().fit(Z_tr[:, keep], y_tr), keep)
        except Exception:
            pass                                   # degenerate column; skip this member

    miss_te = np.isnan(X_te)
    out = np.empty(X_te.shape[0], float)
    for i in range(X_te.shape[0]):
        idx = [int(j) for j in np.flatnonzero(miss_te[i]) if int(j) in members]
        if not idx:
            out[i] = full.predict_proba(Z_te[i:i + 1])[:, 1][0]
            continue
        ps = [m.predict_proba(Z_te[i:i + 1, keep])[:, 1][0] for m, keep in
              (members[j] for j in idx)]
        out[i] = float(np.mean(ps)) if ps else prior
    return out


def refe_tree(X_tr, y_tr, X_te, *, seed: int = 0, max_depth: int = 8):
    """ReFE with the members Saar-Tsechansky & Provost actually used: classification trees.

    WHY THIS EXISTS. `refe` above uses logistic-regression members "to isolate the ensemble
    construction rather than the learner", and on PLCO that isolated it into nothing: excluding
    one column of 188 barely moves a linear model, so averaging 188 such members reproduces the
    full model. Measured on colorectal 2026-08-24, `refe` correlates 0.9995 with mean
    imputation, mean difference -0.000011. The construction was vacuous by construction.

    The source uses trees, where dropping an attribute forces different splits and the members
    genuinely disagree. That makes the ensemble non-trivial at the cost of changing the learner
    class, which is the tension the linear choice was trying to avoid. Both are reported: the
    linear version shows the construction adds nothing to a linear model, and this one tests
    the method as its authors defined it.
    """
    from sklearn.tree import DecisionTreeClassifier

    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float)
    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X_tr)
    Z_tr, Z_te = imp.transform(X_tr), imp.transform(X_te)

    def _tree():
        return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20,
                                      random_state=seed)

    full = _tree().fit(Z_tr, y_tr)
    ever_missing = np.flatnonzero(np.isnan(X_tr).any(axis=0))
    if ever_missing.size == 0:
        return full.predict_proba(Z_te)[:, 1]

    members = {}
    for j in ever_missing:
        keep = np.ones(X_tr.shape[1], bool)
        keep[j] = False
        try:
            members[int(j)] = (_tree().fit(Z_tr[:, keep], y_tr), keep)
        except Exception:
            pass

    miss_te = np.isnan(X_te)
    out = np.empty(X_te.shape[0], float)
    for i in range(X_te.shape[0]):
        idx = [int(j) for j in np.flatnonzero(miss_te[i]) if int(j) in members]
        if not idx:
            out[i] = full.predict_proba(Z_te[i:i + 1])[:, 1][0]
            continue
        ps = [m.predict_proba(Z_te[i:i + 1, keep])[:, 1][0]
              for m, keep in (members[j] for j in idx)]
        out[i] = float(np.mean(ps))
    return out


def learnpp_mf(X_tr, y_tr, X_te, *, seed: int = 0, min_support: int = 30):
    """Learn++.MF (Polikar et al., Pattern Recognition 2010): an unweighted majority vote of
    every model whose feature set is contained in what the test row observes.

    Verified against the source 2026-08-24. One deviation, stated because the paper's novelty
    rests on it: Learn++.MF draws its member feature sets as RANDOM SUBSPACES, whereas the
    members here are the realised missingness patterns. Random subspaces have no sample
    support defined for them, so on this data they would produce an arbitrary comparison; the
    realised patterns are the natural analogue and give the method its best case, since every
    member is one a pattern-conditional estimator would also fit.

    This is the combination rule sitting at kappa = 0 -- the cover, unweighted, no shrinkage --
    which is the top-right cell of the ablation grid in fig_ablation.

    Ported from experiments/family_comparison.py, where it ran as `cover_vote` on the public
    datasets. It had never been run on the PLCO cohorts.
    """
    from scipy.special import expit

    from hgmiss.estimator import HypergraphShrinkage
    from hgmiss.predict import cover
    from hgmiss.shrinkage import INTERCEPT

    X_te = np.asarray(X_te, float)
    est = HypergraphShrinkage(kappa=0.0, min_support=min_support).fit(X_tr, y_tr)
    if not est.raw_theta_:
        return np.full(len(X_te), float(np.mean(y_tr)))

    Xs = est.scaler_.transform(X_te)
    mask = ~np.isnan(X_te)
    out = []
    for i in range(len(X_te)):
        appl = cover(frozenset(np.flatnonzero(mask[i]).tolist()), est.patterns_)
        if not appl:
            out.append(est.prior_)
            continue
        ps = [expit(est.raw_theta_[e][INTERCEPT]
                    + sum(est.raw_theta_[e][v] * Xs[i, v] for v in sorted(e)))
              for e in appl]
        out.append(float(np.mean(ps)))            # unweighted vote
    return np.clip(np.asarray(out), 1e-6, 1 - 1e-6)
