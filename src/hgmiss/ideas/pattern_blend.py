"""B7 -- Paper C's adaptive shrinkage, with "site" replaced by "pattern".

Paper C found that interpolating toward the local model at a weight chosen on held-out local
data is the only policy never harmful on any pool. Missingness patterns have the same shape as
sites: disjoint groups of records, each with its own conditional distribution, each too small to
fit alone. The port has never been run here.

Four arms, in increasing order of what they are allowed to know:

    pooled        one model on all training rows. The frozen `mean_indicator` design, which is
                  the baseline this method family currently LOSES to -- so this composes with
                  the winner rather than competing with it (METHOD-IDEAS section B).
    local         each group's own model wherever support allows, pooled elsewhere. lam = 0.
    blend_cv      per-group lam chosen on an inner split of that group's OWN training rows.
                  The attainable policy, and Paper C's `blend`.
    blend_oracle  per-group lam chosen on the OUTER TEST rows. Not attainable. It is the
                  ceiling, and it is the decisive arm: if the oracle cannot clear the floor
                  then no selection rule can, and the noisiness of `blend_cv` is beside the
                  point.

**The local design is the pooled design, not the pattern's observed columns.** Within a pattern
you know exactly which variables are observed, so the natural pattern-conditional model needs no
imputation at all -- but that makes a null impossible, because a record assigned to a random
pattern has NaN in columns that pattern calls observed. Using one design for both lets the null
be exactly the right question: do real patterns beat RANDOM groups of the same size distribution?
Result 142 is why that trade is taken; an arm without a null is an arm that cannot be interpreted.

Interpolation is on the logit scale, as in Paper C: `z = (1 - lam) * z_local + lam * z_pooled`,
so lam = 1 recovers `pooled` exactly and the oracle is weakly above it by construction. What the
oracle's MARGIN is worth is the question; that it is non-negative is arithmetic.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from ..baselines.simple import _fit_impute, _lr

# Paper C's grid, unchanged, so the port is a port. lam = weight on the POOLED model.
LAMBDAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
MIN_SUPPORT = 30      # the fittable threshold this project already uses (METHOD-IDEAS C8)
INNER_FRAC = 0.35     # share of a group's training rows held out to choose its lam
EPS = 1e-6


def _design(imp, X):
    """The frozen `mean_indicator_lr` design: imputed values beside the raw mask."""
    return np.column_stack([imp.transform(X), ~np.isnan(X)])


def _safe_logit(p):
    return logit(np.clip(p, EPS, 1.0 - EPS))


def _fit_predict(D_tr, y_tr, D_te):
    """Fit the frozen estimator, return test logits. None when the rows cannot support a fit."""
    if len(D_tr) < 8 or len(np.unique(y_tr)) < 2:
        return None
    try:
        m = _lr().fit(D_tr, y_tr)
    except Exception:
        return None
    return _safe_logit(m.predict_proba(D_te)[:, 1])


def _auprc(y, s):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def group_ids(X, rng=None):
    """Integer group id per row: the missingness pattern, or a random regrouping under the null.

    The null draws a permutation of the real ids, so the group SIZE DISTRIBUTION is preserved
    exactly and only the assignment of records to groups is destroyed. That is the one thing the
    method depends on, and the only thing the null removes.
    """
    M = ~np.isnan(np.asarray(X, float))
    _, ids = np.unique(M, axis=0, return_inverse=True)
    if rng is not None:
        ids = ids[rng.permutation(len(ids))]
    return ids


def pattern_blend(X_tr, y_tr, X_te, *, arm: str, g_tr, g_te, y_te=None, seed: int = 0):
    """Predicted probabilities on `X_te` for one arm.

    `blend_oracle` needs `y_te`; it is the ceiling and is explicitly allowed to see it.
    """
    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float)
    y_tr = np.asarray(y_tr).ravel()
    imp = _fit_impute(X_tr)
    D_tr, D_te = _design(imp, X_tr), _design(imp, X_te)

    z_pool_te = _fit_predict(D_tr, y_tr, D_te)
    if arm == "pooled" or z_pool_te is None:
        return expit(z_pool_te) if z_pool_te is not None else np.full(len(X_te), y_tr.mean())

    z_out = z_pool_te.copy()
    rng = np.random.default_rng(seed)

    # ONE inner split shared by every group, not one per group. Two reasons, and the second is
    # the important one. Cost: the pooled model is identical for every group, so refitting it
    # inside the loop was 152 fits on 32,000x376 per fold where 1 would do. Fairness: the
    # obvious cheap fix -- fit pooled once on ALL training rows -- lets the pooled arm score
    # validation rows it was trained on while the local arm is held out from them, biasing
    # every lambda toward pooled. Here both models are fit on the inner-fit rows only and both
    # are scored on inner-val rows neither has seen.
    _pending: dict = {}
    zl_inner = zp_inner = val_all = None
    if arm == "blend_cv":
        perm = rng.permutation(len(y_tr))
        n_val = int(round(INNER_FRAC * len(y_tr)))
        val_all, fit_all = perm[:n_val], perm[n_val:]
        zp_inner = _fit_predict(D_tr[fit_all], y_tr[fit_all], D_tr[val_all])
        if zp_inner is None:
            arm = "pooled_fallback"

    for g in np.unique(g_te):
        te_i = np.where(g_te == g)[0]
        tr_i = np.where(g_tr == g)[0]
        if len(tr_i) < MIN_SUPPORT or len(np.unique(y_tr[tr_i])) < 2:
            continue                      # too small to fit: keep the pooled score
        if arm == "local":
            zl = _fit_predict(D_tr[tr_i], y_tr[tr_i], D_te[te_i])
            if zl is not None:
                z_out[te_i] = zl
            continue

        if arm == "blend_cv":
            # this group's slice of the shared inner split -- Paper C's rule, one level down
            in_val = val_all[np.isin(val_all, tr_i)]
            in_fit = np.setdiff1d(tr_i, val_all, assume_unique=False)
            if len(in_val) < 4 or len(in_fit) < 8 or len(np.unique(y_tr[in_fit])) < 2:
                continue
            zl_val = _fit_predict(D_tr[in_fit], y_tr[in_fit], D_tr[in_val])
            if zl_val is None:
                continue
            # `zp_inner` is indexed by position within `val_all`, and `in_val` was taken as
            # `val_all[isin(...)]`, so it is already in `val_all` order and this mask lines the
            # two up. Asserted rather than assumed: a silent misalignment here would choose
            # every lambda against the wrong rows and look like a result.
            sel = np.isin(val_all, in_val)
            assert np.array_equal(val_all[sel], in_val), "inner-val rows are misaligned"
            zp_val = zp_inner[sel]
            best, best_s = 1.0, -np.inf
            for lam in LAMBDAS:
                sc = _auprc(y_tr[in_val], (1 - lam) * zl_val + lam * zp_val)
                if sc is not None and sc > best_s:
                    best, best_s = lam, sc
            lam = best
        elif arm == "blend_oracle":
            lam = None                    # decided below, on the outer test rows
        elif arm == "pooled_fallback":
            continue
        else:
            raise ValueError(f"unknown arm {arm!r}")

        zl_te = _fit_predict(D_tr[tr_i], y_tr[tr_i], D_te[te_i])
        if zl_te is None:
            continue
        if arm == "blend_oracle":
            # Deferred: a per-group argmax on THIS group's own AUPRC is the wrong objective
            # (Result 143). Collected here and ascended globally below.
            _pending[g] = (te_i, zl_te)
            continue
        z_out[te_i] = (1 - lam) * zl_te + lam * z_pool_te[te_i]

    if arm == "blend_oracle" and _pending:
        # Coordinate ascent on the GLOBAL AUPRC -- the number this arm is scored on. Result 143
        # showed a per-group argmax can improve every group and still lose overall, because
        # AUPRC ranks all test records together. Starting from lam = 1 everywhere means the
        # ascent begins exactly at `pooled` and only ever accepts a strict improvement, so the
        # arm is a genuine ceiling by construction rather than by assumption.
        lam_of = {g: 1.0 for g in _pending}
        cur = _auprc(y_te, z_out)
        if cur is not None:
            for _ in range(3):                       # passes; converges well before this
                moved = False
                for g, (te_i, zl_te) in _pending.items():
                    keep_lam, keep_z = lam_of[g], z_out[te_i].copy()
                    for cand in LAMBDAS:
                        if cand == lam_of[g]:
                            continue
                        z_out[te_i] = (1 - cand) * zl_te + cand * z_pool_te[te_i]
                        s = _auprc(y_te, z_out)
                        if s is not None and s > cur + 1e-12:
                            cur, keep_lam, keep_z = s, cand, z_out[te_i].copy()
                            moved = True
                    lam_of[g], z_out[te_i] = keep_lam, keep_z
                if not moved:
                    break
    return expit(z_out)


def selftest() -> int:
    """Guards that were each confirmed to fail when the behaviour they check is removed."""
    from ..baselines.simple import mean_indicator_lr

    rng = np.random.default_rng(0)
    n, d = 3000, 10
    X = rng.normal(size=(n, d))
    # Only three columns may be missing, so there are at most 8 patterns and each clears
    # MIN_SUPPORT. The first run of this selftest let all ten columns go missing, giving up to
    # 1024 patterns for 2000 training rows -- no group reached support 30, every arm silently
    # fell back to `pooled`, and the arms were identical. The distinctness check below caught
    # it. A group structure too fine to fit is a real failure mode of this method and it must
    # not be the accidental state of its own test.
    for j in range(3):
        X[rng.random(n) < 0.3, j] = np.nan
    # give one pattern a genuinely different coefficient, or `local` has nothing to find
    y = (rng.random(n) < expit(0.6 * X[:, 5] + 0.9 * np.nan_to_num(X[:, 0]))).astype(int)
    tr, te = np.arange(2000), np.arange(2000, n)
    g = group_ids(X)
    kw = dict(g_tr=g[tr], g_te=g[te], y_te=y[te])

    p = pattern_blend(X[tr], y[tr], X[te], arm="pooled", **kw)
    q = mean_indicator_lr(X[tr], y[tr], X[te])
    assert np.allclose(p, q, atol=1e-9), "'pooled' does not reproduce the frozen mean_indicator_lr"
    print("    'pooled' reproduces the frozen mean_indicator_lr")

    # This assertion previously read "lam=1 recovers pooled, so the oracle cannot lose". That
    # is false for a PER-GROUP argmax (Result 143) and the assertion passed on this synthetic
    # data by luck while the real cohort violated it. It is true of the GLOBAL ascent, which
    # starts at lam=1 everywhere and accepts only strict improvements -- so it now checks a
    # property the code actually has, not one it was assumed to have.
    o = pattern_blend(X[tr], y[tr], X[te], arm="blend_oracle", **kw)
    assert _auprc(y[te], o) >= _auprc(y[te], p) - 1e-12, "the oracle scored BELOW pooled"
    print("    'blend_oracle' is weakly above 'pooled' (global ascent from lam=1)")

    for arm in ("local", "blend_cv"):
        v = pattern_blend(X[tr], y[tr], X[te], arm=arm, **kw)
        assert np.all(np.isfinite(v)) and v.shape == (len(te),), f"{arm} produced bad output"
    print("    'local' and 'blend_cv' produce finite predictions")

    # the null must change the grouping, and must preserve the size distribution exactly
    g2 = group_ids(X, rng=np.random.default_rng(3))
    assert not np.array_equal(g, g2), "the null did not regroup anything"
    a = np.sort(np.bincount(g))
    b = np.sort(np.bincount(g2))
    assert np.array_equal(a, b), "the null changed the group SIZE distribution"
    print("    the null regroups records and preserves group sizes exactly")

    # arms must not collapse onto one another (Result 142)
    preds = {a: pattern_blend(X[tr], y[tr], X[te], arm=a, **kw)
             for a in ("pooled", "local", "blend_cv", "blend_oracle")}
    ks = sorted(preds)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            gap = float(np.abs(preds[a] - preds[b]).max())
            assert gap > 1e-9, f"arms {a!r} and {b!r} are the same function (gap {gap:.3g})"
    print("    every pair of arms is genuinely distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
