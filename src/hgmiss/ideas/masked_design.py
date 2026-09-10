"""One design builder for every mask-using arm, and the only correct null for them.

Result 141 discarded the first null because it rebuilt the feature matrix from the permuted mask,
filling newly-observed cells from column means. That changes the VALUES, so it moved `mean_impute`
-- a method that never sees the mask -- by +0.0207 on lung. A mask null that moves a mask-blind
method is measuring something else.

Here the values are built once from the real `X` and never touched. Only the indicator block is
permuted. That makes the null falsifiable in the one way that matters: with `kind="none"` the
permutation must change the output by exactly zero, and `selftest()` asserts it.

The frozen imputer and estimator are imported from `baselines.simple`, not reimplemented, so the
arms cannot drift from the published baselines. `selftest()` also checks that this builder
reproduces `mean_impute_lr` and `mean_indicator_lr` bit-for-bit.
"""
from __future__ import annotations

import numpy as np

from ..baselines.simple import _fit_impute, _lr
from .residual_mask import residual_mask

# "residual_scaled" was removed after Result 142: the estimator standardises every column
# before the L2 penalty, so rescaling a column is undone before the model sees it. It was
# identical to "residual" at 1.7e-16 -- an arm that could not differ. Do not re-add a kind
# that only changes column scale; the selftest below now refuses one.
KINDS = ("none", "raw", "residual")


def masked_lr(X_tr, y_tr, X_te, *, kind: str = "raw", perm_seed: int | None = None):
    """Fit `[imputed values, <mask block>]` -> LR. `perm_seed` shuffles only the mask block.

    kind="none"            values only; the mask block is absent, so the permutation is a no-op
    kind="raw"             the mask as-is (the frozen `mean_indicator_lr` design)
    kind="residual"        the mask minus its leave-one-column-out linear prediction
    kind="residual_scaled" that residual rescaled to unit train variance
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float)
    imp = _fit_impute(X_tr)
    A_tr, A_te = imp.transform(X_tr), imp.transform(X_te)
    if kind == "none":
        return _lr().fit(A_tr, y_tr).predict_proba(A_te)[:, 1]

    M_tr = ~np.isnan(X_tr)
    M_te = ~np.isnan(X_te)
    if perm_seed is not None:
        rng = np.random.default_rng(perm_seed)
        # permute mask rows within each split; the values stay attached to their own records
        M_tr = M_tr[rng.permutation(len(M_tr))]
        M_te = M_te[rng.permutation(len(M_te))]

    if kind == "raw":
        B_tr, B_te = M_tr, M_te
    else:
        B_tr, B_te = residual_mask(X_tr, X_te, M_tr=M_tr.astype(float),
                                   M_te=M_te.astype(float))

    tr = np.column_stack([A_tr, B_tr])
    te = np.column_stack([A_te, B_te])
    return _lr().fit(tr, y_tr).predict_proba(te)[:, 1]


def selftest() -> int:
    """Prove the builder matches the frozen baselines and that the null cannot move a mask-blind
    arm. Every assertion here has been checked to fail when its guard is removed."""
    from ..baselines.simple import mean_impute_lr, mean_indicator_lr

    rng = np.random.default_rng(0)
    n, d = 900, 12
    X = rng.normal(size=(n, d))
    X[rng.random((n, d)) < 0.3] = np.nan
    y = (rng.random(n) < 0.25).astype(int)
    tr, te = np.arange(600), np.arange(600, n)

    a = masked_lr(X[tr], y[tr], X[te], kind="none")
    b = mean_impute_lr(X[tr], y[tr], X[te])
    assert np.allclose(a, b, atol=0, rtol=0), "kind='none' does not reproduce mean_impute_lr"
    print("    kind='none' reproduces mean_impute_lr exactly")

    a = masked_lr(X[tr], y[tr], X[te], kind="raw")
    b = mean_indicator_lr(X[tr], y[tr], X[te])
    assert np.allclose(a, b, atol=0, rtol=0), "kind='raw' does not reproduce mean_indicator_lr"
    print("    kind='raw'  reproduces mean_indicator_lr exactly")

    # the check the first null failed: permuting the mask must not move a mask-blind arm at all
    p = masked_lr(X[tr], y[tr], X[te], kind="none", perm_seed=7)
    q = masked_lr(X[tr], y[tr], X[te], kind="none")
    assert np.array_equal(p, q), "the null moved a mask-blind arm -- it is touching the values"
    print("    permuting the mask leaves the mask-blind arm bit-for-bit identical")

    # and it must move an arm that DOES use the mask, or it is not a null at all
    p = masked_lr(X[tr], y[tr], X[te], kind="raw", perm_seed=7)
    q = masked_lr(X[tr], y[tr], X[te], kind="raw")
    assert not np.array_equal(p, q), "the null did not move the mask arm -- it is not permuting"
    print("    permuting the mask does move the raw-mask arm")

    preds = {}
    for k in KINDS:
        v = masked_lr(X[tr], y[tr], X[te], kind=k)
        assert np.all(np.isfinite(v)) and v.shape == (len(te),), f"{k} produced bad output"
        preds[k] = v
    print(f"    all {len(KINDS)} kinds produce finite predictions")

    # Result 142: two arms declared different were the same function, because the estimator
    # standardises away the only thing that distinguished them. An arm that cannot differ from
    # another cannot be evidence about anything, and the run that compares them is wasted.
    ks = sorted(preds)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            gap = float(np.abs(preds[a] - preds[b]).max())
            assert gap > 1e-9, f"kinds {a!r} and {b!r} are the same function (max gap {gap:.3g})"
    print("    every pair of kinds is genuinely distinct (none collapses onto another)")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
