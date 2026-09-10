"""Two consequences of the finding that site heterogeneity inverts the theory.

`merge_rules.py` established that combining estimates beats combining data here, and
that the one-step refinement degrades because it targets the pooled optimum. Both point
at the same cause: the sites are not draws from a common population. Meta-analysis has
handled exactly that for fifty years, and two of its standard devices have never been
tried on this problem.

**Random-effects weighting.** Inverse-variance (Fisher) weighting is the *fixed-effect*
estimator: it assumes every site estimates the same quantity and weights purely by
precision. When sites genuinely differ it is known to be overconfident, and the
DerSimonian-Laird random-effects weight ``1/(v_k + tau^2)`` -- with ``tau^2`` the
between-site variance estimated from Cochran's Q -- is the standard correction. It
moves weight away from a single very precise site toward consensus. Computed per
coordinate over the sites measuring it, and from aggregates alone.

**Intercept recalibration.** Ovarian and prostate have prevalences 0.026 and 0.051, and
a merged model carries one intercept. Refitting only the intercept on a site's own
training rows, with the merged coefficients held fixed, is the first step of Cox
recalibration and is standard practice in prediction-model transportability. It is a
one-parameter local fit that shares nothing.

Cochran's Q is also reported for its own sake: it is computable without raw data and is
the natural candidate for a *pre-merge test* -- decide whether to merge at all before
paying for it.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge, pooled  # noqa: E402

warnings.simplefilter("ignore")


def merge_random_effects(models):
    """DerSimonian-Laird combination, per coordinate, over the sites measuring it."""
    by_v = defaultdict(list)
    for m in models:
        if m is None:
            continue
        for j, v in enumerate(m["cols"]):
            var = 1.0 / max(m["info"][j], 1e-12)          # inverse observed information
            by_v[v].append((m["raw"][j], var, m["n"]))
    cols = sorted(by_v)
    coef, qstats = [], []
    for v in cols:
        th = np.array([t for t, _, _ in by_v[v]])
        vr = np.array([s for _, s, _ in by_v[v]])
        w = 1.0 / vr
        fe = float(w @ th / w.sum())
        Q = float(w @ (th - fe) ** 2)
        m_k = len(th)
        denom = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (Q - (m_k - 1)) / denom) if denom > 0 and m_k > 1 else 0.0
        wr = 1.0 / (vr + tau2)
        coef.append(float(wr @ th / wr.sum()))
        if m_k > 1:
            qstats.append(Q)
    # intercept and scalers follow the n-weighted convention, unchanged
    base = merge(models, "raw", "fisher")
    out = dict(base)
    out["raw"] = np.array(coef)
    return out, np.array(qstats)


def recalibrate_intercept(model, X, y, apply_fn):
    """Refit only the intercept on this site's own training rows; coefficients fixed."""
    p = np.clip(apply_fn(model, X), 1e-6, 1 - 1e-6)
    off = logit(p)
    # Newton on the single shift parameter, with the linear predictor as a fixed offset.
    # Damped and bounded for the same reason as the calibration intercept in
    # `multisite.metrics`: when a model's predictions saturate, the curvature vanishes and
    # an undamped step runs away, which then pushes every prediction to one extreme and
    # destroys the ranking. A pure intercept shift must leave AUPRC exactly unchanged, so
    # any run where it does not is this divergence and nothing else.
    shift = 0.0
    for _ in range(200):
        q = expit(np.clip(off + shift, -35, 35))
        h = float(np.sum(q * (1 - q)))
        if h < 1e-8:
            break
        step = float(np.clip(np.sum(y - q) / h, -1.0, 1.0))
        shift = float(np.clip(shift + step, -20, 20))
        if abs(step) < 1e-10:
            break
    out = dict(model)
    key = "raw_b" if "raw_b" in model else "intercept"
    out[key] = model[key] + shift
    return out, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-site", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/merge_heterogeneity.csv")
    a = ap.parse_args()

    from hgmiss.data.plco import load_cohort
    R = Path(os.environ["PLCO_ROOT"])
    co = {c: load_cohort(c, R, feature_set="baseline") for c in ["ovarian", "prostate"]}
    union = sorted(set(co["ovarian"].columns) | set(co["prostate"].columns))
    ix = {v: i for i, v in enumerate(union)}
    sites = []
    for nm, c in co.items():
        keep, _ = train_test_split(np.arange(len(c.y)), train_size=a.n_per_site,
                                   stratify=c.y, random_state=0)
        M = np.full((len(keep), len(union)), np.nan)
        for j, v in enumerate(c.columns):
            M[:, ix[v]] = c.X[keep, j]
        sites.append((M, c.y[keep], [ix[v] for v in c.columns], nm))
    shared = len(set(co["ovarian"].columns) & set(co["prostate"].columns))
    print(f"union d={len(union)}  shared d={shared}  "
          + ", ".join(f"{nm} n={len(y)} prev={y.mean():.3f}" for _, y, _, nm in sites))

    A = defaultdict(list)
    Qs, shifts = [], []
    for seed in range(a.seeds):
        tr, te = [], []
        for X, y, cols, nm in sites:
            i1, i2 = train_test_split(np.arange(len(y)), test_size=0.3,
                                      stratify=y, random_state=seed)
            tr.append((X[i1], y[i1], cols))
            te.append((X[i2], y[i2], cols, nm))
        loc = [fit_local(X, y, c) for X, y, c in tr]
        m_std = merge(loc, "std", "n")
        m_re, q = merge_random_effects(loc)
        Qs.append(q)
        pl = pooled(tr)
        for i, (Xt, yt, cols, nm) in enumerate(te):
            Xtr_i, ytr_i, _ = tr[i]
            A["merge (ours)"].append(average_precision_score(yt, apply_std(m_std, Xt)))
            A["merge_random_effects"].append(average_precision_score(yt, apply_raw(m_re, Xt)))
            rc, sh = recalibrate_intercept(m_std, Xtr_i, ytr_i, apply_std)
            shifts.append(sh)
            A["merge + local intercept"].append(
                average_precision_score(yt, apply_std(rc, Xt)))
            rc2, _ = recalibrate_intercept(m_re, Xtr_i, ytr_i, apply_raw)
            A["merge_RE + local intercept"].append(
                average_precision_score(yt, apply_raw(rc2, Xt)))
            A["local_only"].append(average_precision_score(yt, apply_raw(loc[i], Xt)))
            A["pooled_raw_data"].append(average_precision_score(yt, pl(Xt)))

    q = np.concatenate(Qs)
    print(f"\nCochran's Q over the {len(q) // a.seeds} shared coefficients "
          f"(1 df, chi2 crit 3.84):")
    print(f"  median {np.median(q):.2f}   mean {q.mean():.2f}   "
          f"fraction significant {np.mean(q > 3.84):.2f}")
    print(f"local intercept shifts applied: mean {np.mean(shifts):+.3f} "
          f"(range {np.min(shifts):+.3f} to {np.max(shifts):+.3f})")

    base = np.array(A["merge (ours)"])
    print(f"\n{'method':30s} {'AUPRC':>8s} {'vs merge':>9s} {'p':>8s}")
    rows = []
    for k, v in sorted(A.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v)
        d = v - base
        p = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
        flag = ("  BETTER" if d.mean() > 0 and p < 0.05
                else "  worse" if d.mean() < 0 and p < 0.05 else "")
        print(f"{k:30s} {v.mean():8.4f} {d.mean():+9.4f} {p:8.1e}{flag}")
        rows.append(dict(method=k, auprc=float(v.mean()), delta=float(d.mean()),
                         p=float(p), n_paired=len(v)))
    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
