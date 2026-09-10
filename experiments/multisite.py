"""A neutral comparison of merge strategies on real multi-site data, on four metrics.

The prior-art pass found the gap: every block-wise-missingness method is evaluated on
**synthetic data** and every one-shot federated method on **images**. Nobody has run
these families against each other on real multi-cohort clinical data with genuine schema
mismatch. This is that comparison, and it is deliberately a comparison rather than a
proposal -- no method here is ours to sell.

Two genuinely multi-site datasets, no simulated site structure:

  heart-4site   Cleveland, Hungary, Switzerland, Long Beach VA. Four hospitals that
                really did measure different things -- Switzerland records no cholesterol
                at all, and `ca`/`thal` are near-absent outside Cleveland. Small (n~920,
                d=13) but four sites, which is what makes random-effects weighting
                estimable at all.
  plco-2cohort  PLCO ovarian and prostate: disjoint populations, schema Jaccard 0.42,
                union d=146, 20,000 records each.

Four metrics, because three of the findings so far were metric-specific and one earlier
arm was invalidated by using the wrong one. AUPRC and AUROC are rank measures; Brier and
the calibration intercept/slope are not, and a per-site intercept shift is invisible to
the first pair by construction.

Cochran's Q over the shared coefficients is recorded per dataset. It needs no raw data
and no extra communication round, which makes it the natural candidate for a *pre-merge
test*: decide whether combining these sites can help before paying to do it.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_heterogeneity import merge_random_effects, recalibrate_intercept  # noqa: E402
from merge_rules import (apply_raw, apply_std, fit_local, merge,  # noqa: E402
                         newton_iterate)

warnings.simplefilter("ignore")

SCHEMA_THRESH = 0.5          # a variable is "in a site's schema" if half its rows have it
FED_ROUNDS = (1, 2, 5, 30)


# ------------------------------------------------------------------- datasets
def heart_4site():
    """Four hospitals, kept apart. The public loader concatenates them and loses the
    site label, which is the only thing that makes this dataset multi-site."""
    import io
    import urllib.request

    import pandas as pd
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]
    base = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/")
    Xs, ys, gs = [], [], []
    for k, site in enumerate(["cleveland", "hungarian", "switzerland", "va"]):
        raw = urllib.request.urlopen(base + f"processed.{site}.data",
                                     timeout=60).read().decode()
        d = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
        d.loc[d["chol"] == 0, "chol"] = np.nan            # documented sentinel
        num = d[cols[:-1]].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        Xs.append(num.to_numpy(float))
        ys.append((d["num"] > 0).astype(int).to_numpy())
        gs.append(np.full(len(d), k))
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs),
            ["cleveland", "hungarian", "switzerland", "va"])


def plco_2cohort(n_per_site=20000):
    from hgmiss.data.plco import load_cohort
    R = Path(os.environ["PLCO_ROOT"])
    co = {c: load_cohort(c, R, feature_set="baseline") for c in ["ovarian", "prostate"]}
    union = sorted(set(co["ovarian"].columns) | set(co["prostate"].columns))
    ix = {v: i for i, v in enumerate(union)}
    Xs, ys, gs = [], [], []
    for k, (nm, c) in enumerate(co.items()):
        keep, _ = train_test_split(np.arange(len(c.y)), train_size=n_per_site,
                                   stratify=c.y, random_state=0)
        M = np.full((len(keep), len(union)), np.nan)
        for j, v in enumerate(c.columns):
            M[:, ix[v]] = c.X[keep, j]
        Xs.append(M); ys.append(c.y[keep]); gs.append(np.full(len(keep), k))
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs),
            list(co.keys()))


def mimic_careunits():
    """MIMIC-III first ICU care unit as the site. Built by ``mimic_sites.py``; see that
    file for the cohort definition and the DUA note. One hospital, so this is care-unit
    heterogeneity rather than inter-institutional -- but five sites, unequal in size
    (4.9k-16.4k) and in prevalence (0.037-0.147), with schema differences that arise
    because units genuinely order different tests."""
    from mimic_sites import load
    X, y, g, names, _ = load(verbose=False)
    return X, y, g, names


def site_schema(X):
    """Columns a site actually measures. Scattered missingness inside the schema is
    imputed locally by ``fit_local``; a column below the threshold is simply absent."""
    obs = (~np.isnan(X)).mean(0)
    return [j for j in range(X.shape[1]) if obs[j] >= SCHEMA_THRESH]


# ------------------------------------------------------------- other estimators
def prep(X, cols):
    Z = np.asarray(X[:, cols], float)
    cm = np.nanmean(Z, 0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    Z = np.where(np.isnan(Z), cm, Z)
    sc = StandardScaler().fit(Z)
    return sc.transform(Z), sc


def fedavg(tr, d_union, rounds, checkpoints, local_epochs=5, lr=1.0):
    w = np.zeros(d_union + 1)
    prepped = [(*prep(X, c), np.asarray(y, float), list(c)) for X, y, c in tr]
    snaps = {}
    for r in range(1, rounds + 1):
        acc = np.zeros_like(w); tot = 0.0
        for Z, sc, y, c in prepped:
            idx = np.array([0] + [j + 1 for j in c])
            wl = w[idx].copy()
            for _ in range(local_epochs):
                g = expit(wl[0] + Z @ wl[1:]) - y
                wl -= lr * np.concatenate([[g.mean()], Z.T @ g / len(y)])
            upd = w.copy(); upd[idx] = wl
            acc += len(y) * upd; tot += len(y)
        w = acc / tot
        if r in checkpoints:
            snaps[r] = w.copy()
    return snaps, [(p[1], p[3]) for p in prepped]


def apply_fed(w, X, cols, sc):
    A = np.asarray(X[:, cols], float)
    sd = np.where(sc.scale_ > 0, sc.scale_, 1.0)
    Z = np.where(np.isnan(A), 0.0, (A - sc.mean_) / sd)
    idx = np.array([0] + [j + 1 for j in cols])
    return expit(w[idx][0] + Z @ w[idx][1:])


def pooled_union(tr):
    X = np.vstack([t[0] for t in tr]); y = np.concatenate([t[1] for t in tr])
    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)

    def D(A):
        return np.column_stack([imp.transform(A), (~np.isnan(A)).astype(float)])
    sc = StandardScaler().fit(D(X))
    m = LogisticRegression(max_iter=3000).fit(sc.transform(D(X)), y)
    return lambda A: m.predict_proba(sc.transform(D(A)))[:, 1]


# -------------------------------------------------------------------- metrics
def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}
    out["AUROC"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    lp = logit(p)
    try:                       # calibration slope: y ~ logit(p); ideal 1.0
        out["cal_slope"] = float(
            LogisticRegression(max_iter=1000, C=1e12).fit(lp[:, None], y).coef_[0, 0])
    except Exception:                                            # noqa: BLE001
        out["cal_slope"] = np.nan
    # Calibration intercept at fixed slope; ideal 0.0. Damped and bounded: when a model
    # saturates (every p at the clip) the curvature vanishes and an undamped Newton step
    # runs off to 1e9, which is a solver artefact rather than a calibration measurement.
    shift = 0.0
    for _ in range(200):
        q = expit(np.clip(lp + shift, -35, 35))
        h = float(np.sum(q * (1 - q)))
        if h < 1e-8:
            break
        step = float(np.clip(np.sum(y - q) / h, -1.0, 1.0))
        shift = float(np.clip(shift + step, -20, 20))
        if abs(step) < 1e-10:
            break
    out["cal_intercept"] = shift
    return out


def cochran_q(models):
    """Heterogeneity per shared coefficient, from aggregates only -- no raw data, no
    extra communication round.

    Returns Q, its degrees of freedom, and I-squared. Q and I-squared answer different
    questions and the distinction turns out to matter: **Q grows with sample size**, so a
    large Q at large n can mean a tiny real difference measured precisely. I-squared,
    ``max(0, (Q - df) / Q)``, is the fraction of total variation attributable to genuine
    between-site difference and is scale-free. If a pre-merge test is ever built, this is
    the pair to build it from.
    """
    by_v = defaultdict(list)
    for m in models:
        if m is None:
            continue
        for j, v in enumerate(m["cols"]):
            by_v[v].append((m["raw"][j], 1.0 / max(m["info"][j], 1e-12)))
    qs, dfs, i2 = [], [], []
    for v, lst in by_v.items():
        if len(lst) < 2:
            continue
        th = np.array([t for t, _ in lst]); w = 1.0 / np.array([s for _, s in lst])
        fe = float(w @ th / w.sum())
        Q = float(w @ (th - fe) ** 2)
        qs.append(Q); dfs.append(len(lst) - 1)
        i2.append(max(0.0, (Q - (len(lst) - 1)) / Q) if Q > 0 else 0.0)
    return np.array(qs), np.array(dfs), np.array(i2)


# ------------------------------------------------------------------- one dataset
def run_dataset(name, X, y, g, site_names, seeds, newton_iters):
    d = X.shape[1]
    groups = [np.flatnonzero(g == k) for k in range(len(site_names))]
    print(f"\n=== {name}: {len(site_names)} sites, union d={d} ===")
    for k, nm in enumerate(site_names):
        Xi = X[groups[k]]
        print(f"    {nm:12s} n={len(groups[k]):6d} prev={y[groups[k]].mean():.3f} "
              f"schema={len(site_schema(Xi)):3d}/{d}")

    A = defaultdict(lambda: defaultdict(list))
    allq, alldf, alli2 = [], [], []
    for seed in range(seeds):
        tr, te = [], []
        ok = True
        for k in range(len(site_names)):
            idx = groups[k]
            try:
                i1, i2 = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                          random_state=seed)
            except ValueError:
                ok = False
                break
            cols = site_schema(X[i1])
            if len(cols) < 2 or len(np.unique(y[i1])) < 2:
                ok = False
                break
            tr.append((X[i1], y[i1], cols)); te.append((X[i2], y[i2], cols, k))
        if not ok:
            continue

        loc = [fit_local(Xi, yi, c) for Xi, yi, c in tr]
        if any(m is None for m in loc):
            continue
        q, df, i2 = cochran_q(loc)
        allq.append(q); alldf.append(df); alli2.append(i2)

        m_n = merge(loc, "std", "n")
        m_f = merge(loc, "raw", "fisher")
        m_re, _ = merge_random_effects(loc)
        pl = pooled_union(tr)
        nw = newton_iterate(loc, m_f, checkpoints=(newton_iters,))[newton_iters]
        snaps, prepped = fedavg(tr, d, max(FED_ROUNDS), set(FED_ROUNDS))

        for i, (Xt, yt, cols, k) in enumerate(te):
            Xtr_i, ytr_i, _ = tr[i]
            preds = {
                "local_only": apply_raw(loc[i], Xt),
                "merge_n": apply_std(m_n, Xt),
                "merge_fisher": apply_raw(m_f, Xt),
                "merge_randomeffects": apply_raw(m_re, Xt),
                "ensemble_of_locals": np.mean([apply_raw(m, Xt) for m in loc], 0),
                "pooled_raw_data": pl(Xt),
                f"newton_{newton_iters}it": apply_raw(nw, Xt),
            }
            rc, _ = recalibrate_intercept(m_n, Xtr_i, ytr_i, apply_std)
            preds["merge_n + local intercept"] = apply_std(rc, Xt)
            for r, w in snaps.items():
                preds[f"fedavg_{r}r"] = apply_fed(w, Xt, cols, prepped[i][0])
            for meth, p in preds.items():
                for mk, mv in metrics(yt, p).items():
                    A[meth][mk].append(mv)

    if not A:
        print("    (no usable folds)")
        return []

    q = np.concatenate(allq); df = np.concatenate(alldf); i2 = np.concatenate(alli2)
    crit = stats.chi2.ppf(0.95, df)
    print(f"    heterogeneity over {len(q) // max(len(allq), 1)} shared coefficients: "
          f"Q median {np.median(q):.2f}, {np.mean(q > crit) * 100:.0f}% significant, "
          f"I2 median {np.median(i2):.2f}")

    base = np.array(A["merge_n"]["AUPRC"])
    print(f"    {'method':28s} {'AUPRC':>7s} {'AUROC':>7s} {'Brier':>7s} "
          f"{'cal_int':>8s} {'cal_slp':>8s} {'dAUPRC':>8s} {'p':>8s}")
    rows = []
    for meth in sorted(A, key=lambda m: -np.mean(A[m]["AUPRC"])):
        v = np.array(A[meth]["AUPRC"]); dd = v - base
        p = stats.wilcoxon(dd).pvalue if not np.allclose(dd, 0) else 1.0
        mm = {k: float(np.nanmean(A[meth][k])) for k in
              ("AUPRC", "AUROC", "Brier", "cal_intercept", "cal_slope")}
        print(f"    {meth:28s} {mm['AUPRC']:7.4f} {mm['AUROC']:7.4f} {mm['Brier']:7.4f} "
              f"{mm['cal_intercept']:+8.3f} {mm['cal_slope']:8.3f} "
              f"{dd.mean():+8.4f} {p:8.1e}")
        rows.append(dict(dataset=name, method=meth, **mm, dAUPRC=float(dd.mean()),
                         p=float(p), q_median=float(np.median(q)),
                         q_frac_sig=float(np.mean(q > crit)),
                         i2_median=float(np.median(i2)), n_sites=len(site_names)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="heart-4site,mimic-careunits,plco-2cohort")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--newton-iters", type=int, default=25)
    ap.add_argument("--out", default="results/multisite.csv")
    a = ap.parse_args()
    want = set(a.datasets.split(","))
    rows = []
    if "heart-4site" in want:
        X, y, g, nm = heart_4site()
        rows += run_dataset("heart-4site", X, y, g, nm, max(a.seeds, 20), a.newton_iters)
    if "mimic-careunits" in want:
        X, y, g, nm = mimic_careunits()
        rows += run_dataset("mimic-careunits", X, y, g, nm, a.seeds, a.newton_iters)
    if "plco-2cohort" in want:
        X, y, g, nm = plco_2cohort()
        rows += run_dataset("plco-2cohort", X, y, g, nm, a.seeds, a.newton_iters)
    if rows:
        import csv
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
