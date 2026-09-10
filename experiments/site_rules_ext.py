"""Is EPV < 10 a fact about our merge rule, or about the local model's adequacy?

`site_rules.py` found that whether combining beats staying local is predicted by the
target site's events per variable, with a cut at ~10 -- the classical threshold for
fitting a logistic regression at all. That was measured for **one combiner** (n-weighted
coefficient averaging) and **one model class** (logistic regression), which leaves the
interesting question open.

Two predictions follow from the mechanistic reading, and both are falsifiable:

  combiner invariance   If the threshold is a property of the *local* model being
                        underpowered, it should not care how the sites are combined.
                        Ensembling, pooling the raw data and iterative FedAvg should all
                        show the same cut. If instead each combiner has its own
                        threshold, the rule is about our merge and travels no further.

  complexity scaling    EPV counts events per estimated parameter. A gradient-boosted
                        tree estimates far more than a logistic regression, so if the
                        mechanism is adequacy it needs *more* events before going solo --
                        the threshold should move up. A threshold that does not move
                        would say EPV is a proxy for something else.

The second needs a model class that cannot be coefficient-averaged, which is the point:
`HistGradientBoostingClassifier` consumes NaN natively, so block-wise missingness needs
no imputation and the only available combiners are ensembling and pooling -- exactly the
one-shot-FL families. That also closes the gap noted in the benchmark: our comparison has
been linear-only while the one-shot FL literature is deep-learning-only.

Local models are cached per (site, seed): they do not depend on which partners are being
considered, and that is what makes a tree-based sweep affordable.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

SUBSETS_PER_TARGET = 10
MAX_PARTNERS = 4


# --------------------------------------------------------------------- learners
def gbdt(X, y, cols, seed, early_stopping=None):
    """Trees on a site's own schema. NaN is handled natively -- no imputation.

    ``early_stopping`` defaults to on, which is what produced the Result 10 numbers. It
    carves a stratified 15% validation slice out of the training rows, and that split
    fails outright once a class has fewer than two members in the slice -- which happens
    on a quarter of a small, highly imbalanced site's inner fold. Callers that subsample
    (the learning curve) pass it explicitly rather than letting the fit raise.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    if early_stopping is None:
        early_stopping = True
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       early_stopping=early_stopping,
                                       validation_fraction=0.15,
                                       random_state=seed)
    try:
        m.fit(np.asarray(X[:, cols], float), y)
    except Exception:      # tiny/degenerate sites break the histogram binner outright
        return None
    return dict(model=m, cols=list(cols))


def gbdt_apply(fit, X):
    """Columns the scoring site lacks arrive as NaN, which the trees accept."""
    return fit["model"].predict_proba(np.asarray(X[:, fit["cols"]], float))[:, 1]


def pooled_logistic(parts):
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)

    def D(A):
        return np.column_stack([imp.transform(A), (~np.isnan(A)).astype(float)])
    sc = StandardScaler().fit(D(X))
    m = LogisticRegression(max_iter=2000).fit(sc.transform(D(X)), y)
    return lambda A: m.predict_proba(sc.transform(D(A)))[:, 1]


def pooled_gbdt(parts, seed):
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    if len(np.unique(y)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15,
                                       random_state=seed)
    try:
        m.fit(X, y)
    except Exception:      # same binner failure as gbdt(), on the pooled design
        return None
    return lambda A: m.predict_proba(A)[:, 1]


def fedavg(parts, d_union, rounds=5, local_epochs=5, lr=1.0):
    w = np.zeros(d_union + 1)
    prepped = []
    for X, y, cols, *_ in parts:
        Z = np.asarray(X[:, cols], float)
        cm = np.nanmean(Z, 0); cm = np.where(np.isfinite(cm), cm, 0.0)
        sc = StandardScaler().fit(np.where(np.isnan(Z), cm, Z))
        prepped.append((sc.transform(np.where(np.isnan(Z), cm, Z)), sc,
                        np.asarray(y, float), list(cols)))
    for _ in range(rounds):
        acc = np.zeros_like(w); tot = 0.0
        for Z, sc, y, cols in prepped:
            idx = np.array([0] + [j + 1 for j in cols])
            wl = w[idx].copy()
            for _ in range(local_epochs):
                g = expit(wl[0] + Z @ wl[1:]) - y
                wl -= lr * np.concatenate([[g.mean()], Z.T @ g / len(y)])
            upd = w.copy(); upd[idx] = wl
            acc += len(y) * upd; tot += len(y)
        w = acc / tot
    sc0, cols0 = prepped[0][1], prepped[0][3]

    def score(A):
        B = np.asarray(A[:, cols0], float)
        sd = np.where(sc0.scale_ > 0, sc0.scale_, 1.0)
        Z = np.where(np.isnan(B), 0.0, (B - sc0.mean_) / sd)
        idx = np.array([0] + [j + 1 for j in cols0])
        return expit(w[idx][0] + Z @ w[idx][1:])
    return score


# ------------------------------------------------------------------ observations
def run_dataset(name, X, y, g, site_names, seeds=2, rng_seed=0, do_gbdt=True):
    rng = np.random.default_rng(rng_seed)
    groups = [np.flatnonzero(g == k) for k in range(len(site_names))]
    d = X.shape[1]
    split, loc_lr, loc_gb = {}, {}, {}

    for k in range(len(site_names)):
        for seed in range(seeds):
            idx = groups[k]
            try:
                i1, i2 = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                          random_state=seed)
            except ValueError:
                continue
            cols = site_schema(X[i1])
            if len(cols) < 2 or len(np.unique(y[i1])) < 2:
                continue
            split[(k, seed)] = (X[i1], y[i1], cols, X[i2], y[i2])
            loc_lr[(k, seed)] = fit_local(X[i1], y[i1], cols)
            if do_gbdt:
                loc_gb[(k, seed)] = gbdt(X[i1], y[i1], cols, seed)

    rows = []
    for k in range(len(site_names)):
        others = [j for j in range(len(site_names)) if j != k]
        if not others:
            continue
        subsets = set()
        for _ in range(SUBSETS_PER_TARGET):
            m = rng.integers(1, min(MAX_PARTNERS, len(others)) + 1)
            subsets.add(tuple(sorted(rng.choice(others, size=m, replace=False))))
        for S in sorted(subsets):
            for seed in range(seeds):
                keys = [(j, seed) for j in (k,) + S]
                if any(q not in split for q in keys):
                    continue
                parts = [split[q] for q in keys]
                Xte, yte = parts[0][3], parts[0][4]
                if len(np.unique(yte)) < 2:
                    continue
                ytr0 = parts[0][1]
                epv = float(min(np.bincount(ytr0)) / max(len(parts[0][2]), 1))

                r = dict(dataset=name, target=site_names[k], n_partners=len(S),
                         seed=seed, epv_target=epv, n_target=len(ytr0))

                lrs = [loc_lr[q] for q in keys]
                if all(m is not None for m in lrs):
                    a_loc = average_precision_score(yte, apply_raw(lrs[0], Xte))
                    r["lr_local"] = a_loc
                    r["lr_merge"] = average_precision_score(
                        yte, apply_std(merge(lrs, "std", "n"), Xte)) - a_loc
                    r["lr_ensemble"] = average_precision_score(
                        yte, np.mean([apply_raw(m, Xte) for m in lrs], 0)) - a_loc
                    r["lr_pooled"] = average_precision_score(
                        yte, pooled_logistic(parts)(Xte)) - a_loc
                    try:
                        r["lr_fedavg5"] = average_precision_score(
                            yte, fedavg(parts, d)(Xte)) - a_loc
                    except Exception:                               # noqa: BLE001
                        pass

                if do_gbdt:
                    gbs = [loc_gb.get(q) for q in keys]
                    if all(m is not None for m in gbs):
                        a_loc = average_precision_score(yte, gbdt_apply(gbs[0], Xte))
                        r["gb_local"] = a_loc
                        r["gb_ensemble"] = average_precision_score(
                            yte, np.mean([gbdt_apply(m, Xte) for m in gbs], 0)) - a_loc
                        pg = pooled_gbdt(parts, seed)
                        if pg is not None:
                            r["gb_pooled"] = average_precision_score(yte, pg(Xte)) - a_loc
                rows.append(r)
        print(f"  {name}: {site_names[k]} done ({len(rows)} rows)", flush=True)
    return rows


ARMS = ["lr_merge", "lr_ensemble", "lr_pooled", "lr_fedavg5",
        "gb_ensemble", "gb_pooled"]


def analyse(rows):
    print(f"\n{'combiner':14s} {'n':>5s} {'rho(EPV,benefit)':>17s} {'p':>9s} "
          f"{'best EPV cut':>13s} {'acc':>6s} {'base':>6s}")
    out = []
    for arm in ARMS:
        v = np.array([r.get(arm, np.nan) for r in rows], dtype=float)
        e = np.array([r["epv_target"] for r in rows], dtype=float)
        ok = np.isfinite(v) & np.isfinite(e)
        if ok.sum() < 30:
            continue
        rho, p = stats.spearmanr(e[ok], v[ok])
        helped = v[ok] > 0
        base = max(helped.mean(), 1 - helped.mean())
        best = (np.nan, -1.0)
        for t in np.quantile(e[ok], np.linspace(0.05, 0.95, 37)):
            acc = ((e[ok] < t) == helped).mean()
            if acc > best[1]:
                best = (float(t), float(acc))
        print(f"{arm:14s} {int(ok.sum()):5d} {rho:+17.3f} {p:9.1e} "
              f"{best[0]:13.1f} {best[1]:6.3f} {base:6.3f}")
        out.append(dict(combiner=arm, n=int(ok.sum()), rho=float(rho), p=float(p),
                        epv_cut=best[0], acc=best[1], majority=float(base),
                        helped_rate=float(helped.mean())))
    return out


def benefit_curve(rows):
    print(f"\nexpected benefit by EPV band (AUPRC gain over the local model):")
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    hdr = "".join(f"{f'{lo}-{hi if hi < 1e9 else chr(8734)}':>14s}" for lo, hi in bands)
    print(f"{'combiner':14s}{hdr}")
    for arm in ARMS:
        v = np.array([r.get(arm, np.nan) for r in rows], dtype=float)
        e = np.array([r["epv_target"] for r in rows], dtype=float)
        cells = []
        for lo, hi in bands:
            m = np.isfinite(v) & (e >= lo) & (e < hi)
            cells.append(f"{np.mean(v[m]):+.4f}" if m.sum() >= 5 else "     -")
        print(f"{arm:14s}" + "".join(f"{c:>14s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="heart-4site,mimic-11site,plco-2cohort,eicu")
    ap.add_argument("--matrix", default=None,
                    help="run on a derived matrix from MIMIC_CACHE instead of the named "
                         "datasets. Needed because the complexity-scaling claim can only "
                         "be tested where sites reach past EPV 10, which among the "
                         "collections on disk means ACS at d=10 and nothing else.")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-gbdt", action="store_true")
    ap.add_argument("--out", default="results/site_rules_ext.csv")
    a = ap.parse_args()
    want = set(a.datasets.split(","))
    rows = []
    if a.matrix:
        from matrix_loader import load_matrix
        X, y, g, nm = load_matrix(a.matrix)
        rows += run_dataset(a.matrix, X, y, g, nm, a.seeds, do_gbdt=not a.no_gbdt)
        want = set()          # --matrix is exclusive; mixing it with the named sets
                              # would pool incomparable schemas into one fitted cut
    if "heart-4site" in want:
        from multisite import heart_4site
        X, y, g, nm = heart_4site()
        rows += run_dataset("heart-4site", X, y, g, nm, a.seeds, do_gbdt=not a.no_gbdt)
    if "mimic-11site" in want:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        X, y, g, nm, _ = mimic_sites.load(verbose=False)
        rows += run_dataset("mimic-11site", X, y, g, nm, a.seeds, do_gbdt=not a.no_gbdt)
    if "eicu" in want:
        from eicu_sites import load as loade
        X, y, g, nm, _c = loade(verbose=False)
        rows += run_dataset("eicu", X, y, g, nm, a.seeds, do_gbdt=not a.no_gbdt)
    if "plco-2cohort" in want:
        from multisite import plco_2cohort
        X, y, g, nm = plco_2cohort()
        rows += run_dataset("plco-2cohort", X, y, g, nm, a.seeds, do_gbdt=not a.no_gbdt)

    res = analyse(rows)
    benefit_curve(rows)

    import csv
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(a.out.replace(".csv", "_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
