"""Merging across two databases and two eras: the closest proxy for inter-institutional.

Every multi-site result here carries the same caveat -- MIMIC's sites are care units
inside one hospital, so the heterogeneity is organisational rather than institutional, and
eICU was named as the missing piece. There is a cheaper proxy sitting on disk. MIMIC-III
(2001-2012, CareVue and MetaVision) and MIMIC-IV (2008-2022) are separate database
generations with **different item dictionaries**, different extraction code, overlapping
but non-identical lab panels and a decade of practice drift between their midpoints.
Merging a MIMIC-III unit with a MIMIC-IV unit crosses all of that at once.

The two are aligned by human-readable lab **label** rather than by item id, which is the
only thing they share, and the union is exactly the block-wise-missingness structure the
whole project studies -- a label present in one dictionary and absent from the other is a
variable one site simply does not have.

Nineteen sites, and the partner sets divide naturally into three kinds: within MIMIC-III,
within MIMIC-IV, and crossing. Matched on the target's own adequacy, the question is
whether crossing costs anything.

**Prediction, recorded before running.** Cross-database merging should be *worse* at
matched EPV -- era drift and dictionary mismatch add bias that within-database partners do
not carry. If it is not worse, the EPV rule is more robust than the single-hospital
caveat implies, and that is worth more than another within-hospital confirmation.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

SUBSETS_PER_TARGET = 24
MAX_PARTNERS = 4


def _norm(label):
    """Lab names differ in punctuation and case between the two dictionaries."""
    s = str(label).lower().strip()
    for ch in ",.()-/":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def build():
    os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
    import mimic_sites
    mimic_sites.SITE_KEY = "unit_x_system"
    X3, y3, g3, n3, c3 = mimic_sites.load(verbose=False)
    from mimic4_sites import load as load4
    X4, y4, g4, n4, c4 = load4(verbose=False)

    k3 = [_norm(c) for c in c3]
    k4 = [_norm(c) for c in c4]
    union = sorted(set(k3) | set(k4))
    ix = {v: i for i, v in enumerate(union)}
    shared = sorted(set(k3) & set(k4))

    def remap(X, keys):
        M = np.full((X.shape[0], len(union)), np.nan)
        for j, k in enumerate(keys):
            M[:, ix[k]] = X[:, j]
        return M

    from eicu_sites import load as loade
    Xe, ye, ge, ne, ce = loade(verbose=False)
    ke = [_norm(c) for c in ce]
    union = sorted(set(k3) | set(k4) | set(ke))
    ix = {v: i for i, v in enumerate(union)}
    shared = sorted(set(k3) & set(k4) & set(ke))

    M3, M4, ME = remap(X3, k3), remap(X4, k4), remap(Xe, ke)
    names = ([f"m3:{s}" for s in n3] + [f"m4:{s}" for s in n4]
             + [f"ei:{s}" for s in ne])
    X = np.vstack([M3, M4, ME])
    y = np.concatenate([y3, y4, ye])
    g = np.concatenate([g3, g4 + len(n3), ge + len(n3) + len(n4)])
    db = np.array([0] * len(y3) + [1] * len(y4) + [2] * len(ye))
    print(f"union of lab labels: {len(union)}  shared by both dictionaries: {len(shared)}"
          f"  (MIMIC-III {len(set(k3))}, MIMIC-IV {len(set(k4))})")
    print(f"sites: {len(names)}  records: {len(y)}  overall prevalence {y.mean():.3f}")
    return X, y, g, db, names, union, shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/cross_database.csv")
    a = ap.parse_args()

    X, y, g, db, names, union, shared = build()
    site_db = np.array([db[g == k][0] for k in range(len(names))])
    groups = [np.flatnonzero(g == k) for k in range(len(names))]

    split, loc = {}, {}
    for k in range(len(names)):
        for seed in range(a.seeds):
            idx = groups[k]
            try:
                i1, i2 = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                          random_state=seed)
            except ValueError:
                continue
            cols = site_schema(X[i1])
            if len(cols) < 2 or len(np.unique(y[i1])) < 2:
                continue
            m = fit_local(X[i1], y[i1], cols)
            if m is None:
                continue
            split[(k, seed)] = (X[i1], y[i1], cols, X[i2], y[i2])
            loc[(k, seed)] = m
        print(f"  local fit: {names[k]}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for k in range(len(names)):
        others = [j for j in range(len(names)) if j != k]
        subsets = set()
        for _ in range(SUBSETS_PER_TARGET):
            m = rng.integers(1, MAX_PARTNERS + 1)
            subsets.add(tuple(sorted(rng.choice(others, size=m, replace=False))))
        for S in sorted(subsets):
            for seed in range(a.seeds):
                keys = [(j, seed) for j in (k,) + S]
                if any(q not in split for q in keys):
                    continue
                Xte, yte = split[keys[0]][3], split[keys[0]][4]
                if len(np.unique(yte)) < 2:
                    continue
                ytr0, cols0 = split[keys[0]][1], split[keys[0]][2]
                a_loc = average_precision_score(yte, apply_raw(loc[keys[0]], Xte))
                a_mrg = average_precision_score(
                    yte, apply_std(merge([loc[q] for q in keys], "std", "n"), Xte))
                n_cross = int(sum(site_db[j] != site_db[k] for j in S))
                rows.append(dict(
                    target=names[k], target_db=int(site_db[k]), seed=seed,
                    n_partners=len(S), n_cross=n_cross,
                    frac_cross=n_cross / len(S),
                    epv_target=float(min(np.bincount(ytr0)) / max(len(cols0), 1)),
                    n_target=len(ytr0),
                    benefit=a_mrg - a_loc, helped=int(a_mrg > a_loc)))
        print(f"  decisions for {names[k]} ({len(rows)} total)", flush=True)

    e = np.array([r["epv_target"] for r in rows])
    b = np.array([r["benefit"] for r in rows])
    fc = np.array([r["frac_cross"] for r in rows])
    print(f"\n{len(rows)} decisions across {len({r['target'] for r in rows})} sites")
    print(f"Spearman(EPV, benefit) = {stats.spearmanr(e, b)[0]:+.3f} "
          f"p={stats.spearmanr(e, b)[1]:.1e}  -- does the rule survive the union schema?")

    print(f"\n{'target EPV':>12s} {'partners':>22s} {'helped':>8s} "
          f"{'mean benefit':>13s} {'n':>5s}")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 1e9)]:
        m0 = (e >= lo) & (e < hi)
        for lbl, m1 in [("all within-database", fc == 0.0),
                        ("mixed", (fc > 0) & (fc < 1)),
                        ("all cross-database", fc == 1.0)]:
            m = m0 & m1
            if m.sum() >= 8:
                print(f"{f'{lo}-{hi if hi < 1e9 else 999}':>12s} {lbl:>22s} "
                      f"{np.mean(b[m] > 0) * 100:7.1f}% {b[m].mean():+13.4f} "
                      f"{int(m.sum()):5d}")

    print("\npartial correlation of cross-database fraction with benefit, "
          "given target EPV:")
    from sklearn.linear_model import LinearRegression
    rv, rb = stats.rankdata(fc), stats.rankdata(b)
    re = stats.rankdata(e)[:, None]
    rv_r = rv - LinearRegression().fit(re, rv).predict(re)
    rb_r = rb - LinearRegression().fit(re, rb).predict(re)
    pr, pp = stats.pearsonr(rv_r, rb_r)
    print(f"  partial rho = {pr:+.3f}  p={pp:.1e}")

    print(f"\npartner-count curve at this scale (19 sites), by target EPV:")
    npn = np.array([r["n_partners"] for r in rows])
    for lo, hi in [(0, 5), (5, 10), (10, 1e9)]:
        m0 = (e >= lo) & (e < hi)
        cells = [f"{b[m0 & (npn == q)].mean():+.4f}" if (m0 & (npn == q)).sum() >= 8
                 else "   -" for q in (1, 2, 3, 4)]
        print(f"  EPV {lo:>2}-{str(hi if hi < 1e9 else 999):<4} " +
              "  ".join(f"{q}p {c}" for q, c in zip((1, 2, 3, 4), cells)))

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
