"""Is the EPV result an artefact of an untuned local baseline?

Rule 1 of this project's hunt log is *tune the comparator, always* -- an untuned baseline
once turned a +0.006 gap into +0.056. Every merge-versus-local comparison so far has used
`fit_local`, which is `LogisticRegression(max_iter=2000)` at sklearn's **default C=1.0**.
That is an untuned baseline, and it is untuned in exactly the place that matters: a site
with few events per variable is precisely the site whose default-penalty model overfits
most, so tuning the penalty should help low-EPV sites more than high-EPV sites -- the same
gradient the merge benefit follows.

If that is what has been measured, the headline is an artefact.

There is also a second option a starved site has that we never gave it. EPV is events over
*variables*, so a site can raise its own EPV by fitting a smaller model instead of by
finding partners. "Should I merge, or should I just use fewer variables?" is the question
a reviewer asks first, and it has never been tested here.

Four local baselines, each an honest attempt at the site's best solo model, all selected by
inner cross-validation on the site's own training rows:

  default      C = 1.0, what every result so far used
  tuned        C chosen from a grid by inner CV
  selected     univariate top-k features, k chosen by inner CV, C tuned with it
  best-solo    whichever of the above wins the site's inner CV -- the strongest honest
               local competitor, and the one the merge should be judged against

Prediction, recorded before running: tuning will lift the low-EPV local models most, so
the merge benefit at low EPV will shrink. The question is whether it survives at all.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

CS = np.logspace(-4, 2, 10)
INNER = 3
SUBSETS_PER_TARGET = 8
MAX_PARTNERS = 4


def _pipe(C, k=None):
    steps = [("imp", SimpleImputer(strategy="mean", keep_empty_features=True)),
             ("sc", StandardScaler())]
    if k is not None:
        steps.append(("sel", SelectKBest(f_classif, k=k)))
    steps.append(("lr", LogisticRegression(C=C, max_iter=3000)))
    return Pipeline(steps)


def _cv_score(est, X, y, seed):
    try:
        folds = list(StratifiedKFold(INNER, shuffle=True, random_state=seed).split(X, y))
    except ValueError:
        return -1.0
    sc = []
    for itr, iva in folds:
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
            continue
        try:
            est.fit(X[itr], y[itr])
            sc.append(average_precision_score(y[iva], est.predict_proba(X[iva])[:, 1]))
        except Exception:                                        # noqa: BLE001
            return -1.0
    return float(np.mean(sc)) if sc else -1.0


def local_variants(Xtr, ytr, cols, seed):
    """Fit the four solo baselines; each returns a scorer over the full feature matrix."""
    A = np.asarray(Xtr[:, cols], float)
    out = {}

    default = _pipe(1.0).fit(A, ytr)
    out["default"] = (lambda M, e=default: e.predict_proba(np.asarray(M[:, cols],
                                                                     float))[:, 1])
    s_def = _cv_score(_pipe(1.0), A, ytr, seed)

    best_c, s_tuned = 1.0, -1.0
    for C in CS:
        s = _cv_score(_pipe(C), A, ytr, seed)
        if s > s_tuned:
            best_c, s_tuned = C, s
    tuned = _pipe(best_c).fit(A, ytr)
    out["tuned"] = (lambda M, e=tuned: e.predict_proba(np.asarray(M[:, cols],
                                                                  float))[:, 1])

    ks = sorted({max(2, int(round(len(cols) * f))) for f in (0.25, 0.5, 0.75)}
                | {len(cols)})
    best_k, best_kc, s_sel = len(cols), best_c, -1.0
    for k in ks:
        for C in CS[::2]:
            s = _cv_score(_pipe(C, k), A, ytr, seed)
            if s > s_sel:
                best_k, best_kc, s_sel = k, C, s
    sel = _pipe(best_kc, best_k).fit(A, ytr)
    out["selected"] = (lambda M, e=sel: e.predict_proba(np.asarray(M[:, cols],
                                                                   float))[:, 1])

    # best-solo picks by inner CV only -- the test fold is never consulted
    order = sorted((("default", s_def), ("tuned", s_tuned), ("selected", s_sel)),
                   key=lambda z: -z[1])
    out["best_solo"] = out[order[0][0]]
    return out, dict(chosen=order[0][0], C=float(best_c), k=int(best_k),
                     n_cols=len(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="heart-4site,mimic-11site,mimic4-9site,eicu")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/local_tuning_check.csv")
    a = ap.parse_args()

    datasets = []
    want = set(a.datasets.split(","))
    if "heart-4site" in want:
        from multisite import heart_4site
        datasets.append(("heart-4site", *heart_4site()))
    if "mimic-11site" in want:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        X, y, g, nm, _ = mimic_sites.load(verbose=False)
        datasets.append(("mimic-11site", X, y, g, nm))
    if "mimic4-9site" in want:
        from mimic4_sites import load as load4
        X, y, g, nm, _ = load4(verbose=False)
        datasets.append(("mimic4-9site", X, y, g, nm))
    if "eicu" in want:
        from eicu_sites import load as loade
        X, y, g, nm, _c = loade(verbose=False)
        datasets.append(("eicu", X, y, g, nm))

    rows = []
    for name, X, y, g, site_names in datasets:
        rng = np.random.default_rng(0)
        groups = [np.flatnonzero(g == k) for k in range(len(site_names))]
        split, variants, meta, loc0 = {}, {}, {}, {}
        for k in range(len(site_names)):
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
                split[(k, seed)] = (X[i1], y[i1], cols, X[i2], y[i2])
                loc0[(k, seed)] = fit_local(X[i1], y[i1], cols)
                variants[(k, seed)], meta[(k, seed)] = local_variants(
                    X[i1], y[i1], cols, seed)
            print(f"  {name}: {site_names[k]} tuned", flush=True)

        for k in range(len(site_names)):
            others = [j for j in range(len(site_names)) if j != k]
            if not others:
                continue
            subsets = set()
            for _ in range(SUBSETS_PER_TARGET):
                m = rng.integers(1, min(MAX_PARTNERS, len(others)) + 1)
                subsets.add(tuple(sorted(rng.choice(others, size=m, replace=False))))
            for S in sorted(subsets):
                for seed in range(a.seeds):
                    keys = [(j, seed) for j in (k,) + S]
                    if any(q not in split for q in keys):
                        continue
                    if any(loc0[q] is None for q in keys):
                        continue
                    Xte, yte = split[keys[0]][3], split[keys[0]][4]
                    if len(np.unique(yte)) < 2:
                        continue
                    ytr0, cols0 = split[keys[0]][1], split[keys[0]][2]
                    a_mrg = average_precision_score(
                        yte, apply_std(merge([loc0[q] for q in keys], "std", "n"), Xte))
                    r = dict(dataset=name, target=site_names[k], seed=seed,
                             n_partners=len(S), n_target=len(ytr0),
                             epv_target=float(min(np.bincount(ytr0)) / max(len(cols0), 1)),
                             merge=a_mrg, **{f"chose_{kk}": vv for kk, vv
                                             in meta[keys[0]].items()
                                             if kk == "chosen"})
                    for vname, fn in variants[keys[0]].items():
                        r[f"local_{vname}"] = average_precision_score(yte, fn(Xte))
                        r[f"gain_vs_{vname}"] = a_mrg - r[f"local_{vname}"]
                    rows.append(r)

    e = np.array([r["epv_target"] for r in rows])
    print(f"\n{len(rows)} decisions. Merge benefit against each local baseline, by EPV:")
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    hdr = "".join(f"{f'{lo}-{hi if hi < 1e9 else 999}':>13s}" for lo, hi in bands)
    print(f"{'merge vs':22s}{hdr}{'  rho':>9s}{'  p':>9s}")
    for vname in ("default", "tuned", "selected", "best_solo"):
        v = np.array([r[f"gain_vs_{vname}"] for r in rows], float)
        cells = []
        for lo, hi in bands:
            m = (e >= lo) & (e < hi)
            cells.append(f"{v[m].mean():+.4f}" if m.sum() >= 5 else "    -")
        rho, p = stats.spearmanr(e, v)
        print(f"{'local_' + vname:22s}" + "".join(f"{c:>13s}" for c in cells)
              + f"{rho:+9.3f}{p:9.1e}")

    print(f"\nhow often each solo strategy won the site's own inner CV:")
    from collections import Counter
    c = Counter(r.get("chose_chosen") for r in rows)
    for k, v in c.most_common():
        print(f"  {str(k):10s} {v / len(rows) * 100:5.1f}%")

    print(f"\nlocal AUPRC by baseline (the tuning gain itself), by EPV band:")
    print(f"{'baseline':22s}{hdr}")
    for vname in ("default", "tuned", "selected", "best_solo"):
        v = np.array([r[f"local_{vname}"] for r in rows], float)
        cells = []
        for lo, hi in bands:
            m = (e >= lo) & (e < hi)
            cells.append(f"{v[m].mean():.4f}" if m.sum() >= 5 else "    -")
        print(f"{'local_' + vname:22s}" + "".join(f"{c:>13s}" for c in cells))

    import csv
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
