"""Why is one partner the worst choice? Measure the bias and variance, do not assert them.

Results 16 and 18 found the same non-obvious shape: merging with a single partner is
consistently worse than merging with two or three, in every band of target adequacy. The
explanation offered was arithmetic hand-waving -- "you halve the variance but import the
partner's bias at full strength" -- and hand-waving is exactly what this project has
learned to distrust. The quantities are estimable, so estimate them.

For a target site, treat the model fitted on **all of its training rows** as the best
available stand-in for that site's own population parameters. Draw repeated subsamples of
those rows, fit a local model on each, merge each with k randomly chosen partners, and
watch what happens to the resulting coefficient vectors:

    variance   how much the merged estimate moves from draw to draw
    bias^2     how far its average sits from the target's own parameters
    total      bias^2 + variance, which is what the error actually tracks

**Predictions, recorded before running.**
  1. Variance falls monotonically in k -- more sites averaged, less sampling noise.
  2. Bias-squared is *worst at k=1* and falls as k grows, because independent partner
     biases partially cancel when averaged; with one partner nothing cancels.
  3. Bias-squared does not fall to zero but to a floor: the target's own deviation from
     the consensus of the partner population, which no amount of averaging removes.
  4. Total error is therefore U-shaped, matching the measured benefit curve.

If instead bias-squared is flat in k, the observed curve has some other cause and the
explanation in Results 16 and 18 must be withdrawn.

Everything is computed on training rows only; the held-out AUPRC is reported alongside so
the decomposition can be lined up against the benefit it is supposed to explain.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

N_DRAWS = 24
SUBSAMPLE = 0.6
MAX_K = 4


def coef_on(model, cols):
    """Standardised-unit coefficients restricted to `cols`, zero where unmeasured.

    Standardised rather than raw units so that coordinates with very different scales
    contribute comparably to a squared distance -- otherwise one variable measured in
    thousands would dominate the decomposition entirely.
    """
    ix = {v: j for j, v in enumerate(model["cols"])}
    out = np.zeros(len(cols))
    for i, v in enumerate(cols):
        if v in ix:
            out[i] = (model["raw"][ix[v]] * model["sd"][ix[v]]
                      if "raw" in model else model["coef"][ix[v]])
    return out


def run_site(X, y, groups, k_target, others, seed, rng):
    idx = groups[k_target]
    try:
        tr, te = train_test_split(idx, test_size=0.3, stratify=y[idx], random_state=seed)
    except ValueError:
        return None
    cols_t = site_schema(X[tr])
    if len(cols_t) < 3 or len(np.unique(y[tr])) < 2:
        return None

    # the target's own parameters, as well as its training data can determine them
    ref = fit_local(X[tr], y[tr], cols_t)
    if ref is None:
        return None
    theta_ref = coef_on(ref, cols_t)

    # partners, each fitted on its own training split
    partners = {}
    for j in others:
        jdx = groups[j]
        try:
            jtr, _ = train_test_split(jdx, test_size=0.3, stratify=y[jdx],
                                      random_state=seed)
        except ValueError:
            continue
        cj = site_schema(X[jtr])
        if len(cj) < 3 or len(np.unique(y[jtr])) < 2:
            continue
        m = fit_local(X[jtr], y[jtr], cj)
        if m is not None:
            partners[j] = m
    if len(partners) < MAX_K:
        return None
    pkeys = list(partners)

    out = {}
    for k in range(0, MAX_K + 1):
        vecs, aucs = [], []
        for b in range(N_DRAWS):
            sub, _ = train_test_split(tr, train_size=SUBSAMPLE, stratify=y[tr],
                                      random_state=1000 * seed + b)
            cols_b = site_schema(X[sub])
            if len(cols_b) < 3 or len(np.unique(y[sub])) < 2:
                continue
            loc = fit_local(X[sub], y[sub], cols_b)
            if loc is None:
                continue
            if k == 0:
                mdl = loc
            else:
                chosen = rng.choice(pkeys, size=k, replace=False)
                mdl = merge([loc] + [partners[j] for j in chosen], "std", "n")
            vecs.append(coef_on(mdl, cols_t))
            aucs.append(average_precision_score(y[te], apply_std(mdl, X[te]))
                        if k > 0 else
                        average_precision_score(y[te], apply_std(mdl, X[te])))
        if len(vecs) < 8:
            continue
        V = np.vstack(vecs)
        mean = V.mean(0)
        out[k] = dict(bias2=float(np.sum((mean - theta_ref) ** 2)),
                      variance=float(np.mean(np.sum((V - mean) ** 2, axis=1))),
                      auprc=float(np.mean(aucs)), n_draws=len(vecs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mimic-11site,mimic4-9site,eicu")
    ap.add_argument("--matrix", default=None,
                    help="run the decomposition on a derived matrix from MIMIC_CACHE, so "
                         "the k=1 mechanism can be checked outside clinical data")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/bias_variance.csv")
    a = ap.parse_args()

    datasets = []
    want = set(a.datasets.split(",")) if not a.matrix else set()
    if a.matrix:
        from matrix_loader import load_matrix
        Xm, ym, gm, nmm = load_matrix(a.matrix)
        datasets.append((a.matrix, Xm, ym, gm, nmm))
    if "mimic-11site" in want:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        Xa, ya, ga, na, _ = mimic_sites.load(verbose=False)
        datasets.append(("mimic-11site", Xa, ya, ga, na))
    if "mimic4-9site" in want:
        from mimic4_sites import load as load4
        Xb, yb, gb, nb, _ = load4(verbose=False)
        datasets.append(("mimic4-9site", Xb, yb, gb, nb))
    if "eicu" in want:
        from eicu_sites import load as loade
        Xb, yb, gb, nb, _c = loade(verbose=False)
        datasets.append(("eicu", Xb, yb, gb, nb))

    rows = []
    for name, X, y, g, names in datasets:
        groups = [np.flatnonzero(g == k) for k in range(len(names))]
        rng = np.random.default_rng(0)
        for k_t in range(len(names)):
            others = [j for j in range(len(names)) if j != k_t]
            for seed in range(a.seeds):
                res = run_site(X, y, groups, k_t, others, seed, rng)
                if not res:
                    continue
                ytr = y[groups[k_t]]
                epv = float(min(np.bincount(ytr)) * 0.7
                            / max(len(site_schema(X[groups[k_t]])), 1))
                for k, v in res.items():
                    rows.append(dict(dataset=name, target=names[k_t], seed=seed,
                                     epv_target=epv, k_partners=k, **v))
            print(f"  {name}: {names[k_t]}", flush=True)

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["k_partners"]]["bias2"].append(r["bias2"])
        agg[r["k_partners"]]["variance"].append(r["variance"])
        agg[r["k_partners"]]["auprc"].append(r["auprc"])

    print(f"\n{len(rows)} (site, seed, k) cells from "
          f"{len({(r['dataset'], r['target']) for r in rows})} target sites, "
          f"{N_DRAWS} subsample draws each")
    print(f"\n{'partners':>9s} {'bias^2':>10s} {'variance':>10s} {'total':>10s} "
          f"{'AUPRC':>8s}")
    base = None
    for k in sorted(agg):
        b = float(np.mean(agg[k]["bias2"])); v = float(np.mean(agg[k]["variance"]))
        au = float(np.mean(agg[k]["auprc"]))
        if k == 0:
            base = (b, v)
        print(f"{k:9d} {b:10.4f} {v:10.4f} {b + v:10.4f} {au:8.4f}")
    if base:
        print(f"\nrelative to going solo (k=0):")
        for k in sorted(agg):
            if k == 0:
                continue
            b = float(np.mean(agg[k]["bias2"])); v = float(np.mean(agg[k]["variance"]))
            print(f"  k={k}: bias^2 x{b / max(base[0], 1e-12):5.2f}   "
                  f"variance x{v / max(base[1], 1e-12):5.2f}")

    print(f"\nsplit by target adequacy:")
    for lo, hi in [(0, 5), (5, 1e9)]:
        sel = [r for r in rows if lo <= r["epv_target"] < hi]
        if len(sel) < 20:
            continue
        print(f"  target EPV {lo}-{hi if hi < 1e9 else 999}:")
        by = defaultdict(lambda: defaultdict(list))
        for r in sel:
            by[r["k_partners"]]["bias2"].append(r["bias2"])
            by[r["k_partners"]]["variance"].append(r["variance"])
            by[r["k_partners"]]["auprc"].append(r["auprc"])
        for k in sorted(by):
            b = float(np.mean(by[k]["bias2"])); v = float(np.mean(by[k]["variance"]))
            print(f"    k={k}  bias^2 {b:8.4f}  variance {v:8.4f}  "
                  f"total {b + v:8.4f}  AUPRC {float(np.mean(by[k]['auprc'])):.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
