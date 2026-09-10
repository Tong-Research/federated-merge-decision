"""If one partner imports bias without reducing variance, can a different weight fix it?

Result 19 measured why a single partner is the worst choice: it cuts variance by 0.7%
while nearly quintupling bias-squared, because only shared coordinates are averaged and
one partner shares few. That diagnosis has an obvious implication -- the trouble is not
that the partner is present, it is that sample-size weighting gives it too much say.

Under n-weighting a target of 700 stays merging with a partner of 3,000 keeps 19% of its
own coefficient. Nothing about that number came from a bias-variance argument; it is
simply how many records each side happened to have, which is the right weight only when
both are estimating the same quantity. They are not.

So: sweep the weight the target keeps on its own model, from 0 (pure merge) to 1 (stay
local), with n-weighting marked for reference, and find where the optimum actually sits at
each partner count.

**Predictions, recorded before running.**
  1. The optimal own-weight is **higher** than n-weighting gives, at every k.
  2. The gap is largest at k=1, since that is where the bias-per-unit-variance-reduction is
     worst -- and applying the optimum should remove the k=1 deficit.
  3. The optimal own-weight *falls* as k rises, because averaging more partners cancels
     their independent biases and makes the consensus worth more.

If the optimum sits at the n-weighted value throughout, sample-size weighting is already
doing the right thing and Result 19's diagnosis, though correct about the mechanism, has
no actionable consequence.

The optimum is reported two ways: chosen on the test fold (an oracle, upper bound only)
and chosen by inner cross-validation on the target's own training rows, which is what a
site could actually do.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SUBSETS_PER_TARGET = 8
MAX_PARTNERS = 4
INNER = 3


def blend(local, merged, w):
    """Convex combination in the standardised space the merge already uses.

    w = 1 keeps the local model exactly; w = 0 is the merge. Coordinates the local model
    does not carry stay at the merged value, since the target has no opinion about them.
    """
    ix = {v: j for j, v in enumerate(merged["cols"])}
    coef = merged["coef"].copy()
    mu, sd = merged["mu"].copy(), merged["sd"].copy()
    lix = {v: j for j, v in enumerate(local["cols"])}
    lcoef = local["raw"] * local["sd"]
    for v, j in lix.items():
        if v in ix:
            coef[ix[v]] = w * lcoef[j] + (1 - w) * coef[ix[v]]
            mu[ix[v]] = w * local["mu"][j] + (1 - w) * mu[ix[v]]
            sd[ix[v]] = w * local["sd"][j] + (1 - w) * sd[ix[v]]
    b_local = local["raw_b"] + float(local["raw"] @ local["mu"])
    return dict(cols=merged["cols"], coef=coef, mu=mu, sd=sd,
                intercept=w * b_local + (1 - w) * merged["intercept"])


def pick_weight_inner(Xtr, ytr, cols, partners, seed):
    """Choose the own-weight by inner CV on the target's own training rows only."""
    try:
        folds = list(StratifiedKFold(INNER, shuffle=True,
                                     random_state=seed).split(Xtr, ytr))
    except ValueError:
        return None
    scores = defaultdict(list)
    for itr, iva in folds:
        if len(np.unique(ytr[itr])) < 2 or len(np.unique(ytr[iva])) < 2:
            continue
        lo = fit_local(Xtr[itr], ytr[itr], cols)
        if lo is None:
            continue
        mg = merge([lo] + partners, "std", "n")
        for w in WEIGHTS:
            scores[w].append(average_precision_score(
                ytr[iva], apply_std(blend(lo, mg, w), Xtr[iva])))
    if not scores:
        return None
    return max(scores, key=lambda w: np.mean(scores[w]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mimic-11site,mimic4-9site,eicu")
    ap.add_argument("--matrix", default=None,
                    help="run on a derived matrix from MIMIC_CACHE instead. The weighting "
                         "rule w = min(1, EPV/10) was fitted entirely on clinical sites; "
                         "this is how it gets tested out of domain.")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/merge_weight.csv")
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
        split, loc = {}, {}
        for k in range(len(names)):
            for seed in range(a.seeds):
                idx = groups[k]
                try:
                    i1, i2 = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                              random_state=seed)
                except ValueError:
                    continue
                c = site_schema(X[i1])
                if len(c) < 3 or len(np.unique(y[i1])) < 2:
                    continue
                m = fit_local(X[i1], y[i1], c)
                if m is not None:
                    split[(k, seed)] = (X[i1], y[i1], c, X[i2], y[i2])
                    loc[(k, seed)] = m
        rng = np.random.default_rng(0)
        for k in range(len(names)):
            others = [j for j in range(len(names)) if j != k]
            subsets = set()
            for _ in range(SUBSETS_PER_TARGET):
                m = rng.integers(1, min(MAX_PARTNERS, len(others)) + 1)
                subsets.add(tuple(sorted(rng.choice(others, size=m, replace=False))))
            for S in sorted(subsets):
                for seed in range(a.seeds):
                    keys = [(j, seed) for j in (k,) + S]
                    if any(q not in split for q in keys):
                        continue
                    Xtr, ytr, cols0, Xte, yte = split[keys[0]]
                    if len(np.unique(yte)) < 2:
                        continue
                    partners = [loc[q] for q in keys[1:]]
                    lo = loc[keys[0]]
                    mg = merge([lo] + partners, "std", "n")
                    curve = {w: average_precision_score(
                        yte, apply_std(blend(lo, mg, w), Xte)) for w in WEIGHTS}
                    n_t = len(ytr)
                    n_p = sum(len(split[q][1]) for q in keys[1:])
                    w_n = n_t / (n_t + n_p)          # what n-weighting implies
                    w_cv = pick_weight_inner(Xtr, ytr, cols0, partners, seed)
                    rows.append(dict(
                        dataset=name, target=names[k], seed=seed, k_partners=len(S),
                        epv_target=float(min(np.bincount(ytr)) / max(len(cols0), 1)),
                        w_nweight=float(w_n),
                        w_oracle=float(max(curve, key=curve.get)),
                        w_cv=float(w_cv) if w_cv is not None else np.nan,
                        auprc_merge=curve[0.0], auprc_local=curve[1.0],
                        auprc_oracle=max(curve.values()),
                        auprc_cv=(curve[w_cv] if w_cv is not None else np.nan),
                        **{f"auprc_w{int(w * 10)}": curve[w] for w in WEIGHTS}))
            print(f"  {name}: {names[k]}", flush=True)

    print(f"\n{len(rows)} decisions across "
          f"{len({(r['dataset'], r['target']) for r in rows})} sites")
    print(f"\nmean AUPRC by own-weight (0 = pure merge, 1 = stay local):")
    hdr = "".join(f"{w:>8.1f}" for w in WEIGHTS)
    print(f"{'partners':>9s}{hdr}")
    for k in sorted({r["k_partners"] for r in rows}):
        sel = [r for r in rows if r["k_partners"] == k]
        cells = "".join(f"{np.mean([r[f'auprc_w{int(w * 10)}'] for r in sel]):8.4f}"
                        for w in WEIGHTS)
        print(f"{k:9d}{cells}")

    print(f"\n{'partners':>9s} {'n-weight':>9s} {'oracle w':>9s} {'CV w':>7s} "
          f"{'merge':>8s} {'local':>8s} {'CV':>8s} {'oracle':>8s}")
    for k in sorted({r["k_partners"] for r in rows}):
        sel = [r for r in rows if r["k_partners"] == k]
        print(f"{k:9d} {np.mean([r['w_nweight'] for r in sel]):9.2f} "
              f"{np.mean([r['w_oracle'] for r in sel]):9.2f} "
              f"{np.nanmean([r['w_cv'] for r in sel]):7.2f} "
              f"{np.mean([r['auprc_merge'] for r in sel]):8.4f} "
              f"{np.mean([r['auprc_local'] for r in sel]):8.4f} "
              f"{np.nanmean([r['auprc_cv'] for r in sel]):8.4f} "
              f"{np.mean([r['auprc_oracle'] for r in sel]):8.4f}")

    print("\ndoes CV weighting remove the one-partner deficit? "
          "(benefit over staying local)")
    for k in sorted({r["k_partners"] for r in rows}):
        sel = [r for r in rows if r["k_partners"] == k]
        dm = np.array([r["auprc_merge"] - r["auprc_local"] for r in sel])
        dc = np.array([r["auprc_cv"] - r["auprc_local"] for r in sel])
        pm = stats.wilcoxon(dm).pvalue if not np.allclose(dm, 0) else 1.0
        pc = (stats.wilcoxon(dc[np.isfinite(dc)]).pvalue
              if np.isfinite(dc).any() and not np.allclose(dc[np.isfinite(dc)], 0)
              else 1.0)
        print(f"  k={k}:  n-weighted merge {dm.mean():+.4f} (p={pm:.1e})   "
              f"CV-weighted {np.nanmean(dc):+.4f} (p={pc:.1e})")

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
