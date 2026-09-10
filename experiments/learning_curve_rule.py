"""A self-calibrating replacement for EPV: measure starvation instead of counting it.

EPV works (711 decisions, rho -0.475) but it has a practical defect that Result 10 exposed:
**its threshold depends on the model class.** The cut is 7.5 for a logistic regression,
17.7 for a gradient-boosted ensemble and 32.7 for pooled trees. A site deciding whether to
join a federation would therefore need to know which threshold applies to the model it
intends to fit -- and for anything other than a linear model, "number of estimated
parameters" is not even well defined.

A learning curve measures the same thing directly and needs no such knowledge. A site
fits its own model on 25%, 50%, 75% and 100% of its own training rows, by internal
cross-validation, and looks at whether performance is still climbing. A curve that has
flattened says *more data will not help me*; a curve still rising steeply says *it will*.
That is precisely the quantity the merge decision turns on, it is computed from the site's
own data alone, and it is expressed in the units of whatever model is actually being used.

**Prediction, recorded before running.** The learning-curve slope should predict merge
benefit about as well as EPV for logistic regression, and -- the part that matters -- it
should do so with **one threshold that works for both model classes**, where EPV needs
two. If instead the slope also needs a per-class threshold, it is no improvement and EPV
stays.

The comparison is deliberately fair: both statistics are computed from the target site's
training rows only, and both are scored by leave-one-target-site-out cross-validation.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402
from site_rules_ext import gbdt, gbdt_apply  # noqa: E402

warnings.simplefilter("ignore")

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
INNER_FOLDS = 3
SUBSETS_PER_TARGET = 8
MAX_PARTNERS = 4


def learning_curve(X, y, cols, kind, seed):
    """AUPRC by training-set fraction, from the site's own rows only.

    Each fraction is evaluated on held-out inner folds, so the curve measures
    generalisation rather than fit. Returns the curve and two summaries: the final slope
    (how much the last quarter of the data bought) and the total climb from half data.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2 * INNER_FOLDS:
        return None
    try:
        folds = list(StratifiedKFold(INNER_FOLDS, shuffle=True,
                                     random_state=seed).split(X, y))
    except ValueError:
        return None
    curve = []
    for f in FRACTIONS:
        sc = []
        for itr, iva in folds:
            if f < 1.0:
                try:
                    itr, _ = train_test_split(itr, train_size=f, stratify=y[itr],
                                              random_state=seed)
                except ValueError:
                    continue
            if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
                continue
            if kind == "lr":
                m = fit_local(X[itr], y[itr], cols)
                if m is None:
                    continue
                p = apply_raw(m, X[iva])
            else:
                # 15% of this subsample must still contain both classes for the
                # internal early-stopping split to exist.
                es = min(np.bincount(y[itr])) * 0.15 >= 2
                m = gbdt(X[itr], y[itr], cols, seed, early_stopping=es)
                if m is None:
                    continue
                p = gbdt_apply(m, X[iva])
            sc.append(average_precision_score(y[iva], p))
        if not sc:
            return None
        curve.append(float(np.mean(sc)))
    c = np.array(curve)
    # normalise by the achieved level: a curve climbing 0.01 from a base of 0.05 is
    # starved, the same climb from a base of 0.9 is not.
    denom = max(c[-1], 1e-6)
    return dict(curve=c, slope_last=float((c[-1] - c[-2]) / denom),
                climb_half=float((c[-1] - c[1]) / denom),
                level=float(c[-1]))


def run(name, X, y, g, site_names, seeds=2, rng_seed=0, kinds=("lr", "gb")):
    rng = np.random.default_rng(rng_seed)
    groups = [np.flatnonzero(g == k) for k in range(len(site_names))]
    split, loc, lc = {}, {}, {}
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
            loc[("lr", k, seed)] = fit_local(X[i1], y[i1], cols)
            if "gb" in kinds:
                loc[("gb", k, seed)] = gbdt(X[i1], y[i1], cols, seed)
            for kind in kinds:
                lc[(kind, k, seed)] = learning_curve(X[i1], y[i1], cols, kind, seed)
        print(f"  {name}: curves for {site_names[k]}", flush=True)

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
                ytr0, cols0 = parts[0][1], parts[0][2]
                base = dict(dataset=name, target=site_names[k], seed=seed,
                            n_partners=len(S), n_target=len(ytr0),
                            epv_target=float(min(np.bincount(ytr0)) / max(len(cols0), 1)))
                for kind in kinds:
                    ms = [loc.get((kind, j, seed)) for j in (k,) + S]
                    curve = lc.get((kind, k, seed))
                    if any(m is None for m in ms) or curve is None:
                        continue
                    if kind == "lr":
                        a_loc = average_precision_score(yte, apply_raw(ms[0], Xte))
                        a_cmb = average_precision_score(
                            yte, apply_std(merge(ms, "std", "n"), Xte))
                    else:
                        a_loc = average_precision_score(yte, gbdt_apply(ms[0], Xte))
                        a_cmb = average_precision_score(
                            yte, np.mean([gbdt_apply(m, Xte) for m in ms], 0))
                    rows.append(dict(base, kind=kind, benefit=a_cmb - a_loc,
                                     helped=int(a_cmb > a_loc),
                                     slope_last=curve["slope_last"],
                                     climb_half=curve["climb_half"],
                                     level=curve["level"]))
    return rows


def loso(rows, feature, kinds_together):
    """Leave-one-target-site-out threshold accuracy. `kinds_together` fits ONE threshold
    across both model classes -- the property being tested."""
    sel = rows if kinds_together else None
    y = np.array([r["helped"] for r in sel])
    v = np.array([r[feature] for r in sel], dtype=float)
    keys = sorted({(r["dataset"], r["target"], r["kind"]) if not kinds_together
                   else (r["dataset"], r["target"]) for r in sel})
    grp = np.array([keys.index((r["dataset"], r["target"])) for r in sel])
    ok = np.isfinite(v)
    correct = []
    for gi in np.unique(grp):
        tr, te = ok & (grp != gi), ok & (grp == gi)
        if te.sum() == 0 or tr.sum() < 10 or len(np.unique(y[tr])) < 2:
            continue
        best = (None, -1.0)
        for t in np.quantile(v[tr], np.linspace(0.05, 0.95, 25)):
            for sign in (+1, -1):
                pred = (v[tr] > t) if sign > 0 else (v[tr] <= t)
                acc = (pred == y[tr]).mean()
                if acc > best[1]:
                    best = ((t, sign), acc)
        (t, sign), _ = best
        pred = (v[te] > t) if sign > 0 else (v[te] <= t)
        correct.extend((pred == y[te]).tolist())
    return float(np.mean(correct)) if correct else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="heart-4site,mimic-11site,mimic4-9site")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/learning_curve_rule.csv")
    a = ap.parse_args()
    want = set(a.datasets.split(","))
    rows = []
    if "heart-4site" in want:
        from multisite import heart_4site
        X, y, g, nm = heart_4site()
        rows += run("heart-4site", X, y, g, nm, a.seeds)
    if "mimic-11site" in want:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        X, y, g, nm, _ = mimic_sites.load(verbose=False)
        rows += run("mimic-11site", X, y, g, nm, a.seeds)
    if "mimic4-9site" in want:
        from mimic4_sites import load as load4
        X, y, g, nm, _ = load4(verbose=False)
        rows += run("mimic4-9site", X, y, g, nm, a.seeds)

    print(f"\n{len(rows)} decisions "
          f"({sum(r['kind'] == 'lr' for r in rows)} logistic, "
          f"{sum(r['kind'] == 'gb' for r in rows)} boosted)")

    print(f"\n{'statistic':14s} {'model class':>12s} {'rho':>8s} {'p':>9s} "
          f"{'own-class cut':>14s}")
    for feat in ("epv_target", "slope_last", "climb_half"):
        for kind in ("lr", "gb"):
            sub = [r for r in rows if r["kind"] == kind]
            v = np.array([r[feat] for r in sub], float)
            b = np.array([r["benefit"] for r in sub], float)
            ok = np.isfinite(v) & np.isfinite(b)
            rho, p = stats.spearmanr(v[ok], b[ok])
            h = b[ok] > 0
            best = max(((float(t), float(((v[ok] > t) == h).mean()))
                        for t in np.quantile(v[ok], np.linspace(0.05, 0.95, 25))),
                       key=lambda z: z[1])
            print(f"{feat:14s} {kind:>12s} {rho:+8.3f} {p:9.1e} "
                  f"{best[0]:9.3f}/{best[1]:.2f}")

    print("\nTHE TEST: one threshold shared across BOTH model classes, "
          "leave-one-target-site-out")
    base = max(np.mean([r["helped"] for r in rows]),
               1 - np.mean([r["helped"] for r in rows]))
    for feat in ("epv_target", "slope_last", "climb_half"):
        acc = loso(rows, feat, kinds_together=True)
        print(f"  {feat:14s} shared-threshold LOSO accuracy {acc:.3f}   "
              f"(majority {base:.3f})")

    import csv
    keys = [k for k in rows[0] if k != "curve"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
