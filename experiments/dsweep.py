"""Does the merge decision follow EPV when EPV is moved by changing d alone?

The sharpest internal-validity test available. EPV is events over variables, so rebuilding
the same hospitals with fewer features multiplies every site's EPV without touching a
single patient, a single event, or the site structure. If the decision to merge tracks EPV
through that manipulation, EPV is doing causal work. If benefit stays put while EPV moves,
EPV was only ever a proxy for site size and the "per variable" half of the story is
decoration.

Feature sets are nested -- the top-d most-measured labs -- so smaller d is a strict subset
and nothing but dimensionality changes.

Prediction, recorded before running: at d=8 the median hospital has EPV 8.5 and 39 of 100
sit ABOVE the threshold, so the d=8 arm should show the sign change that eICU at d=42
could not (Result 22). At d=42, median EPV 1.6 and every site below, benefit should be
uniformly positive.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import site_schema  # noqa: E402
warnings.simplefilter("ignore")

SUBSETS, MAXP = 8, 4

def run(d, seeds, rng_seed=0):
    z = np.load(f"{os.environ['MIMIC_CACHE']}/eicu_d{d}.npz")
    X, y, g = z["X"], z["y"], z["g"]; names = [str(s) for s in z["names"]]
    groups = [np.flatnonzero(g == k) for k in range(len(names))]
    split, loc = {}, {}
    for k in range(len(names)):
        for s in range(seeds):
            idx = groups[k]
            try:
                i1, i2 = train_test_split(idx, test_size=.3, stratify=y[idx], random_state=s)
            except ValueError:
                continue
            c = site_schema(X[i1])
            if len(c) < 2 or len(np.unique(y[i1])) < 2:
                continue
            m = fit_local(X[i1], y[i1], c)
            if m is not None:
                split[(k, s)] = (X[i1], y[i1], c, X[i2], y[i2]); loc[(k, s)] = m
    rng = np.random.default_rng(rng_seed); rows = []
    for k in range(len(names)):
        others = [j for j in range(len(names)) if j != k]
        subs = set()
        for _ in range(SUBSETS):
            subs.add(tuple(sorted(rng.choice(others, size=rng.integers(1, MAXP + 1), replace=False))))
        for S in sorted(subs):
            for s in range(seeds):
                keys = [(j, s) for j in (k,) + S]
                if any(q not in split for q in keys):
                    continue
                Xte, yte = split[keys[0]][3], split[keys[0]][4]
                if len(np.unique(yte)) < 2:
                    continue
                ytr, c0 = split[keys[0]][1], split[keys[0]][2]
                al = average_precision_score(yte, apply_raw(loc[keys[0]], Xte))
                am = average_precision_score(yte, apply_std(merge([loc[q] for q in keys], "std", "n"), Xte))
                rows.append(dict(d=d, target=names[k], seed=s, n_partners=len(S),
                                 epv_target=float(min(np.bincount(ytr)) / max(len(c0), 1)),
                                 n_target=len(ytr), benefit=am - al, helped=int(am > al)))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/dsweep_eicu.csv"); a = ap.parse_args()
    rows = []
    for d in (8, 16, 24, 42):
        r = run(d, a.seeds); rows += r
        e = np.array([x["epv_target"] for x in r]); b = np.array([x["benefit"] for x in r])
        print(f"d={d:2d}: {len(r):5d} decisions, median EPV {np.median(e):5.2f}, "
              f"mean benefit {b.mean():+.4f}, helped {100*np.mean(b>0):4.1f}%, "
              f"rho(EPV,benefit) {stats.spearmanr(e,b)[0]:+.3f}", flush=True)
    e = np.array([x["epv_target"] for x in rows]); b = np.array([x["benefit"] for x in rows])
    print(f"\nPOOLED across d ({len(rows)} decisions): rho(EPV, benefit) = "
          f"{stats.spearmanr(e,b)[0]:+.3f}  p={stats.spearmanr(e,b)[1]:.1e}")
    print(f"\n{'EPV band':>10s} {'helped':>8s} {'mean benefit':>13s} {'n':>6s}  {'which d contribute':<22s}")
    import collections
    for lo, hi in [(0,1),(1,2),(2,5),(5,10),(10,20),(20,1e9)]:
        m = (e >= lo) & (e < hi)
        if m.sum() >= 15:
            ds = ",".join(str(x) for x in sorted({rows[i]["d"] for i in np.flatnonzero(m)}))
            print(f"{f'{lo}-{hi if hi<1e9 else 999}':>10s} {100*np.mean(b[m]>0):7.1f}% "
                  f"{b[m].mean():+13.4f} {int(m.sum()):6d}  {ds:<22s}")
    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
