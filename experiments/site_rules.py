"""Fit the pre-merge decision rule on real sites instead of importing it from a simulator.

The rule "should this site merge with those sites?" has so far been calibrated on
generated data and tested on three real datasets, which is three observations. That was
the binding constraint, and it looked like it needed eICU. It does not: a *pair* of
(target site, partner set) is one observation of the decision, and the sites already on
disk supply hundreds of them.

  heart-4site       4 hospitals
  mimic-11site      MIMIC-III care unit crossed with EHR system (CareVue 2001-2008 vs
                    MetaVision 2008-2012) -- eleven sites, different eras, different
                    ordering cultures, different case mix
  plco-2cohort      2 cohorts

For each target site k, each sampled partner subset S of the other sites, and each seed,
the question is asked exactly as a data owner would ask it: *I have my own records; would
merging with these partners beat keeping my own model?* Everything the rule is allowed to
use is computable from aggregates -- sizes, schema overlap, Cochran's Q, I-squared -- and
never from the partners' raw data.

Rules are then scored by **leave-one-target-site-out** cross-validation, so a threshold is
never evaluated on the site it was fitted on. That is the part the three-dataset test could
not do, and it is what separates "this statistic correlates" from "this rule would have
helped you decide".
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed  # noqa: E402
from merge_rules import apply_raw, apply_std, fit_local, merge  # noqa: E402
from multisite import cochran_q, metrics as _metrics, site_schema  # noqa: E402

warnings.simplefilter("ignore")

# Env-overridable so a larger-federation sweep needs no code edit. Every result so far
# caps partners at 4, which is small for a federation and is itself an untested choice.
MAX_PARTNERS = int(os.environ.get("MAX_PARTNERS", "4"))
SUBSETS_PER_TARGET = int(os.environ.get("SUBSETS_PER_TARGET", "12"))
N_JOBS = int(os.environ.get("N_JOBS", "-1"))



def _pack_groups(groups):
    """Flatten per-site indices into (concatenated, offsets), both plain arrays.

    joblib memmaps large numpy arrays once and shares them across workers, but a *list* of
    arrays is pickled into every dispatched batch. Passing `groups` directly sent 24MB of
    index arrays per batch and the run stalled in serialisation with zero workers alive --
    measured at 4.5 minutes before a single task started. Two flat arrays are memmapped;
    the worker rebuilds the list with slicing, which is views and costs nothing.
    """
    off = np.zeros(len(groups) + 1, dtype=np.int64)
    for i, ix in enumerate(groups):
        off[i + 1] = off[i] + len(ix)
    flat = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    return flat.astype(np.int64), off


def _one_packed(X, y, flat, off, k, S, seed, names):
    groups = [flat[off[j]:off[j + 1]] for j in range(len(off) - 1)]
    return _one(X, y, groups, k, S, seed, names)


def observations(name, X, y, g, site_names, seeds=3, rng_seed=0):
    """One row per (target site, partner subset, seed): did merging beat staying local?

    Two phases. Drawing the subsets is cheap and stays sequential, because `rng` is a single
    stream advanced across targets and parallelising that loop would change which subsets are
    drawn -- silently invalidating every comparison against results already in results/.
    Evaluating them is where the time goes and every task is independent, so that part runs
    across all cores. Verified to reproduce the sequential output row for row.

    Measured before this change: 98.9% CPU on a ten-core machine, i.e. one core.
    """
    rng = np.random.default_rng(rng_seed)
    groups = [np.flatnonzero(g == k) for k in range(len(site_names))]
    tasks = []
    for k in range(len(site_names)):
        others = [j for j in range(len(site_names)) if j != k]
        if not others:
            continue
        subsets = []
        for _ in range(SUBSETS_PER_TARGET):
            m = rng.integers(1, min(MAX_PARTNERS, len(others)) + 1)
            subsets.append(tuple(sorted(rng.choice(others, size=m, replace=False))))
        for S in sorted(set(subsets)):
            for seed in range(seeds):
                tasks.append((k, S, seed))
    if not tasks:
        return []
    flat, off = _pack_groups(groups)
    out = Parallel(n_jobs=N_JOBS, inner_max_num_threads=1, batch_size=64)(
        delayed(_one_packed)(X, y, flat, off, k, S, seed, site_names)
        for k, S, seed in tasks)
    return [dict(dataset=name, target=site_names[k], **r)
            for (k, _, _), r in zip(tasks, out) if r]


def _one(X, y, groups, k, S, seed, site_names):
    parts = []
    for j in (k,) + tuple(S):
        idx = groups[j]
        try:
            i1, i2 = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                      random_state=seed)
        except ValueError:
            return None
        cols = site_schema(X[i1])
        if len(cols) < 2 or len(np.unique(y[i1])) < 2:
            return None
        parts.append((X[i1], y[i1], cols, X[i2], y[i2]))
    locs = [fit_local(a, b, c) for a, b, c, _, _ in parts]
    if any(m is None for m in locs):
        return None
    Xte, yte = parts[0][3], parts[0][4]
    if len(np.unique(yte)) < 2:
        return None

    merged = merge(locs, "std", "n")
    a_local = average_precision_score(yte, apply_raw(locs[0], Xte))
    a_merge = average_precision_score(yte, apply_std(merged, Xte))

    q, df, i2 = cochran_q(locs)
    sizes = [len(p[1]) for p in parts]
    schemas = [set(p[2]) for p in parts]
    inter = set.intersection(*schemas)
    union = set.union(*schemas)
    return dict(
        n_partners=len(S), n_target=sizes[0], n_min=int(min(sizes)),
        n_partner_total=int(sum(sizes[1:])), d_target=len(schemas[0]),
        ratio_partner_to_target=float(sum(sizes[1:]) / max(sizes[0], 1)),
        events_target=int(y[groups[k]].sum()),
        epv_target=float(min(np.bincount(parts[0][1])) / max(len(schemas[0]), 1)),
        schema_jaccard=float(len(inter) / max(len(union), 1)),
        q_median=float(np.median(q)) if len(q) else np.nan,
        i2_median=float(np.median(i2)) if len(i2) else np.nan,
        benefit=float(a_merge - a_local), helped=int(a_merge > a_local),
        **{f"loc_{k}": v for k, v in _metrics(yte, apply_raw(locs[0], Xte)).items()},
        **{f"mrg_{k}": v for k, v in _metrics(yte, apply_std(merged, Xte)).items()})


FEATURES = ["epv_target", "n_target", "n_min", "n_partner_total",
            "ratio_partner_to_target", "n_partners", "schema_jaccard",
            "q_median", "i2_median"]


def evaluate_rules(rows):
    """Leave-one-target-site-out: fit a threshold on other sites, test on the held-out one."""
    keys = sorted({(r["dataset"], r["target"]) for r in rows})
    y = np.array([r["helped"] for r in rows])
    grp = np.array([keys.index((r["dataset"], r["target"])) for r in rows])
    print(f"\n{len(rows)} real merge decisions across {len(keys)} target sites; "
          f"merging helped in {y.mean() * 100:.0f}%")

    print(f"\n{'feature':26s} {'Spearman rho':>13s} {'p':>9s} "
          f"{'LOSO acc':>9s} {'base':>6s} {'direction':>10s}")
    base = max(y.mean(), 1 - y.mean())
    out = []
    for f in FEATURES:
        v = np.array([r[f] for r in rows], dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        rho, p = stats.spearmanr(v[ok], np.array([r["benefit"] for r in rows])[ok])
        # leave-one-target-site-out threshold rule, both directions
        correct = []
        for gi in np.unique(grp):
            tr = ok & (grp != gi)
            te = ok & (grp == gi)
            if te.sum() == 0 or tr.sum() < 10 or len(np.unique(y[tr])) < 2:
                continue
            best = (None, -1.0)
            for t in np.quantile(v[tr], np.linspace(0.05, 0.95, 19)):
                for sign in (+1, -1):
                    pred = (v[tr] < t) if sign > 0 else (v[tr] >= t)
                    acc = (pred == y[tr]).mean()
                    if acc > best[1]:
                        best = ((t, sign), acc)
            (t, sign), _ = best
            pred = (v[te] < t) if sign > 0 else (v[te] >= t)
            correct.extend((pred == y[te]).tolist())
        acc = float(np.mean(correct)) if correct else np.nan
        arrow = "lower→merge" if rho < 0 else "higher→merge"
        print(f"{f:26s} {rho:+13.3f} {p:9.1e} {acc:9.3f} {base:6.3f} {arrow:>10s}")
        out.append(dict(feature=f, rho=float(rho), p=float(p), loso_acc=acc,
                        majority=float(base)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="heart-4site,mimic-11site,mimic4-9site,eicu,plco-2cohort")
    ap.add_argument("--matrix", default=None,
                    help="fit the rule on a derived matrix from MIMIC_CACHE instead of the "
                         "named datasets")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/site_rules.csv")
    a = ap.parse_args()
    want = set(a.datasets.split(",")) if not a.matrix else set()
    rows = []
    if a.matrix:
        from matrix_loader import load_matrix
        X, y, g, nm = load_matrix(a.matrix)
        rows += observations(a.matrix, X, y, g, nm, a.seeds)
        print(f"{a.matrix}: {len(rows)} decisions", flush=True)
    if "heart-4site" in want:
        from multisite import heart_4site
        X, y, g, nm = heart_4site()
        rows += observations("heart-4site", X, y, g, nm, a.seeds)
        print(f"heart-4site: {len(rows)} decisions", flush=True)
    if "mimic-11site" in want:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        X, y, g, nm, _ = mimic_sites.load(verbose=False)
        n0 = len(rows)
        rows += observations("mimic-11site", X, y, g, nm, a.seeds)
        print(f"mimic-11site ({len(nm)} sites): {len(rows) - n0} decisions", flush=True)
    if "mimic4-9site" in want:
        from mimic4_sites import load as load4
        X, y, g, nm, _ = load4(verbose=False)
        n0 = len(rows)
        rows += observations("mimic4-9site", X, y, g, nm, a.seeds)
        print(f"mimic4-9site ({len(nm)} sites): {len(rows) - n0} decisions", flush=True)
    if "eicu" in want:
        from eicu_sites import load as loade
        X, y, g, nm, _c = loade(verbose=False)
        n0 = len(rows)
        rows += observations("eicu", X, y, g, nm, a.seeds)
        print(f"eicu ({len(nm)} hospitals): {len(rows) - n0} decisions", flush=True)
    if "plco-2cohort" in want:
        from multisite import plco_2cohort
        X, y, g, nm = plco_2cohort()
        n0 = len(rows)
        rows += observations("plco-2cohort", X, y, g, nm, a.seeds)
        print(f"plco-2cohort: {len(rows) - n0} decisions", flush=True)

    res = evaluate_rules(rows)

    print("\nper-dataset breakdown of how often merging helped:")
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r["helped"])
    for k, v in by.items():
        print(f"  {k:14s} {np.mean(v) * 100:5.1f}%  of {len(v)} decisions")

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(a.out.replace(".csv", "_rules.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
