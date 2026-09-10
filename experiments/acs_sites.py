"""US Census ACS by state: is the merge rule about medicine, or about tabular merging?

Every result in this project comes from ICU mortality prediction. A reviewer will ask
whether the events-per-variable threshold is a fact about clinical data -- low prevalence,
correlated labs, block-wise missingness from data interfaces -- or a fact about combining
tabular models across heterogeneous sources. The question is answerable, and cheaply.

`folktables` (Ding et al., NeurIPS 2021) turns US Census ACS microdata into prediction
tasks with **states as natural sites**: 51 of them, genuinely different populations and
base rates, millions of rows, fully open with no data use agreement. Nothing about it is
clinical.

Two departures from the clinical setup are deliberate and both make the test harder:
states are far larger than hospitals, so most sit high on the EPV scale where the rule
predicts merging should *hurt*; and ACS has no block-wise missingness, so every site sees
the same schema. If the threshold reappears here it is a property of model merging under
heterogeneity, not of clinical data.

Sites are subsampled to a configurable size so the EPV range covers the decision boundary
rather than sitting entirely above it -- the same problem eICU had at d=42 (Result 22),
solved the same way.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

CACHE = os.environ.get("MIMIC_CACHE", ".")

def build(task_name="ACSIncome", year=2018, per_state=None, seed=0, states=None,
          log_uniform=None):
    """Build a state-partitioned matrix for one folktables task.

    `per_state=None` (or 0) keeps every state at its NATURAL size, which is the point of the
    full-size build: real ACS states run from a few thousand records to a couple of hundred
    thousand, a spread that puts sites on BOTH sides of the adequacy boundary. The earlier
    build capped every state at 1,200 so EPV would span the cut -- which engineered the harm
    regime out of the data, and is why Result 39 found no boundary to locate.

    The task choice is a prevalence lever: at fixed n a rarer outcome means fewer events and
    so lower EPV, the same causal manipulation the d-sweep achieved by moving the denominator.
    """
    import folktables as _ft
    from folktables import ACSDataSource
    task = getattr(_ft, task_name)
    src = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person",
                        root_dir=os.path.join(CACHE, "acs_raw"))
    data = src.get_data(states=states, download=True)
    Xs, ys, gs, names = [], [], [], []
    rng = np.random.default_rng(seed)
    for st, sub in data.groupby("ST"):
        try:
            x, yv, _ = task.df_to_numpy(sub)
        except Exception:                                        # noqa: BLE001
            continue
        yv = np.asarray(yv).astype(int)
        if len(np.unique(yv)) < 2 or len(yv) < 200:
            continue
        if log_uniform:
            # Each state keeps a log-uniform draw between `log_uniform` and its natural
            # size. Neither fixed cap works: capping every state at 1,200 puts them all
            # BELOW the harm regime (Result 39 found no boundary), while natural sizes put
            # them all far ABOVE it (EPV 96-8,033). A threshold is only fittable in a pool
            # holding sites on both sides of it (Result 42), so the sizes must span.
            hi = len(yv)
            if hi > log_uniform:
                target = int(np.exp(rng.uniform(np.log(log_uniform), np.log(hi))))
                keep = rng.choice(hi, min(target, hi), replace=False)
                x, yv = x[keep], yv[keep]
        elif per_state and len(yv) > per_state:
            keep = rng.choice(len(yv), per_state, replace=False)
            x, yv = x[keep], yv[keep]
        Xs.append(x.astype(float)); ys.append(yv)
        gs.append(np.full(len(yv), len(names))); names.append(str(int(st)))
    X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    cols = [f"f{j}" for j in range(X.shape[1])]
    return X, y, g, names, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ACSIncome")
    ap.add_argument("--per-state", type=int, default=1200,
                    help="subsample each state; 0 keeps natural state sizes")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--log-uniform", type=int, default=0,
                    help="draw each state's size log-uniformly between this and its "
                         "natural size, so the pool spans the boundary")
    ap.add_argument("--out", default="acs_sites")
    a = ap.parse_args()
    X, y, g, names, cols = build(a.task, year=a.year,
                                 per_state=(a.per_state or None),
                                 log_uniform=(a.log_uniform or None))
    np.savez_compressed(f"{CACHE}/{a.out}.npz", X=X, y=y, g=g,
                        names=np.array(names, dtype=np.str_),
                        cols=np.array(cols, dtype=np.str_))
    d = X.shape[1]
    ev = np.array([min(np.bincount(y[g == k])) for k in range(len(names))])
    print(f"{a.task}: {X.shape}, {len(names)} states, prevalence {y.mean():.3f}")
    print(f"  minority events per state: {ev.min()}-{ev.max()} (median {int(np.median(ev))})")
    print(f"  EPV at d={d}: {ev.min()/d:.1f}-{ev.max()/d:.1f} (median {np.median(ev)/d:.1f})")
    for lo, hi in [(0,2),(2,5),(5,10),(10,20),(20,1e9)]:
        n = int(((ev/d >= lo) & (ev/d < hi)).sum())
        if n: print(f"    EPV {lo}-{hi if hi<1e9 else 999}: {n} states")

if __name__ == "__main__":
    main()
