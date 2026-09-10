"""eICU-CRD: 100 real hospitals, which is what every previous caveat asked for.

Every multi-site result in this project so far rests on care units inside one or two
hospitals. eICU is 208 hospitals across the United States, 2014-2015, and the authors
document the missingness mechanism themselves: *"different care units may have different
interfaces in place, and the lack of an interface will result in no data being available
for a given patient, even if those measurements were made in reality."* That is exactly
the block-wise structure this project studies, arising from data pipelines rather than
from anything we imposed.

Two things make it worth more than another confirmation.

**The sites are the right size.** MIMIC's care units hold thousands of deaths because they
aggregate a decade at one large academic centre; real hospitals do not. eICU hospitals
with at least 300 qualifying stays have 12 to 381 deaths, median 68 -- so at ~40 features
essentially all of them sit *below* the EPV threshold, where MIMIC's units sat above it.
The same rule therefore gives the opposite advice on realistic sites, and that contrast is
a finding rather than an inconsistency.

**Institutional characteristics are available.** `hospital.csv.gz` carries bed count,
teaching status and region, so "prefer similar partners" can be tested at the level it was
always meant to apply to.

Cohort matches `mimic_sites.py` so the two are comparable: first unit stay of a hospital
admission, adults, at least 24 hours in the unit, features are the mean of each frequent
lab over the first 24 hours, outcome is in-hospital mortality. eICU records lab times as
minutes from unit admission, so the window needs no timestamp arithmetic.

Credentialed PhysioNet data under its own DUA. Only the aggregated matrix is cached, to
scratch, never to the repo.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get(
    "EICU_ROOT",
    Path.home() / "Works/tong/phd/datasets/physionet.org/files/eicu-crd/2.0"))
_SCRATCH = Path(os.environ.get(
    "MIMIC_CACHE",
    Path.home() / ".cache" / "phd-sites"))

MIN_SITE_N = int(os.environ.get("EICU_MIN_SITE_N", "300"))  # floor on stays per hospital
N_LABS = 40               # match mimic_sites.py so feature counts are comparable
WINDOW_MIN = 1440.0       # 24 hours, in minutes from unit admission


def _cohort():
    p = pd.read_csv(ROOT / "patient.csv.gz",
                    usecols=["patientunitstayid", "hospitalid", "unittype", "age",
                             "gender", "unitvisitnumber", "unitdischargeoffset",
                             "hospitaldischargestatus"])
    p = p[(p["unitvisitnumber"] == 1) & (p["unitdischargeoffset"] >= WINDOW_MIN)]
    # eICU masks ages over 89 as the string "> 89"; keep them at 90 as MIMIC does.
    p["age_n"] = pd.to_numeric(p["age"].replace("> 89", "90"), errors="coerce")
    p = p[p["age_n"] >= 18].dropna(subset=["hospitaldischargestatus"])
    p["died"] = (p["hospitaldischargestatus"] == "Expired").astype(int)
    keep = p["hospitalid"].value_counts().loc[lambda s: s >= MIN_SITE_N].index
    return p[p["hospitalid"].isin(keep)].reset_index(drop=True)


def _build(verbose=True):
    coh = _cohort()
    if verbose:
        print(f"cohort: {len(coh):,} first unit stays across "
              f"{coh['hospitalid'].nunique()} hospitals "
              f"(>= {MIN_SITE_N} stays each), mortality {coh['died'].mean():.3f}")

    wanted = set(coh["patientunitstayid"])
    counts: dict[str, int] = {}
    sums: dict[tuple[int, str], float] = {}
    ns: dict[tuple[int, str], int] = {}
    reader = pd.read_csv(ROOT / "lab.csv.gz",
                         usecols=["patientunitstayid", "labresultoffset", "labname",
                                  "labresult"],
                         chunksize=4_000_000)
    for ci, ch in enumerate(reader):
        ch = ch.dropna(subset=["labresult"])
        ch = ch[(ch["labresultoffset"] >= 0) & (ch["labresultoffset"] <= WINDOW_MIN)]
        ch = ch[ch["patientunitstayid"].isin(wanted)]
        if ch.empty:
            continue
        for sid, name, val in zip(ch["patientunitstayid"].to_numpy(dtype=np.int64),
                                  ch["labname"].to_numpy(),
                                  ch["labresult"].to_numpy(dtype=float)):
            counts[name] = counts.get(name, 0) + 1
            k = (sid, name)
            sums[k] = sums.get(k, 0.0) + val
            ns[k] = ns.get(k, 0) + 1
        if verbose:
            print(f"  chunk {ci}: {len(ns):,} (stay, lab) cells so far", flush=True)

    top = [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:N_LABS]]
    tix = {n: j for j, n in enumerate(top)}
    sids = coh["patientunitstayid"].to_numpy(dtype=np.int64)
    six = {s: i for i, s in enumerate(sids)}
    X = np.full((len(sids), len(top) + 2), np.nan)
    for (sid, name), tot in sums.items():
        if name in tix and sid in six:
            X[six[sid], tix[name]] = tot / ns[(sid, name)]
    X[:, -2] = coh["age_n"].to_numpy()
    X[:, -1] = (coh["gender"].to_numpy() == "Male").astype(float)

    y = coh["died"].to_numpy().astype(int)
    hosp = coh["hospitalid"].to_numpy()
    names = [str(h) for h in sorted(set(hosp))]
    g = np.array([names.index(str(h)) for h in hosp])
    cols = list(top) + ["age", "male"]
    return X, y, g, names, cols


def load(rebuild=False, verbose=True):
    # The cache key must carry every parameter that changes the cohort. It did not, so a
    # run with EICU_MIN_SITE_N=540 silently returned the 300-stay matrix and reported
    # "sites 100" as though the sensitivity check had passed. A cache whose key omits an
    # input does not fail loudly -- it returns the wrong answer confidently, which is the
    # most expensive kind of bug this project produces.
    suffix = "" if MIN_SITE_N == 300 else f"_min{MIN_SITE_N}"
    cache = _SCRATCH / f"eicu_sites{suffix}.npz"
    if cache.exists() and not rebuild:
        z = np.load(cache)
        return (z["X"], z["y"], z["g"],
                [str(s) for s in z["names"]], [str(s) for s in z["cols"]])
    X, y, g, names, cols = _build(verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, g=g, names=np.array(names, dtype=np.str_),
                        cols=np.array(cols, dtype=np.str_))
    return X, y, g, names, cols


def hospital_meta():
    """Bed count, teaching status and region -- for testing partner choice at the
    institutional level, which no other dataset here supports."""
    return pd.read_csv(ROOT / "hospital.csv.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    X, y, g, names, cols = load(rebuild=a.rebuild)
    print(f"\nX {X.shape}  missing {np.isnan(X).mean():.2f}  prevalence {y.mean():.3f}")

    d = X.shape[1]
    rows = []
    for k, nm in enumerate(names):
        m = g == k
        obs = (~np.isnan(X[m])).mean(0)
        rows.append((nm, int(m.sum()), int(y[m].sum()), float(y[m].mean()),
                     int((obs >= 0.5).sum()), float(y[m].sum() / d)))
    df = pd.DataFrame(rows, columns=["hospital", "n", "deaths", "prev", "schema", "epv"])
    print(f"\n{len(df)} hospitals. Size {df.n.min()}-{df.n.max()} (median "
          f"{int(df.n.median())}), deaths {df.deaths.min()}-{df.deaths.max()}, "
          f"mortality {df.prev.min():.3f}-{df.prev.max():.3f}")
    print(f"schema size (variables observed in >=50% of a hospital's stays): "
          f"{df.schema.min()}-{df.schema.max()} of {d}")

    print("\nEPV distribution -- the quantity the decision rule turns on:")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
        k = int(((df.epv >= lo) & (df.epv < hi)).sum())
        if k:
            print(f"  EPV {lo:>3}-{hi if hi < 1e9 else 999:<4}: {k:3d} hospitals")

    print("\nvariables whose availability differs most across hospitals:")
    frac = np.array([[(~np.isnan(X[g == k, j])).mean() for k in range(len(names))]
                     for j in range(d)])
    spread = frac.max(1) - frac.min(1)
    for j in np.argsort(-spread)[:8]:
        print(f"  {cols[j][:26]:26s} spread {spread[j]:.2f}  "
              f"(median availability {np.median(frac[j]):.2f})")

    meta = hospital_meta()
    mg = df.assign(hospitalid=df.hospital.astype(int)).merge(meta, on="hospitalid",
                                                             how="left")
    print("\nby region (institutional heterogeneity, testable for the first time):")
    print(mg.groupby("region", dropna=False)
            .agg(hospitals=("n", "size"), median_n=("n", "median"),
                 mortality=("prev", "mean"), median_epv=("epv", "median"))
            .to_string())


if __name__ == "__main__":
    main()
