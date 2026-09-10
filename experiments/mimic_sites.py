"""Build a third real multi-site dataset: MIMIC-III ICU care units.

The multi-site benchmark is limited by having only two real datasets, and the one
diagnostic that looked promising failed because our simulator gives every site the same
size and the same prevalence. What is needed is a real dataset whose sites differ in
*both*, with block-wise missingness that nobody imposed.

MIMIC-III's first ICU care unit supplies exactly that. CCU, CSRU, MICU, SICU and TSICU
admit different populations, differ in size by an order of magnitude, differ in mortality
by a factor of two or more, and -- the part that matters here -- **order different
laboratory tests**. Cardiac units draw troponin and CK-MB as a matter of routine; surgical
units draw lactate and coagulation studies. No site label was invented and no missingness
was simulated: a variable is absent from a unit's schema because that unit does not order
that test.

It is one hospital, so this is care-unit heterogeneity rather than inter-institutional
heterogeneity, and that limitation is stated wherever the results are used. What it buys
is the configuration the other two datasets and the generator all lack: five sites,
unequal in size and prevalence, with naturally arising schema differences.

Cohort: first ICU stay of each hospital admission, adults, at least 24 hours in the unit.
Features: the mean of each frequent laboratory measurement over the first 24 hours after
ICU admission. Outcome: in-hospital mortality.

Writes only the aggregated feature matrix to the scratch cache -- never to the repo. MIMIC
is governed by a PhysioNet DUA and `datasets/` is git-ignored for that reason.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get(
    "MIMIC3_ROOT",
    Path.home() / "Works/tong/phd/datasets/physionet.org/files/mimiciii/1.4"))
_SCRATCH = Path(os.environ.get(
    "MIMIC_CACHE",
    Path.home() / ".cache" / "phd-sites"))

MIN_UNIT_N = 300          # a site needs enough stays to fit a local model
SITE_KEY = os.environ.get("MIMIC_SITE_KEY", "careunit")   # or "unit_x_system"
N_LABS = 40               # most frequently measured labs in the cohort
WINDOW_H = 24.0


def _build(verbose=True):
    icu = pd.read_csv(ROOT / "ICUSTAYS.csv.gz",
                      usecols=["HADM_ID", "ICUSTAY_ID", "FIRST_CAREUNIT",
                               "DBSOURCE", "INTIME", "LOS"],
                      parse_dates=["INTIME"])
    icu = icu.sort_values("INTIME").drop_duplicates("HADM_ID", keep="first")
    icu = icu[icu["LOS"] >= 1.0]

    adm = pd.read_csv(ROOT / "ADMISSIONS.csv.gz",
                      usecols=["HADM_ID", "SUBJECT_ID", "ADMITTIME",
                               "HOSPITAL_EXPIRE_FLAG"],
                      parse_dates=["ADMITTIME"])
    pat = pd.read_csv(ROOT / "PATIENTS.csv.gz",
                      usecols=["SUBJECT_ID", "DOB", "GENDER"], parse_dates=["DOB"])
    coh = icu.merge(adm, on="HADM_ID").merge(pat, on="SUBJECT_ID")

    # MIMIC-III shifts ages over 89 to ~300 years; keep adults, cap the shifted ones.
    age = (coh["ADMITTIME"].dt.year - coh["DOB"].dt.year)
    coh = coh.assign(AGE=np.where(age > 120, 90.0, age.astype(float)))
    coh = coh[coh["AGE"] >= 18]

    # Two ways to define a site. FIRST_CAREUNIT gives five units within one EHR era;
    # crossing it with DBSOURCE splits each unit by the record system in use (CareVue
    # 2001-2008 against MetaVision 2008-2012), which is a different era, a different
    # ordering culture and a different case mix -- eleven sites rather than five, from
    # data already on disk. `both` is dropped: 136 stays that span the changeover.
    coh = coh[coh["DBSOURCE"].isin(["carevue", "metavision"])]
    coh["SITE"] = (coh["FIRST_CAREUNIT"] + "-" + coh["DBSOURCE"].str[:4]
                   if SITE_KEY == "unit_x_system" else coh["FIRST_CAREUNIT"])
    keep_units = (coh["SITE"].value_counts()
                  .loc[lambda s: s >= MIN_UNIT_N].index.tolist())
    coh = coh[coh["SITE"].isin(keep_units)]
    if verbose:
        print(f"cohort: {len(coh)} first ICU stays across {len(keep_units)} sites "
              f"(site key = {SITE_KEY})")

    intime = dict(zip(coh["HADM_ID"], coh["INTIME"]))
    wanted = set(coh["HADM_ID"])

    # LABEVENTS is 27M rows; stream it and keep only the cohort's first 24 ICU hours.
    counts: dict[int, int] = {}
    sums: dict[tuple[int, int], float] = {}
    ns: dict[tuple[int, int], int] = {}
    reader = pd.read_csv(ROOT / "LABEVENTS.csv.gz",
                         usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
                         parse_dates=["CHARTTIME"], chunksize=2_000_000)
    for ci, ch in enumerate(reader):
        ch = ch.dropna(subset=["HADM_ID", "VALUENUM"])
        ch = ch[ch["HADM_ID"].isin(wanted)]
        if ch.empty:
            continue
        t0 = ch["HADM_ID"].map(intime)
        hrs = (ch["CHARTTIME"] - t0).dt.total_seconds() / 3600.0
        ch = ch[(hrs >= 0) & (hrs <= WINDOW_H)]
        for hadm, item, val in zip(ch["HADM_ID"].to_numpy(dtype=np.int64),
                                   ch["ITEMID"].to_numpy(dtype=np.int64),
                                   ch["VALUENUM"].to_numpy()):
            counts[item] = counts.get(item, 0) + 1
            k = (hadm, item)
            sums[k] = sums.get(k, 0.0) + val
            ns[k] = ns.get(k, 0) + 1
        if verbose:
            print(f"  chunk {ci}: {len(ch)} in-window rows, "
                  f"{len(ns)} (stay, lab) cells so far", flush=True)

    top = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:N_LABS]]
    tix = {it: j for j, it in enumerate(top)}
    hadms = coh["HADM_ID"].to_numpy(dtype=np.int64)
    hix = {h: i for i, h in enumerate(hadms)}
    X = np.full((len(hadms), len(top) + 2), np.nan)
    for (hadm, item), s in sums.items():
        if item in tix and hadm in hix:
            X[hix[hadm], tix[item]] = s / ns[(hadm, item)]
    X[:, -2] = coh["AGE"].to_numpy()
    X[:, -1] = (coh["GENDER"].to_numpy() == "M").astype(float)

    y = coh["HOSPITAL_EXPIRE_FLAG"].to_numpy().astype(int)
    units = coh["SITE"].to_numpy()
    names = sorted(set(units))
    g = np.array([names.index(u) for u in units])

    lab = pd.read_csv(ROOT / "D_LABITEMS.csv.gz", usecols=["ITEMID", "LABEL"])
    lmap = dict(zip(lab["ITEMID"], lab["LABEL"]))
    cols = [str(lmap.get(i, i)) for i in top] + ["age", "male"]
    return X, y, g, names, cols


def load(rebuild=False, verbose=True):
    """Cached build. The cache lives in scratch, never in the repo."""
    # Names are stored as a fixed-width unicode array rather than object dtype, so the
    # cache loads without allow_pickle -- no code path here can execute cached content.
    CACHE = _SCRATCH / f"mimic_sites_{SITE_KEY}.npz"
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE)
        return (z["X"], z["y"], z["g"],
                [str(s) for s in z["names"]], [str(s) for s in z["cols"]])
    X, y, g, names, cols = _build(verbose=verbose)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, g=g, names=np.array(names, dtype=np.str_),
                        cols=np.array(cols, dtype=np.str_))
    return X, y, g, names, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    X, y, g, names, cols = load(rebuild=a.rebuild)
    print(f"\nX {X.shape}  overall missing {np.isnan(X).mean():.2f}  "
          f"prevalence {y.mean():.3f}")
    print(f"{'care unit':10s} {'n':>7s} {'prev':>6s} {'schema>=50%':>12s} {'missing':>8s}")
    for k, nm in enumerate(names):
        m = g == k
        obs = (~np.isnan(X[m])).mean(0)
        print(f"{nm:10s} {int(m.sum()):7d} {y[m].mean():6.3f} "
              f"{int((obs >= 0.5).sum()):12d} {np.isnan(X[m]).mean():8.2f}")

    # which variables are schema differences rather than scattered gaps?
    print("\nvariables whose availability differs most across units "
          "(fraction of stays measured):")
    frac = np.array([[(~np.isnan(X[g == k, j])).mean() for k in range(len(names))]
                     for j in range(X.shape[1])])
    spread = frac.max(1) - frac.min(1)
    for j in np.argsort(-spread)[:10]:
        per = "  ".join(f"{names[k]}:{frac[j, k]:.2f}" for k in range(len(names)))
        print(f"  {cols[j][:28]:28s} spread {spread[j]:.2f}   {per}")


if __name__ == "__main__":
    main()
