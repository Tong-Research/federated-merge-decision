"""MIMIC-IV ICU care units: more sites, a later era, and a second hospital system generation.

MIMIC-IV 3.0 has been sitting on disk unused. It is worth adding for three reasons that
MIMIC-III cannot supply on its own: its `careunit` field is more granular (neuro, trauma,
cardiac vascular and medical/surgical units are separated rather than collapsed), it
covers 2008-2022 rather than 2001-2012, and it is a different database generation with
different item identifiers -- so pairing it with MIMIC-III gives an era split on top of a
unit split.

Same cohort definition as `mimic_sites.py` so the two are comparable: first ICU stay per
hospital admission, adults, at least 24 hours, features are the mean of each frequent lab
over the first 24 ICU hours, outcome is in-hospital mortality.

Still one hospital system. That caveat does not go away by adding sites, and it is the
reason eICU remains worth acquiring.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get(
    "MIMIC4_ROOT",
    Path.home() / "Works/tong/phd/datasets/physionet.org/files/mimiciv/3.0"))
_SCRATCH = Path(os.environ.get(
    "MIMIC_CACHE",
    Path.home() / ".cache" / "phd-sites"))
CACHE = _SCRATCH / "mimic4_sites.npz"

MIN_UNIT_N = 300
N_LABS = 40
WINDOW_H = 24.0


def _build(verbose=True):
    icu = pd.read_csv(ROOT / "icu" / "icustays.csv.gz",
                      usecols=["hadm_id", "stay_id", "first_careunit", "intime", "los"],
                      parse_dates=["intime"])
    icu = icu.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    icu = icu[icu["los"] >= 1.0]

    adm = pd.read_csv(ROOT / "hosp" / "admissions.csv.gz",
                      usecols=["hadm_id", "subject_id", "admittime",
                               "hospital_expire_flag"],
                      parse_dates=["admittime"])
    pat = pd.read_csv(ROOT / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
    coh = icu.merge(adm, on="hadm_id").merge(pat, on="subject_id")
    # MIMIC-IV gives an anchor age and anchor year rather than a date of birth.
    coh["AGE"] = (coh["anchor_age"]
                  + (coh["admittime"].dt.year - coh["anchor_year"])).astype(float)
    coh = coh[(coh["AGE"] >= 18) & (coh["AGE"] <= 100)]

    keep = (coh["first_careunit"].value_counts()
            .loc[lambda s: s >= MIN_UNIT_N].index.tolist())
    coh = coh[coh["first_careunit"].isin(keep)]
    if verbose:
        print(f"cohort: {len(coh)} first ICU stays across {len(keep)} care units")

    intime = dict(zip(coh["hadm_id"], coh["intime"]))
    wanted = set(coh["hadm_id"])
    counts: dict[int, int] = {}
    sums: dict[tuple[int, int], float] = {}
    ns: dict[tuple[int, int], int] = {}
    reader = pd.read_csv(ROOT / "hosp" / "labevents.csv.gz",
                         usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                         parse_dates=["charttime"], chunksize=2_000_000)
    for ci, ch in enumerate(reader):
        ch = ch.dropna(subset=["hadm_id", "valuenum"])
        ch = ch[ch["hadm_id"].isin(wanted)]
        if ch.empty:
            continue
        hrs = (ch["charttime"] - ch["hadm_id"].map(intime)).dt.total_seconds() / 3600.0
        ch = ch[(hrs >= 0) & (hrs <= WINDOW_H)]
        for hadm, item, val in zip(ch["hadm_id"].to_numpy(dtype=np.int64),
                                   ch["itemid"].to_numpy(dtype=np.int64),
                                   ch["valuenum"].to_numpy()):
            counts[item] = counts.get(item, 0) + 1
            k = (hadm, item)
            sums[k] = sums.get(k, 0.0) + val
            ns[k] = ns.get(k, 0) + 1
        if verbose and ci % 5 == 0:
            print(f"  chunk {ci}: {len(ns)} (stay, lab) cells so far", flush=True)

    top = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:N_LABS]]
    tix = {it: j for j, it in enumerate(top)}
    hadms = coh["hadm_id"].to_numpy(dtype=np.int64)
    hix = {h: i for i, h in enumerate(hadms)}
    X = np.full((len(hadms), len(top) + 2), np.nan)
    for (hadm, item), s in sums.items():
        if item in tix and hadm in hix:
            X[hix[hadm], tix[item]] = s / ns[(hadm, item)]
    X[:, -2] = coh["AGE"].to_numpy()
    X[:, -1] = (coh["gender"].to_numpy() == "M").astype(float)

    y = coh["hospital_expire_flag"].to_numpy().astype(int)
    units = coh["first_careunit"].to_numpy()
    names = sorted(set(units))
    g = np.array([names.index(u) for u in units])

    lab = pd.read_csv(ROOT / "hosp" / "d_labitems.csv.gz", usecols=["itemid", "label"])
    lmap = dict(zip(lab["itemid"], lab["label"]))
    cols = [str(lmap.get(i, i)) for i in top] + ["age", "male"]
    return X, y, g, names, cols


def load(rebuild=False, verbose=True):
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
    print(f"\nX {X.shape}  missing {np.isnan(X).mean():.2f}  prevalence {y.mean():.3f}")
    print(f"{'care unit':42s} {'n':>7s} {'prev':>6s} {'schema>=50%':>12s}")
    for k, nm in enumerate(names):
        m = g == k
        obs = (~np.isnan(X[m])).mean(0)
        print(f"{nm[:42]:42s} {int(m.sum()):7d} {y[m].mean():6.3f} "
              f"{int((obs >= 0.5).sum()):12d}")


if __name__ == "__main__":
    main()
