"""Carve one population of people several ways, and measure the boundary under each.

Result 75 found that the size at which merging stops helping varies six-fold across five
partitions of the same three million mortgages, and concluded the boundary belongs to a
(domain, partition) pair rather than to a domain. That claim rests on ONE population, which
Section 10 lists as the limitation most worth removing.

ACS removes it, and removes it with a stronger design than the collection that produced it.
Freddie Mac's partitions are geography or corporate structure -- seller, state, metro, ZIP-2,
ZIP-3 -- so they largely nest inside one another. The Census microdata carries orthogonal
axes: the same respondents can be grouped by *where they live* (state, PUMA) or by *what they
do* (occupation, industry), and those have no nesting relationship at all. If a boundary is a
property of the carving, two orthogonal carvings should disagree at least as much as two
nested ones.

The design point that makes this a test of carving rather than of data:

    EVERY PARTITION USES EXACTLY THE SAME ROWS.

The row set is fixed once -- respondents with a usable value on every partition variable --
and each partition only relabels them. Freddie Mac could not do this: its metro partition
dropped records with no metro code, so its five partitions covered slightly different loans
and a sceptic could attribute part of the six-fold spread to that. Here nothing differs but
the labels.

Partitions
----------
    state       ST     -- 51 sites, the existing convention
    puma        PUMA   -- Public Use Microdata Areas within state, a finer geography
    occupation  OCCP   -- occupation code, orthogonal to geography
    industry    INDP   -- industry code, orthogonal to geography
    education   SCHL   -- attainment, a non-geographic non-economic axis

Usage:
    acs_repartition.py --by occupation --out acs_rp_occupation
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

CACHE = pathlib.Path(os.environ.get("MIMIC_CACHE", "."))
RAW = CACHE / "acs_raw" / "2018" / "1-Year"

# partition name -> (PUMS column, minimum records for a site to be kept)
PARTITIONS = {
    "state":      ("ST", 200),
    "puma":       ("PUMA", 200),
    "occupation": ("OCCP", 200),
    "industry":   ("INDP", 200),
    "education":  ("SCHL", 200),
}

# ACSIncome's own definition, restated here rather than imported: folktables applies its row
# filter and feature list inside `df_to_numpy`, and this script needs the SAME rows to
# survive into every partition, which means doing the filtering once and up front.
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
TARGET = "PINCP"
THRESHOLD = 50000


def load_rows(nmax=None):
    """One row set, used by every partition. Filters follow folktables' ACSIncome."""
    files = sorted(glob.glob(str(RAW / "psam_p*.csv")))
    if not files:
        raise FileNotFoundError(f"no PUMS files under {RAW}")
    keep = sorted(set(FEATURES + [TARGET] + [c for c, _ in PARTITIONS.values()]))
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in keep, low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # ACSIncome's filter: employed adults with positive income and recorded hours.
    df = df[(df["AGEP"] > 16) & (df[TARGET].notna()) & (df[TARGET] > 100)
            & (df["WKHP"] > 0) & (df["PWGTP"] > 0)] if "PWGTP" in df else \
        df[(df["AGEP"] > 16) & (df[TARGET].notna()) & (df[TARGET] > 100) & (df["WKHP"] > 0)]
    # A respondent must have a usable value on EVERY partition variable, or the partitions
    # would cover different people and the comparison would confound carving with coverage.
    for col, _ in PARTITIONS.values():
        df = df[df[col].notna()]
    df = df.dropna(subset=FEATURES)
    if nmax and len(df) > nmax:
        df = df.sample(nmax, random_state=0)
    return df.reset_index(drop=True)


def build(df, by, cap=0, seed=0, max_sites=0, prevalence=0.0):
    """`cap` equalises SITE SIZE across partitions, which is the only way to separate the
    effect of carving from the effect of the site-size distribution it induces.

    Without it the partitions are not comparable: grouping by PUMA gives a median site of 562
    records and grouping by state gives 15,343, so any difference in merge benefit is
    consistent with either the carving or the sizes. With every kept site subsampled to
    exactly `cap` records, the partitions differ ONLY in which people are grouped together.

    `prevalence` equalises the OUTCOME RATE, which is the residual confound Section 10 names
    and which capping alone does not remove: equalising site size shifts the pooled rate
    between 0.269 and 0.396 across the five carvings, because different groupings retain
    different people. Under `--prevalence r` the `cap` records drawn from each site are drawn
    STRATIFIED on the outcome -- round(cap * r) positives and the rest negatives -- so every
    site in every carving has the same size AND the same rate. A site without enough of either
    class is dropped, which costs rows; that is the price, and the paper's claim that rows,
    site size and prevalence cannot be fixed simultaneously is true only if all rows are kept.

    `max_sites` equalises SITE COUNT, which Result 83 showed R82 had left uncontrolled. The
    capped partitions run from 20 sites (education) to 496 (PUMA), and site count sets the
    partner pool -- Result 45 established that partner count sets the size of the prize. So
    education's benefit, which is the counterexample currently defeating two proposed
    mechanisms, is confounded with having a twentieth of the partners available. Holding both
    size and count fixed is what makes the five carvings actually comparable.
    """
    col, floor = PARTITIONS[by]
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df[TARGET].to_numpy() > THRESHOLD).astype(np.int8)
    codes = df[col].astype("int64").astype(str).to_numpy()
    uniq, counts = np.unique(codes, return_counts=True)
    keep = set(uniq[counts >= floor])
    if cap:
        keep = {u for u, c in zip(uniq, counts) if c >= cap}
    if cap and prevalence:
        # A site can only be forced to the target rate if it holds enough of both classes.
        npos, nneg = int(round(cap * prevalence)), cap - int(round(cap * prevalence))
        keep = {u for u in keep
                if int((y[codes == u] == 1).sum()) >= npos and int((y[codes == u] == 0).sum()) >= nneg}
    if max_sites and len(keep) > max_sites:
        # Drawn at random rather than taking the largest: "the largest N" would reintroduce a
        # size ordering across partitions, which is the confound the cap exists to remove.
        # Seeded, and meant to be run at several seeds -- one draw of 20 sites is one sample.
        keep = set(np.random.default_rng(seed).choice(sorted(keep), max_sites, replace=False))
    m = np.array([c in keep for c in codes])
    X, y, codes = X[m], y[m], codes[m]
    if cap:
        rng = np.random.default_rng(seed)
        if prevalence:
            npos = int(round(cap * prevalence)); nneg = cap - npos
            parts = []
            for u in sorted(keep):
                at = np.flatnonzero(codes == u)
                parts.append(rng.choice(at[y[at] == 1], npos, replace=False))
                parts.append(rng.choice(at[y[at] == 0], nneg, replace=False))
            sel = np.concatenate(parts)
        else:
            sel = np.concatenate([rng.choice(np.flatnonzero(codes == u), cap, replace=False)
                                  for u in sorted(keep)])
        sel.sort()
        X, y, codes = X[sel], y[sel], codes[sel]
    names = sorted(set(codes))
    idx = {n: i for i, n in enumerate(names)}
    g = np.array([idx[c] for c in codes], dtype=np.int32)
    return X, y, g, names


def emit(df, by, out, cap, seed, max_sites, prevalence=0.0):
    X, y, g, names = build(df, by, cap=cap, seed=seed, max_sites=max_sites, prevalence=prevalence)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{out}.npz", X=X, y=y, g=g, names=np.array(names))
    s = np.bincount(g)
    print(f"{out}: {X.shape[0]:,} rows x {X.shape[1]} features, {len(names)} sites, "
          f"prevalence {y.mean():.3f}")
    print(f"  shared row set {len(df):,}; this partition keeps {X.shape[0]:,} "
          f"({100 * X.shape[0] / len(df):.1f}%)")
    print(f"  site sizes {s.min():,}-{s.max():,} (median {int(np.median(s)):,})")
    if cap:
        # Capping equalises size but shifts prevalence, because different carvings retain
        # different people. Section 10 names that as the residual confound, so print it
        # here rather than leaving it to be recovered from the decision file later.
        if prevalence:
            print(f"  CAPPED at {cap} AND MATCHED to prevalence {prevalence:.3f}: realised "
                  f"{y.mean():.3f}. Site size, site count and outcome rate are now all fixed "
                  f"across carvings; the price is rows.")
        else:
            print(f"  CAPPED at {cap}: report prevalence {y.mean():.3f} beside the benefit --"
                  f" it is the confound that size-matching does not remove")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=sorted(PARTITIONS),
                    help="one partition; omit and pass --all to build every partition")
    ap.add_argument("--all", action="store_true",
                    help="build every partition from ONE read of the PUMS files. Loading the "
                         "row set dominates the runtime, so building the five carvings "
                         "separately re-reads 1.66M rows five times for no reason.")
    ap.add_argument("--prefix", default="",
                    help="with --all, name outputs <prefix><partition>")
    ap.add_argument("--out")
    ap.add_argument("--nmax", type=int, default=0,
                    help="subsample the shared row set; 0 keeps all of it")
    # `build` has taken a cap since it was written, but main() did not expose it until
    # 2026-08-11, so the acs_cap_* matrices behind Result 82 -- the paper's causal claim
    # about carving -- existed on disk with no committed command that rebuilds them.
    ap.add_argument("--cap", type=int, default=0,
                    help="subsample every kept site to exactly this many records, and drop "
                         "sites smaller than it; 0 keeps natural sizes. This is what "
                         "separates the effect of carving from the site-size distribution "
                         "the carving induces (Result 82 used 800).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the per-site subsample drawn under --cap")
    ap.add_argument("--max-sites", type=int, default=0, dest="max_sites",
                    help="keep at most this many sites, drawn at random under --seed; 0 keeps "
                         "all of them. Equalises the partner pool across carvings, the "
                         "confound Result 83 found uncontrolled in Result 82 (20 sites for "
                         "education against 496 for PUMA).")
    ap.add_argument("--prevalence", type=float, default=0.0,
                    help="with --cap, draw each site's records stratified on the outcome so "
                         "every site hits this rate. Removes the residual confound Section 10 "
                         "names; 0 leaves prevalence to fall where the carving puts it.")
    a = ap.parse_args()
    if not a.all and not (a.by and a.out):
        ap.error("pass --by and --out, or --all")

    df = load_rows(a.nmax or None)
    if a.all:
        for by in sorted(PARTITIONS):
            emit(df, by, f"{a.prefix}{by}", a.cap, a.seed, a.max_sites, a.prevalence)
        return 0
    return emit(df, a.by, a.out, a.cap, a.seed, a.max_sites, a.prevalence)


if __name__ == "__main__":
    raise SystemExit(main())
