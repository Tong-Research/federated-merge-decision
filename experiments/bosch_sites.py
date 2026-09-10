"""Bosch production line: parts grouped by the route they took through the factory.

Result 55 is the reason this dataset is here. Our ACS conclusions depend on a size
distribution we chose, and Lending Club supplied only one natural alternative. Bosch supplies
a second, and a stranger one: **the missingness is the routing**. A part carries a value at
station S only if it physically passed through S, so the observed-column pattern of a row IS
its path through the factory. Grouping parts by that pattern gives a partition nobody
designed -- not us, and not Bosch either. It falls out of how the parts were built.

The line prefix in `L0_S0_F0` groups *features*, not rows: a part traverses many stations
across lines, so "line" is not a partition of parts and cannot be the site. The routing path
is.

**Why the file is streamed.** train_numeric.csv is roughly 2 GB with 970 columns, most of
them empty on any given row. Reading it whole is possible but wasteful, and the machine this
runs on has ~13 GB free. Two passes: one to learn the pattern of each row and count patterns,
a second to materialise only the rows belonging to groups that clear the floor.

**Predictions, recorded before running** (see NEW-DATASETS-PLAN.md):
  1. Routing groups span more than three orders of magnitude in size.
  2. EPV is below 1 for almost every group, because the positive rate is ~0.6% and there are
     968 features -- in which case Bosch cannot test the rule, exactly as eICU could not
     (Result 46), and that is the finding rather than a failure.

If prediction 2 holds, do not force an analysis. Report it as a third example of a collection
that cannot detect its own regime and move on.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("MIMIC_CACHE", Path.home() / ".cache/phd-matrices"))
SRC = CACHE / "bosch" / "train_numeric.csv"
MIN_GROUP = int(os.environ.get("BOSCH_MIN_GROUP", "300"))
# Only stations matter for the route; grouping on individual features would produce a pattern
# per part and no groups at all.
STATION_LEVEL = os.environ.get("BOSCH_STATION_LEVEL", "1") == "1"


def _station_of(col):
    """`L0_S12_F345` -> `L0_S12`. Returns None for Id / Response."""
    bits = col.split("_")
    return f"{bits[0]}_{bits[1]}" if len(bits) >= 3 and bits[0].startswith("L") else None


def pass1(path, chunk=100_000, verbose=True):
    """Learn each row's route signature and count how many parts share it.

    Vectorised with pandas rather than looped in Python: 1.18M rows x 970 columns is 1.15
    billion cells, and a per-cell loop takes hours. Per chunk we take `notna()` once and then
    reduce to one boolean per station with a column-slice `any`, which is ~50 vector
    operations per chunk instead of a million scalar ones.
    """
    import pandas as pd

    header = pd.read_csv(path, nrows=0).columns.tolist()
    stations = [_station_of(c) for c in header]
    uniq = sorted({s for s in stations if s})
    cols_of = {s: [c for c, st in zip(header, stations) if st == s] for s in uniq}
    if verbose:
        print(f"  {len(header)} columns, {len(uniq)} stations", flush=True)

    counts = Counter()
    sig_of_row = []
    seen_rows = 0
    for ch in pd.read_csv(path, chunksize=chunk, low_memory=False):
        obs = ch.notna()
        # one column per station: did this part touch it at all?
        st = np.column_stack([obs[cols_of[s]].any(axis=1).to_numpy() for s in uniq])
        packed = np.packbits(st, axis=1)
        for row in packed:
            sig = row.tobytes()
            counts[sig] += 1
            sig_of_row.append(sig)
        seen_rows += len(ch)
        if verbose:
            print(f"  pass1 {seen_rows:,} rows, {len(counts):,} distinct routes", flush=True)
    return header, uniq, counts, sig_of_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--out", default="bosch_routes")
    ap.add_argument("--top-d", type=int, default=100,
                    help="keep only the D most-often-observed features. Not a convenience: "
                         "at the full 968 the array is 4.6 GB and EPV 10 would need 1.67M "
                         "parts in one route, 1.4x the whole dataset, so full width cannot "
                         "reach the boundary anyway (NEW-DATASETS-PLAN.md).")
    ap.add_argument("--spread", action="store_true",
                    help="pick one feature per station, from stations spanning the full "
                         "range of how often they are visited, instead of the D "
                         "most-observed features. Taking the most-observed features selects "
                         "ones that CO-OCCUR: the first d=20 build came out with every "
                         "feature seen in exactly the same 278 routes, degree 278-278, which "
                         "engineers away the block-wise missingness that makes this dataset "
                         "interesting. Spreading across stations keeps it.")
    ap.add_argument("--survey-only", action="store_true",
                    help="report the route-size distribution and stop, without building a "
                         "matrix -- this is the check that decides whether Bosch is usable")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        sys.exit(f"{src} not found.\n"
                 f"Bosch needs a Kaggle API token at ~/.kaggle/kaggle.json and the "
                 f"competition rules accepted in a browser, then:\n"
                 f"  kaggle competitions download -c bosch-production-line-performance "
                 f"-f train_numeric.csv.zip -p {src.parent}")

    print(f"pass 1: routes in {src.name}")
    header, stations, counts, sig_of_row = pass1(src)

    sizes = np.array(sorted(counts.values())[::-1])
    keep = {s for s, c in counts.items() if c >= a.min_group}
    print(f"\n{len(counts):,} distinct routes over {sizes.sum():,} parts")
    print(f"  sizes: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max():,}")
    print(f"  spread: {sizes.max() / max(sizes.min(), 1):,.0f}x")
    print(f"  routes with >= {a.min_group} parts: {len(keep):,} "
          f"covering {sum(counts[s] for s in keep):,} parts")
    for thr in (1000, 5000, 20000, 100000):
        print(f"    >= {thr:>6,} parts: {sum(1 for c in counts.values() if c >= thr):,} routes")

    if a.survey_only:
        print("\nsurvey only -- no matrix written. Decide from the EPV arithmetic above "
              "whether Bosch can test the rule before building anything.")
        return 0

    # ---- pass 2: materialise the kept groups
    #
    # Also vectorised. The first version looped over 968 columns per row twice -- once to
    # count observations and once to fill -- which is 2.3 billion Python iterations. pandas
    # does both as column reductions.
    import pandas as pd
    names = sorted(keep, key=lambda s: -counts[s])
    gidx = {s: i for i, s in enumerate(names)}
    feat_all = [c for c in header if c not in ("Id", "Response") and _station_of(c)]
    sig_arr = np.array(sig_of_row, dtype=object)
    keep_mask_all = np.array([s in keep for s in sig_arr])

    # which features are observed most often, among the rows we are keeping
    obs_count = pd.Series(0, index=feat_all, dtype="int64")
    off = 0
    for ch in pd.read_csv(src, chunksize=100_000, low_memory=False):
        m = keep_mask_all[off:off + len(ch)]
        obs_count += ch.loc[m, feat_all].notna().sum()
        off += len(ch)
    if a.spread:
        # One feature per station, from stations chosen to span the visit-rate spectrum.
        # A station visited by every route carries no schema information; one visited by
        # half of them carries the most. Sampling evenly across the spectrum gives a degree
        # sequence that actually varies, which is the whole point of using Bosch.
        st_of = {c: _station_of(c) for c in feat_all}
        by_st = {}
        for c in feat_all:
            by_st.setdefault(st_of[c], []).append(c)
        rate = {s: max(obs_count[c] for c in cs) / max(int(keep_mask_all.sum()), 1)
                for s, cs in by_st.items()}
        ordered = sorted(rate, key=lambda s: -rate[s])
        pick = [ordered[int(round(i * (len(ordered) - 1) / max(a.top_d - 1, 1)))]
                for i in range(min(a.top_d, len(ordered)))]
        seen_st, feat = set(), []
        for s in pick:
            if s in seen_st:
                continue
            seen_st.add(s)
            feat.append(max(by_st[s], key=lambda c: obs_count[c]))
        print(f"\n  spread mode: one feature from each of {len(feat)} stations, "
              f"visit rates {min(rate[st_of[c]] for c in feat):.2f}-"
              f"{max(rate[st_of[c]] for c in feat):.2f}")
    else:
        feat = list(obs_count.sort_values(ascending=False).index[:a.top_d])
    print(f"  {len(feat)} features observed in {obs_count[feat].min():,}-"
          f"{obs_count[feat].max():,} of {keep_mask_all.sum():,} kept parts")

    n_keep = int(keep_mask_all.sum())
    X = np.full((n_keep, len(feat)), np.nan, dtype=np.float32)
    y = np.zeros(n_keep, dtype=np.int8)
    g = np.zeros(n_keep, dtype=np.int32)
    off = r = 0
    for ch in pd.read_csv(src, chunksize=100_000, low_memory=False):
        m = keep_mask_all[off:off + len(ch)]
        k = int(m.sum())
        if k:
            X[r:r + k] = ch.loc[m, feat].to_numpy(dtype=np.float32)
            y[r:r + k] = ch.loc[m, "Response"].to_numpy(dtype=np.int8)
            g[r:r + k] = [gidx[s] for s in sig_arr[off:off + len(ch)][m]]
            r += k
        off += len(ch)
        print(f"  pass2 {off:,} scanned, {r:,} kept", flush=True)

    out = CACHE / f"{a.out}.npz"
    np.savez_compressed(out, X=X, y=y, g=g,
                        names=np.array([f"route{i}" for i in range(len(names))],
                                       dtype=np.str_),
                        cols=np.array(feat, dtype=np.str_))
    d = X.shape[1]
    ev = np.array([min(np.bincount(y[g == k], minlength=2)) for k in range(len(names))])
    print(f"\nwrote {out}: {X.shape}, {len(names)} routes, positive rate {y.mean():.4f}")
    print(f"  EPV at d={d}: {ev.min() / d:.3f} - {ev.max() / d:.3f} "
          f"(median {np.median(ev) / d:.3f})")
    above = int((ev / d >= 10).sum())
    print(f"  routes above EPV 10: {above}"
          + ("" if above else "  <-- as predicted, Bosch cannot test the threshold"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
