"""Freddie Mac single-family loans, partitioned at whatever granularity you ask for.

Result 55 is the standing problem: our ACS conclusions depend on a size distribution *we*
chose, and moving that distribution meant subsampling, which is a knob nobody has in reality.
Lending Club gave one natural alternative. Freddie Mac gives something better -- **a hierarchy
of natural partitions over the same loans**:

    seller     large institutions only; the rest collapse to 'OTHER' (see below)
    state      ~52 units, sizes spanning orders of magnitude
    zip2       ~100 two-digit postal regions
    MSA        ~400 metropolitan areas
    zip3       ~720 three-digit postal areas

Refining the partition shrinks the sites without touching a single row. That is the harm
fraction moved by a mechanism that exists in the world -- a federation of state agencies
versus one of metro-area lenders -- rather than by our subsampling. It is the closest thing
to a natural experiment on Result 64's x-axis that this project has found.

**The servicer caveat, which matters.** Freddie Mac discloses Seller and Servicer names only
for institutions holding at least 1% of a quarter's total original UPB; everyone else appears
as 'OTHER'. So a servicer partition is truncated at the *small* end -- exactly the end our
rule is about -- and its 'OTHER' bucket is an artefact rather than a site. Servicer is the
best governance story here but the worst size distribution, and both facts belong in the
paper if we use it.

Outcome: default, defined the standard way for this dataset -- ever 180+ days delinquent, or
a zero-balance code indicating short sale, repurchase, third-party sale or REO disposition.

Schema per the published layout (origination file, 26 fields; performance file keyed by
LoanSequenceNumber). Verified against a working public parser rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import zipfile
from collections import defaultdict
import pathlib
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("MIMIC_CACHE", Path.home() / ".cache/phd-matrices"))
SRC = CACHE / "freddiemac"

# Verified against the 2017 files by value distribution, not assumed from an older layout.
# The earlier version of this list was right to index 23 and wrong after it: it placed
# ServicerName at 24, where the data actually holds a Y/N super-conforming flag. Partitioning
# on that would have produced two "sites" and looked like a result.
#
# There is no servicer in the 2017 origination file. The servicer appears in the PERFORMANCE
# file, because it changes over the life of a loan. SellerName is the institutional field
# available at origination, and it carries the same >=1% disclosure floor: 17 distinct values,
# sixteen named lenders and 'OTHER'.
ORIG_COLS = [
    "CreditScore", "FirstPaymentDate", "FirstTimeHomeBuyerFlag", "MaturityDate", "MSA",
    "MortgageInsurancePercentage", "NumberOfUnits", "OccupancyStatus", "OriginalCLTV",
    "OriginalDTIRatio", "OriginalUPB", "OriginalLTV", "OriginalInterestRate", "Channel",
    "PPMFlag", "ProductType", "PropertyState", "PropertyType", "PostalCode",
    "LoanSequenceNumber", "LoanPurpose", "OriginalLoanTerm", "NumberOfBorrowers",
    "SellerName", "SuperConformingFlag", "PreHARPLoanSequenceNumber", "ProgramIndicator",
    "HARPIndicator", "PropertyValuationMethod", "InterestOnlyIndicator",
    "MICancellationIndicator",
]
# Application-time only. Everything in the performance file describes what happened AFTER
# origination and would leak the outcome -- the same trap avoided in lendingclub_sites.py.
NUMERIC = ["CreditScore", "MortgageInsurancePercentage", "NumberOfUnits", "OriginalCLTV",
           "OriginalDTIRatio", "OriginalUPB", "OriginalLTV", "OriginalInterestRate",
           "OriginalLoanTerm", "NumberOfBorrowers"]
CATEGORICAL = ["FirstTimeHomeBuyerFlag", "OccupancyStatus", "Channel", "PPMFlag",
               "PropertyType", "LoanPurpose"]
# zero-balance codes meaning the loan ended badly
BAD_ZB = {"02", "03", "09", "15"}
MIN_SITE_N = int(os.environ.get("FM_MIN_SITE_N", "200"))


def _num(v):
    if v is None:
        return np.nan
    v = v.strip()
    # the layout uses sentinel values for unknown; treating them as numbers would be a bug
    if not v or v in {"999", "9999", "99", "9", "NA", "Unknown"}:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def _members(paths):
    """Yield (label, text-stream) for every data file, recursing into nested archives.

    The Standard Dataset ships one zip per YEAR, and that zip contains one zip per QUARTER,
    which in turn holds the two text files. So a single level of unwrapping is not enough:
    without recursion the quarterly archives are handed back as members and parsed as text,
    which yields nothing and reports no error.

    Nothing is written to disk at any level. A quarterly archive is read into memory because
    ZipFile needs a seekable object, which costs a few hundred megabytes and avoids
    unpacking about 7 GB of performance data per year.
    """
    for p in paths:
        s = str(p)
        if s.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for nm in z.namelist():
                    if nm.endswith("/") or nm.startswith("__MACOSX"):
                        continue
                    if nm.lower().endswith(".zip"):
                        inner = io.BytesIO(z.read(nm))
                        with zipfile.ZipFile(inner) as z2:
                            for nm2 in z2.namelist():
                                if nm2.endswith("/") or nm2.startswith("__MACOSX"):
                                    continue
                                with z2.open(nm2) as raw2:
                                    yield nm2, io.TextIOWrapper(raw2, errors="replace")
                    else:
                        with z.open(nm) as raw:
                            yield nm, io.TextIOWrapper(raw, errors="replace")
        elif s.endswith(".gz"):
            yield p.name, gzip.open(p, "rt", errors="replace")
        else:
            yield p.name, open(p, errors="replace")


def _is_perf(name):
    """Tell the performance file from the origination file.

    The 2017 release names them `perf_2017Q1.txt` and `orig_2017Q1.txt`. Older releases use
    `historical_data_time_*`. Both spellings are matched, because a check that silently
    returns False for every performance file reads the origination data twice and finds no
    defaults at all.
    """
    n = name.lower()
    return "_time_" in n or pathlib.PurePath(n).name.startswith("perf")


def defaults_from_performance(paths, verbose=True):
    """LoanSequenceNumber -> defaulted, streamed straight out of the archives."""
    bad = set()
    for name, fh in _members(paths):
        if not _is_perf(name):
            continue
        for i, row in enumerate(csv.reader(fh, delimiter="|")):
            if len(row) < 9:
                continue
            lsn, dq, zb = row[0], row[3].strip(), row[8].strip()
            if zb in BAD_ZB:
                bad.add(lsn)
            elif dq.isdigit() and int(dq) >= 6:      # 6 monthly periods = 180+ days
                bad.add(lsn)
            if verbose and (i + 1) % 5_000_000 == 0:
                print(f"    {name}: {i + 1:,} rows, {len(bad):,} defaults", flush=True)
        if verbose:
            print(f"    {name}: done, {len(bad):,} defaults so far", flush=True)
    return bad


def build(partition="PropertyState", verbose=True):
    # one zip per year holds both files, so the same list feeds both passes
    archives = sorted(SRC.glob("*.zip")) or sorted(SRC.glob("historical_data*"))
    if not archives:
        raise FileNotFoundError(
            f"no origination files under {SRC}. Freddie Mac requires registration at "
            f"Clarity Data Intelligence; download the sample or a standard year and place "
            f"the files here.")
    if verbose:
        print(f"  {len(archives)} archive(s): {', '.join(p.name for p in archives)}")

    bad = defaults_from_performance(archives, verbose)
    if verbose:
        print(f"  {len(bad):,} loans ever defaulted")

    ix = {c: i for i, c in enumerate(ORIG_COLS)}
    rows, ys, keys = [], [], []
    for name, fh in _members(archives):
        if _is_perf(name):
            continue
        for row in csv.reader(fh, delimiter="|"):
            if len(row) < len(ORIG_COLS):
                continue
            lsn = row[ix["LoanSequenceNumber"]]
            if partition == "zip3":
                key = row[ix["PostalCode"]].strip()[:3]
            elif partition == "zip2":
                # The two-digit postal region sits between MSA and state in granularity, and
                # that gap is where Result 64's curve peaks. The other four partitions cluster
                # at f ~ 0.1 and f ~ 0.77, so without this the law is only tested at its ends,
                # where it predicts small lifts and is hardest to falsify.
                key = row[ix["PostalCode"]].strip()[:2]
            elif partition == "seller_state":
                key = f"{row[ix['SellerName']].strip()}|{row[ix['PropertyState']].strip()}"
            else:
                key = row[ix[partition]].strip()
            if not key or (key == "OTHER" and partition in {"SellerName", "seller_state"}):
                # 'OTHER' is not a site: it is every institution below the 1% UPB
                # disclosure floor pooled together, which is an artefact of the release
                # rather than an organisation that could join a federation.
                continue
            vec = [_num(row[ix[c]]) for c in NUMERIC]
            for c in CATEGORICAL:
                v = row[ix[c]].strip()
                vec.append(float(abs(hash(v)) % 997) if v else np.nan)
            rows.append(vec)
            ys.append(1 if lsn in bad else 0)
            keys.append(key)

    X = np.array(rows, dtype=np.float32)
    y = np.array(ys, dtype=np.int8)
    k = np.array(keys)
    counts = defaultdict(int)
    for v in k:
        counts[v] += 1
    names = sorted(s for s in counts
                   if counts[s] >= MIN_SITE_N and len(np.unique(y[k == s])) > 1)
    m = np.isin(k, names)
    X, y, k = X[m], y[m], k[m]
    g = np.array([names.index(s) for s in k])

    if verbose:
        cnt = np.bincount(g)
        d = X.shape[1]
        ev = np.array([min(np.bincount(y[g == i], minlength=2)) for i in range(len(names))])
        print(f"\nfreddiemac[{partition}]: {X.shape}, {len(names)} sites, "
              f"default rate {y.mean():.4f}")
        print(f"  site sizes {cnt.min():,}-{cnt.max():,} (median {int(np.median(cnt)):,})")
        print(f"  EPV at d={d}: {ev.min()/d:.2f}-{ev.max()/d:.2f} "
              f"(median {np.median(ev)/d:.2f})")
        print(f"  sites above EPV 10: {(ev/d >= 10).sum()} / {len(names)}")
    return X, y, g, names, NUMERIC + CATEGORICAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="PropertyState",
                    choices=["PropertyState", "MSA", "zip2", "zip3", "SellerName",
                             "seller_state"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, y, g, names, cols = build(a.partition)
    out = CACHE / f"{a.out or 'freddiemac_' + a.partition.lower()}.npz"
    np.savez_compressed(out, X=X, y=y, g=g,
                        names=np.array(names, dtype=np.str_),
                        cols=np.array(cols, dtype=np.str_))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
