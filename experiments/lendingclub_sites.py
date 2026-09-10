"""Lending Club loans, US states as sites: a third domain whose site sizes we did not choose.

Result 55 is the reason this dataset exists in the project. It showed that our ACS
conclusions depend on a size distribution *we* selected -- capping states at 1,200 made
merging look universally good, natural sizes made it look mostly bad, and a log-uniform draw
made the rule look useful. ACS cannot resolve that, because every ACS size distribution is
one of our decisions.

Lending Club's is not. The 2007-2011 file holds 42,538 loans across 51 states, and the state
sizes run from California's 7,429 to Maine's 3 -- a spread of roughly 2,400x that arrived
with the data. Whatever the rule does here, nobody chose the distribution to make it do that.

Outcome: `loan_status` collapsed to charged-off versus fully paid, which is the standard
binary target for this dataset and runs about 13% -- close enough to ICU mortality's 9% that
EPV means something comparable, and far from ACS income's 35%.

Features are the numeric application-time fields only. Anything recorded *after* origination
(recoveries, total payments, last payment date) is excluded, since it leaks the outcome --
the single most common error in published analyses of this dataset.
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("MIMIC_CACHE", Path.home() / ".cache/phd-matrices"))
LC_DIR = CACHE / "lendingclub"
SRC = LC_DIR / "LoanStats3a.csv"
MIN_SITE_N = int(os.environ.get("LC_MIN_SITE_N", "50"))

# Application-time numeric fields. Deliberately conservative: every one of these is known to
# the lender before the loan is funded, so none can leak repayment behaviour.
FEATURES = [
    "loan_amnt", "funded_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "fico_range_low", "fico_range_high",
    "pub_rec_bankruptcies", "collections_12_mths_ex_med", "acc_now_delinq",
]
# Encoded separately because they carry the credit signal and are ordinal, not numeric.
GRADE = "ABCDEFG"


def _num(v):
    if v is None:
        return np.nan
    v = v.strip().rstrip("%")
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def build(verbose=True, sources=None):
    """Parse one or many LoanStats files into a single state-partitioned matrix.

    `sources=None` reads the 2007-2011 file alone (harm fraction 0.023, Result 63). Passing
    every LoanStats file instead is the point of Result 63's follow-up: across all years the
    large states hold hundreds of thousands of loans, which is the first NATURAL size
    distribution in this project that could reach the harm regime.
    """
    srcs = sorted(sources) if sources else [SRC]
    missing = [s for s in srcs if not pathlib.Path(s).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} — run experiments/fetch_lendingclub_full.sh")
    rows, states, ys = [], [], []
    for src in srcs:
        with open(src, errors="replace") as fh:
            first = fh.readline()
            if "," in first and first.count('"') > 4:
                fh.seek(0)                          # some files have no notes banner
            for r in csv.DictReader(fh):
                status = (r.get("loan_status") or "").replace(
                    "Does not meet the credit policy. Status:", "").strip()
                if status not in ("Fully Paid", "Charged Off"):
                    continue                           # in-flight loans have no outcome yet
                st = (r.get("addr_state") or "").strip()
                if len(st) != 2:
                    continue
                vec = [_num(r.get(f)) for f in FEATURES]
                g = (r.get("grade") or "").strip()
                vec.append(float(GRADE.index(g)) if g in GRADE else np.nan)
                term = (r.get("term") or "")
                vec.append(60.0 if "60" in term else 36.0 if "36" in term else np.nan)
                vec.append(_num(r.get("int_rate")))
                rows.append(vec)
                states.append(st)
                ys.append(1 if status == "Charged Off" else 0)

    X = np.array(rows, float)
    y = np.array(ys, int)
    st = np.array(states)
    keep_states = [s for s in sorted(set(st))
                   if (st == s).sum() >= MIN_SITE_N and len(np.unique(y[st == s])) > 1]
    m = np.isin(st, keep_states)
    X, y, st = X[m], y[m], st[m]
    names = sorted(set(st))
    g = np.array([names.index(s) for s in st])

    if verbose:
        cnt = np.bincount(g)
        ev = np.array([min(np.bincount(y[g == k])) for k in range(len(names))])
        d = X.shape[1]
        print(f"lendingclub: {X.shape}, {len(names)} states, default rate {y.mean():.3f}")
        print(f"  site sizes {cnt.min()}-{cnt.max()} (median {int(np.median(cnt))})")
        print(f"  EPV at d={d}: {ev.min()/d:.1f}-{ev.max()/d:.1f} "
              f"(median {np.median(ev)/d:.1f})")
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
            n = int(((ev / d >= lo) & (ev / d < hi)).sum())
            if n:
                print(f"    EPV {lo}-{hi if hi < 1e9 else 999}: {n} states")
    return X, y, g, names, FEATURES + ["grade", "term", "int_rate"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every LoanStats file, not just 3a")
    ap.add_argument("--out", default="lendingclub")
    a = ap.parse_args()
    srcs = sorted(str(p) for p in LC_DIR.glob("LoanStats*.csv")) if a.all else None
    if a.all:
        print(f"reading {len(srcs)} files")
    X, y, g, names, cols = build(sources=srcs)
    out = CACHE / f"{a.out}.npz"
    np.savez_compressed(out, X=X, y=y, g=g,
                        names=np.array(names, dtype=np.str_),
                        cols=np.array(cols, dtype=np.str_))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
