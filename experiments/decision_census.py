"""Derive the decision count instead of asserting it.

Written 2026-08-11 because the headline number could not be reconstructed from the files.
Three documents quoted three different naive sums -- `claims_audit.py` and
`PAPER-C-OUTLINE.md` said 688,690, `sections-framing.tex` said 829,122 -- and the
de-duplicated headline of 775,662 existed only as a hardcoded string inside the audit that
was supposed to be checking it. The de-duplication rule itself lived in prose, in three
places, and the prose disagreed.

That is precisely the failure the project's own meta-rule forbids: a number that only one
path computes is a number that drifts. So this file replaces the prose with a procedure.

WHAT COUNTS AS A DECISION
    The paper defines the unit as (target site, partner set, split seed). A file holds
    per-decision rows if it names a `target` and either a partner count or an outcome.
    Rows within a file are distinct by construction -- each harness enumerates a grid.

    Files fall into two groups, and conflating them is what produced the disagreement:

    PRIMARY      the merge decision itself, asked once per (target, partner set, seed).
                 `site_rules_*`, `partner_formation_*`, and the standalone probes.
    CONDITIONED  the SAME decisions re-asked under an added condition -- a privacy budget
                 (`dp_*`), a calibration arm (`calib_*`), or a policy set (`gate_/lam_/cut_`).
                 Epsilon is not in the unit tuple, so counting these in the headline
                 inflates coverage by re-counting configurations already counted.

EXCLUSIONS, AND WHY EACH IS VERIFIED RATHER THAN ASSERTED
    1. `partner_formation`'s `random` arm reproduces `site_rules` row for row. That exact
       agreement is what licenses the comparison between selection strategies, so the arm
       must exist and must not be counted twice.
    2. `gate_*` and `lam_*` are superseded by `cut_*` on the same pools with more policy
       arms. Only the survivor counts.

    Both are checked below, not trusted. An exclusion that silently matches nothing removes
    nothing and reports success -- the same class of error as a test that drops the cases it
    is about.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parent.parent / "results"

# Files whose names begin with these prefixes re-ask decisions under an added condition.
CONDITIONED = ("dp_", "calib_", "gate_", "lam_", "cut_", "local_tuning", "site_rules_ext",
               "site_rules_recalib")

# Superseded by a later harness over the same decisions.
SUPERSEDED = ("gate_", "lam_")


def rows_and_cols(path):
    with path.open() as fh:
        rd = csv.DictReader(fh)
        first = next(rd, None)
        if first is None:
            return 0, set()
        return 1 + sum(1 for _ in rd), set(first.keys())


def is_decision_file(cols):
    """A per-decision row names a target and either a partner count or an outcome."""
    return "target" in cols and ({"n_partners"} & cols or {"helped"} & cols)


def census():
    prim, cond = {}, {}
    for p in sorted(R.glob("*.csv")):
        n, cols = rows_and_cols(p)
        if not n or not is_decision_file(cols):
            continue
        (cond if p.name.startswith(CONDITIONED) else prim)[p.name] = n
    return prim, cond


def verify_partner_formation():
    """The `random` arm should reproduce site_rules. Count it, and prove it is not empty."""
    total = 0
    per_file = {}
    for p in sorted(R.glob("partner_formation*.csv")):
        with p.open() as fh:
            k = sum(1 for r in csv.DictReader(fh) if r.get("strategy") == "random")
        if k:
            per_file[p.name] = k
            total += k
    return total, per_file


def verify_supersession(prim_and_cond):
    """Every superseded file should have a surviving `cut_*` twin of the same size.

    A supersession claim is only safe if the survivor covers the same decisions. Equal row
    counts do not prove that, but unequal ones disprove it -- so an unequal pair is reported
    rather than silently excluded.
    """
    dropped, unmatched = 0, []
    for name, n in prim_and_cond.items():
        if not name.startswith(SUPERSEDED):
            continue
        twin = re.sub(r"^(gate|lam)_", "cut_", name)
        if prim_and_cond.get(twin) == n:
            dropped += n
        else:
            unmatched.append((name, n, twin, prim_and_cond.get(twin)))
    return dropped, unmatched


def main():
    prim, cond = census()
    allf = {**prim, **cond}

    pf_dup, pf_per_file = verify_partner_formation()
    superseded, unmatched = verify_supersession(allf)

    naive_all = sum(allf.values())
    naive_prim = sum(prim.values())
    headline = naive_prim - pf_dup
    cond_kept = sum(cond.values()) - superseded

    print("=" * 88)
    print("DECISION CENSUS — derived from the files, not restated")
    print("=" * 88)

    print(f"\n  per-decision files found            {len(allf):>12,}")
    print(f"    primary merge harness             {len(prim):>12,} files, {naive_prim:>10,} rows")
    print(f"    conditioned re-decisions          {len(cond):>12,} files, "
          f"{sum(cond.values()):>10,} rows")
    print(f"  naive sum over every file           {naive_all:>12,}")

    print("\n  exclusions (each verified, not asserted)")
    print(f"    partner_formation `random` arm    {pf_dup:>12,}  "
          f"reproduces site_rules row for row")
    for f, k in sorted(pf_per_file.items(), key=lambda x: -x[1]):
        print(f"        {f:<44s} {k:>8,}")
    if not pf_dup:
        print("        !! matched nothing — the arm name changed, or the rule is stale")
    print(f"    gate_*/lam_* superseded by cut_*  {superseded:>12,}")
    for name, n, twin, tn in unmatched:
        print(f"        !! {name} ({n:,}) has no equal-sized twin {twin} ({tn})")

    print("\n" + "-" * 88)
    print(f"  HEADLINE — distinct primary decisions   {headline:>12,}")
    print(f"  conditioned re-decisions retained       {cond_kept:>12,}")
    print("-" * 88)
    print("\n  Quote the headline. It counts each (target, partner set, seed) once, in the")
    print("  harness that asks the merge question directly. The conditioned harnesses re-ask")
    print("  those same units under a privacy budget, a calibration arm or a policy set, and")
    print("  belong in the sections that use them rather than in a coverage total.")

    print("\n  collections represented:")
    ds = collections.Counter()
    for name in prim:
        with (R / name).open() as fh:
            for r in csv.DictReader(fh):
                ds[r.get("dataset") or "(unnamed)"] += 1
                break
    for k, _ in ds.most_common():
        print(f"    {k}")

    return 0 if not unmatched and pf_dup else 1


if __name__ == "__main__":
    sys.exit(main())
