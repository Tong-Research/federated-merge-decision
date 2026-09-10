"""Do the results survive partners a real federation would actually have?

Section 8 admits the limitation this script exists to close: every partner set in this project
is drawn uniformly at random from the available sites. No federation forms that way. They form
by region, by corporate parent, by vendor, by referral network -- and those partners are
*correlated*, both with each other and with the target.

Correlation could cut either way and the file has no evidence:

  * Partners like the target may be worth more (comparable populations, transferable
    coefficients) -- which would mean random sampling has been UNDERSTATING federation value.
  * Or they may be worth less (the same blind spots, nothing new to contribute) -- which would
    mean the benefit numbers throughout are optimistic.

Freddie Mac is the only collection here that can answer it, because its site labels carry real
containment: zip3 '010' lies inside zip2 '01', and the leading digit is a coarse US region band.
So a regional federation can be built from the labels alone, with no extra data.

Strategies
----------
random         partners drawn uniformly -- the existing behaviour, run here as the control so
               the comparison is within one script rather than across two runs
same_prefix    partners share the target's leading `--prefix` characters: a regional federation
cross_prefix   partners deliberately from other regions: the contrast that isolates correlation
size_similar   partners nearest the target in record count: a federation of peers
size_largest   the largest available partners: "join the big consortium"

The `size_*` strategies work on any collection. The `*_prefix` strategies need labels with
containment and are skipped, loudly, where the labels have none.

What this deliberately does NOT do
----------------------------------
It does not re-implement scoring. `_one` from site_rules is imported and called unchanged, so a
benefit measured here is comparable to every benefit already in results/ -- if the scoring
drifted, the comparison against the random control would be meaningless and the whole point is
that comparison. Output columns match site_rules exactly, plus `strategy`, so
trivial_baseline.py and crossing_point.py run on the output without special-casing.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed                       # noqa: E402
from matrix_loader import load_matrix                      # noqa: E402
from site_rules import _one, site_schema                   # noqa: E402

MAX_PARTNERS = int(os.environ.get("MAX_PARTNERS", "4"))
SUBSETS_PER_TARGET = int(os.environ.get("SUBSETS_PER_TARGET", "12"))
# -1 means every core. The runner gives a machine one job at a time, so taking the whole
# box is correct: pilot was measured at 98.9% CPU (one core of ten) on this script.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

STRATEGIES = ("random", "same_prefix", "cross_prefix", "size_similar", "size_largest",
              "schema_complement", "schema_similar", "large_complement")

# `schema_complement` and `schema_similar` added 2026-08-11, after Result 86.
#
# Result 17 measured that partner signals computable in advance recover at most 4% of the
# oracle gap, and that ceiling has been quoted since as a bound on partner selection in
# general. It is not. Every strategy it tested was a flavour of SIMILARITY -- same region,
# similar size, largest. Result 86 found that the axis which predicts merge benefit is
# COMPLEMENTARITY: on all four eICU pools, in every adequacy quartile, a target gains more
# from partners whose schemas overlap it LESS, and the effect is orthogonal to adequacy
# (rho with EPV -0.036, with n -0.024). In the best-resourced quartile, similar-schema
# partners leave the target at a coin flip (49% helped, -0.00001) while complementary ones
# give 60% and +0.0107.
#
# So the ceiling was measured on the wrong axis. `schema_complement` selects on the right
# one; `schema_similar` is its mirror and exists so the contrast is measured rather than
# assumed -- if both beat random by the same margin, the effect is "selecting on schema at
# all" rather than complementarity, and that must be visible.


# Matrices whose labels nest GEOGRAPHICALLY, and the prefix length that does it.
#
# This is an allowlist rather than a structural test, because a structural test cannot tell a
# real region from an alphabetical accident and both look identical to it:
#
#   propertystate at prefix=1 groups AK, AL, AR, AZ -- by first LETTER, not by geography.
#   MSA codes are assigned in alphabetical order of the metro name, so '10180' and '10420'
#   share a prefix for no spatial reason whatever.
#
# Either would pass "the prefix groups several labels together" and produce a confident
# "regional federation" result that is a random partition wearing a geographic label. Only
# ZIP prefixes actually nest in space: '010' lies inside '01', and the leading digit is a
# coarse US band. Add an entry here only when the containment is real.
CONTAINMENT = {
    "freddiemac_zip3": 2,     # zip3 within zip2
    "freddiemac_zip2": 1,     # zip2 within the leading-digit band
}



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


def has_containment(matrix: str, names: list[str], prefix: int) -> bool:
    """Is grouping these labels by `prefix` a real region, and does it actually group?"""
    if CONTAINMENT.get(matrix) != prefix:
        return False
    groups: dict[str, int] = {}
    for s in names:
        groups[str(s)[:prefix]] = groups.get(str(s)[:prefix], 0) + 1
    multi = sum(1 for v in groups.values() if v > 1)
    return multi >= 3 and len(groups) >= 2


def site_schemas(X, groups) -> list[set]:
    """Which variables each site measures, as sets of column indices.

    Taken from the site's full rows rather than a train split: a site can publish which
    variables it collects without publishing any record, so this stays inside the
    admissibility rule in section 3, and it does not move with the seed.
    """
    return [set(site_schema(X[ix])) for ix in groups]


def candidates(strategy: str, k: int, names: list[str], sizes: np.ndarray,
               prefix: int, schemas: list[set] | None = None) -> list[int]:
    """The pool `k` may draw partners from, under this strategy."""
    others = [j for j in range(len(names)) if j != k]
    if strategy == "large_complement":
        # C1 measured the two signals separately on eICU: complementarity beats random by
        # +48% and its mirror LOSES 14%, so the direction is real; size_largest beats random
        # by +74%, so size is the stronger single signal. R86 measured them as ORTHOGONAL
        # (rho(overlap, n) = -0.024), which is the whole reason to expect a combination to
        # beat either alone -- two independent signals, neither subsuming the other.
        #
        # Construction: take a generous pool of the largest partners, then keep the most
        # COMPLEMENTARY within it. Size first because it is the stronger signal and must not
        # be given away; complementarity as the tie-break it is orthogonal enough to be.
        if schemas is None:
            raise ValueError(f"{strategy} needs site schemas")
        wide = [others[i] for i in np.argsort(-sizes[others])[:max(MAX_PARTNERS * 6, 16)]]
        mine = schemas[k]
        jac = np.array([len(mine & schemas[j]) / max(len(mine | schemas[j]), 1) for j in wide])
        return [wide[i] for i in np.argsort(jac)[:max(MAX_PARTNERS * 3, 8)]]
    if strategy in ("schema_complement", "schema_similar"):
        if schemas is None:
            raise ValueError(f"{strategy} needs site schemas")
        mine = schemas[k]
        jac = np.array([len(mine & schemas[j]) / max(len(mine | schemas[j]), 1)
                        for j in others])
        # Ascending Jaccard = most complementary first. Same pool width as the size
        # strategies so the comparison is not confounded by how much choice each has.
        order = np.argsort(jac if strategy == "schema_complement" else -jac)
        return [others[i] for i in order[:max(MAX_PARTNERS * 3, 8)]]
    if strategy == "random":
        return others
    if strategy in ("same_prefix", "cross_prefix"):
        p = str(names[k])[:prefix]
        same = [j for j in others if str(names[j])[:prefix] == p]
        return same if strategy == "same_prefix" else [j for j in others if j not in same]
    if strategy == "size_similar":
        # Nearest in log size: "similar scale" must not mean "the biggest few" just because
        # the size distribution is heavy-tailed.
        d = np.abs(np.log(sizes[others] + 1) - np.log(sizes[k] + 1))
        return [others[i] for i in np.argsort(d)[:max(MAX_PARTNERS * 3, 8)]]
    if strategy == "size_largest":
        return [others[i] for i in np.argsort(-sizes[others])[:max(MAX_PARTNERS * 3, 8)]]
    raise ValueError(strategy)


def run(matrix: str, strategy: str, seeds: int, prefix: int,
        rng_seed: int = 0) -> list[dict]:
    X, y, g, names = load_matrix(matrix)
    names = [str(s) for s in names]
    groups = [np.flatnonzero(g == k) for k in range(len(names))]
    sizes = np.array([len(ix) for ix in groups], dtype=float)
    schemas = (site_schemas(X, groups)
               if strategy in ("schema_complement", "schema_similar",
                               "large_complement") else None)

    if strategy in ("same_prefix", "cross_prefix") and not has_containment(matrix, names, prefix):
        want = CONTAINMENT.get(matrix)
        why = (f"prefix={prefix} is not the geographic one ({want})" if want
               else "its labels do not nest geographically (see CONTAINMENT)")
        print(f"  SKIP {matrix} / {strategy}: {why}; e.g. {names[:4]}", file=sys.stderr)
        return []

    # Two phases, and the split is what makes parallelism safe here.
    #
    # Phase 1 draws the partner subsets. It is cheap and it MUST stay sequential, because
    # `rng` is one stream advanced across targets -- parallelising this loop would change
    # which subsets are drawn, and the exact reproduction of site_rules by the `random` arm
    # is the only reason the strategy contrast can be believed.
    #
    # Phase 2 evaluates them. It is where all the time goes (fitting and scoring one model
    # per site per seed) and every task is independent, so it parallelises without touching
    # a single drawn subset. Same tuples, same order, same numbers -- N times faster.
    rng = np.random.default_rng(rng_seed)
    tasks = []
    for k in range(len(names)):
        pool = candidates(strategy, k, names, sizes, prefix, schemas)
        if not pool:
            continue
        subsets = set()
        for _ in range(SUBSETS_PER_TARGET):
            m = int(rng.integers(1, min(MAX_PARTNERS, len(pool)) + 1))
            subsets.add(tuple(sorted(rng.choice(pool, size=m, replace=False))))
        for S in sorted(subsets):
            for seed in range(seeds):
                tasks.append((k, S, seed))
    if not tasks:
        return []

    # inner_max_num_threads=1: each worker fits small logistic models, so letting BLAS
    # spawn its own threads inside N processes oversubscribes the box and runs slower than
    # serial. One core per task, N tasks at once.
    flat, off = _pack_groups(groups)
    out = Parallel(n_jobs=N_JOBS, inner_max_num_threads=1, batch_size=64)(
        delayed(_one_packed)(X, y, flat, off, k, S, seed, names)
        for k, S, seed in tasks)
    return [dict(dataset=matrix, strategy=strategy, target=names[k], **r)
            for (k, _, _), r in zip(tasks, out) if r]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--prefix", type=int, default=2,
                    help="leading characters that define a region (zip3 '010' -> zip2 '01')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    allrows = []
    for s in [x.strip() for x in a.strategies.split(",") if x.strip()]:
        rows = run(a.matrix, s, a.seeds, a.prefix)
        print(f"  {a.matrix} / {s:13s} {len(rows):6,d} decisions"
              + (f"  helped {np.mean([r['helped'] for r in rows]):.3f}"
                 f"  mean benefit {np.mean([r['benefit'] for r in rows]):+.4f}" if rows else ""))
        allrows += rows

    if not allrows:
        print("no decisions produced", file=sys.stderr)
        return 1
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {a.out} ({len(allrows):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
