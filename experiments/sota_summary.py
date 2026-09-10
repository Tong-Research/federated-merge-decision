"""Compare aggregation algorithms on identical pools: FedAvg, FedProx, SCAFFOLD, pFedMe.

The axis here is BASELINES.md's first question -- how is the shared model built -- so every
mode runs the same policies, the same oracles and the same floor, and differs only in the local
update inside `fedavg_iterative.fed_train`. A difference between columns is therefore a
difference in the aggregation algorithm and nothing else.

Two rules inherited from Results 103-110, both of which changed a conclusion when first
applied:

  * Report ABOVE THE ATTAINABLE-GAIN FLOOR. Every share divides by (oracle - never_merge),
    which spans two orders of magnitude across pools. Averaging over a pool whose denominator
    is below the smallest effect the paper reports inflates every mode's number.
  * Print the ABSOLUTE AUPRC beside every share, because a share is a ratio to a quantity the
    reader cannot otherwise see.

Modes are compared only on pools ALL of them have run, so a mode is never advantaged by
having landed on an easier subset.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parent.parent / "results"
MIN_GAIN_AP = 0.002

# (tags, display name). A mode can have SEVERAL file tags because the comparison was extended
# in a second batch -- cut_fa2_* holds the same FedAvg configuration as cut_fa_*, on four more
# pools. Listing both here is what makes the extension visible to this script; without it the
# new files would land in results/ and be silently ignored, which is the quieter version of the
# glob-rot that had fig_policy depicting a different run from the table beside it.
MODES = [(("fa", "fa2"), "FedAvg"), (("fx", "fx2"), "FedProx"),
         (("sc", "sc2"), "SCAFFOLD"), (("pm",), "pFedMe"),
         (("pl1", "pl1b"), "pFedMe(lam=1)")]
# What a site actually deploys, plus the ceiling each mode implies.
ARMS = ["never_merge", "always_merge", "blend", "oracle_lambda"]


def load():
    out: dict[str, dict[str, dict]] = {}
    for tags, name in MODES:
        for tag in tags:
          for f in sorted(glob.glob(str(R / f"cut_{tag}_*.csv"))):
            pool = os.path.basename(f)[len(f"cut_{tag}_"):-4]
            d = pd.read_csv(f)
            nev = d["ap_never_merge"].mean()
            den = d["ap_oracle_lambda"].mean() - nev
            rec = {"_n": len(d), "_den": den, "_nev": nev}
            for a in ARMS:
                if f"ap_{a}" in d.columns:
                    rec[a] = d[f"ap_{a}"].mean()
            if "ap_pfedme_personal" in d.columns and d["ap_pfedme_personal"].notna().any():
                rec["pfedme_personal"] = d["ap_pfedme_personal"].mean()
            out.setdefault(name, {})[pool] = rec
    return out


def main() -> int:
    data = load()
    if not data:
        print("no cut_{fa,fx,sc,pm}_*.csv yet")
        return 0

    present = {m: set(v) for m, v in data.items()}
    common = set.intersection(*present.values()) if present else set()
    print("=" * 92)
    print("AGGREGATION AXIS — how the shared model is built")
    print("=" * 92)
    for m, pools in present.items():
        print(f"  {m:<10} {len(pools)} pools: {', '.join(sorted(pools))}")
    if not common:
        print("\nNo pool has been run by every mode yet; nothing comparable to report.")
        return 0
    print(f"\n  compared on the {len(common)} pool(s) every mode has run: {sorted(common)}")
    # COMPOSITION OF THE INTERSECTION. Comparing only on pools every mode has run stops a mode
    # being flattered by an easier subset -- but it cannot stop the intersection ITSELF being
    # biased. On 2026-08-24 the lambda=50 arm had finished exactly two pools, both of them ones
    # where merging helps, so the intersection contained no harmful pool and pFedMe appeared to
    # beat adaptive shrinkage 45.0 to 33.3. On the full four-pool set it loses, -23.4 to 15.8.
    # A mid-run intersection is a biased sample of pools, not a fair subset of them.
    harmful = [pl for pl in sorted(common)
               if any(data[m][pl].get("always_merge", 0) < data[m][pl]["_nev"] for m in data)]
    print(f"    of which merging HARMS on {len(harmful)}: {harmful or 'NONE'}")
    if not harmful:
        print("    *** WARNING: no harmful pool in the comparison set. Any policy that gains")
        print("        on the pools where merging helps will look dominant here. Do not")
        print("        compare downside behaviour from this set; wait for a harmful pool.")

    above = sorted(p for p in common
                   if all(data[m][p]["_den"] >= MIN_GAIN_AP for m in data))
    below = sorted(set(common) - set(above))
    if below:
        print(f"  EXCLUDED, attainable gain below {MIN_GAIN_AP} AUPRC: {below}")
    if not above:
        print("\n  Every common pool is below the floor. No share here can rank the modes.")
        return 0

    # PROVISIONAL BELOW EIGHT. Three claims moved on 2026-08-23/24 when pools arrived: the
    # Result 108 retraction, the biased-intersection episode, and "pFedMe wins the helping
    # pools", which was true at four pools and a tie at seven. Four pools has been enough to be
    # wrong three times, twice in the direction that flattered us. The banner is unconditional
    # for that reason -- a provisional finding that happens to be favourable reads as settled.
    if len(above) < 8:
        print(f"\n  *** {len(above)} above-floor pools. Any PER-POOL pattern below is")
        print("      PROVISIONAL, whichever way it points. Three claims already moved this")
        print("      week when pools arrived, two of them in our favour before they moved. ***")

    print(f"\n  {'mode':<10}{'merged AP':>11}{'blend AP':>10}{'oracle AP':>11}"
          f"{'attainable':>12}{'blend share':>13}")
    print("  " + "-" * 78)
    for _tags, name in MODES:
        if name not in data:
            continue
        rows = [data[name][p] for p in above]
        alw = sum(r["always_merge"] for r in rows) / len(rows)
        bl = sum(r["blend"] for r in rows) / len(rows)
        orc = sum(r["oracle_lambda"] for r in rows) / len(rows)
        nev = sum(r["_nev"] for r in rows) / len(rows)
        share = 100.0 * (bl - nev) / (orc - nev) if orc > nev else float("nan")
        print(f"  {name:<10}{alw:>11.4f}{bl:>10.4f}{orc:>11.4f}{orc - nev:>12.5f}"
              f"{share:>12.1f}%")
        if "pfedme_personal" in rows[0]:
            pp = sum(r["pfedme_personal"] for r in rows) / len(rows)
            print(f"  {'  personal':<10}{pp:>11.4f}{'':>10}{'':>11}{'':>12}"
                  f"   <- pFedMe's own personalised model")

    print("\n" + "=" * 92)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 92)
    ref = "FedAvg"
    if ref in data:
        def merged_share(name):
            rows = [data[name][p] for p in above]
            nev = sum(r["_nev"] for r in rows) / len(rows)
            orc = sum(r["oracle_lambda"] for r in rows) / len(rows)
            alw = sum(r["always_merge"] for r in rows) / len(rows)
            return 100.0 * (alw - nev) / (orc - nev)
        base = merged_share(ref)
        for name, rule in (("FedProx", "within 1 share point of FedAvg"),
                           ("SCAFFOLD", "does NOT beat FedAvg")):
            if name not in data:
                print(f"\n  {name}: not yet run")
                continue
            d = merged_share(name) - base
            # ABSOLUTE difference too, and judged against the same 0.002 floor the panel above
            # applies to pools. A share difference between MODES is a ratio to each mode's own
            # attainable gain, and those differ -- SCAFFOLD's is smaller, which inflates its
            # share at equal absolute performance. Scoring a prediction on the share alone
            # while the panel above warns against exactly that would be inconsistent.
            def abs_merged(nm):
                rows = [data[nm][pl] for pl in above]
                return sum(r["always_merge"] for r in rows) / len(rows)
            da = abs_merged(name) - abs_merged(ref)
            print(f"\n  {name} merged-model share {merged_share(name):.1f} vs FedAvg {base:.1f} "
                  f"({d:+.1f})   [{rule}]")
            print(f"    absolute AP {abs_merged(name):.4f} vs {abs_merged(ref):.4f} ({da:+.5f})"
                  f"   attainable {name} vs FedAvg: "
                  f"{sum(data[name][pl]['_den'] for pl in above)/len(above):.5f} vs "
                  f"{sum(data[ref][pl]['_den'] for pl in above)/len(above):.5f}")
            if abs(da) < MIN_GAIN_AP:
                print(f"    -> IMMATERIAL: the absolute difference is {abs(da):.5f} AP, below the "
                      f"{MIN_GAIN_AP} floor.")
                print(f"       The share gap of {d:+.1f} is mostly a denominator effect. Neither "
                      f"'{rule}' nor its negation is established here.")
            elif name == "FedProx":
                print(f"    -> {'HOLDS' if abs(d) <= 1.0 else 'FAILS: differs by more than 1 point'}")
            else:
                print(f"    -> {'HOLDS' if d <= 0 else 'FAILS: SCAFFOLD beats FedAvg'}")
    if "pFedMe" in data:
        rows = [data["pFedMe"][p] for p in above]
        if "pfedme_personal" in rows[0]:
            # PER POOL. The prediction was "beats its own global model on EVERY pool", and an
            # earlier version of this block tested the mean instead -- the same substitution
            # that produced the Result 108 retraction, where a per-pool claim was confirmed
            # from an average that one pool carried.
            print("\n  pFedMe personalised vs its own global, per pool "
                  "(prediction: beats it on EVERY pool)")
            wins = 0
            for pool in above:
                r = data["pFedMe"][pool]
                d = r["pfedme_personal"] - r["always_merge"]
                wins += d > 0
                print(f"    {pool:<26}{r['always_merge']:>9.4f}{r['pfedme_personal']:>10.4f}"
                      f"{d:>+10.4f}   {100 * d / r['_den']:>6.1f}% of attainable")
            if wins == len(above):
                print(f"    -> HOLDS, {wins} of {len(above)}")
            else:
                print(f"    -> FAILS: beats its global on {wins} of {len(above)}.")
                print("       Before concluding the implementation is wrong, check lambda. At")
                print("       lam=15 the inner step is lr/(1+lam) ~ 0.03 over 5 epochs, so the")
                print("       personalised model barely leaves the global one and the method is")
                print("       under-powered BY CONFIGURATION. Our own blend selects its weight")
                print("       on held-out data; pFedMe here does not, which is not a fair")
                print("       comparison until lambda is selected the same way.")

    print("\n  Prediction 4 (shrinkage stays non-negative above the floor under every mode):")
    for _tags, name in MODES:
        if name not in data:
            continue
        neg = [p for p in above if data[name][p]["blend"] < data[name][p]["_nev"]]
        print(f"    {name:<10} negative on {len(neg)} of {len(above)}"
              + (f"  {neg}  <-- WITHDRAWAL CONDITION" if neg else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
