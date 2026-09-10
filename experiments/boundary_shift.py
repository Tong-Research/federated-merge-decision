"""Does heterogeneity move *where* the boundary sits, rather than how strong the rule is?

Result 60 withdrew the claim that heterogeneity explains ACSTravelTime's inversion, and left
an after-the-fact observation flagged explicitly as untested: the three most heterogeneous
pools had the three strongest adequacy effects. The candidate mechanism is geometric rather
than statistical. With the minimax boundary at R^2 n, a larger R puts the crossing at a
*smaller* n -- so the boundary moves down into the range of site sizes a pool actually holds,
more decisions fall on either side of it, and there is more for a rule to separate.

That mechanism makes a prediction about **where the cut sits**, which is a different and much
sharper claim than "the correlation is stronger". Correlations can strengthen for many
reasons; a cut that slides monotonically down as R rises is specific.

Testing it across the seven outcome pools would be a seven-point correlation -- thin, and
exactly the kind of reading that produced the withdrawn explanation. So the test is run
**within** pools instead: split each pool's decisions into I-squared terciles and fit a cut in
each. That holds the outcome, the states and the size distribution fixed, varies only measured
heterogeneity, and yields 21 cut estimates from thousands of decisions rather than 7 from
seven pools.

**Predictions, recorded before running.**
  1. Within a pool, the fitted EPV cut **falls** from the low-I-squared tercile to the high one.
     This is the whole claim; if cuts are flat or rise, the mechanism is wrong and the
     observation in Result 60 stays an observation.
  2. The effect is consistent in sign across at least five of the seven pools. A mechanism that
     holds in three and reverses in four is noise.
  3. The high-I-squared tercile has the larger lift over its own trivial rule, since its
     boundary sits inside the observed range.
  4. `ACSTravelTime` does **not** follow the pattern, because Result 58's inversion there is
     unexplained by anything in this file and heterogeneity has already been ruled out (R60).

If prediction 1 holds and 4 fails -- that is, if TravelTime does follow -- then the mechanism
may explain the inversion after all, and Result 58 gets a replacement account rather than an
open question.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent / "results"


def fit_cut(v, h):
    """Best 'merge if EPV < t' cut and its lift over that subset's own trivial rule."""
    m = np.isfinite(v) & np.isfinite(h)
    if m.sum() < 60 or len(np.unique(h[m])) < 2:
        return None
    v, h = v[m], h[m]
    triv = max(h.mean(), 1 - h.mean())
    best_t, best_a = np.nan, -1.0
    for t in np.quantile(v, np.linspace(0.05, 0.95, 49)):
        a = ((v < t) == h).mean()
        if a > best_a:
            best_t, best_a = float(t), float(a)
    return dict(cut=best_t, acc=best_a, lift=best_a - triv, n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+")
    a = ap.parse_args()

    print(f"\n{'pool':24s} {'I2 tercile':>11s} {'med I2':>7s} {'cut':>8s} "
          f"{'acc':>6s} {'lift':>7s} {'n':>6s}")
    print("-" * 78)
    verdicts = []
    for name in a.pools:
        path = R / name
        if not path.exists():
            print(f"  missing: {name}")
            continue
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "i2_median" not in rows[0]:
            continue
        g = lambda k: np.array([float(r.get(k, "nan") or "nan") for r in rows])
        i2, epv, h = g("i2_median"), g("epv_target"), g("helped")
        ok = np.isfinite(i2)
        if ok.sum() < 200:
            continue
        q = np.quantile(i2[ok], [0, 1 / 3, 2 / 3, 1.0])
        label = name.replace("site_rules_", "").replace(".csv", "")
        cuts = []
        for i, tag in enumerate(("low", "mid", "high")):
            sel = ok & (i2 >= q[i]) & (i2 <= q[i + 1])
            r = fit_cut(epv[sel], h[sel])
            if r is None:
                continue
            cuts.append((tag, float(np.median(i2[sel])), r))
            print(f"{label:24s} {tag:>11s} {np.median(i2[sel]):7.3f} {r['cut']:8.2f} "
                  f"{r['acc']:6.3f} {r['lift']:+7.3f} {r['n']:6d}")
        if len(cuts) == 3:
            lo, hi = cuts[0][2]["cut"], cuts[2][2]["cut"]
            falls = hi < lo
            verdicts.append((label, lo, hi, falls,
                             cuts[2][2]["lift"] > cuts[0][2]["lift"]))
            print(f"{'':24s} {'-> cut':>11s} {lo:7.2f} -> {hi:.2f}   "
                  f"{'FALLS (predicted)' if falls else 'rises or flat'}")
        print()

    if not verdicts:
        raise SystemExit("no pool produced three terciles")

    n_fall = sum(v[3] for v in verdicts)
    n_lift = sum(v[4] for v in verdicts)
    print("=" * 78)
    print(f"P1/P2  cut falls with I2 in {n_fall}/{len(verdicts)} pools -> "
          f"{'CONFIRMED' if n_fall >= 5 else 'NOT SUPPORTED — the mechanism is wrong'}")
    print(f"P3     high-I2 tercile has the larger lift in {n_lift}/{len(verdicts)} pools -> "
          f"{'CONFIRMED' if n_lift >= 5 else 'not supported'}")
    tt = [v for v in verdicts if "traveltime" in v[0].lower()]
    if tt:
        print(f"P4     ACSTravelTime cut {tt[0][1]:.2f} -> {tt[0][2]:.2f}: "
              f"{'FOLLOWS the pattern — Result 58 may have a replacement account' if tt[0][3] else 'does NOT follow, as predicted; its inversion stays unexplained'}")


if __name__ == "__main__":
    main()
