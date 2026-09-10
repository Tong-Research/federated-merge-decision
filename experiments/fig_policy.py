"""The figure for the policy comparison: return against downside.

Section 6's argument is not that adaptive shrinkage scores highest. It is that every
alternative with a comparable return has a regime that destroys it, and shrinkage does not.
Two tables carry that, and a reader has to hold both in mind at once to see it -- which is
exactly the job a figure should take over.

Panel A plots every policy's score on every pool, so the catastrophic tails are visible as
points rather than inferred from a "worst pool" column. The eye lands on the left-hand
outliers, which is correct: they are the finding.

Panel B is the argument in one geometry. Mean on one axis, worst pool on the other. A policy
that is both high-return and safe sits in the upper right, and only one does. The diagonal is
not a fit; it is the line where a policy's worst pool equals its mean, which nothing reaches
and which marks how much of the return each policy pays for in its bad regime.

Reads the eleven files named in ELEVEN below -- not a glob -- and asserts their combined
decision count against the one Section 6 reports, so figure and text cannot disagree. It used
to glob, which is how it came to depict a different run from the table beside it.
"""
from __future__ import annotations

import csv
import glob
import os
import pathlib
import sys

import math

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# papers/common/, found by searching upward: a fixed hop count is right in the
# monorepo and wrong in an extracted release
for _d in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
    if (_d / "common" / "pgfemit.py").exists():
        sys.path.insert(0, str(_d / "common")); break
else:
    raise SystemExit("could not find common/pgfemit.py above " + __file__)

from pgfemit import Fig, declutter, fmt                     # noqa: E402

R = HERE.parent / "results"

POLICIES = [
    ("blend",        "adaptive shrinkage", "hgblue"),
    ("local_cv",     "local validation",   "hggreen"),
    ("gate_bestcut", "gate, best cut",     "hgpurple"),
    ("epv_gate",     "gate, fixed cut",    "hgamber"),
    ("fix_075",      "fixed 0.75",         "hgred"),
    ("fix_050",      "fixed 0.50",         "hgslate"),
    ("always_merge", "always merge",       "hggrey"),
]
# Pools whose mean benefit is below -0.002, marked so the reader can see that the tails all
# come from the same few collections rather than being spread across the corpus.
#
# DERIVED, not listed. The literal used here until 2026-08-23 named the tags of the Aug 11
# low-seed run -- "acs-nat", "ladder10k", "ladder5k", "mimic4", "traveltime" -- and none of
# them match the fifteen-seed files the table now uses, so every pool silently tested as
# not-harmful and the figure would have drawn none. The membership is unchanged when derived
# (the same five pools, renamed); it is the names that rotted.
HARM_CUT = -0.002


def _harmful():
    out = set()
    for name in ELEVEN:
        rows = list(csv.DictReader(open(str(R / name))))
        b = np.mean([float(x["ap_always_merge"]) - float(x["ap_never_merge"]) for x in rows])
        if b < HARM_CUT:
            out.add(os.path.basename(name)[4:-4])
    return out

# The eleven pools section 6 reports, named explicitly.
#
# This was `glob("cut_*.csv")` until 2026-08-23, on the reasoning that figure and table then
# could not disagree. That held when eleven files matched. Thirty-seven match now -- the DT,
# RB, HS and FA batches all landed in the same directory -- so the glob silently redefined
# what "the eleven-pool run" meant, and regenerating the figure today would have produced a
# different set from the one the caption describes. An explicit list cannot rot that way, and
# the assertion below fails loudly if it ever stops matching the published decision count.
ELEVEN = [f"cut_hs_{p}.csv" for p in (
    "acs_full_income", "acs_ladder_10000", "acs_ladder_5000", "acs_span_income",
    "acs_span_traveltime", "eicu_sites", "freddiemac_propertystate", "freddiemac_zip2",
    "lendingclub", "mimic_sites_unit_x_system")] + ["cut_mimic4_hs.csv"]
PUBLISHED_DECISIONS = 101_865


def load():
    per = {k: {} for k, _, _ in POLICIES}
    seen = 0
    for name in ELEVEN:
        f = str(R / name)
        rows = list(csv.DictReader(open(f)))
        seen += len(rows)
        tag = os.path.basename(f)[4:-4]
        base = np.mean([float(x["ap_never_merge"]) for x in rows])
        gain = np.mean([float(x["ap_oracle_lambda"]) for x in rows]) - base
        if gain <= 1e-9:
            continue
        for key, _, _ in POLICIES:
            per[key][tag] = 100.0 * (np.mean([float(x[f"ap_{key}"]) for x in rows])
                                     - base) / gain
    assert seen == PUBLISHED_DECISIONS, (
        f"pool set drifted: {seen:,} decisions, section 6 reports "
        f"{PUBLISHED_DECISIONS:,}. Do not regenerate the figure until this agrees.")
    return per


CLIP = -120.0          # panel A's left edge; clipped points are labelled with their value
LINTHRESH = 20.0       # panel B's symmetric-log threshold


def sym(y):
    """Symmetric log, matching matplotlib's symlog with linscale=1.

    pgfplots has no symlog, so the transform is applied here and the axis is
    given explicit ticks at the transformed positions carrying the original
    labels.  A policy losing 455% of the attainable gain and one losing 5%
    cannot share a linear axis without hiding the difference between the safe
    policies, which is the comparison panel B exists to make.
    """
    if abs(y) <= LINTHRESH:
        return y / LINTHRESH
    return math.copysign(1.0 + math.log10(abs(y) / LINTHRESH), y)


def main() -> int:
    per = load()
    harmful = _harmful()
    pools = sorted(next(iter(per.values())))
    fig = Fig()

    # ---------- Panel A: every pool, every policy --------------------------
    ax = fig.axis(
        "A", "hg axis",
        width=r"0.56\textwidth", height="6.4cm",
        xlabel="share of the attainable gain (\\%), one point per pool",
        title=r"\textbf{A}\quad Every policy on every pool",
        xmin=-158, xmax=72, enlarge_x_limits=False,
        ymin=-0.62, ymax=len(POLICIES) - 1 + 1.00, enlarge_y_limits=False,
        y_dir="reverse",
        ytick=",".join(str(i) for i in range(len(POLICIES))),
        yticklabels=",".join("{%s}" % lab for _, lab, _ in POLICIES),
        y_tick_label_style="font=\\small, align=right",
        legend_style=("font=\\scriptsize, draw=none, fill=none, "
                      "at={(0.5,-0.26)}, anchor=north, legend columns=2, "
                      "column sep=1.4em"))
    ax.raw(r"\draw[hg rule] (axis cs:0,-0.62) -- (axis cs:0,"
           f"{len(POLICIES) - 1 + 0.62});")

    for i, (key, label, colour) in enumerate(POLICIES):
        vals = np.array([per[key][p] for p in pools])
        harm = np.array([p in harmful for p in pools])
        y = np.full(len(vals), i, float) + np.linspace(-0.16, 0.16, len(vals))
        shown = np.clip(vals, CLIP, None)
        ax.plot(f"{colour}, only marks, mark=*, mark size=1.5pt, "
                f"mark options={{draw=none, fill={colour}, fill opacity=0.75}}",
                shown[~harm], y[~harm], digits=2, forget=True)
        ax.plot(f"{colour}, only marks, mark=o, mark size=1.9pt, "
                f"mark options={{line width=0.6pt}}",
                shown[harm], y[harm], digits=2, forget=True)
        ax.raw(f"\\draw[{colour}, line width=1.1pt] "
               f"(axis cs:{fmt(vals.mean(), 2)},{i - 0.30}) -- "
               f"(axis cs:{fmt(vals.mean(), 2)},{i + 0.30});")
        # A clipped point is labelled with its true value, so the axis limit
        # never hides a magnitude -- the -455 is the most informative number
        # in the panel.  Stacked on the row, not on the jittered y.
        clipped = sorted(v for v in vals if v < CLIP)
        for j, v in enumerate(clipped):
            dy = 8.2 * (j - (len(clipped) - 1) / 2)
            ax.raw(f"\\node[font=\\tiny, text={colour}, anchor=east, "
                   f"xshift=-3pt, yshift={dy:.1f}pt] "
                   f"at (axis cs:{CLIP:.0f},{i}) {{{v:.0f}}};")

    ax.raw(r"\addlegendimage{hgslate, only marks, mark=o, mark size=1.9pt, "
           r"mark options={line width=0.6pt}}")
    ax.raw(r"\addlegendentry{pool where merging harms}")
    ax.raw(r"\addlegendimage{hgslate, line width=1.1pt, mark=none}")
    ax.raw(r"\addlegendentry{mean}")

    # ---------- Panel B: return against downside ---------------------------
    pts = []
    for key, label, colour in POLICIES:
        v = np.array([per[key][p] for p in pools])
        pts.append((key, label, colour, float(v.mean()), float(v.min())))

    ylo = min(sym(w) for *_, w in pts) - 0.25
    yhi = sym(95)
    ticks = [50, 0, -20, -50, -150, -450]

    bx = fig.right_of(
        "B", "hg axis", "A", gap="1.9cm",
        width=r"0.40\textwidth", height="6.4cm",
        xlabel="mean across pools (\\%)",
        ylabel="worst pool (\\%), symmetric log",
        title=r"\textbf{B}\quad Return against downside",
        xmin=-132, xmax=78, enlarge_x_limits=False,
        ymin=fmt(ylo, 3), ymax=fmt(yhi, 3), enlarge_y_limits=False,
        ytick=",".join(fmt(sym(v), 3) for v in ticks),
        yticklabels=",".join("{%+d}" % v if v else "{0}" for v in ticks))

    # the claim is a half-plane -- never worse than keeping the local model
    bx.raw(f"\\fill[hggreen, opacity=0.10] (axis cs:-132,0) rectangle "
           f"(axis cs:78,{fmt(yhi, 3)});")
    bx.raw(r"\draw[hg rule] (axis cs:-132,0) -- (axis cs:78,0);")
    bx.raw(r"\node[font=\scriptsize, text=hggreen, anchor=north west, align=left] "
           f"at (axis cs:-128,{fmt(yhi, 3)}) "
           r"{never worse than\\keeping the local model};")

    for key, label, colour, m, w in pts:
        big = key == "blend"
        style = (f"{colour}, only marks, mark=*, "
                 f"mark size={'3.0' if big else '2.1'}pt, "
                 f"mark options={{fill={colour}, draw="
                 f"{'hgslate' if big else 'white'}, line width=0.6pt}}")
        bx.plot(style, [m], [sym(w)], digits=3, forget=True)

    # placed by rule, not by hand: the previous fixed offsets were re-tuned
    # twice after the points moved, and collided both times
    lab = [(m, sym(w), label) for _, label, _, m, w in pts]
    for (m, sw, label), (anchor, dx, dy) in zip(
            lab, declutter(lab, (-132, 78), (ylo, yhi), box=(143, 137),
                           font_pt=8.0, radius=8.0)):
        colour = next(c for k, l, c, _, _ in pts if l == label)
        weight = r"\bfseries" if label == "adaptive shrinkage" else ""
        bx.raw(f"\\node[font=\\scriptsize{weight}, text={colour}, anchor={anchor}, "
               f"xshift={dx}pt, yshift={dy}pt] at (axis cs:{fmt(m, 3)},{fmt(sw, 3)}) "
               f"{{{label}}};")

    out = R.parent / "figures" / "fig_policy.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.render(__file__))
    print(f"wrote {out}\n")

    print(f"{'policy':22s} {'mean':>7s} {'median':>7s} {'worst':>8s} {'pools<0':>8s}")
    for key, label, _ in POLICIES:
        v = np.array([per[key][p] for p in pools])
        print(f"  {label:20s} {v.mean():7.1f} {np.median(v):7.1f} {v.min():8.1f} "
              f"{int((v < 0).sum()):8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
