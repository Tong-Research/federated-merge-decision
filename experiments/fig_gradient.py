"""Section 5's figure: the gradient, and the three controls that bound it.

Section 5 carries four subsections and no picture. Its argument is not one number but a
contrast between what the gradient survives and what it does not, and that is a shape rather
than a table.

Two panels.

LEFT -- the gradient across eleven pools, as a caterpillar of rank correlations. Sorted, with
the sign boundary drawn, so the eight-negative / one-null / two-positive split is visible at a
glance and the two positive pools are identifiable rather than buried in a table row.

RIGHT -- the controls, each an orthogonal lever on the same axis:
  dimension ladder   adequacy moved by PARAMETER COUNT at fixed data (removes the n confound)
  temporal           the same partitions in a held-out origination year
  outcome            the same rows and sites, a common outcome instead of a rare one
The right panel is what makes the section's claim falsifiable rather than merely replicated:
three of the four levers leave the gradient intact and the fourth extinguishes it.
"""
from __future__ import annotations

import glob
import pathlib
import re
import sys

import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
# papers/common/, found by searching upward: a fixed hop count is right in the
# monorepo and wrong in an extracted release
for _d in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
    if (_d / "common" / "pgfemit.py").exists():
        sys.path.insert(0, str(_d / "common")); break
else:
    raise SystemExit("could not find common/pgfemit.py above " + __file__)

from pgfemit import Fig, fmt   # noqa: E402

# The eleven pools of Table "gradient", with the rho as published.
POOLS = [("MIMIC-IV", -0.296), ("Freddie Mac, zip3", -0.232), ("ACS, spanning", -0.214),
         ("Lending Club, 42k", -0.181), ("Freddie Mac, state", -0.176), ("MIMIC-III", -0.175),
         ("Lending Club, 1.9M", -0.170), ("eICU", -0.163), ("Bosch", -0.012),
         ("ACS, TravelTime", +0.027), ("ACS, natural size", +0.371)]


def _rho(path):
    d = pd.read_csv(path)
    return stats.spearmanr(d.epv_target, d.benefit)[0]


def controls():
    out = []
    # dimension ladder: adequacy by parameter count, data fixed
    for f in sorted(glob.glob(str(R / "site_rules_eicu_d*.csv"))):
        if f.endswith("_rules.csv"):
            continue
        out.append(("parameters", f"eICU d={re.search(r'_d(\d+)', f).group(1)}", _rho(f)))
    # temporal: a held-out origination year
    for f in sorted(glob.glob(str(R / "site_rules_freddiemac_*_2017.csv"))):
        if f.endswith("_rules.csv"):
            continue
        out.append(("year", re.search(r"freddiemac_(\w+)_2017", f).group(1)[:9], _rho(f)))
    # outcome: same rows and sites, a common outcome
    for lbl, f in (("mortality", R / "site_rules_hs_eicu_sites.csv"),
                   ("long stay", R / "site_rules_eicu_longstay.csv")):
        if f.exists():
            out.append(("outcome", lbl, _rho(f)))
    return out


def main() -> int:
    ctl = controls()
    if not ctl:
        print("no control runs found", file=sys.stderr)
        return 1

    fig = Fig()

    # ---- A: the eleven pools, sorted, sign-coloured -----------------------
    pl = sorted(POOLS, key=lambda t: t[1])
    groups = [("hgblue", "starved sites gain ($\\rho<-0.05$)", lambda r: r < -0.05),
              ("hggrey", "no gradient", lambda r: abs(r) <= 0.05),
              ("hgred",  "wrong sign ($\\rho>0.05$)", lambda r: r > 0.05)]
    ax = fig.axis(
        "A", "hg axis, hg decimal x=1",
        width=r"0.50\textwidth", height=f"{0.42 * len(pl) + 1.6:.2f}cm",
        xbar=True, bar_width="7pt", area_legend=True,
        xlabel=r"$\rho$(events per variable, merge benefit)",
        title=r"\textbf{A}\quad The gradient, eleven pools",
        ymin=-0.8, ymax=len(pl) - 0.2, enlarge_y_limits=False,
        ytick=",".join(str(i) for i in range(len(pl))),
        yticklabels=",".join("{%s}" % n for n, _ in pl),
        y_tick_label_style="font=\\small, align=right",
        legend_style=("font=\\scriptsize, draw=none, fill=none, "
                      "at={(0.5,-0.26)}, anchor=north, legend columns=1, row sep=1pt"))
    for colour, label, keep in groups:
        rows = [(i, r) for i, (_, r) in enumerate(pl) if keep(r)]
        if not rows:
            continue
        ax.plot(f"fill={colour}, draw=none, bar shift=0pt",
                [r for _, r in rows], [i for i, _ in rows], digits=3, legend=label)
    ax.raw(r"\draw[hgslate, line width=0.6pt] (axis cs:0,-0.8) -- (axis cs:0,"
           f"{len(pl) - 0.2});")

    # ---- B: three orthogonal controls -------------------------------------
    # a blank row between groups, so the three levers read as three blocks
    order = [("parameters", "hgblue", "adequacy by parameter count"),
             ("year", "hggreen", "held-out origination year"),
             ("outcome", "hgred", "a common outcome")]
    rows, i = [], 0
    for key, colour, label in order:
        block = [(name, r) for g, name, r in ctl if g == key]
        for name, r in block:
            rows.append((i, name, r, colour, key))
            i += 1
        i += 1                      # a blank row between levers
    top = max(r[0] for r in rows)
    # every control is negative, so the axis must still reach zero: without
    # that the bars start at the panel edge and their length means nothing
    xlo, xhi = min(r[2] for r in rows) * 1.08, 0.02

    bx = fig.right_of(
        "B", "hg axis, hg decimal x=1", "A", gap="2.4cm",
        width=r"0.42\textwidth", height=f"{0.42 * len(pl) + 1.6:.2f}cm",
        xbar=True, bar_width="7pt", area_legend=True,
        xlabel=r"$\rho$(events per variable, merge benefit)",
        title=r"\textbf{B}\quad Three orthogonal controls",
        ymin=-0.8, ymax=top + 0.8, enlarge_y_limits=False, y_dir="reverse",
        xmin=fmt(xlo, 3), xmax=fmt(xhi, 3), enlarge_x_limits=False,
        ytick=",".join(str(r[0]) for r in rows),
        yticklabels=",".join("{%s}" % r[1] for r in rows),
        y_tick_label_style="font=\\small, align=right",
        legend_style=("font=\\scriptsize, draw=none, fill=none, "
                      "at={(0.5,-0.26)}, anchor=north, legend columns=1, row sep=1pt"))
    for key, colour, label in order:
        sel = [r for r in rows if r[4] == key]
        if not sel:
            continue
        bx.plot(f"fill={colour}, draw=none, bar shift=0pt",
                [r[2] for r in sel], [r[0] for r in sel], digits=3, legend=label)
    bx.raw(r"\draw[hgslate, line width=0.6pt] (axis cs:0,-0.8) -- (axis cs:0,"
           f"{top + 0.8});")

    out = HERE.parent / "figures" / "fig_gradient.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fig.render(__file__))
    print(f"  wrote {out}  ({len(pl)} pools, {len(ctl)} control runs)")
    for g, n, r in ctl:
        print(f"    {g:<11}{n:<12}{r:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
