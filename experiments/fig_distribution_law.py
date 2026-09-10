"""Section 5's figure: the rule's value as a function of the federation it is evaluated in.

This is the paper's centrepiece and the thing a reader is meant to take away. It carries two
panels because the finding has two halves that are easy to conflate.

**Panel A** plots lift over the best trivial policy against the fraction of decisions whose
target sits past the harm threshold. Every point is a decision pool; the fitted quadratic is
Result 64's high-seed law. The shape is the claim: a rule can only earn its keep where the
pool contains sites on both sides of the boundary, so lift rises, peaks near f = 0.45, and
falls again once nearly every site is adequate and "never merge" is almost always right.

**Panel B** is the honesty panel, and it is why the figure has two. The x-axis of Panel A is
"fraction above 5,000 records", and Result 65 showed that threshold is domain-specific by a
factor of four -- Lending Club crosses zero near n = 20,000 while ACS is harmful at 2,000.
Result 67 then found the per-domain threshold is not predictable from heterogeneity. So Panel
B plots each pool's own measured crossing point, showing directly that the x-axis of Panel A
needs a per-domain calibration the paper does not have. Publishing Panel A alone would imply
a reader can compute their own harm fraction, which they cannot.

Points are labelled by collection rather than by pool name so the reader can see that the
clinical, Census and credit pools land on the same curve.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# papers/common/, found by searching upward: a fixed hop count is right in the
# monorepo and wrong in an extracted release
for _d in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
    if (_d / "common" / "pgfemit.py").exists():
        sys.path.insert(0, str(_d / "common")); break
else:
    raise SystemExit("could not find common/pgfemit.py above " + __file__)

from pgfemit import Fig, declutter, fmt   # noqa: E402

R = HERE.parent / "results"
HARM_N = 5000

# pool file -> (display label, family) ; family drives colour so the reader can see that
# three unrelated domains lie on one curve.
# (file, label, family, in_fit). Only the high-seed pools enter the quadratic: the text
# reports that fit, and a figure whose curve is fitted on a different set than the equation
# it illustrates is a discrepancy waiting to be found in review.
POOLS = [
    ("site_rules_acs_d10.csv",            "ACS capped 1.2k",      "ACS"),
    ("site_rules_lendingclub.csv",        "LendingClub 42k",      "credit"),
    ("site_rules_hs_acs_span_income.csv", "ACS spanning",         "ACS"),
    ("site_rules_hsladder_1000.csv",      "ACS ladder 1k",        "ACS"),
    ("site_rules_hsladder_500.csv",       "ACS ladder 0.5k",      "ACS"),
    ("site_rules_hsladder_2000.csv",      "ACS ladder 2k",        "ACS"),
    ("site_rules_hsladder_5000.csv",      "ACS ladder 5k",        "ACS"),
    ("site_rules_lc_full.csv",            "LendingClub 1.9M",     "credit"),
    ("site_rules_hsladder_10000.csv",     "ACS ladder 10k",       "ACS"),
    ("site_rules_hs_acs_full_income.csv", "ACS natural",          "ACS"),
    ("site_rules_all.csv",                "clinical spanning",    "clinical"),
    ("site_rules_hs_eicu_sites.csv",      "eICU",                 "clinical"),
]
COLOURS = {"ACS": "hgblue", "clinical": "hgred", "credit": "hggreen"}


def pool_stats(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows or "helped" not in rows[0]:
        return None
    g = lambda k: np.array([float(r.get(k, "nan") or "nan") for r in rows])
    n, e, b = g("n_target"), g("epv_target"), g("benefit")
    h = np.array([int(r["helped"]) for r in rows])
    triv = max(h.mean(), 1 - h.mean())
    best = -1.0
    for v in (e, n):
        ok = np.isfinite(v)
        if ok.sum() < 50:
            continue
        for t in np.quantile(v[ok], np.linspace(0.05, 0.95, 37)):
            for pred in ((v[ok] < t), (v[ok] >= t)):
                best = max(best, (pred == h[ok]).mean())
    # each pool's OWN crossing point: where mean benefit changes sign against log n
    ok = np.isfinite(n) & np.isfinite(b) & (n > 0)
    cross = np.nan
    if ok.sum() > 300:
        q = np.quantile(n[ok], np.linspace(0, 1, 7))
        xs, ys = [], []
        for i in range(6):
            s = ok & (n >= q[i]) & (n <= q[i + 1])
            if s.sum() >= 30:
                xs.append(np.log10(np.median(n[s])))
                ys.append(b[s].mean())
        if len(xs) >= 4:
            ys_a = np.array(ys)
            if (ys_a > 0).any() and (ys_a < 0).any():
                sl, ic = np.polyfit(np.array(xs), ys_a, 1)
                if abs(sl) > 1e-9:
                    c_ = 10 ** (-ic / sl)
                    # A crossing OUTSIDE the pool's observed size range is an extrapolation,
                    # not a measurement. Result 61 saw ladder-2000 return 2.1 million records
                    # this way -- a near-flat benefit curve divided by a near-zero slope. It
                    # would dominate a log axis and imply a precision that is not there.
                    lo, hi = n[ok].min(), n[ok].max()
                    cross = c_ if lo <= c_ <= hi else np.nan
    return dict(harm=float((n > HARM_N).mean()), lift=float(best - triv),
                helped=float(h.mean()), n=len(rows), cross=float(cross),
                # Keep the raw target sizes so the harm fraction can be recomputed at other
                # thresholds without re-reading every CSV. peak_range() needs exactly this.
                n_target=n)


def peak_range(fit_pts, lo=1000, hi=7500, steps=9):
    """Where does the fitted maximum sit, across thresholds where the fit still means anything?

    Result 74 as amended: the curvature stays negative from 1,000 to 50,000 records, so the
    sign alone is a weak bound -- an earlier version claimed it flipped above 10,000 and that
    was a clipping bug in the sweep, not a property of the data. The real bound is fit
    quality: R^2 falls from 0.81 at 3,000 to 0.24 at 7,500 and 0.004 at 15,000. Past that a
    negative coefficient explains nothing. So the supported range is 1,000-7,500, and across
    it the maximum still moves from f=0.73 to f=0.36 -- which is the point.
    """
    peaks = []
    for T in np.linspace(lo, hi, steps):
        f = np.array([float((p[2]["n_target"] > T).mean()) for p in fit_pts])
        y = np.array([p[2]["lift"] for p in fit_pts])
        if len(set(np.round(f, 3))) < 3:
            continue
        a, b, _ = np.polyfit(f, y, 2)
        if a >= 0:
            continue
        pk = -b / (2 * a)
        if 0 <= pk <= 1:
            peaks.append(pk)
    if not peaks:
        return float("nan"), float("nan")
    return min(peaks), max(peaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R.parent / "figures" / "fig_distribution_law.tex"))
    a = ap.parse_args()

    pts = []
    for f, label, fam in POOLS:
        s = pool_stats(R / f)
        if s:
            pts.append((label, fam, s))
        else:
            print(f"  missing: {f}")
    if len(pts) < 5:
        raise SystemExit("too few pools to draw the law")

    HIGH_SEED = {"ACS spanning", "ACS natural", "ACS ladder 0.5k", "ACS ladder 1k",
                 "ACS ladder 2k", "ACS ladder 5k", "ACS ladder 10k", "LendingClub 42k"}
    fit_pts = [p for p in pts if p[0] in HIGH_SEED]
    hf = np.array([p[2]["harm"] for p in fit_pts])
    lift = np.array([p[2]["lift"] for p in fit_pts])

    fig = Fig()

    # ---- Panel A: the law -------------------------------------------------
    c = np.polyfit(hf, lift, 2)
    xs = np.linspace(0, max(hf.max(), 0.85), 120)
    ys = np.polyval(c, xs)
    allx = [s["harm"] for _, _, s in pts]
    ally = [s["lift"] for _, _, s in pts]
    xlo, xhi = -0.12, max(max(allx), xs.max()) + 0.14
    ylo = min(min(ally), ys.min()) - 0.030
    yhi = max(max(ally), ys.max()) + 0.022

    ax = fig.axis(
        "A", "hg axis, hg legend in",
        width=r"0.78\textwidth", height="5.8cm", scale_only_axis=True,
        xlabel="fraction of decisions with target $n>5{,}000$ (harm fraction)",
        ylabel="lift over the best trivial policy",
        title=r"\textbf{A}\quad The rule earns its keep in the middle",
        xmin=fmt(xlo, 3), xmax=fmt(xhi, 3), ymin=fmt(ylo, 4), ymax=fmt(yhi, 4),
        enlarge_x_limits=False, enlarge_y_limits=False, clip=False)

    # The peak is NOT drawn as a line. Refitting these pools at harm thresholds
    # from 1k to 7.5k keeps the curvature negative -- the shape is real -- but
    # moves the maximum from f=0.73 to f=0.36. A single dotted "peak f=0.45"
    # would report a property of the 5,000-record convention as a measurement.
    peak_lo, peak_hi = peak_range(fit_pts)
    if np.isfinite(peak_lo) and np.isfinite(peak_hi):
        ax.raw(f"\\fill[hgpale, opacity=0.75] (axis cs:{fmt(peak_lo,3)},{fmt(ylo,4)}) "
               f"rectangle (axis cs:{fmt(peak_hi,3)},{fmt(yhi,4)});")
        ax.raw(f"\\node[font=\\tiny, text=hgslate, anchor=south, align=center] "
               f"at (axis cs:{fmt((peak_lo + peak_hi) / 2, 3)},{fmt(ylo, 4)}) "
               f"{{maximum lies anywhere in $[{peak_lo:.2f},\\,{peak_hi:.2f}]$\\\\"
               f"across harm thresholds 1k--7.5k}};")
    ax.raw(f"\\draw[hg rule] (axis cs:{fmt(xlo,3)},0) -- (axis cs:{fmt(xhi,3)},0);")
    ax.plot("hgslate, line width=1.2pt, mark=none, smooth", xs, ys, digits=4,
            legend=f"quadratic fit, {len(fit_pts)} high-seed pools")

    for fam, colour in COLOURS.items():
        rows = [(s["harm"], s["lift"]) for lab, f2, s in pts
                if f2 == fam and lab in HIGH_SEED]
        if rows:
            ax.plot(f"{colour}, only marks, mark=*, mark size=2.1pt, "
                    f"mark options={{fill={colour}, draw=white, line width=0.3pt}}",
                    [r[0] for r in rows], [r[1] for r in rows], digits=4, legend=fam)
    for fam, colour in COLOURS.items():
        rows = [(s["harm"], s["lift"]) for lab, f2, s in pts
                if f2 == fam and lab not in HIGH_SEED]
        if rows:
            ax.plot(f"{colour}, only marks, mark=o, mark size=2.1pt, "
                    f"mark options={{line width=0.8pt}}",
                    [r[0] for r in rows], [r[1] for r in rows], digits=4, forget=True)
    ax.raw(r"\addlegendimage{hgslate, only marks, mark=o, mark size=2.1pt, "
           r"mark options={line width=0.8pt}}")
    ax.raw(r"\addlegendentry{not in fit (3 seeds)}")

    lab = [(s["harm"], s["lift"], label) for label, _, s in pts]
    for (x, y, text), (anchor, dx, dy) in zip(
            lab, declutter(lab, (xlo, xhi), (ylo, yhi), box=(366, 165),
                           font_pt=6.0, radius=7.5)):
        colour = COLOURS[next(f2 for l2, f2, _ in pts if l2 == text)]
        ax.raw(f"\\node[font=\\tiny, text={colour}, anchor={anchor}, "
               f"fill=white, fill opacity=0.72, text opacity=1, inner sep=0.6pt, "
               f"xshift={dx}pt, yshift={dy}pt] at (axis cs:{fmt(x,4)},{fmt(y,4)}) "
               f"{{{text}}};")

    # ---- Panel B: the shape survives the convention, the peak does not ----
    Ts = np.linspace(1000, 20000, 20)
    peaks, curvs, keep, r2s = [], [], [], []
    for T in Ts:
        f = np.array([float((q[2]["n_target"] > T).mean()) for q in fit_pts])
        y = np.array([q[2]["lift"] for q in fit_pts])
        if len(set(np.round(f, 3))) < 3:
            continue
        aq, bq, cq = np.polyfit(f, y, 2)
        pred = aq * f * f + bq * f + cq
        r2s.append(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        keep.append(T); curvs.append(aq)
        peaks.append(-bq / (2 * aq) if aq < 0 else np.nan)
    keep = np.array(keep); curvs = np.array(curvs); peaks = np.array(peaks)

    blo, bhi = float(curvs.min()) * 1.10, 0.02

    # Two stacked panels rather than one panel with a twin y axis. The section's
    # claim is that the shape survives the threshold convention and the peak does
    # not, which is two statements; on a twin axis a reader has to work out which
    # curve belongs to which scale before they can read either one.
    shared = dict(width=r"0.33\textwidth", height="3.2cm",
                  scale_only_axis=True, xmode="log",
                  xmin=900, xmax=23000, enlarge_x_limits=False,
                  enlarge_y_limits=False)

    bx = fig.axis(
        "B", "hg axis", at="($(A.south west)-(0,2.15cm)$)", anchor="north west",
        ylabel="curvature of the fit",
        xlabel="harm threshold used to compute $f$ (records)",
        title=r"\textbf{B}\quad The shape survives",
        ymin=fmt(blo, 4), ymax=fmt(bhi, 4), **shared)

    cx = fig.axis(
        "C", "hg axis", at="($(A.south east)-(0,2.15cm)$)", anchor="north east",
        ymin=0, ymax=1,
        xlabel="harm threshold used to compute $f$ (records)",
        ylabel="location of the maximum, $f$",
        title=r"\textbf{C}\quad The peak does not", **shared)

    # Shade where the FIT is informative, not where the curvature is merely
    # negative: a<0 holds all the way to 50,000 and is not the binding
    # constraint.  R^2 is.
    ok = np.array(r2s) >= 0.20
    for axis, lo, hi in ((bx, blo, bhi), (cx, 0.0, 1.0)):
        if ok.any():
            axis.raw(f"\\fill[hgpale, opacity=0.75] "
                     f"(axis cs:{keep[ok].min():.0f},{fmt(lo, 4)}) rectangle "
                     f"(axis cs:{keep[ok].max():.0f},{fmt(hi, 4)});")
        axis.raw(f"\\draw[hgred, line width=0.8pt, densely dotted] "
                 f"(axis cs:{HARM_N},{fmt(lo, 4)}) -- (axis cs:{HARM_N},{fmt(hi, 4)});")
    bx.raw(f"\\node[font=\\tiny, text=hgslate, anchor=south] "
           f"at (axis cs:{(keep[ok].min() * keep[ok].max()) ** 0.5:.0f},{fmt(blo, 4)}) "
           r"{$R^2\geq0.2$};")
    cx.raw(f"\\node[font=\\tiny, text=hgred, anchor=south west, xshift=2pt] "
           f"at (axis cs:{HARM_N},0.04) "
           r"{$5{,}000$, used by \textbf{A}};")
    bx.raw(r"\draw[hg rule] (axis cs:900,0) -- (axis cs:23000,0);")
    bx.plot("hgslate, line width=1.0pt, mark=*, mark size=1.4pt, "
            "mark options={fill=hgslate, draw=white, line width=0.3pt}",
            keep, curvs, digits=5, forget=True)

    good = np.isfinite(peaks)
    cx.plot("hgblue, line width=1.0pt, mark=square*, mark size=1.4pt, "
            "mark options={fill=hgblue, draw=white, line width=0.3pt}",
            keep[good], peaks[good], digits=4, forget=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.render(__file__))
    print(f"wrote {out}")

    print(f"\n{'pool':22s} {'harm frac':>9s} {'helped':>7s} {'lift':>7s} {'crossing n':>11s}")
    for label, fam, s in sorted(pts, key=lambda p: p[2]["harm"]):
        cs = f"{s['cross']:,.0f}" if np.isfinite(s["cross"]) else "—"
        print(f"{label:22s} {s['harm']:9.3f} {s['helped']:7.3f} {s['lift']:+7.3f} {cs:>11s}")


if __name__ == "__main__":
    main()
