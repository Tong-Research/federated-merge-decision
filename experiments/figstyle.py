"""One publication style for every figure, so the set reads as a set.

Journals reject figure sets that look assembled from different sessions: three font sizes,
four palettes, axes boxed in one panel and open in the next. This module fixes those choices
once so no individual figure has to decide.

**Typography.** STIX Two Text is a Times clone with matching maths, which is what most
biomedical and statistics venues expect, and it is present on this machine (Latin Modern and
CMU are not). `mathtext` is set to `stix` so symbols in axis labels match the body text
instead of falling back to DejaVu, which is the usual tell that a figure came out of
matplotlib untouched.

**Colour.** A four-hue palette chosen to stay distinguishable in greyscale and under the
common forms of colour blindness, because the domain contrast — clinical against Census
against credit — is load-bearing in several figures and must survive printing.

**Sizes.** Widths are given in inches to match a two-column journal page: `ONE_COL` 3.4in and
`TWO_COL` 7.0in. Figures are authored at final size rather than scaled afterwards, which is
what keeps the type in a figure the same size as the type around it.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ONE_COL = 3.4
TWO_COL = 7.0

# Domain colours. Blue/red/green/grey, checked for greyscale separation: the luminances are
# roughly 0.45 / 0.42 / 0.55 / 0.70, far enough apart to read when printed monochrome.
C = {
    "ACS":      "#3B6EA5",
    "clinical": "#B4444E",
    "credit":   "#3E8E64",
    "other":    "#7A7A7A",
    "accent":   "#C77A2B",
    "fit":      "#333333",
    "grid":     "#D8D8D8",
    "zero":     "#9A9A9A",
}
ORDER = ["clinical", "ACS", "credit", "other"]


def use():
    """Apply the house style. Call once at the top of a figure script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        # Open axes: no top/right spine. Standard in statistics journals and it removes two
        # lines per panel that carry no information.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,          # data draws over the grid, never under it
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,              # embed TrueType, not Type3: many venues reject Type3
        "ps.fonttype": 42,
    })


def save(fig, path, also_png=True):
    """Write a vector PDF for the manuscript and a PNG for looking at."""
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    if also_png:
        fig.savefig(p.with_suffix(".png"))
    print(f"  wrote {p.name}" + (f" and {p.with_suffix('.png').name}" if also_png else ""))
    return p


def panel_label(ax, letter, dx=-0.13, dy=1.06):
    """Bold panel letter in the corner, positioned in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")


def zeroline(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=C["zero"], lw=0.8, ls=(0, (4, 3)),
                                               zorder=0)
