"""Generate a synthetic federation, so the site structure can be the independent variable.

Every claim in Paper C about federations is measured on nine real collections, and none of
the four mechanisms this project proposed for *why* the boundary sits where it does has
survived its own test (R60, R61, R67, and the bias-variance decomposition). That is the
signature of a missing controlled experiment: with real data you cannot hold the
data-generating process fixed and move one factor.

Result 75 is the sharpest case. It found that the harm threshold varies six-fold across
partitions of the *same* 3.07M loans, and concluded the threshold belongs to a (domain,
partition) pair. That is a causal claim about carving, and the only way to test it is to
carve identical rows two different ways -- which no real collection permits and this
generator does, via `partition_align`.

`synthgen.py` already generates realistic *within-dataset* structured missingness and is
reused here for that. What it has no notion of is a site: no per-site coefficients, no size
distribution, no partition. This adds exactly that layer.

Output is `(X, y, g, names)` in the same shape every builder writes, so a synthetic
federation goes through `site_rules.py`, `gate_baselines.py` and `crossing_point.py`
unchanged. That matters more than it sounds: a synthetic result measured by a different code
path than the real ones would not be comparable to them.

The factors
-----------
Site structure
    n_sites, size_dist (uniform | lognormal | zipf | bimodal), size_median, size_spread

Between-site variation -- the part meta-analysis theory says governs pooling
    tau              spread of the per-site coefficient vector around the global one.
                     tau = 0 means every site shares one truth and merging is free.
    covariate_shift  per-site shift in the FEATURE distribution, holding coefficients
                     fixed. Real collections confound this with tau; here they separate.

Partition
    partition_align  0 carves rows into sites independently of their coefficients;
                     1 sorts by the latent coefficient modulator first, so sites differ
                     systematically. The SAME rows, carved two ways -- Result 75's
                     experiment, run causally.

Task difficulty
    signal           target AUROC of the true model. Result 70 found Bosch's local models
                     at 0.576 made the merge decision inconsequential; below about 0.6
                     nothing here should matter, and that is a prediction.
    prevalence, label_noise, d

Schema
    schema_overlap   1.0 gives every site every feature; below that, sites see blocks,
                     which is the block-wise missingness Paper A studies.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from matrix_loader import CACHE  # noqa: E402

SIZE_DISTS = ("uniform", "lognormal", "zipf", "bimodal")


def _site_sizes(rng, n_sites, dist, median, spread):
    """Per-site row counts. The distribution is the point, not an implementation detail:
    it sets the fraction of sites past the harm threshold, which is the x-axis of the
    distribution law (R56, R64, R74)."""
    if dist == "uniform":
        s = np.full(n_sites, float(median))
    elif dist == "lognormal":
        s = median * np.exp(rng.normal(0, spread, n_sites))
    elif dist == "zipf":
        r = np.arange(1, n_sites + 1)
        s = median * n_sites / (r ** max(spread, 0.1)) / 2
    elif dist == "bimodal":
        # Half starved, half adequate: the boundary-spanning pool that R42/R48 found is the
        # only kind where a rule beats the trivial policy.
        s = np.where(rng.random(n_sites) < 0.5, median / 8, median * 8.0)
    else:
        raise ValueError(dist)
    return np.clip(s, 60, None).astype(int)


def make_federation(n_sites=20, size_dist="lognormal", size_median=2000, size_spread=1.0,
                    d=16, tau_site=0.3, tau_row=0.0, partition_align=0.0, signal=0.75,
                    prevalence=0.1, covariate_shift=0.0, schema_overlap=1.0,
                    label_noise=0.0, seed=0):
    """Two kinds of coefficient heterogeneity, deliberately separated.

    `tau_site` is a per-site random effect: beta_s = beta + N(0, tau_site^2). This is what
    meta-analysis means by tau, and it produces between-site heterogeneity unconditionally.

    `tau_row` is a per-ROW modulator, and on its own it produces NONE: with rows assigned to
    sites at random, each site draws a representative mix and the site means coincide. It
    becomes site heterogeneity only when `partition_align` carves along it. That is Result
    75's claim in generator form -- the same rows, carved two ways, giving different
    between-site structure and therefore a different boundary. Measured on the first
    version, which conflated the two: tau 0.3 and tau 1.5 gave between-site spreads of 0.077
    and 0.075, indistinguishable, until alignment was turned on.
    """
    rng = np.random.default_rng(seed)
    sizes = _site_sizes(rng, n_sites, size_dist, size_median, size_spread)
    n = int(sizes.sum())

    # Correlated features: a latent factor model, because independent Gaussians make
    # merging look better than it is -- redundancy is what lets one site's coefficients
    # stand in for another's.
    n_fac = max(2, d // 4)
    L = rng.normal(0, 1, (d, n_fac)) / np.sqrt(n_fac)
    X = (rng.normal(0, 1, (n, n_fac)) @ L.T + rng.normal(0, 0.6, (n, d))).astype(np.float64)

    # One latent modulator per ROW. Coefficients depend on it, so "which rows share a site"
    # and "which sites share coefficients" can be coupled or decoupled at will.
    u = rng.normal(0, 1, n)

    # Assign rows to sites. Sorting by u before splitting makes the partition ALIGNED with
    # the coefficient structure; shuffling makes it orthogonal. Intermediate values
    # interpolate, so alignment is a dial rather than a switch.
    key = partition_align * u + (1 - partition_align) * rng.normal(0, 1, n)
    order = np.argsort(key)
    g = np.empty(n, dtype=np.int32)
    for k, idx in enumerate(np.split(order, np.cumsum(sizes)[:-1])):
        g[idx] = k

    beta0 = rng.normal(0, 1, d)
    v = rng.normal(0, 1, d)
    v /= np.linalg.norm(v)
    beta_row = beta0[None, :] + tau_row * u[:, None] * v[None, :]
    if tau_site:
        beta_row = beta_row + rng.normal(0, tau_site, (n_sites, d))[g]

    if covariate_shift:
        # Shift the FEATURES per site while leaving coefficients alone, so covariate shift
        # and concept shift are separately controllable. Every real collection confounds them.
        shift = rng.normal(0, covariate_shift, (n_sites, d))
        X = X + shift[g]

    z = np.einsum("ij,ij->i", X, beta_row)
    z = (z - z.mean()) / (z.std() + 1e-12)
    # Scale the linear predictor to hit a target AUROC. The relation between the SD of a
    # normal linear predictor and AUROC is Phi(sd / sqrt(2)); invert it.
    from scipy.stats import norm
    sd = max(1e-6, np.sqrt(2) * norm.ppf(min(max(signal, 0.51), 0.99)))
    z = z * sd
    # Solve for the intercept that makes the Bernoulli MEAN equal `prevalence`. Setting a
    # quantile of z instead sets the sign threshold, which is a different thing: the first
    # version asked for 0.10 and produced 0.26, because sigmoid is not a step function.
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() < prevalence:
            lo = mid
        else:
            hi = mid
    z = z + (lo + hi) / 2
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int8)
    if label_noise:
        flip = rng.random(n) < label_noise
        y[flip] = 1 - y[flip]

    if schema_overlap < 1.0:
        # Block-wise missingness: each site sees a contiguous block of features plus a
        # shared core. This is the structure `synthgen` models within a dataset, lifted to
        # the site level, and it is what makes a schema hypergraph non-trivial.
        # A shared core of features every site records, plus a rotating block that only
        # some do. `np.ix_` is required: indexing with a row mask and a column mask
        # together is element-wise pairing, not the outer product intended, and it raised
        # a shape error the moment n_sites and d disagreed.
        core = max(1, min(d - 1, int(round(d * schema_overlap))))
        extra = d - core
        Xf = X.astype(np.float32)
        for k in range(n_sites):
            keep = np.zeros(d, bool)
            keep[:core] = True
            if extra:
                start = core + (k * max(1, extra // 2)) % extra
                keep[start:start + max(1, extra // 2)] = True
            Xf[np.ix_(g == k, ~keep)] = np.nan
        X = Xf

    names = [f"s{k:03d}" for k in range(n_sites)]
    return X.astype(np.float32), y.astype(np.int8), g, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="written to MIMIC_CACHE as <name>.npz")
    for k, v in [("n-sites", 20), ("d", 16), ("size-median", 2000)]:
        ap.add_argument(f"--{k}", type=int, default=v)
    for k, v in [("size-spread", 1.0), ("tau-site", 0.3), ("tau-row", 0.0),
                 ("partition-align", 0.0),
                 ("signal", 0.75), ("prevalence", 0.1), ("covariate-shift", 0.0),
                 ("schema-overlap", 1.0), ("label-noise", 0.0)]:
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--size-dist", default="lognormal", choices=SIZE_DISTS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y, g, names = make_federation(
        n_sites=a.n_sites, size_dist=a.size_dist, size_median=a.size_median,
        size_spread=a.size_spread, d=a.d, tau_site=a.tau_site, tau_row=a.tau_row,
        partition_align=a.partition_align, signal=a.signal, prevalence=a.prevalence,
        covariate_shift=a.covariate_shift, schema_overlap=a.schema_overlap,
        label_noise=a.label_noise, seed=a.seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{a.name}.npz", X=X, y=y, g=g, names=np.array(names))
    sizes = np.bincount(g)
    print(f"{a.name}: {X.shape[0]:,} rows x {X.shape[1]} features, {len(names)} sites, "
          f"prevalence {y.mean():.3f}, sites {sizes.min():,}-{sizes.max():,} "
          f"(median {int(np.median(sizes)):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
