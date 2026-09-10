"""Latent-factor generator with directly controlled mask informativeness."""

from __future__ import annotations

import numpy as np


def make_synthetic(
    n: int,
    d: int,
    *,
    beta: float,
    missing_rate: float,
    seed: int,
    n_latent: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(X_with_nan, y, mask)``.

    Features are a sparse linear map of ``n_latent`` Gaussian factors. The
    target is a thresholded linear function of the factors. Missingness follows
    ``P[m_ij = 0] = sigmoid(alpha_j + beta * y_i)``: ``beta = 0`` is MCAR with
    iota = 0, and larger ``beta`` raises iota monotonically.
    """
    rng = np.random.default_rng(seed)

    z = rng.normal(size=(n, n_latent))
    w = rng.normal(size=(n_latent, d)) * (rng.random((n_latent, d)) < 0.6)
    X = z @ w + rng.normal(0, 0.5, size=(n, d))

    coef = rng.normal(size=n_latent)
    logits = z @ coef
    y = (logits > np.median(logits)).astype(int)

    alpha = _alpha_for_rate(missing_rate, beta, y)
    p_missing = _sigmoid(alpha + beta * y[:, None])
    mask = rng.random((n, d)) >= p_missing

    X_nan = X.astype(float).copy()
    X_nan[~mask] = np.nan
    return X_nan, y, mask


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _alpha_for_rate(rate: float, beta: float, y: np.ndarray) -> float:
    """Offset making the marginal missing rate approximately ``rate``.

    Solved by bisection because ``beta`` shifts the mean of the sigmoid.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"missing_rate must be in [0, 1), got {rate}")
    if rate == 0.0:
        return -50.0
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        got = _sigmoid(mid + beta * y).mean()
        if got < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
