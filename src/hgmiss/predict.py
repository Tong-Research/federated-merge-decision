"""Cover rule and support-weighted combination of pattern models."""

from __future__ import annotations

from collections.abc import Mapping


def cover(
    observed: frozenset[int], patterns: list[frozenset[int]]
) -> list[frozenset[int]]:
    """Patterns applicable to a sample: every ``e`` with ``e subset-eq observed``."""
    return [e for e in patterns if e <= observed]


def combine(
    preds: Mapping[frozenset[int], float],
    supports: Mapping[frozenset[int], int],
    fallback: float,
) -> float:
    """Support-weighted mean of per-pattern predictions.

    Returns ``fallback`` (the marginal predictor) when no pattern applies or the
    applicable patterns carry no support. Callers must count how often this
    happens: the no-match rate is reported per cohort.
    """
    if not preds:
        return fallback
    total = sum(max(supports.get(e, 0), 0) for e in preds)
    if total <= 0:
        return fallback
    return sum(preds[e] * max(supports.get(e, 0), 0) for e in preds) / total
