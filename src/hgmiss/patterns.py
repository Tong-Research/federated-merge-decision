"""Missingness patterns, their support, and the containment hierarchy."""

from __future__ import annotations

import numpy as np


def extract_patterns(mask: np.ndarray) -> dict[frozenset[int], int]:
    """Map each *exactly* realised observed-variable set to its row count.

    ``mask[i, j]`` is True when variable ``j`` is observed in row ``i``.
    """
    _validate(mask)
    counts: dict[frozenset[int], int] = {}
    for row in mask:
        key = frozenset(np.flatnonzero(row).tolist())
        counts[key] = counts.get(key, 0) + 1
    return counts


def support(pattern: frozenset[int], mask: np.ndarray) -> int:
    """Number of rows observing every variable in ``pattern`` (superset rule).

    This is ``|S(e)|`` in the paper: ``S(e) = {i : O_i superset-eq e}``. It is
    NOT the count of rows whose pattern equals ``e`` -- see ``extract_patterns``.
    """
    _validate(mask)
    if not pattern:
        return int(mask.shape[0])
    cols = sorted(pattern)
    return int(mask[:, cols].all(axis=1).sum())


def _validate(mask: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be boolean, got dtype {mask.dtype}")


def immediate_ancestors(
    patterns: list[frozenset[int]],
) -> dict[frozenset[int], list[frozenset[int]]]:
    """Maximal realised patterns strictly contained in each pattern.

    This is the transitive reduction of the containment order restricted to the
    realised patterns. ``e'`` is an immediate ancestor of ``e`` when
    ``e' subset e`` and no realised ``e''`` satisfies ``e' subset e'' subset e``.

    Each pattern is represented as an integer bitmask so containment becomes a
    single ``a & b == a`` machine-word test instead of a Python set
    comparison. Patterns are also indexed by size so that, for a given ``e``,
    only realised patterns strictly smaller than ``e`` are ever considered.
    Candidates are then visited in decreasing size and accepted only if they
    are not already contained in a previously accepted (necessarily larger)
    ancestor -- which keeps the accepted set small and collapses the inner
    quadratic term from the straightforward implementation in practice.
    """
    uniq = sorted(set(patterns), key=len)
    masks = {p: _bitmask(p) for p in uniq}

    by_size: dict[int, list[tuple[frozenset[int], int]]] = {}
    for p in uniq:
        by_size.setdefault(len(p), []).append((p, masks[p]))
    sizes_desc = sorted(by_size, reverse=True)

    out: dict[frozenset[int], list[frozenset[int]]] = {}
    for e in uniq:
        e_mask = masks[e]
        e_size = len(e)
        accepted: list[frozenset[int]] = []
        accepted_masks: list[int] = []
        for size in sizes_desc:
            if size >= e_size:
                continue
            for c, c_mask in by_size[size]:
                if c_mask & e_mask != c_mask:
                    continue  # c is not a subset of e
                if any(c_mask & am == c_mask for am in accepted_masks):
                    continue  # c is already dominated by an accepted ancestor
                accepted.append(c)
                accepted_masks.append(c_mask)
        out[e] = accepted
    return out


def _bitmask(pattern: frozenset[int]) -> int:
    mask = 0
    for i in pattern:
        mask |= 1 << i
    return mask


def supports_for(
    patterns: list[frozenset[int]], mask: np.ndarray
) -> dict[frozenset[int], int]:
    """Batch version of :func:`support`: supports for many patterns at once.

    Equivalent to calling :func:`support` on each pattern individually, but
    vectorised with numpy: for a batch of patterns, each represented as a
    0/1 indicator column, ``mask @ indicator`` counts -- for every row and
    every pattern -- how many of the pattern's variables are observed in that
    row. A row supports a pattern exactly when that count equals the
    pattern's size (since each term is 0 or 1, hitting the size means every
    variable in the pattern was observed). Patterns are processed in
    memory-bounded batches since a full ``n_rows x n_patterns`` matrix can be
    large for the pattern counts seen in practice (thousands of patterns over
    tens of thousands of rows).
    """
    _validate(mask)
    n_rows, n_vars = mask.shape
    uniq = list(dict.fromkeys(patterns))
    if not uniq:
        return {}

    mask_f = mask.astype(np.float64, copy=False)

    # Keep each batch's working set (n_rows x batch_size float64) to a modest
    # memory budget while still amortising Python-level overhead over many
    # patterns per matmul call.
    budget_elems = 20_000_000
    batch_size = max(1, min(len(uniq), budget_elems // max(n_rows, 1)))

    out: dict[frozenset[int], int] = {}
    for start in range(0, len(uniq), batch_size):
        batch = uniq[start:start + batch_size]
        sizes = np.fromiter((len(p) for p in batch), dtype=np.float64, count=len(batch))
        indicator = np.zeros((n_vars, len(batch)), dtype=np.float64)
        for j, p in enumerate(batch):
            if p:
                indicator[sorted(p), j] = 1.0
        counts = mask_f @ indicator  # (n_rows, batch_size)
        hits = counts == sizes[np.newaxis, :]
        batch_supports = hits.sum(axis=0)
        for p, s in zip(batch, batch_supports):
            out[p] = int(s)
    return out
