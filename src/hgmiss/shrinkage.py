"""Hierarchical shrinkage of per-pattern parameters toward realised ancestors."""

from __future__ import annotations

INTERCEPT = -1

Params = dict[int, float]


def shrink(
    theta_hat: dict[frozenset[int], Params],
    supports: dict[frozenset[int], int],
    ancestors: dict[frozenset[int], list[frozenset[int]]],
    kappa: float,
    coordinatewise: bool = False,
) -> dict[frozenset[int], Params]:
    """Shrink each pattern's coefficients toward its immediate ancestors.

    Coefficients are keyed by variable index (plus ``INTERCEPT``), so ancestors
    and descendants align on the variables they share without any positional
    bookkeeping. A variable the ancestors do not observe keeps the descendant's
    own estimate -- there is nothing to borrow for it.

    Patterns are processed smallest-first, so every ancestor is already shrunk
    when a descendant consumes it. ``kappa = 0`` recovers independent pattern
    submodels; large ``kappa`` pulls each pattern onto its ancestors.
    """
    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")

    out: dict[frozenset[int], Params] = {}
    for e in sorted(theta_hat, key=len):
        own = dict(theta_hat[e])
        parents = [p for p in ancestors.get(e, []) if p in out]
        if not parents:
            out[e] = own
            continue

        n_e = supports[e]
        lam = n_e / (n_e + kappa) if (n_e + kappa) > 0 else 0.0

        # C4, the coordinate-wise weight. The scalar rule above gives every coefficient of
        # this pattern the same lambda = n_e / (n_e + kappa), which asks only how much
        # evidence the pattern has for ITSELF. It never asks how much evidence stands
        # behind the target it is shrinking toward.
        #
        # Within a pattern every observed variable shares n_e, so the pattern side cannot
        # vary by coordinate. The ancestor side can, and by a lot: some variables are
        # carried by heavily-supported ancestors and some by one small one. Shrinking both
        # at the same rate over-trusts the second.
        #
        # So the prior's strength is scaled by the evidence behind it -- the support-weighted
        # degree of that vertex among the contributing ancestors, which is a quantity of the
        # pattern hypergraph and not of any one pattern:
        #
        #     lambda_j = n_e / (n_e + kappa * m_j / mean(m))
        #
        # Normalising by the mean over this pattern's own keys is what makes this a
        # REDISTRIBUTION of shrinkage across coordinates rather than a different kappa. When
        # every key is equally supported, m_j / mean(m) = 1 and the rule reduces EXACTLY to
        # the scalar one -- the C3 control is structural here rather than an empirical check.
        #
        # Unlike C2 (Result 88), the borrowed estimate comes from the SAME cohort, differing
        # only in which variables were recorded. C2 failed because shrinking a partner's
        # coefficient toward zero bought protection against unknown transportability between
        # populations; that objection is much weaker here, which is the reason to expect a
        # different answer rather than a repeat.
        # The INTERCEPT is deliberately excluded and keeps the scalar lambda.
        #
        # It is carried by every ancestor, so its mass is always the largest and the rule
        # above would shrink it hardest of all -- the opposite of what this project's
        # calibration results say. Result 53 and its 2026-08-11 restatement are explicit:
        # refit the intercept locally whatever weight is placed on the borrowed model. The
        # intercept is a calibration parameter and the coefficients are slopes; a rule about
        # how much slope evidence stands behind a variable has nothing to say about it.
        mass: dict[int, float] = {}
        if coordinatewise:
            for key in own:
                if key == INTERCEPT:
                    continue
                contributors = [p for p in parents if key in out[p]]
                mass[key] = sum(float(max(supports[p], 0)) for p in contributors)
            live = [v for v in mass.values() if v > 0]
            mbar = sum(live) / len(live) if live else 0.0

        shrunk: Params = {}
        for key, own_val in own.items():
            contributors = [p for p in parents if key in out[p]]
            if not contributors:
                shrunk[key] = own_val
                continue
            weights = [float(max(supports[p], 0)) for p in contributors]
            total = sum(weights)
            if total <= 0:
                weights = [1.0] * len(contributors)
                total = float(len(contributors))
            target = sum(
                w * out[p][key] for w, p in zip(weights, contributors)
            ) / total
            lam_j = lam
            if coordinatewise and mbar > 0 and mass.get(key, 0.0) > 0:
                k_j = kappa * (mass[key] / mbar)
                lam_j = n_e / (n_e + k_j) if (n_e + k_j) > 0 else 0.0
            shrunk[key] = lam_j * own_val + (1.0 - lam_j) * target

        out[e] = shrunk
    return out
