"""Detect missingness, and separately values, that encode the outcome.

A variable recorded only after diagnosis (a tumour grade, a treatment field) is
observed almost exclusively for cases. Its missingness indicator is then a copy
of the label. Any method that conditions on the missingness pattern -- which is
exactly what this paper proposes -- will exploit that and appear excellent.
``assert_no_outcome_conditional_missingness`` guards against that failure mode.

A second, distinct failure mode is leakage carried in a column's observed
VALUES rather than its missingness pattern: an administrative or process
field that is recorded for (almost) everyone, but whose value itself
determines the outcome -- e.g. PLCO's ``reconsent_outcome_days``, which is
100% observed and has a univariate AUROC of 1.0000 against 5-year mortality
on real prostate data. The mask-based guard above is blind to this, because
it only ever looks at *whether* a column was observed, never at *what* it
says. ``assert_no_value_leakage`` guards against that failure mode instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class LeakageError(RuntimeError):
    """Raised when a column's missingness is outcome-conditional."""


def _validate(
    mask: np.ndarray,
    y: np.ndarray,
    columns: list[str] | None = None,
) -> None:
    """Validate inputs to leakage detection functions.

    Checks:
    - mask is 2-D boolean array
    - mask and y have same number of rows
    - y contains no non-finite values
    - columns length matches mask.shape[1] (if provided)
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be boolean, got dtype {mask.dtype}")

    y_array = np.asarray(y).ravel()
    if y_array.shape[0] != mask.shape[0]:
        raise ValueError(
            f"mask and y must have same number of rows: "
            f"mask has {mask.shape[0]}, y has {y_array.shape[0]}"
        )

    # Check for non-finite values in y
    if np.issubdtype(y_array.dtype, np.floating):
        non_finite_mask = ~np.isfinite(y_array)
        if non_finite_mask.any():
            count = non_finite_mask.sum()
            raise ValueError(
                f"y contains {count} non-finite value(s) (NaN or inf); "
                f"these must be resolved before checking for leakage"
            )

    if columns is not None:
        if len(columns) != mask.shape[1]:
            raise ValueError(
                f"columns must have length matching mask.shape[1]: "
                f"got {len(columns)} columns but mask has {mask.shape[1]} features"
            )


def observation_rate_by_class(mask: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-column observation rate within each outcome class."""
    _validate(mask, y)
    y = np.asarray(y).ravel().astype(int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("need both classes present to assess leakage")

    rate_pos = mask[pos].mean(axis=0)
    rate_neg = mask[neg].mean(axis=0)
    return pd.DataFrame({
        "column": np.arange(mask.shape[1]),
        "rate_pos": rate_pos,
        "rate_neg": rate_neg,
        "gap": np.abs(rate_pos - rate_neg),
    })


def assert_no_outcome_conditional_missingness(
    mask: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    max_gap: float = 0.5,
) -> None:
    """Raise if any column's observation rate differs across classes by ``max_gap``.

    ``max_gap=0.5`` is deliberately permissive: it catches the catastrophic
    post-diagnosis fields without flagging ordinary informative missingness,
    which is the phenomenon the paper is *about* and must be preserved.
    """
    _validate(mask, y, columns)
    df = observation_rate_by_class(mask, y)
    bad = df[df.gap >= max_gap]
    if bad.empty:
        return

    lines = [
        f"  {columns[int(r.column)]}: observed in "
        f"{r.rate_pos:.1%} of positives vs {r.rate_neg:.1%} of negatives "
        f"(gap {r.gap:.1%})"
        for r in bad.itertuples()
    ]
    raise LeakageError(
        "Outcome-conditional missingness detected in "
        f"{len(bad)} column(s); their missingness indicator largely reproduces "
        "the label:\n" + "\n".join(lines) +
        "\n\nThese are typically post-diagnosis fields. Drop them from the "
        "feature set, or justify each one explicitly in the paper."
    )


def _validate_values(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str] | None = None,
) -> None:
    """Validate inputs to the value-leakage detection functions.

    Mirrors ``_validate`` above, but for a numeric feature matrix rather than
    a boolean observation mask:
    - X is 2-D and numeric (missing values are represented as NaN)
    - X and y have the same number of rows
    - y contains no non-finite values
    - columns length matches X.shape[1] (if provided)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"X must be numeric, got dtype {X.dtype}")

    y_array = np.asarray(y).ravel()
    if y_array.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y must have same number of rows: "
            f"X has {X.shape[0]}, y has {y_array.shape[0]}"
        )

    if np.issubdtype(y_array.dtype, np.floating):
        non_finite_mask = ~np.isfinite(y_array)
        if non_finite_mask.any():
            count = non_finite_mask.sum()
            raise ValueError(
                f"y contains {count} non-finite value(s) (NaN or inf); "
                f"these must be resolved before checking for leakage"
            )

    if columns is not None:
        if len(columns) != X.shape[1]:
            raise ValueError(
                f"columns must have length matching X.shape[1]: "
                f"got {len(columns)} columns but X has {X.shape[1]} features"
            )


def value_leakage_report(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    min_observed: int = 30,
) -> pd.DataFrame:
    """Per-column univariate AUROC of X's VALUES against y.

    For each column, uses only the rows where that column is observed
    (non-NaN) and computes ``max(auc, 1 - auc)`` so a perfectly *inverted*
    predictor (raw AUROC near 0) is scored identically to a perfectly
    *aligned* one (raw AUROC near 1) -- both are equally leaky.

    THE SKIP RULE (explicit, so silent skipping cannot hide anything): a
    column is skipped -- ``skipped=True``, ``auroc=NaN`` -- rather than
    scored when either:
    - fewer than ``min_observed`` rows have it observed, because a handful of
      observed values can separate the classes by chance and produce a
      spuriously extreme AUROC; or
    - fewer than 2 outcome classes are present among its observed rows,
      because AUROC is undefined without both classes to rank against each
      other.
    Every input column appears in the returned frame regardless -- skipped
    columns are never silently dropped, only flagged -- so
    ``report.skipped.sum()`` always gives the true skip count.

    Returned frame has one row per column: ``column``, ``auroc`` (NaN if
    skipped), ``n_observed``, ``skipped``; sorted by ``auroc`` descending
    (skipped rows, with NaN auroc, sort last).
    """
    _validate_values(X, y, columns)
    y = np.asarray(y).ravel().astype(int)

    rows = []
    for j, col in enumerate(columns):
        x = X[:, j]
        obs = ~np.isnan(x)
        n_observed = int(obs.sum())
        yy = y[obs]
        n_classes = len(np.unique(yy))
        skipped = n_observed < min_observed or n_classes < 2
        if skipped:
            rows.append({
                "column": col, "auroc": np.nan,
                "n_observed": n_observed, "skipped": True,
            })
            continue
        auc = roc_auc_score(yy, x[obs])
        auc = max(auc, 1.0 - auc)
        rows.append({
            "column": col, "auroc": float(auc),
            "n_observed": n_observed, "skipped": False,
        })

    df = pd.DataFrame(rows, columns=["column", "auroc", "n_observed", "skipped"])
    return df.sort_values("auroc", ascending=False, na_position="last").reset_index(drop=True)


def assert_no_value_leakage(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    max_auroc: float = 0.90,
    min_observed: int = 30,
) -> None:
    """Raise if any column's own VALUES let it univariately predict y too well.

    ``assert_no_outcome_conditional_missingness`` only ever looks at whether a
    column was observed; it is blind to leakage carried in what an observed
    column actually *says*. This guard closes that gap: for each column, it
    fits nothing more than a one-variable ranking (univariate AUROC, see
    ``value_leakage_report``) using only that column's observed rows, and
    raises if any column scores at or above ``max_auroc``.

    ``max_auroc=0.90`` is deliberately permissive, mirroring the mask-based
    guard's ``max_gap=0.5``. Measured on real PLCO prostate data (5-year
    mortality target):

    - ``reconsent_outcome_days`` (an administrative reconsent-process field,
      100% observed): univariate AUROC **1.0000** -- perfectly determines the
      label. Clearly not a clinical measurement.
    - ``reconsent_outcome``: **0.8599**.
    - ``in_TGWAS_population`` (GWAS sub-study membership): **0.7547**.
    - By contrast, the strongest genuine clinical predictors in the same
      cohort -- ``age`` (0.645), ``cig_years`` (0.647), ``pack_years``
      (0.632) -- all sit far below 0.90. Real clinical signal in this dataset
      tops out around 0.65; anything crossing 0.90 is not that.

    0.90 sits well above the real predictors' ceiling (~0.65) and well below
    the administrative leaks' floor (~0.86-1.00) among the cases actually
    observed in this dataset, so it separates the two regimes cleanly without
    being tuned to any single column.

    See ``value_leakage_report`` for the skip rule (too few observed rows, or
    only one outcome class observed) and for inspecting the full screen
    without raising.
    """
    _validate_values(X, y, columns)
    report = value_leakage_report(X, y, columns, min_observed=min_observed)
    n_skipped = int(report.skipped.sum())

    bad = report[(~report.skipped) & (report.auroc >= max_auroc)]
    if bad.empty:
        return

    lines = [
        f"  {r.column}: univariate AUROC {r.auroc:.4f} "
        f"(n_observed={r.n_observed})"
        for r in bad.itertuples()
    ]
    raise LeakageError(
        "Value leakage detected in "
        f"{len(bad)} column(s); their observed VALUES alone predict the "
        f"outcome above the max_auroc={max_auroc} threshold:\n" +
        "\n".join(lines) +
        f"\n\n({n_skipped} of {len(columns)} column(s) were skipped -- too "
        "few observed rows or only one outcome class present -- and are not "
        "assessed by this check; see value_leakage_report for the full "
        "per-column breakdown.)"
        "\n\nThese are typically administrative or process fields (e.g. "
        "consent/reconsent status, sub-study membership, record-linkage "
        "flags), not clinical measurements. Drop them from the feature set, "
        "or justify each one explicitly in the paper."
    )
