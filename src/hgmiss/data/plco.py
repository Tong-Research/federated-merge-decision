"""PLCO cohort loading with post-diagnosis columns excluded."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..leakage import assert_no_outcome_conditional_missingness, assert_no_value_leakage
from .plco_excluded import EXCLUDED_PATTERNS, is_followup_screening_round

FEATURE_SETS: tuple[str, ...] = ("baseline", "full")

COHORTS = {
    "colorectal": ("Colorectal", "colo_data_mar22_d032222.csv.zip"),
    "lung": ("Lung", "lung_data_mar22_d032222.csv.zip"),
    "ovarian": ("Ovarian", "ovar_data_mar22_d032222.csv.zip"),
    "prostate": ("Prostate", "pros_data_mar22_d032222.csv.zip"),
}


@dataclass(frozen=True)
class Cohort:
    name: str
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    columns: list[str]
    feature_set: str


def load_cohort(
    name: str,
    root: Path,
    *,
    check_leakage: bool = True,
    feature_set: str = "baseline",
) -> Cohort:
    """Load one PLCO cohort as numeric features, target, and observation mask.

    ``feature_set`` selects which columns are eligible to enter the feature
    matrix (after the leakage exclusions in ``plco_excluded.EXCLUDED_PATTERNS``,
    which always apply):

    - ``"baseline"`` (default, the PRIMARY analysis): keeps only variables
      measured at or before randomisation -- the baseline questionnaire and
      screening round 0. Later screening rounds and follow-up questionnaire
      waves carry survivorship-conditional missingness with respect to the
      5-year mortality target (see ``plco_excluded.is_followup_screening_round``
      for the measured numbers and the exact rule).
    - ``"full"``: keeps everything that currently survives
      ``EXCLUDED_PATTERNS`` (the prior, unchanged behaviour). Retained as a
      sensitivity analysis only.

    ``check_leakage`` (default True) gates TWO independent guards, run for
    both feature sets: ``assert_no_outcome_conditional_missingness`` (a
    column's missingness pattern reproduces the label) and
    ``assert_no_value_leakage`` (a column's observed VALUES alone predict the
    label, e.g. a fully-observed administrative field -- see
    ``hgmiss.leakage`` and the "value leakage" block in
    ``plco_excluded.EXCLUDED_PATTERNS`` for the real numbers that motivated
    this second guard).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS!r}, got {feature_set!r}"
        )
    if name not in COHORTS:
        raise KeyError(f"unknown cohort {name!r}; expected one of {sorted(COHORTS)}")
    subdir, fname = COHORTS[name]
    path = Path(root) / subdir / fname

    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as fh:
            df = pd.read_csv(fh, low_memory=False)

    y = _derive_target(df, name)
    keep = df.index[y.notna()]
    df, y = df.loc[keep], y.loc[keep].astype(int)

    features = [c for c in df.columns if not _is_excluded(c)]
    if feature_set == "baseline":
        features = [c for c in features if not is_followup_screening_round(c)]
    Xdf = df[features].apply(pd.to_numeric, errors="coerce")

    # Drop columns that carry no information after coercion.
    usable = [c for c in Xdf.columns if Xdf[c].notna().any() and Xdf[c].nunique() > 1]
    Xdf = Xdf[usable]

    X = Xdf.to_numpy(dtype=float)
    mask = ~np.isnan(X)

    if check_leakage:
        yv = y.to_numpy()
        assert_no_outcome_conditional_missingness(mask, yv, usable)
        # Whole-cohort rates dilute leakage confined to a subset of rows: a field
        # that perfectly copies the label for <=45% of the cohort scores a gap below
        # the 0.5 threshold and passes. PLCO randomises to a screening arm and a
        # control arm, so a field recorded only in the screening arm affects ~50% of
        # rows -- exactly that regime. Re-run the guard within each arm.
        for arm_val, rows in _arm_strata(df.loc[keep]):
            if rows.sum() < 100 or len(np.unique(yv[rows])) < 2:
                continue
            assert_no_outcome_conditional_missingness(
                mask[rows], yv[rows], usable
            )
        # Second, independent guard: the mask-based checks above only ever see
        # WHETHER a column was observed. A fully-observed administrative field
        # (e.g. PLCO's reconsent_outcome_days) has no missingness gap to
        # detect at all, yet its VALUE can still perfectly determine the
        # outcome. assert_no_value_leakage screens each column's univariate
        # AUROC on its own observed rows, so -- unlike the mask guard -- it
        # needs no separate per-arm pass: a field recorded only within one
        # randomisation arm is already restricted to that arm's rows by
        # "observed", not diluted by the other arm's universal non-observation.
        assert_no_value_leakage(X, yv, usable)

    return Cohort(
        name=name, X=X, y=y.to_numpy(), mask=mask, columns=usable,
        feature_set=feature_set,
    )


def _is_excluded(col: str) -> bool:
    low = col.lower()
    return any(pat in low for pat in EXCLUDED_PATTERNS)


def _arm_strata(df: pd.DataFrame):
    """Yield ``(arm_value, boolean_row_mask)`` for each randomisation arm.

    PLCO assigns participants to a screening arm or a control arm, and many
    fields are only ever recorded in one of them. Leakage confined to one arm is
    diluted below the whole-cohort detection threshold, so the guard is re-run
    per arm. Yields nothing if no arm column is present, which makes the
    per-arm check a no-op rather than an error.

    The real column is named ``arm`` in all four PLCO cohorts (verified via
    each cohort's ``*_sas_formats_*.sas``: ``value armf; 1 = "Intervention"
    2 = "Control";`` -- identical coding across colorectal, lung, ovarian, and
    prostate). ``rndgroup``/``study_arm`` are kept as fallback candidate names
    in case a future data release renames it.
    """
    for candidate in ("arm", "rndgroup", "study_arm"):
        if candidate in df.columns:
            col = pd.to_numeric(df[candidate], errors="coerce")
            for val in sorted(col.dropna().unique()):
                yield val, (col == val).to_numpy()
            return


FIVE_YEARS_DAYS = 1826  # 5 * 365 + 1 leap day


def _derive_target(df: pd.DataFrame, name: str) -> pd.Series:
    """Death within five years of enrolment. 1 = died, 0 = survived.

    Coding verified 2026-07-19 against all four cohorts' SAS format files,
    which define ``mortality_exitstat`` identically::

        1 = Death      2 = Last NDI/Cutoff      3 = Refusal      4 = Other

    and ``mortality_exitdays`` as days from entry to mortality exit.

    - died within 5y  -> 1   (exitstat == 1 and exitdays <= 1826)
    - alive past 5y   -> 0   (exitdays > 1826, whatever happened afterwards)
    - censored before 5y without dying -> NaN, and the row is DROPPED by the
      caller, because five-year status is genuinely unknown for them.

    Measured on prostate: 3,940 died / 72,721 survived / 17 censored early,
    giving n = 76,661 at 5.14% prevalence. The 5.14% supersedes the paper's
    mock "6.9% to 12.1%" claim.
    """
    stat = pd.to_numeric(df["mortality_exitstat"], errors="coerce")
    days = pd.to_numeric(df["mortality_exitdays"], errors="coerce")

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y[(stat == 1) & (days <= FIVE_YEARS_DAYS)] = 1.0
    y[days > FIVE_YEARS_DAYS] = 0.0
    return y
