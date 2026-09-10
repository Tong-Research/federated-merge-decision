"""Repeated stratified CV and paired significance testing."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from .metrics import evaluate

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _run_one_fold(fit_predict, X, y, seed, fold, tr, te):
    """One (seed, fold) evaluation. BLAS pinned to a single thread so that
    when many folds run as parallel processes they do not oversubscribe --
    the per-pattern workload is thousands of tiny fits, where single-threaded
    BLAS is fastest and multi-threaded BLAS is pure overhead."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        prob = np.asarray(fit_predict(X[tr], y[tr], X[te])).ravel()
    if prob.shape[0] != te.shape[0]:
        raise ValueError(
            f"fit_predict returned {prob.shape[0]} predictions "
            f"for {te.shape[0]} test rows"
        )
    return {"seed": seed, "fold": fold, **evaluate(y[te], prob)}


def run_cv(
    fit_predict: FitPredict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seeds: Sequence[int],
    n_splits: int = 5,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Repeated stratified CV.

    ``fit_predict`` receives training data only and returns probabilities for
    the test rows. Anything fitted on the full table -- imputers, scalers, the
    pattern set, kappa -- must be fitted inside this callback.

    ``n_jobs`` parallelises the (seed, fold) evaluations across processes.
    Folds are independent and each split is fixed by ``StratifiedKFold(
    random_state=seed)``, so the result is identical to the serial order
    regardless of ``n_jobs`` (rows are sorted back to (seed, fold) order).
    Each worker runs single-threaded BLAS (see ``_run_one_fold``). Keep
    ``n_jobs`` low for a memory-heavy ``fit_predict`` (e.g. the mask-interaction
    baseline, whose design matrix is tens of GB) -- each worker holds its own
    copy.
    """
    y = np.asarray(y).ravel()
    tasks = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            tasks.append((seed, fold, tr, te))

    if n_jobs == 1:
        rows = [_run_one_fold(fit_predict, X, y, s, f, tr, te)
                for (s, f, tr, te) in tasks]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_run_one_fold)(fit_predict, X, y, s, f, tr, te)
            for (s, f, tr, te) in tasks
        )
    rows.sort(key=lambda r: (r["seed"], r["fold"]))
    return pd.DataFrame(rows)


def _check_no_duplicate_keys(s: pd.Series, name: str) -> None:
    """Raise if ``s`` (indexed by (seed, fold)) has a repeated key.

    Duplicate (seed, fold) keys make ``.loc[common]`` a many-to-many join,
    which silently inflates the paired sample with non-independent rows.
    """
    dup = s.index[s.index.duplicated(keep=False)]
    if len(dup):
        raise ValueError(
            f"duplicate (seed, fold) keys for method {name!r}: "
            f"e.g. {dup[0]} -- fix the source data, do not deduplicate here"
        )


def paired_holm(
    df: pd.DataFrame, reference: str, metric: str
) -> pd.DataFrame:
    """Wilcoxon signed-rank vs ``reference`` on paired folds, Holm-corrected.

    Pairing is on (seed, fold): both methods must have been evaluated on the
    identical split for the comparison to be paired. Pairs where either side's
    metric is non-finite (e.g. NaN from a single-class fold, see
    ``metrics.evaluate``) are dropped before testing -- a non-finite value
    reaching ``scipy.stats.wilcoxon`` produces a NaN p-value that the Holm
    step-down loop is not safe against. The number of dropped pairs is
    reported in ``n_dropped`` so the loss is never silent.
    """
    ref = df[df.method == reference].set_index(["seed", "fold"])[metric]
    _check_no_duplicate_keys(ref, reference)
    out = []
    for name, grp in df[df.method != reference].groupby("method"):
        other = grp.set_index(["seed", "fold"])[metric]
        _check_no_duplicate_keys(other, name)
        common = ref.index.intersection(other.index)
        ref_common = ref.loc[common].to_numpy()
        other_common = other.loc[common].to_numpy()
        finite = np.isfinite(ref_common) & np.isfinite(other_common)
        n_pairs = int(finite.sum())
        n_dropped = int(len(common) - n_pairs)
        if n_pairs < 2:
            raise ValueError(
                f"not enough usable paired folds for {name}: only {n_pairs} "
                f"remained after dropping non-finite metric values "
                f"({len(common)} paired folds total)"
            )
        diff = ref_common[finite] - other_common[finite]
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(diff, zero_method="zsplit").pvalue
        out.append({
            "method": name,
            "median_diff": float(np.median(diff)),
            "p_raw": float(p),
            "n_pairs": n_pairs,
            "n_dropped": n_dropped,
        })

    if not out:
        # The loop above yields nothing when `df` holds the reference method alone --
        # which is what a partially-populated results file looks like on resume, and is
        # exactly the state colorectal was left in. Without this the function raised
        # `KeyError: 'p_raw'` from the sort below, naming a column instead of the
        # problem. A comparison with nothing to compare against must say so.
        present = sorted(df.method.unique())
        raise ValueError(
            f"no methods to compare against reference {reference!r}: the results "
            f"contain only {present}. Run the baseline methods for this "
            f"(cohort, feature_set) before the paired tests."
        )

    res = pd.DataFrame(out).sort_values("p_raw").reset_index(drop=True)
    m = len(res)
    # Holm: step-down, enforcing monotonicity of adjusted p-values
    adj, running = [], 0.0
    for i, p in enumerate(res.p_raw):
        running = max(running, min((m - i) * p, 1.0))
        adj.append(running)
    res["p_holm"] = adj
    return res
