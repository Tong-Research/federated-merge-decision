"""One loader for every derived site matrix, so a new site collection costs no code.

Four datasets each had a bespoke loader, which was fine while there were four. It stopped
being fine once the analysis started manipulating the *data* as the independent variable:
the d-sweep (eICU at 8/16/24/42 labs), the temporal split, the long-stay outcome, the
cohort-floor sensitivity, ACS at two feature widths. Each is a legitimate `(X, y, g,
names)` collection and none of them deserves its own `if` branch in five scripts.

So every builder writes `<name>.npz` into `MIMIC_CACHE` with the same four arrays, and
every analysis script takes `--matrix <name>`. Adding a site collection now means running
its builder; no analysis code changes.

`MIMIC_CACHE` defaults to `~/.cache/phd-matrices` rather than a scratch directory. The
matrices are hours of derivation from restricted data and must not live somewhere a
temp-file reaper can take them -- which was very nearly what happened.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("MIMIC_CACHE", Path.home() / ".cache/phd-matrices"))


def load_matrix(name: str, verbose: bool = True):
    """Return (X, y, g, site_names) for a derived matrix by name."""
    path = CACHE / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.stem for p in CACHE.glob('*.npz')) or '(cache empty)'}")
    # No allow_pickle: every array here is numeric or fixed-width unicode, so the default
    # refusal to unpickle costs nothing and means a tampered .npz cannot execute code.
    z = np.load(path)
    X, y, g = z["X"], z["y"], z["g"]
    names = [str(v) for v in z["names"]]
    if verbose:
        print(f"{name}: {X.shape[0]} records x {X.shape[1]} features, "
              f"{len(names)} sites, prevalence {float(np.mean(y)):.3f}", flush=True)
    return X, y, g, names
