"""Experimental estimators. Nothing here may be imported by the frozen method's code path.

The published method is defined by `hgmiss.estimator`, `hgmiss.shrinkage` and `hgmiss.patterns`.
Those three modules are FROZEN: every number in Paper 1 and Paper 3 was produced by them, and
FINGERPRINTS.md records SHA checksums over the results they generate. Editing them to accommodate
an idea would silently invalidate three papers' worth of committed numbers, and the invalidation
would not announce itself -- the fingerprints would simply stop matching, months later, with no
way to tell which change caused it.

So ideas subclass or wrap; they do not patch. `frozen_guard.py` enforces this.
"""
