"""Our merge rule against the estimators the distributed-statistics literature already has.

A prior-art pass on one-shot federated learning placed our merge exactly: averaging
locally-fitted coefficients once is **one-shot averaging**, the classical distributed
estimator (Zhang, Duchi & Wainwright 2013), and two established refinements dominate it
in theory:

  Fisher weighting     weight each site's estimate by its observed information rather
                       than its sample size -- inverse-variance / fixed-effect
                       meta-analysis, and in the model-merging literature Matena &
                       Raffel (NeurIPS 2022). Still one round.
  one-step Newton      average, ship the average back, let each site return its score
                       and information at that point, take one Fisher-scoring step
                       (Huang & Huo, Math. Prog.; Jordan, Lee & Yang's CSL). Two rounds.

Neither is defined for sites holding **different covariate sets**, which is our setting,
so both need an extension before they can be run: a coordinate is combined only over the
sites that measure it, weighted by those sites' information for it, and the Newton step
is taken on the union with each site contributing only its own block.

A third axis is ours alone and is not a weighting question at all. ``composable._merge``
averages coefficients expressed in each site's **own standardised units**, then averages
the scalers -- so a variable with different spread at different sites has its
coefficients averaged in incommensurable units. Converting each local model to raw units
before combining removes that entirely, and costs nothing.

Baselines beyond the merges: each site's own local model, an unweighted ensemble of the
applicable local models (the one-shot FL family that needs no public data, unlike
distillation-based OSFL), and pooled training on the raw data.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

RIDGE = 1e-6


# ----------------------------------------------------------------- local fitting
def fit_local(X, y, cols):
    """Fit on a site's own rows and columns; return the model in BOTH parametrisations.

    ``coef``/``intercept`` are in the site's standardised units (what ``_merge`` uses);
    ``raw``/``raw_b`` are the same model rewritten in the variables' natural units, so
    that two sites' coefficients for one variable are directly comparable. ``info`` is
    the observed Fisher information in raw units, the inverse-variance weight.
    """
    A = np.asarray(X[:, cols], float)
    mu = np.nanmean(A, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    A = np.where(np.isnan(A), mu, A)
    if len(np.unique(y)) < 2:
        return None
    sc = StandardScaler().fit(A)
    sd = np.where(sc.scale_ > 0, sc.scale_, 1.0)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(A), y)
    coef = m.coef_.ravel().copy()
    b = float(m.intercept_[0])

    raw = coef / sd                      # z = b + sum coef*(x-mu)/sd = raw_b + raw . x
    raw_b = b - float(raw @ mu)
    p = expit(raw_b + A @ raw)
    w = p * (1 - p)
    info = np.einsum("ij,i,ij->j", A, w, A)     # diag of X'WX, raw units
    return dict(cols=list(cols), mu=mu, sd=sd, coef=coef, intercept=b,
                raw=raw, raw_b=raw_b, info=info, n=len(y), A=A, y=np.asarray(y, float))


def apply_raw(model, X):
    """Score in raw units; an absent coordinate contributes nothing."""
    A = np.asarray(X[:, model["cols"]], float)
    Z = np.where(np.isnan(A), model["mu"], A)
    return expit(model["raw_b"] + Z @ model["raw"])


def apply_std(model, X):
    """Score in the site's standardised units -- the ``composable._merge`` convention."""
    A = np.asarray(X[:, model["cols"]], float)
    Z = np.where(np.isnan(A), 0.0, (A - model["mu"]) / model["sd"])
    return expit(model["intercept"] + Z @ model["coef"])


# ----------------------------------------------------------------------- merges
def merge(models, space, weight):
    """Coordinate-wise combination. ``space`` in {std, raw}; ``weight`` in {n, fisher}.

    A coordinate is combined only over the sites that measure it, which is the whole
    point of the rule; the weighting is what this experiment varies.
    """
    num, den = defaultdict(float), defaultdict(float)
    mu_n, sd_n, mu_d = defaultdict(float), defaultdict(float), defaultdict(float)
    b_n = b_d = 0.0
    for m in models:
        if m is None:
            continue
        c = m["raw"] if space == "raw" else m["coef"]
        b = m["raw_b"] if space == "raw" else m["intercept"]
        for j, v in enumerate(m["cols"]):
            w = m["info"][j] if weight == "fisher" else m["n"]
            num[v] += w * c[j]
            den[v] += w
            mu_n[v] += m["n"] * m["mu"][j]
            sd_n[v] += m["n"] * m["sd"][j]
            mu_d[v] += m["n"]
        b_n += m["n"] * b
        b_d += m["n"]
    cols = sorted(den)
    out = dict(cols=cols, mu=np.array([mu_n[v] / mu_d[v] for v in cols]),
               sd=np.array([max(sd_n[v] / mu_d[v], 1e-9) for v in cols]),
               n=int(b_d))
    coefs = np.array([num[v] / den[v] for v in cols])
    if space == "raw":
        out["raw"], out["raw_b"] = coefs, b_n / max(b_d, 1e-9)
    else:
        out["coef"], out["intercept"] = coefs, b_n / max(b_d, 1e-9)
    return out


def one_step_newton(models, start, C=1.0):
    """Two rounds: ship ``start`` back, aggregate each site's score and information there.

    Each site contributes only the block of coordinates it measures, so the assembled
    system is the union's; coordinates no site measures never enter it.

    Two details decide whether this works at all, and getting either wrong makes the
    established estimator look far worse than the naive average it is supposed to beat:

    *Units.* The step is taken in one **global** standardised space (the merged scaler),
    not in raw units. On raw PLCO variables the coordinates span several orders of
    magnitude, the information matrix is correspondingly ill-conditioned, and a single
    Newton step overshoots catastrophically.

    *Penalty.* The local fits are sklearn logistic regressions, which minimise
    ``0.5 |w|^2 + C * sum(loss)`` -- they are penalised. Targeting the *unpenalised* MLE
    from a penalised start makes the step chase a different optimum. The same penalty is
    therefore carried into the score and the information (intercept exempt).
    """
    cols = start["cols"]
    ix = {v: i for i, v in enumerate(cols)}
    d = len(cols)
    mu_g, sd_g = start["mu"], start["sd"]

    # start, expressed in the global standardised space
    theta = start["raw"] * sd_g
    b = start["raw_b"] + float(start["raw"] @ mu_g)

    G = np.zeros(d + 1)
    H = np.zeros((d + 1, d + 1))
    for m in models:
        if m is None:
            continue
        loc = np.array([ix[v] for v in m["cols"]])
        Z = (m["A"] - mu_g[loc]) / sd_g[loc]
        p = expit(b + Z @ theta[loc])
        r = m["y"] - p
        w = p * (1 - p)
        D = np.column_stack([np.ones(len(Z)), Z])
        idx = np.concatenate([[d], loc])                       # intercept in the last slot
        G[idx] += C * (D.T @ r)
        H[np.ix_(idx, idx)] += C * (D.T @ (D * w[:, None]))
    G[:d] -= theta                                             # d/dw of 0.5|w|^2
    H[np.diag_indices(d)] += 1.0                               # intercept left unpenalised
    H[np.diag_indices_from(H)] += RIDGE * max(np.trace(H) / (d + 1), 1.0)

    step = np.linalg.solve(H, G)

    # Step-halving on the aggregated penalised objective. The one-step estimator is
    # only guaranteed when the start is consistent for the target, and an average of
    # models fitted on two disjoint populations (women, men) is not consistent for the
    # pooled model over both -- so the undamped step overshoots badly. Each site returns
    # one scalar loss per trial step, which keeps this a two-round protocol.
    def objective(th, bb):
        tot = 0.0
        for m in models:
            if m is None:
                continue
            loc = np.array([ix[v] for v in m["cols"]])
            Z = (m["A"] - mu_g[loc]) / sd_g[loc]
            z = np.clip(bb + Z @ th[loc], -35, 35)
            tot += C * np.sum(np.logaddexp(0, z) - m["y"] * z)
        return tot + 0.5 * float(th @ th)

    best = objective(theta, b)
    out_th, out_b, used = theta, b, 0.0
    t = 1.0
    for _ in range(8):
        cand_th, cand_b = theta + t * step[:d], b + t * step[d]
        val = objective(cand_th, cand_b)
        if val < best:
            best, out_th, out_b, used = val, cand_th, cand_b, t
            break
        t *= 0.5
    theta, b = out_th, out_b

    out = dict(start)
    out["raw"] = theta / sd_g
    out["raw_b"] = b - float(out["raw"] @ mu_g)
    out["step_size"] = used
    return out


def newton_iterate(models, start, checkpoints=(1, 2, 5, 10, 25, 50, 100)):
    """Run the aggregated Fisher scoring to convergence, snapshotting on the way.

    Reporting "the one-step refinement fails" would be worthless without this. One step
    from a poor start can land anywhere; what matters is the estimator's *target*. If the
    iteration converges somewhere worse than the naive average, the finding is about
    which objective the distributed-statistics refinements pursue, not about step size.
    """
    cur = start
    out = {}
    for i in range(1, max(checkpoints) + 1):
        nxt = one_step_newton(models, cur)
        moved = nxt["step_size"] > 0
        cur = nxt
        if i in checkpoints:
            out[i] = dict(cur)
        if not moved:
            for c in checkpoints:
                out.setdefault(c, dict(cur))
            break
    return out


def ensemble(models, X):
    """Unweighted mean of the applicable local models' probabilities."""
    P = [apply_raw(m, X) for m in models if m is not None]
    return np.mean(P, 0)


def pooled(tr):
    X = np.vstack([t[0] for t in tr])
    y = np.concatenate([t[1] for t in tr])
    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)

    def D(A):
        return np.column_stack([imp.transform(A), (~np.isnan(A)).astype(float)])
    sc = StandardScaler().fit(D(X))
    m = LogisticRegression(max_iter=2000).fit(sc.transform(D(X)), y)
    return lambda A: m.predict_proba(sc.transform(D(A)))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-site", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/merge_rules.csv")
    a = ap.parse_args()

    from hgmiss.data.plco import load_cohort
    R = Path(os.environ["PLCO_ROOT"])
    co = {c: load_cohort(c, R, feature_set="baseline") for c in ["ovarian", "prostate"]}
    union = sorted(set(co["ovarian"].columns) | set(co["prostate"].columns))
    ix = {v: i for i, v in enumerate(union)}
    sites = []
    for nm, c in co.items():
        keep, _ = train_test_split(np.arange(len(c.y)), train_size=a.n_per_site,
                                   stratify=c.y, random_state=0)
        M = np.full((len(keep), len(union)), np.nan)
        for j, v in enumerate(c.columns):
            M[:, ix[v]] = c.X[keep, j]
        sites.append((M, c.y[keep], [ix[v] for v in c.columns], nm))
    print(f"union d={len(union)}  " + ", ".join(f"{nm} n={len(y)}" for _, y, _, nm in sites))

    A = defaultdict(list)
    steps = []
    for seed in range(a.seeds):
        tr, te = [], []
        for X, y, cols, nm in sites:
            i1, i2 = train_test_split(np.arange(len(y)), test_size=0.3,
                                      stratify=y, random_state=seed)
            tr.append((X[i1], y[i1], cols))
            te.append((X[i2], y[i2], cols, nm))
        loc = [fit_local(X, y, c) for X, y, c in tr]
        pl = pooled(tr)
        merges = {
            "merge_std_n (ours, current)": (merge(loc, "std", "n"), apply_std),
            "merge_std_fisher": (merge(loc, "std", "fisher"), apply_std),
            "merge_raw_n": (merge(loc, "raw", "n"), apply_raw),
            "merge_raw_fisher": (merge(loc, "raw", "fisher"), apply_raw),
        }
        newton = one_step_newton(loc, merges["merge_raw_fisher"][0])
        steps.append(newton["step_size"])
        iters = newton_iterate(loc, merges["merge_raw_fisher"][0])
        for i, (Xt, yt, cols, nm) in enumerate(te):
            for k, (mdl, fn) in merges.items():
                A[k].append(average_precision_score(yt, fn(mdl, Xt)))
            A["onestep_newton (2 rounds)"].append(
                average_precision_score(yt, apply_raw(newton, Xt)))
            for it, mdl in iters.items():
                A[f"newton_{it}_iters"].append(
                    average_precision_score(yt, apply_raw(mdl, Xt)))
            A["local_only"].append(average_precision_score(yt, apply_raw(loc[i], Xt)))
            A["ensemble_of_locals"].append(average_precision_score(yt, ensemble(loc, Xt)))
            A["pooled_raw_data"].append(average_precision_score(yt, pl(Xt)))

    print(f"\none-step Newton accepted step sizes: {steps}")
    base = np.array(A["merge_std_n (ours, current)"])
    print(f"\n{'method':30s} {'AUPRC':>8s} {'vs ours':>9s} {'p':>8s}")
    rows = []
    for k, v in sorted(A.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v)
        d = v - base
        p = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
        flag = ("  BETTER" if d.mean() > 0 and p < 0.05
                else "  worse" if d.mean() < 0 and p < 0.05 else "")
        print(f"{k:30s} {v.mean():8.4f} {d.mean():+9.4f} {p:8.1e}{flag}")
        rows.append(dict(method=k, auprc=float(v.mean()), delta=float(d.mean()),
                         p=float(p), n_paired=len(v)))

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
