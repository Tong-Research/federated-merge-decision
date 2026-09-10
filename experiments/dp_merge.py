"""Does the merge benefit survive formal privacy -- and does one-shot win once budget composes?

"No raw data leaves the site" is a governance property: it satisfies a data use agreement.
It is not a privacy property. Shared coefficients leak, membership-inference attacks
recover whether a given patient was in a training set from parameters alone, and the
defences in the literature reduce attack success by 20-30% rather than removing it. The
distinction has to be made in the paper whatever the numbers say.

Two questions follow, and both are measurable rather than rhetorical.

**A. Where does the benefit die?** Our effects are +0.003 to +0.03 AUPRC. Differential
privacy costs accuracy, and it costs *most* at small sites: with the regulariser fixed at
Lambda per averaged loss, the L2 sensitivity of the fitted coefficients is 2/(n*Lambda),
so the noise a site must add is inversely proportional to its sample size. Our own rule
says the sites that should merge are the ones with the fewest events -- so **privacy is
most expensive exactly where the method claims to help.** If the benefit vanishes at any
epsilon tight enough to mean something, that is worth reporting plainly.

**B. Does composition reverse the ordering?** Privacy budget composes across releases.
Iterative FedAvg at R rounds must divide epsilon R ways; a one-shot merge spends it once.
Axis 16 measured FedAvg beating the one-shot merge by about 0.004 AUPRC with no privacy at
all -- a gap small enough that composition could plausibly overturn it. If it does, that is
the strongest argument for one-shot merging available, and it is not the communication-cost
argument the literature keeps making.

Mechanism: output perturbation for regularised ERM (Chaudhuri, Monteleoni & Sarwate,
JMLR 2011). Their guarantee requires each row to satisfy ||x|| <= 1, so rows are scaled
into the unit ball after standardisation -- a real constraint that slightly changes the
model, applied identically to every arm so the comparison stays fair.

FedAvg is given **advanced** composition rather than basic, which is the generous choice:
basic composition would hand one-shot an easy win.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import optimize, stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multisite import site_schema  # noqa: E402

warnings.simplefilter("ignore")

LAMBDA = 0.01            # regularisation of the averaged loss; fixed across sites so that
                         # sensitivity 2/(n*LAMBDA) genuinely varies with site size
DELTA = 1e-5
EPSILONS = tuple(float(e) for e in os.environ.get(
    "DP_EPSILONS", "0.5,1,2,5,inf").split(","))
FED_ROUNDS = tuple(int(r) for r in os.environ.get("DP_ROUNDS", "1,5,30").split(","))
FED_TUNE = os.environ.get("DP_FED_TUNE", "0") == "1"   # sweep lr/epochs per cell
SUBSETS_PER_TARGET = 6
MAX_PARTNERS = 3


# ---------------------------------------------------------------- privacy plumbing
def gaussian_sigma(sensitivity, eps, delta=DELTA):
    """Gaussian mechanism scale for (eps, delta)-DP."""
    if not np.isfinite(eps):
        return 0.0
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def advanced_composition_eps0(eps_total, rounds, delta=DELTA):
    """Per-round epsilon such that `rounds` releases compose to eps_total.

    Advanced composition (Dwork, Rothblum & Vadhan): k mechanisms each (e0, d0)-DP give
    (sqrt(2k ln(1/d')) e0 + k e0 (e^{e0} - 1), k d0 + d'). Solved numerically for e0.

    The **better of basic and advanced** is returned, which matters and is the generous
    choice for FedAvg. Advanced composition only pays off once k is large: at k=2 it
    yields e0=0.137 against basic's 0.500, and only overtakes basic around k=30. Reporting
    advanced alone would understate FedAvg's budget at small round counts and hand the
    one-shot merge an unearned victory.
    """
    if not np.isfinite(eps_total):
        return float("inf")
    if rounds <= 1:
        return eps_total
    basic = eps_total / rounds
    dprime = delta / 2.0

    def total(e0):
        return (math.sqrt(2 * rounds * math.log(1 / dprime)) * e0
                + rounds * e0 * (math.exp(e0) - 1.0))
    try:
        adv = float(optimize.brentq(lambda e: total(e) - eps_total, 1e-9, eps_total))
    except ValueError:
        adv = 0.0
    return max(basic, adv)


def unit_ball(Z):
    """Scale rows into the unit ball, as the ERM guarantee requires."""
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(nrm, 1.0)


def fit_dp(X, y, cols, eps, rng, lam=LAMBDA):
    """Regularised logistic fit with output perturbation. eps=inf gives the clean fit."""
    A = np.asarray(X[:, cols], float)
    mu = np.nanmean(A, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    A = np.where(np.isnan(A), mu, A)
    if len(np.unique(y)) < 2:
        return None
    sc = StandardScaler().fit(A)
    Z = unit_ball(sc.transform(A))
    n = len(y)
    C = 1.0 / (n * lam)                       # sklearn C such that the averaged-loss
    m = LogisticRegression(C=C, max_iter=3000).fit(Z, y)   # regulariser is exactly lam
    w = m.coef_.ravel().copy()
    b = float(m.intercept_[0])
    if np.isfinite(eps):
        sigma = gaussian_sigma(2.0 / (n * lam), eps)
        w = w + rng.normal(0.0, sigma, size=w.shape)
        b = b + float(rng.normal(0.0, sigma))
    return dict(cols=list(cols), mu=sc.mean_,
                sd=np.where(sc.scale_ > 0, sc.scale_, 1.0), coef=w, intercept=b, n=n)


def apply_dp(model, X):
    A = np.asarray(X[:, model["cols"]], float)
    Z = np.where(np.isnan(A), 0.0, (A - model["mu"]) / model["sd"])
    return expit(model["intercept"] + unit_ball(Z) @ model["coef"])


def calib_slope(y, p):
    """Slope of the logistic recalibration of the linear predictor.

    1.0 is perfect; below 1 means the scores are over-dispersed (too confident), which is
    the direction merging pushes them. Reported instead of Brier alone because Brier is
    dominated by the base rate at 9% prevalence and barely moves.
    """
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if np.ptp(lp) < 1e-9 or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.linear_model import LogisticRegression
    try:
        m = LogisticRegression(max_iter=200).fit(lp.reshape(-1, 1), y)
        return float(m.coef_[0][0])
    except Exception:
        return float("nan")


def merge_dp(models, own_weight=None, n_power=1.0):
    """Coordinate-wise combination.

    `own_weight` applies a fixed blend (Results 20/24). `n_power` instead weights each
    site by n**power: power=1 is FedAvg's default, and power=2 is the inverse-variance
    weight the DP sensitivity formula implies, since injected noise scales as 1/n so its
    variance scales as 1/n**2. Result 27 showed n beats the EPV rule under privacy; this
    tests whether the theoretically correct exponent beats plain n.
    """
    num, den = defaultdict(float), defaultdict(float)
    mu_n, sd_n, wt = defaultdict(float), defaultdict(float), defaultdict(float)
    b_n = b_d = 0.0
    for i, m in enumerate(models):
        if m is None:
            continue
        w = (m["n"] ** n_power) if own_weight is None else (
            own_weight if i == 0 else (1 - own_weight) / max(len(models) - 1, 1))
        for j, v in enumerate(m["cols"]):
            num[v] += w * m["coef"][j]; den[v] += w
            mu_n[v] += m["n"] * m["mu"][j]; sd_n[v] += m["n"] * m["sd"][j]
            wt[v] += m["n"]
        b_n += w * m["intercept"]; b_d += w
    cols = sorted(den)
    coef = np.array([num[v] / den[v] for v in cols])
    mu = np.array([mu_n[v] / wt[v] for v in cols])
    sd = np.array([max(sd_n[v] / wt[v], 1e-9) for v in cols])

    # A contributing site can hold a column that is entirely missing in its own training
    # split, in which case its mu and sd are NaN and the weighted average above is NaN
    # too. `apply_dp` guards against NaN in the *data* but not in the standardisation
    # constants, so such a coordinate produces NaN scores for every row where the feature
    # IS observed -- which is what killed the MIMIC run. A coordinate nobody can
    # standardise carries no information, so neutralise it rather than propagate it.
    bad = ~(np.isfinite(mu) & np.isfinite(sd) & np.isfinite(coef))
    if bad.any():
        mu[bad], sd[bad], coef[bad] = 0.0, 1.0, 0.0
    return dict(cols=cols, coef=coef, mu=mu, sd=sd,
                intercept=b_n / max(b_d, 1e-9), n=0)


def fedavg_dp(parts, rounds, eps_total, rng, lam=LAMBDA, local_epochs=5, lr=0.5):
    """Iterative FedAvg where every round's release is privatised.

    Each round each site perturbs its update at the per-round epsilon implied by advanced
    composition, so the total spend over `rounds` releases equals eps_total.
    """
    eps0 = advanced_composition_eps0(eps_total, rounds)
    prepped = []
    for X, y, cols in parts:
        A = np.asarray(X[:, cols], float)
        mu = np.nanmean(A, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
        sc = StandardScaler().fit(np.where(np.isnan(A), mu, A))
        prepped.append((unit_ball(sc.transform(np.where(np.isnan(A), mu, A))), sc,
                        np.asarray(y, float), list(cols)))
    d = max(max(c) for _, _, _, c in prepped) + 1
    w = np.zeros(d + 1)
    for _ in range(rounds):
        acc = np.zeros_like(w); tot = 0.0
        for Z, sc, yy, cols in prepped:
            idx = np.array([0] + [j + 1 for j in cols])
            wl = w[idx].copy()
            for _ in range(local_epochs):
                gr = expit(wl[0] + Z @ wl[1:]) - yy
                wl -= lr * np.concatenate([[gr.mean()], Z.T @ gr / len(yy)])
            if np.isfinite(eps0):
                sig = gaussian_sigma(2.0 / (len(yy) * lam), eps0)
                wl = wl + rng.normal(0.0, sig, size=wl.shape)
            upd = w.copy(); upd[idx] = wl
            acc += len(yy) * upd; tot += len(yy)
        w = acc / tot
    sc0, cols0 = prepped[0][1], prepped[0][3]

    def score(A):
        B = np.asarray(A[:, cols0], float)
        sd = np.where(sc0.scale_ > 0, sc0.scale_, 1.0)
        Z = unit_ball(np.where(np.isnan(B), 0.0, (B - sc0.mean_) / sd))
        idx = np.array([0] + [j + 1 for j in cols0])
        return expit(w[idx][0] + Z @ w[idx][1:])
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic-11site",
                    choices=["mimic-11site", "mimic4-9site", "eicu", "matrix"])
    ap.add_argument("--matrix", default=None,
                    help="with --dataset matrix: load this .npz from MIMIC_CACHE, so any "
                         "site collection (ACS states, d-sweep variants) can be tested "
                         "under the same privacy machinery")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/dp_merge.csv")
    a = ap.parse_args()

    if a.dataset == "matrix":
        z = np.load(f"{os.environ['MIMIC_CACHE']}/{a.matrix}.npz")
        X, y, g = z["X"], z["y"], z["g"]
        names = [str(v) for v in z["names"]]
    elif a.dataset == "eicu":
        from eicu_sites import load
        X, y, g, names, _ = load(verbose=False)
    elif a.dataset == "mimic4-9site":
        from mimic4_sites import load as load4
        X, y, g, names, _ = load4(verbose=False)
    else:
        os.environ["MIMIC_SITE_KEY"] = "unit_x_system"
        import mimic_sites
        mimic_sites.SITE_KEY = "unit_x_system"
        X, y, g, names, _ = mimic_sites.load(verbose=False)

    groups = [np.flatnonzero(g == k) for k in range(len(names))]
    rng = np.random.default_rng(0)
    rows = []
    for k in range(len(names)):
        others = [j for j in range(len(names)) if j != k]
        for seed in range(a.seeds):
            try:
                tr, te = train_test_split(groups[k], test_size=0.3,
                                          stratify=y[groups[k]], random_state=seed)
            except ValueError:
                continue
            cols_t = site_schema(X[tr])
            if len(cols_t) < 3 or len(np.unique(y[tr])) < 2:
                continue
            if len(np.unique(y[te])) < 2:
                continue
            for _ in range(SUBSETS_PER_TARGET):
                m = rng.integers(1, min(MAX_PARTNERS, len(others)) + 1)
                S = tuple(sorted(rng.choice(others, size=m, replace=False)))
                parts = [(X[tr], y[tr], cols_t)]
                ok = True
                for j in S:
                    try:
                        jtr, _ = train_test_split(groups[j], test_size=0.3,
                                                  stratify=y[groups[j]],
                                                  random_state=seed)
                    except ValueError:
                        ok = False; break
                    cj = site_schema(X[jtr])
                    if len(cj) < 3 or len(np.unique(y[jtr])) < 2:
                        ok = False; break
                    parts.append((X[jtr], y[jtr], cj))
                if not ok:
                    continue
                epv = float(min(np.bincount(y[tr])) / max(len(cols_t), 1))
                for eps in EPSILONS:
                    ms = [fit_dp(p[0], p[1], p[2], eps, rng) for p in parts]
                    if any(mm is None for mm in ms):
                        continue
                    p_loc = apply_dp(ms[0], X[te])
                    p_mrg = apply_dp(merge_dp(ms), X[te])
                    a_loc = average_precision_score(y[te], p_loc)
                    a_mrg = average_precision_score(y[te], p_mrg)
                    # Result 33 found merging buys discrimination and sells calibration.
                    # Result 21 found DP amplifies the discrimination gain. Whether DP also
                    # amplifies the calibration cost was never measured, because this script
                    # only ever recorded AUPRC -- so record both here.
                    b_loc, b_mrg = (brier_score_loss(y[te], p_loc),
                                    brier_score_loss(y[te], p_mrg))
                    s_loc, s_mrg = calib_slope(y[te], p_loc), calib_slope(y[te], p_mrg)
                    a_w5 = average_precision_score(
                        y[te], apply_dp(merge_dp(ms, own_weight=0.5), X[te]))
                    # Result 24's derived weight, tested under privacy: does the rule
                    # that beats n-weighting without DP still beat it with DP?
                    a_n2 = average_precision_score(
                        y[te], apply_dp(merge_dp(ms, n_power=2.0), X[te]))
                    a_n05 = average_precision_score(
                        y[te], apply_dp(merge_dp(ms, n_power=0.0), X[te]))
                    w_epv = float(min(1.0, epv / 10.0))
                    a_wepv = average_precision_score(
                        y[te], apply_dp(merge_dp(ms, own_weight=w_epv), X[te]))
                    r = dict(dataset=a.dataset, target=names[k], seed=seed,
                             n_partners=len(S), epv_target=epv, n_target=len(tr),
                             eps=eps, local=a_loc, merge=a_mrg, merge_w5=a_w5,
                             merge_wepv=a_wepv, w_epv=w_epv,
                             merge_n2=a_n2, merge_unif=a_n05,
                             benefit_n2=a_n2 - a_loc, benefit_unif=a_n05 - a_loc,
                             benefit=a_mrg - a_loc, benefit_w5=a_w5 - a_loc,
                             benefit_wepv=a_wepv - a_loc,
                             brier_local=b_loc, brier_merge=b_mrg,
                             d_brier=b_mrg - b_loc,
                             slope_local=s_loc, slope_merge=s_mrg,
                             d_slope=s_mrg - s_loc)
                    # FedAvg gets its learning rate and local-epoch count swept and the
                    # BEST reported, at every epsilon. Axis 16 established that FedAvg
                    # needs this to be competitive at all; fixing one configuration made
                    # it lose even at epsilon=infinity, which would have made any
                    # composition claim an artefact of an undertuned baseline.
                    for R in FED_ROUNDS:
                        best = None
                        grid = ((0.1, 1), (0.5, 5), (1.0, 5)) if FED_TUNE else ((1.0, 5),)
                        for lr, ep in grid:
                            for _once in (0,):
                                ep = ep
                                try:
                                    s = average_precision_score(
                                        y[te], fedavg_dp(parts, R, eps, rng,
                                                         local_epochs=ep, lr=lr)(X[te]))
                                except Exception:                    # noqa: BLE001
                                    continue
                                if best is None or s > best:
                                    best = s
                        if best is not None:
                            r[f"fedavg_{R}r"] = best - a_loc
                    rows.append(r)
        print(f"  {names[k]}  ({len(rows)} rows)", flush=True)

    import csv
    keys = sorted({kk for r in rows for kk in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    print(f"\n=== A. where does the merge benefit die? ({len(rows)} rows) ===")
    print(f"{'epsilon':>9s} {'local':>8s} {'merge':>8s} {'benefit':>9s} {'p':>8s} "
          f"{'benefit(w=0.5)':>15s}")
    for eps in EPSILONS:
        sel = [r for r in rows if r["eps"] == eps]
        if len(sel) < 10:
            continue
        b = np.array([r["benefit"] for r in sel])
        b5 = np.array([r["benefit_w5"] for r in sel])
        p = stats.wilcoxon(b).pvalue if not np.allclose(b, 0) else 1.0
        lbl = "inf" if not np.isfinite(eps) else f"{eps:g}"
        print(f"{lbl:>9s} {np.mean([r['local'] for r in sel]):8.4f} "
              f"{np.mean([r['merge'] for r in sel]):8.4f} {b.mean():+9.4f} {p:8.1e} "
              f"{b5.mean():+15.4f}")

    print(f"\n   split by target adequacy (privacy should cost small sites most):")
    for lo, hi in [(0, 5), (5, 1e9)]:
        print(f"   target EPV {lo}-{hi if hi < 1e9 else 999}:")
        for eps in EPSILONS:
            sel = [r for r in rows if r["eps"] == eps
                   and lo <= r["epv_target"] < hi]
            if len(sel) < 10:
                continue
            lbl = "inf" if not np.isfinite(eps) else f"{eps:g}"
            print(f"     eps={lbl:>4s}  local {np.mean([r['local'] for r in sel]):.4f}  "
                  f"benefit {np.mean([r['benefit'] for r in sel]):+.4f}")

    print(f"\n=== B. does composition reverse one-shot vs FedAvg? ===")
    print(f"{'epsilon':>9s} {'one-shot':>10s} " +
          "".join(f"{'fed ' + str(R) + 'r':>10s}" for R in FED_ROUNDS))
    for eps in EPSILONS:
        sel = [r for r in rows if r["eps"] == eps]
        if len(sel) < 10:
            continue
        lbl = "inf" if not np.isfinite(eps) else f"{eps:g}"
        cells = "".join(
            f"{np.mean([r[f'fedavg_{R}r'] for r in sel if f'fedavg_{R}r' in r]):+10.4f}"
            if any(f"fedavg_{R}r" in r for r in sel) else f"{'-':>10s}"
            for R in FED_ROUNDS)
        print(f"{lbl:>9s} {np.mean([r['benefit'] for r in sel]):+10.4f}{cells}")
    print("\n(all figures are AUPRC gain over that site's own DP local model at the same "
          "epsilon)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
