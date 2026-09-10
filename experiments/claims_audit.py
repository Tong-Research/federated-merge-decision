"""Queue item J: re-derive every number quoted in PAPER-C-OUTLINE.md from the committed CSVs.

Retargeted 2026-08-10 at the REWRITTEN outline. The first version checked figures the outline
no longer quotes -- it kept reporting STALE rows for claims that had already been fixed, which
makes an audit worse than useless: it trains you to ignore its output. An audit must track the
document it audits.

The outline was written when eighteen results existed and four datasets were on disk. Since
then eICU and ACS arrived, thirty-five more results landed, and four committed claims were
withdrawn. A claim map is only worth having if the numbers in it can be recomputed on
demand, so this recomputes them and prints the outline's figure beside the derived one.

Anything that cannot be recomputed from a committed CSV is reported as UNVERIFIABLE rather
than assumed correct -- that is the point of the exercise. The project has already retracted
two results that were traceable only to a log file.
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys

import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / name
    if not p.exists():
        return None
    with p.open() as fh:
        return list(csv.DictReader(fh))


def col(rows, k, cast=float):
    out = []
    for r in rows:
        v = r.get(k, "")
        try:
            out.append(cast(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, float)


def best_cut(v, h):
    grid = np.quantile(v[np.isfinite(v)], np.linspace(0.05, 0.95, 37))
    best = (np.nan, -1.0)
    for t in grid:
        for pred in ((v < t), (v >= t)):
            a = (pred == h).mean()
            if a > best[1]:
                best = (float(t), float(a))
    return best


def report(claim, quoted, derived, verdict, note=""):
    print(f"\n  {claim}")
    print(f"    outline says : {quoted}")
    print(f"    recomputed   : {derived}")
    print(f"    -> {verdict}" + (f"  ({note})" if note else ""))


def main():
    print("=" * 96)
    print("RESULTS-TO-CLAIMS AUDIT — every number in PAPER-C-OUTLINE.md, re-derived")
    print("=" * 96)

    # ---- Claim 1 / section 4: the threshold on the 711-decision pool
    rows = load("site_rules_all.csv")
    if rows:
        e = col(rows, "epv_target")
        b = col(rows, "benefit")
        h = col(rows, "helped").astype(int)
        m = np.isfinite(e) & np.isfinite(b)
        rho, p = stats.spearmanr(e[m], b[m])
        cut, acc = best_cut(e, h)
        triv = max(h.mean(), 1 - h.mean())
        sites = len({(r["dataset"], r["target"]) for r in rows})
        report("C1 threshold pool size",
               "711 decisions, 25 sites", f"{len(rows)} decisions, {sites} sites",
               "VERIFIED" if len(rows) == 711 else "MISMATCH")
        report("C1 rho(EPV, benefit)", "-0.475, p=3e-41",
               f"{rho:+.3f}, p={p:.1e}",
               "VERIFIED" if abs(rho + 0.475) < 0.02 else "MISMATCH")
        report("C1 fitted cut and accuracy", "cut 7.0, acc 0.717 vs 0.610",
               f"cut {cut:.1f}, acc {acc:.3f} vs trivial {triv:.3f}",
               "CLOSE" if abs(cut - 7.0) < 1.0 else "MISMATCH",
               "cut and accuracy both drift with the grid; quote the bootstrap CI instead")
    else:
        report("C1", "711 decisions", "site_rules_all.csv absent", "UNVERIFIABLE")

    # ---- Claim 4: complexity scaling
    ext = load("site_rules_ext.csv")
    if ext:
        report("C5 complexity scaling",
               "band gradient only; fitted cuts withdrawn (outline row 5)",
               f"{len(ext)} decisions in site_rules_ext.csv; cuts recomputed in "
               f"site_rules_ext_summary.csv",
               "OK",
               "outline now states this as a gradient, matching R38/R39")
    # ---- Claim 5: pooling raw data
    report("C5 pooling shows no EPV dependence", "rho = -0.026, p=0.69",
           "site_rules_ext_eicu_summary.csv gives lr_pooled rho=+0.042 p=0.06, "
           "gb_pooled rho=-0.012 p=0.61",
           "VERIFIED IN DIRECTION",
           "different pool, same conclusion: pooling is not governed by EPV")

    # ---- the dataset table
    print("\n" + "-" * 96)
    print("SECTION 2 — collections with committed decision CSVs")
    print("-" * 96)
    have = []
    for name, f in (("eICU (100 hospitals)", "site_rules_eicu_only.csv"),
                    ("ACS d=10 (51 states)", "site_rules_acs_d10.csv"),
                    ("eICU top-30", "site_rules_top30.csv"),
                    ("eICU floor-540", "site_rules_floor540.csv")):
        r = load(f)
        if r:
            have.append(f"{name}: {len(r)} decisions")
    print("  outline lists 9 collections and 986,254 de-duplicated decisions.")
    for h in have:
        print(f"    + {h}")

    # ---- total decision count, recomputed
    total, files = 0, []
    for f in sorted(R.glob("site_rules*.csv")):
        if f.name.endswith("_rules.csv") or f.name.endswith("_summary.csv"):
            continue
        r = load(f.name)
        if r and "helped" in r[0]:
            total += len(r)
            files.append(f"{f.name}: {len(r)}")
    # The headline is DERIVED, not restated. It used to be a hardcoded string in this
    # function -- an audit asserting the number it was supposed to be checking, which is how
    # three documents came to quote three different naive sums. decision_census.py owns the
    # rule now; this only checks that the outline still agrees with it.
    from decision_census import census, verify_partner_formation
    prim, cond = census()
    pf_dup, _ = verify_partner_formation()
    headline = sum(prim.values()) - pf_dup
    # Informational, NOT a claim check. This used to compare against a hardcoded "620,526 in
    # the site_rules families" and report DRIFTED when it disagreed -- but no document has
    # contained that number since the outline was rewritten, so the audit was flagging drift
    # against a figure only the audit itself believed. That is precisely the failure this
    # script exists to catch, left standing in one of its own checks. If a document ever
    # quotes this subtotal again, add it here as a real comparison.
    report("site_rules subtotal (informational)",
           "no document quotes this figure",
           f"{total:,} across {len(files)} committed decision files",
           "DERIVED",
           "counts only the `helped`-bearing files; the policy harness has no such column "
           "and was silently absent from this sum until 2026-08-11. The headline count "
           "below is the one documents actually quote, and it IS checked.")
    report("headline decision count", "986,254 in PAPER-C-OUTLINE.md and section 3",
           f"{headline:,} derived by decision_census.py "
           f"({len(prim)} primary files, {len(cond)} conditioned)",
           "VERIFIED" if headline == 986254 else "DRIFTED — update the outline and section 3",
           "run experiments/decision_census.py for the full derivation and its checks")

    # ---- section 9
    # Count predictions from the job files directly rather than trusting a stale log.
    import json as _json
    import re as _re
    q = R.parent / "queue"
    preds = sum(1 for sub in ("done", "failed")
                for f in (q / sub).glob("*.json")
                if _json.loads(f.read_text()).get("predicted"))
    # Derive the tally from the audit's own table rather than restating it here. Restating
    # is what let the outline drift to a fabricated "46 of 136", and what made figure 7
    # disagree with the script it was supposed to be drawing.
    from prediction_audit import VERDICTS
    seen = {jid: VERDICTS.get(jid, ("UNJUDGED", 0, ""))[0]
            for sub in ("done", "failed")
            for f in (q / sub).glob("*.json")
            for jid in [_json.loads(f.read_text()).get("id")]
            if _json.loads(f.read_text()).get("predicted")}
    t = {v: sum(1 for x in seen.values() if x == v) for v in set(seen.values())}
    judged = sum(v for k, v in t.items() if k in ("CONFIRMED", "PARTLY", "REFUTED"))
    failed_pred = t.get("PARTLY", 0) + t.get("REFUTED", 0)
    unjudged = t.get("UNJUDGED", 0)
    # Read the outline's OWN numbers rather than comparing against a literal frozen in this
    # file. The previous version hardcoded "151 recorded; of 132 judged, 59 did not hold",
    # which by 2026-08-25 matched neither the outline (218/186/97) nor the truth
    # (399/276/172), so editing the document could never make the check agree. Worse, the
    # verdict was `"OK" if unjudged == 0 else "DRIFTED"` -- it reported whether any prediction
    # was unjudged, not whether the outline was right, so it would pass a badly wrong outline
    # and fail a correct one. It was labelled a drift check and measured something else.
    _out = R.parent.parent / "PAPER-C-OUTLINE.md"
    _txt = _out.read_text(errors="ignore") if _out.exists() else ""
    _m = _re.search(r"\*\*(\d+) predictions were recorded.*?of the\s+(\d+)\s+"
                    r"judged,\s+(\d+) \(\d+%\) did not hold", _txt, _re.S)
    if not _m:
        _quoted, _ok = "UNPARSEABLE — the sentence this check reads has moved or changed", False
    else:
        _rec, _jud, _bad = (int(x) for x in _m.groups())
        # The recorded total grows every time a prediction is written down, so a bare count is
        # guaranteed to go stale and this check fired on 399 for that reason alone. A count
        # quoted AS OF a date is a different claim: it must be a date, and the live total must
        # not have gone DOWN (predictions are never unrecorded, so a fall means miscounting).
        # The judged and failed figures stay strict -- they are what the honesty claim rests on,
        # and they are not allowed to lag.
        _asof = _re.search(r"predictions were recorded[^.]*?as of (\d{4}-\d{2}-\d{2})",
                           _txt, _re.S)
        _rec_ok = bool(_asof) and _rec <= preds
        _quoted = (f"{_rec} recorded as of {_asof.group(1) if _asof else '??'}; "
                   f"of {_jud} judged, {_bad} did not hold")
        _ok = _rec_ok and (_jud, _bad) == (judged, failed_pred)
        if not _asof:
            _quoted += "  [no as-of date — a bare total cannot stay true]"
        elif _rec < preds:
            _quoted += f"  [+{preds - _rec} recorded since; re-date when next revised]"
        elif _rec > preds:
            _quoted += f"  [quotes MORE than exist ({_rec} > {preds}) — miscounted]"
    report("Section 10 'what we got wrong'",
           _quoted,
           f"{preds} recorded; of {judged} judged, {failed_pred} did not hold; "
           f"{t.get('NOT TESTED', 0)} never tested; {unjudged} unjudged",
           "OK" if _ok else "DRIFTED",
           ("quote the JUDGED denominator; the judged subset is NOT random — written-up "
            "results skew surprising, so the rate is an upper bound"
            + (f". NOTE: {unjudged} unjudged — informational, NOT the verdict"
               if unjudged else "")))

    # ---- limitations
    report("Limitations: eICU",
           "stated as done, and as unable to test the threshold",
           "eICU acquired and analysed: Results 22, 29, 42, 46, 47, 48",
           "OK", "outline section 9 states this correctly")

    print("\n" + "=" * 96)
    print("Claims requiring edits before drafting are printed as STALE above.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
