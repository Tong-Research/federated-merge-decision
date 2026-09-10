"""Substrings marking columns that must never enter the feature set.

Post-diagnosis fields (grade, stage, treatment) exist only for cases, so their
missingness indicator reproduces the label. Follow-up and exit fields encode the
outcome directly. This list was derived from the prostate dictionary and then
verified against the actual column names of all four PLCO cohorts (colo_*,
lung_*, ovar_*, pros_*) -- see the additions below the original starting set.
"""

import re

EXCLUDED_PATTERNS: tuple[str, ...] = (
    # --- original starting set (prostate-derived, per task brief) ---
    # post-diagnosis clinical detail
    "gleason", "stage", "grade", "morphology", "topography", "behavior",
    "curative", "neoadjuvant", "primary_trt", "dx_psa",
    "num_heslide", "has_deliv",
    # outcome / follow-up
    "exitstat", "exitage", "cancer_first", "intstat", "annyr",
    # NOTE: the brief's starting patterns ("reasfollp", "reassympp", ...) are
    # prostate-suffixed and silently fail to match the analogous colo_*c,
    # lung_*l, and ovar_reas* columns in the other three cohorts. Using the
    # suffix-free stems below catches all four. (A bare "reas" was tried and
    # rejected -- it collides with the legitimate ovarian family-history
    # columns "breast_fh"/"breast_fh_cnt"/"breast_fh_age", which contain
    # "reas" as a substring of "breast".)
    "reasfoll", "reassymp", "reassurv", "reasoth",
    "mortality", "dth", "is_dead",

    # --- additions found by inspecting all four cohorts' real columns ---
    # post-diagnosis pathology / histology detail not covered above
    "histtype",          # ovar_histtype, lung_histtype, lung_histtype_cat
    "carcinoid",          # lung_carcinoid, lung_carcinoid_summstage (subtype, cases only)
    "cancer_site",        # ovar_cancer_site
    "cancersite",         # d_cancersite, f_cancersite (cause-of-death cancer site)
    "diagdays",           # {cohort}_cancer_diagdays: days to diagnosis
    "seer",                # {cohort}_seer, d_seer_death, f_seer_death: registry linkage
    "death",               # d_codeath_cat, f_codeath_cat, build_death_cutoff
    "exitdays",            # {cohort}_exitdays, fstcan_exitdays: cancer-specific exit timing
    "num_canc",             # num_cancc, num_cancl: count of cancers diagnosed
    # nested case-control subsampling in the Lung cohort: selection/weighting
    # variables are constructed FROM the cancer outcome, so membership itself
    # is outcome-derived even though it is not a "missingness" pattern.
    "ccsub", "case_study", "caseset", "25k", "eligible_population",
    # bare cohort disease-incidence flags and their cohort-specific relatives:
    # whether the participant was EVER diagnosed with the cancer under study,
    # possibly after the 5-year mortality window used for the target. Using
    # this as an input feature is temporal (value) leakage, not missingness
    # leakage, so the mask-based guard would not catch it -- excluded on
    # inspection rather than because a test tripped.
    "pros_cancer", "lung_cancer", "colo_cancer", "ovar_cancer",
    "appendix_cancer",
    # first-cancer-of-any-site outcome (fstcan_exitstat/_exitage/_exitdays)
    "fstcan",
    # Follow-up-wave questionnaire administration/completion metadata:
    # entryage_*, entrydays_*, ph_any_*, ph_{cohort}_*, and {cohort}_eligible_*
    # are each recorded once per questionnaire instrument (bq = baseline, at
    # enrolment; dqx/dhq/sqx/muq = later follow-up waves, months to years
    # after enrolment). A participant who died before a later wave was mailed
    # cannot have completed it, so that wave's completion indicator is a
    # survivorship-conditional proxy for the mortality label -- structurally
    # the same failure mode as a post-diagnosis field, just driven by
    # follow-up timing instead of diagnosis. Measured directly on prostate:
    # entryage_dhq observed in 23.0% of 5y deaths vs 76.2% of 5y survivors
    # (gap 53.3%), entryage_sqx 0.0% vs 67.8% (gap 67.8%) -- both well past
    # the guard's 0.5 threshold. dqx/muq did not trip the guard on prostate
    # at the whole-cohort level, but share the identical causal mechanism, so
    # all three later waves are excluded uniformly rather than kept until a
    # different cohort happens to trip on them. Only "_bq" (baseline,
    # completed essentially universally near enrolment) is kept.
    "_dqx", "_dhq", "_sqx", "_muq",
    # Later-round screening exam results (prostate PSA/DRE, rounds 3-5, ~3-5y
    # post-enrolment): observation requires surviving to that screening round,
    # so within the intervention arm (screening only happens there -- see
    # _arm_strata) missingness tracks the mortality label directly. Measured
    # per-arm on prostate: psa_level5 observed in 0.0% of 5y deaths vs 71.2%
    # of 5y survivors within the screening arm. Gap grows monotonically with
    # round (round0 ~7%, round1 ~15%, round2 ~22-25%, round3+ >50%) -- an
    # informative-missingness gradient inherent to the screening design, not
    # a data error. Rounds 0-2 stay under the guard's threshold and are kept,
    # consistent with the guard being deliberately permissive of ordinary
    # informative missingness; rounds 3-5 exceed it and are excluded.
    "psa_level3", "psa_level4", "psa_level5",
    "psa_days3", "psa_days4", "psa_days5",
    "dre_days3",
    # Same mechanism, ovarian CA-125 screening (round 5, ~5y post-enrolment):
    # ca125_level5 observed in 0.0% of 5y deaths vs 58.3% of 5y survivors
    # within the screening arm; rounds 0-4 stay under the guard's threshold.
    "ca125_level5", "ca125ii_level5", "ca125_days5",

    # --- value leakage (assert_no_value_leakage / task 13-addendum) ---------
    # Columns below are 100% (or near-100%) OBSERVED -- the mask-based guard
    # above cannot see them at all, because there is no missingness gap to
    # detect. Univariate AUROC against 5-year mortality, measured directly on
    # real PLCO data 2026-07-20 via hgmiss.leakage.value_leakage_report:
    #
    #   reconsent_outcome_days   AUROC 0.9999-1.0000  (all 4 cohorts, ~100% observed)
    #   reconsent_outcome        AUROC 0.8599-0.9001  (all 4 cohorts, ~100% observed)
    #   in_TGWAS_population      AUROC 0.7399-0.7547  (all 4 cohorts, 100% observed)
    #
    # reconsent_outcome[_days] records the outcome of a later re-consent
    # contact attempt -- administratively, participants who died are
    # disproportionately easy (or impossible) to re-contact within a fixed
    # window, so this process field tracks the mortality label almost
    # exactly. in_TGWAS_population is trial-wide GWAS sub-study membership --
    # a study-administration flag, not a health measurement. Both are
    # excluded outright: not clinical variables under any interpretation.
    #
    # NOTE: in_TGWAS_population's 0.75 does not itself cross the guard's
    # max_auroc=0.90 default -- it is excluded on inspection (per the task
    # brief's explicit instruction to drop "GWAS/sub-study membership
    # flags"), the same category of scientific judgement already used above
    # for pros_cancer/lung_cancer/etc. Not every exclusion here is guard-
    # triggered; this one is a judgement call made explicit, not a silent one.
    "reconsent",
    # Incident-adenoma sub-study (colorectal): follow-up findings and sub-study
    # membership flags, not baseline measurements. iaden_aden_days is a
    # days-to-event field of the same character as reconsent_outcome_days
    # (univariate AUROC 0.84); iaden_protocol 0.85. Excluded 2026-07-20.
    "iaden", "gwas",

    # Two follow-up-round variable families that DO cross the value-leakage
    # guard's 0.90 threshold under feature_set="full" (they are already
    # removed under "baseline" by is_followup_screening_round, since both are
    # round-suffixed; this is only reachable via the "full" sensitivity
    # analysis). Same survivorship mechanism already documented above for
    # psa_level3/4/5 and the *_days35 family: only observed in participants
    # who survived to attend that late round, and the observed value itself
    # (not just its presence) is measured close enough to death to track the
    # outcome directly. Measured 2026-07-20:
    #
    #   psa_result5 (prostate)         AUROC 0.9523 (50% observed, screening arm)
    #
    # psa_result3/4 (0.82/0.82) and dre_result0-3 (<=0.82) stay under 0.90 and
    # are NOT excluded here -- consistent with the guard's permissive design,
    # only what actually crosses the line is added.
    "psa_result5",

    # Colorectal round-3-or-5 diagnostic-workup timing/size variables (the
    # colorectal-specific "35" round suffix -- see is_followup_screening_round
    # above). Of colorectal's 23 "_days35" columns, these 8 cross the 0.90
    # value-leakage threshold under feature_set="full"; the other 15 do not
    # and are left in place. Measured 2026-07-20:
    #
    #   sizerght_days35     AUROC 0.9359
    #   largeith_days35     AUROC 0.9229
    #   colsw_rght_days35   AUROC 0.9189
    #   sizeeith_days35     AUROC 0.9123
    #   colsw_left_days35   AUROC 0.9080
    #   hadcolsc_days35     AUROC 0.9079
    #   colsw_eith_days35   AUROC 0.9078
    #   largleft_days35     AUROC 0.9044
    #
    # All are observed in <5% of the cohort (the round-3/5 colonoscopy
    # follow-up subset) yet still clear min_observed -- their leakage is
    # concentrated, not diluted away by a large denominator.
    "sizerght_days35", "largeith_days35", "colsw_rght_days35",
    "sizeeith_days35", "colsw_left_days35", "hadcolsc_days35",
    "colsw_eith_days35", "largleft_days35",
)


# --- baseline-only feature set -------------------------------------------
#
# WHY THIS EXISTS (the survivorship mechanism):
#
# EXCLUDED_PATTERNS above removes the columns whose missingness gap crosses
# the leakage guard's 0.5 threshold. But a gap safely under 0.5 is not the
# same as "no contamination" -- it can simply mean the same survivorship
# mechanism has not yet crossed the guard's cutoff. Every PLCO follow-up
# measurement (a later screening round, a later-mailed questionnaire) can
# only be observed in a participant who was alive to attend or return it.
# The target here is death within five years, so "was this follow-up field
# observed" is itself weak evidence about survival -- contamination, not
# clinical signal, even when the guard passes it.
#
# Measured directly on prostate: round-2 PSA screening (psa_result2) is
# observed in 20% of 5-year deaths vs. 42% of 5-year survivors (a 22-point
# gap) -- well under the 0.5 guard threshold, so EXCLUDED_PATTERNS lets it
# through. Dropping every follow-up column (everything below) from the
# feature set drops whole-cohort mask informativeness (AUROC(mask -> y) -
# 0.5, see hgmiss.maskinfo.mask_informativeness) from 0.214 to 0.134 --
# roughly 40% of the apparent informativeness in the full feature set comes
# from this source. Because the mortality target is exactly what follow-up
# survival encodes, the baseline-only feature set (this rule) is the
# PRIMARY analysis; the full feature set is retained only as a sensitivity
# analysis to show how much the result depends on this contamination.
#
# THE RULE, verified against real column names (2026-07-19):
#
# PLCO screening-exam variables are named with a trailing round-index digit,
# e.g. prostate's ``psa_result0``, ``psa_days1``, ``dre_days2``,
# ``biopplink3``, ``pros_mra_stat4``; ovarian's ``ca125_level0..5``,
# ``tvu_days0..3``; lung's ``xry_result0..3``, ``lung_mra_stat0..3``; and
# colorectal's ``hadcolsc0/3/5/35``, ``colo_mra_stat0/3/5/35`` (colorectal
# uses its own "T[X]" round convention per the colorectal data dictionary,
# Section 12: Diagnostic Workup -- round 0 is baseline, 3 and 5 are the
# year-3/year-5 follow-up exams, and 35 is a derived "occurred in round 3 or
# 5" variable). In every cohort, index/suffix 0 is the round administered
# at or immediately after randomisation (baseline); indices 1 and up, and
# the colorectal "35" suffix, are later follow-up rounds.
#
# The suffix must be read as the FULL trailing digit run, not just the last
# character: ``bmi_20`` / ``bmi_50`` (recalled weight at age 20/50, a
# baseline-questionnaire item, per the prostate data dictionary Question 22)
# and ``race7`` (a 7-category race coding) all end in a digit that looks
# like a round index if you only look at the last character, but their full
# trailing run ("20", "50", "7") is not a valid round suffix, so they are
# correctly left alone.
#
# Note on a hypothesis that did NOT hold up: it was suspected that a
# trailing "_f" suffix (``arthrit_f``, ``weight_f``, ``height_f``, ...)
# marked follow-up-form variables and should also be excluded here. Checking
# the prostate data dictionary shows this is wrong -- every "_f" column
# inspected (arthrit_f, bronchit_f, diabetes_f, weight_f, height_f,
# weight20_f, weight50_f, vasect_f, ...) is listed under "Section 26: BQ
# Diseases" or an equivalent baseline-questionnaire (BQ) section, answering
# a baseline questionnaire item (e.g. Question 22, Question 27, Question
# M40); the trailing "_f" there stands for "field", not "follow-up", and
# ``.F = "No Form"`` is a sentinel for a missing baseline form, not a later
# one. These columns are baseline and are deliberately NOT excluded by this
# rule. Follow-up questionnaire waves genuinely are excluded, but via a
# different, already-existing route: DHQ/SQX/MUQ/DQX wave columns
# (``entryage_dhq``, ``ph_lung_sqx``, ...) all contain the substrings
# "_dhq"/"_sqx"/"_muq"/"_dqx" and are already removed unconditionally by
# EXCLUDED_PATTERNS above, under both "baseline" and "full" feature sets, so
# they need no additional handling here.

BASELINE_ROUND_SUFFIX = "0"
"""The PLCO screening-round suffix that denotes the baseline exam."""

FOLLOWUP_ROUND_SUFFIXES: frozenset[str] = frozenset({"1", "2", "3", "4", "5", "35"})
"""PLCO screening-round suffixes that denote a post-baseline follow-up exam.

"35" is colorectal-specific: a derived variable meaning "round 3 or round 5".
"""

_TRAILING_DIGITS_RE = re.compile(r"(\d+)$")


def is_followup_screening_round(col: str) -> bool:
    """True if ``col`` is a PLCO screening-round variable from round 1 or later.

    Extracts the *full* trailing digit run of the column name (not just its
    last character -- see the module-level note on ``bmi_20``/``bmi_50``/
    ``race7`` above) and checks it against ``FOLLOWUP_ROUND_SUFFIXES``.
    Columns with no trailing digit run, or whose run is ``BASELINE_ROUND_SUFFIX``
    ("0") or anything else outside ``FOLLOWUP_ROUND_SUFFIXES``, return False.
    """
    match = _TRAILING_DIGITS_RE.search(col)
    if match is None:
        return False
    return match.group(1) in FOLLOWUP_ROUND_SUFFIXES
