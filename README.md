# From Binary Gating to Continuous Shrinkage: A Decision-Level Evaluation of Federated Data Merging

Code, pre-registrations and aggregate results for the article of the same title
(Vivatchai Kaveeta, Prompong Sugunnasil, Juggapong Natwichai, Chiang Mai University). Generated from the authors' research monorepo at
commit `d6a46e4` on 2026-09-10 by `build_release.py`; the generator's verification step imported the
library and regenerated the tables and figures in this tree.

## Layout
- `src/hgmiss/` — the library (estimators, baselines, protocol, metrics, data loaders).
- `experiments/` — every experiment script the article uses, and the queue tooling the runs were made with.
- `figures/` — the figure and table generators (run from inside `figures/`) and their aggregate inputs.
- `results/` — aggregate outputs (per seed and fold, or per decision for public collections) that the tables are built from; files over 1 MB are gzip-compressed and `reproduce.sh` inflates them.
- `predictions/` — the job files carrying each pre-registered prediction and the runner's record of the run (the ledger the article's 'What We Got Wrong' section counts).

## What is not here
The manuscript source is not included: this repository carries the code, the
pre-registrations and the aggregate results only. `reproduce.sh` regenerates the tables and
figures from `results/`.

## Reproduce the tables and figures from the shipped results
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
./reproduce.sh
```

## Re-run the experiments
MIMIC-III, MIMIC-IV and eICU: PhysioNet credentialed access; build the site matrices with experiments/mimic_sites.py, mimic4_sites.py, eicu_sites.py. PLCO: CDAS application. ACS, Freddie Mac, Lending Club and Bosch are public; their builders are in experiments/. Per-decision outputs for the four restricted sources are NOT included (their figures are shipped as PDFs); per-decision outputs for the public collections are.
Set `PLCO_ROOT` to the PLCO directory where applicable. Runs are launched through `experiments/spinner_runner.py`
(a queue runner) or directly, e.g. `python experiments/run_h1_h4.py --cohort lung`.

## Data not included
Per-decision or per-record outputs derived from restricted sources are withheld under their data-use agreements: cut_dt_eicu_sites.csv (17940 rows); cut_eicu.csv (3588 rows); cut_fa_eicu_sites.csv (5980 rows); cut_fx_eicu_sites.csv (5980 rows); cut_hs_eicu_sites.csv (17940 rows); cut_pl1_eicu_sites.csv (5980 rows); cut_pl50_eicu_sites.csv (5980 rows); cut_pl5_eicu_sites.csv (5980 rows); cut_pm_eicu_sites.csv (5980 rows); cut_pmcv_eicu_sites.csv (5980 rows); cut_rb_eicu_sites.csv (17940 rows); cut_sc_eicu_sites.csv (5980 rows); lam_eicu.csv (3588 rows); site_rules_eicu_d16.csv (5980 rows); site_rules_eicu_d24.csv (5980 rows); site_rules_eicu_d42.csv (5980 rows); site_rules_eicu_d8.csv (5980 rows); site_rules_eicu_k10.csv (5994 rows); site_rules_eicu_longstay.csv (5980 rows); site_rules_eicu_only.csv (3588 rows); site_rules_hs_eicu_sites.csv (17940 rows); site_rules_hs_eicu_top30.csv (5340 rows); site_rules_recalib_eicu.csv (2994 rows); dp_calib_eicu.csv (7200 rows); dp_calib_eicu2.csv (7200 rows); dp_eicu_4arm.csv (6000 rows); dp_epvweight_eicu.csv (3600 rows); dp_hs_eicu_sites.csv (18000 rows); dp_merge_eicu.csv (6000 rows); dp_npower_eicu.csv (3600 rows); weight_hs_eicu_sites.csv (7980 rows); dsweep_eicu.csv (9576 rows); gate_eicu.csv (3588 rows)

## Licence
MIT (see LICENSE). Please cite the article (CITATION.cff).
