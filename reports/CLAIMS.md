# Claims ledger

Every quantitative claim the project makes, where its evidence lives, and whether it may still be quoted.
Status: **Current** (reproduced, may be quoted), **Diagnostic** (true but not evidence for the headline task),
**Exploratory** (not validated), **Withdrawn** (do not reuse). Last reviewed 2026-11-09.

**Frozen definitions** (from [the sealed-holdout protocol](repro/sealed_holdout_protocol.md), code in
`src/evaluation/frozen_metrics.py`): the **false-alert rate** is the share of safe possessions (no later shot or box
entry) with at least one eligible snapshot at or above the validation-chosen threshold; **ECE** uses ten equal-width
bins, weighted by snapshot count, in absolute probability units. Every false-alert rate and ECE quoted as current uses
these.

## Current

| Claim | Value | Evidence | Note |
|---|---|---|---|
| Event-time XGBoost, all test snapshots | ROC AUC 0.665 [0.647, 0.681] | [repair_retrain/event_time_results.md](repro/repair_retrain/event_time_results.md) | Fresh retrain, 500-resample match bootstrap |
| Event-time GRU, all test snapshots | ROC AUC 0.664 [0.642, 0.687] | same | |
| Ball x/y logistic baseline | ROC AUC 0.642 [0.622, 0.661] | same | The lift of the models over it is about 0.02; over a nonlinear position baseline it is smaller (row below) |
| One snapshot per possession | XGBoost 0.736, GRU 0.739 | same | Different sampling from the all-snapshot figure |
| Committed event-time models re-score | XGBoost 0.6649, GRU 0.6636 | [repro/artefact_check.md](repro/artefact_check.md) | Earlier training run than the fresh retrain; `make verify` |
| Pass progression, event-only / event+360 | ROC AUC 0.7337 / 0.7380 | [repro/artefact_check.md](repro/artefact_check.md) | 360 lift is small; coverage is partial |
| Sealed external holdout, UEFA Euro 2024 (51 matches, scored once) | ROC AUC XGBoost 0.662 [0.650, 0.674], GRU 0.662 [0.650, 0.674]; ball x/y baseline 0.641 [0.631, 0.652]; lift 0.021 [0.015, 0.027] / 0.021 [0.013, 0.028] | [repro/sealed_holdout.md](repro/sealed_holdout.md), [JSON](repro/sealed_holdout.json) | Committed models and thresholds unchanged; protocol committed first. By the pre-specified rule AUC 0.66 generalised to this competition. One men's tournament only |
| Second sealed external holdout, UEFA Euro 2020 (51 matches, scored once) | ROC AUC XGBoost 0.667 [0.654, 0.681], GRU 0.659 [0.647, 0.671]; ball x/y baseline 0.639 [0.629, 0.650]; position-only boosted baseline 0.648 [0.638, 0.658] | [repro/second_holdout.md](repro/second_holdout.md), [JSON](repro/second_holdout.json) | Protocol committed first. AUC 0.66 generalised for both models. GRU ranks slightly below XGBoost here: -0.009 [-0.016, -0.000] |
| Lift over the position-only boosted baseline (all snapshots) | XGBoost / GRU: Euro 2020 0.020 [0.014, 0.026] / 0.011 [0.002, 0.022]; Euro 2024 0.012 [0.005, 0.018] / 0.012 [0.004, 0.019]; in-cohort test 0.011 [-0.003, 0.023] / 0.010 [-0.013, 0.028] | [repro/second_holdout.json](repro/second_holdout.json) | Baseline added after the Euro 2024 scoring, frozen before any scoring. Lift holds on both tournaments, **not shown** in-cohort; smaller than the lift over ball x/y everywhere |
| Alert behaviour at validation-prevalence thresholds (frozen definitions) | Test: precision 0.552 / 0.554, recall 0.520 / 0.518, false-alert rate 48.0% / 39.7% (XGBoost / GRU), 2,284 possessions. Euro 2024: precision 0.575 / 0.576, recall 0.506 / 0.511, false-alert rate 45.2% / 39.3%. Euro 2020: precision 0.529 / 0.527, recall 0.520 / 0.501, false-alert rate 45.4% / 37.2% | [repro/frozen_metrics_test.json](repro/frozen_metrics_test.json), [repro/sealed_holdout.json](repro/sealed_holdout.json), [repro/second_holdout.json](repro/second_holdout.json) | Precision and recall per eligible snapshot; false-alert rate per safe possession |
| Precision-oriented operating point (target 0.65, chosen on validation) | XGBoost threshold 0.574: Euro 2020 precision 0.728, recall 0.050, false-alert rate 6.6%; test 0.699, 0.069, 8.5%. GRU: not reachable on validation | [repro/second_holdout.json](repro/second_holdout.json) | Recall is about 5 to 7% of dangerous snapshots: a high-confidence filter, not a detector |
| Calibration (ECE, snapshot-weighted, absolute probability) | Test 0.039 / 0.039; Euro 2024 0.063 / 0.062; Euro 2020 0.031 / 0.030 (XGBoost / GRU); on Euro 2020 the baselines reach 0.019 (ball x/y) and 0.022 (boosted) | same | Not a calibrated probability: use the score as a ranking aid |
| Fixed research split | 99 matches: 69 train, 15 validation, 15 test | [configs/research_split_manifest.csv](../configs/research_split_manifest.csv) | Guarded by `tests/test_repo_integrity.py` |

## Diagnostic and exploratory (never headline evidence)

| Claim | Value | Evidence | Why it is not early-warning evidence |
|---|---|---|---|
| Retrospective whole-possession models | XGBoost 0.957, GRU 0.957, ensemble 0.977 | [audit/exp023_v2_corrected_labels.md](audit/exp023_v2_corrected_labels.md) | Use completed possession summaries |
| Graph-only pass GNN | ROC AUC 0.595 | [pass_label/gnn_comparison.md](pass_label/gnn_comparison.md) | Positions only, no pass destination; not like-for-like with XGBoost |
| AUC by competition | XGBoost 0.663 (La Liga) vs 0.659 (World Cup); GRU 0.660 vs 0.661 | [event_time/error_analysis.md](event_time/error_analysis.md) | Descriptive, 6 and 9 test matches; same cohort, so not external validation |
| Player attribution | 623 players with at least 100 possessing-team events | [attribution/player_leaderboard.csv](attribution/player_leaderboard.csv) | Retrospective model explanation, not causal value |

## Withdrawn (do not reuse)

| Old claim | Why withdrawn | Where |
|---|---|---|
| Early forecasting at 50% of events: GRU 0.820, XGBoost 0.847 | The prefix length encodes the possession's total length | [feature_timing_audit.md](feature_timing_audit.md) |
| First event-time results 0.668 / 0.670 | Same-moment leak | [audit/same_moment_before.md](audit/same_moment_before.md) |
| Pass model AUC about 0.88 on `dangerous_progression_k` | The label was self-determining; the stale `models/results_summary.json` was deleted (still in git history) | [label_methodology.md](label_methodology.md) |
| GNN near-parity with XGBoost (0.841 vs 0.845) | Old label | [03_methodology.md](03_methodology.md) |
| In-sample v2 attribution leaderboard (453 players) | Not cross-fitted | [attribution/ranking_change_audit.md](attribution/ranking_change_audit.md) |
| Practical-usefulness values precision 0, recall 0, false alerts 89.9%, XGBoost lead time 16 | Labelling bug in `scripts/practical_usefulness.py`, fixed | [audit/final_audit_2026-10.md](audit/final_audit_2026-10.md) |
| False-alert rate 55.9% / 46.6% and ECE 0.082 / 0.053 | Not the frozen definitions: `practical_usefulness.py` keeps possessions of four or more events, counts alerts on the last event of safe possessions, and averages calibration bins unweighted. Superseded by the frozen-definition values above | [repro/practical_usefulness.json](repro/practical_usefulness.json) |

## Open items: inconsistencies found and not yet resolved

Items 1-6 were found during the 2026-11-04 review, 7-8 on 2026-11-09. None changes a headline AUC; each should be settled before a
public release.

1. **Resolved 2026-11-09: false-alert rate quoted three ways.** The 39.4% / 46.9% (retrain report, all possessions,
   eligible snapshots), 55.9% / 46.6% (`practical_usefulness.py`, possessions of four or more events, alerts on the
   last event counted) and the dashboard's typed "39-48%" used different populations and models. One definition is
   now frozen (top of this page) and restated in the README, the dashboard, the case study and the demo report:
   48.0% / 39.7% on the test split, 45.2% / 39.3% on Euro 2024.
2. **Resolved 2026-11-09: ECE quoted two ways.** 0.039 / 0.039 is ten equal-width bins **weighted by snapshot
   count**; 0.082 / 0.053 is the **unweighted** mean over non-empty bins from `practical_usefulness.py` (also on a
   different possession population). It was mislabelled "possession-level" here. Both are in absolute probability.
   The weighted version is the frozen one and is now labelled with its units wherever it is quoted.
3. **GNN comparison file disagrees with itself.** [pass_label/gnn_comparison.md](pass_label/gnn_comparison.md) gives
   0.728 for XGBoost event+360 on the 360 subset in its summary line and 0.715 in its table.
4. **Arrays cache differs slightly from the current builder.** For 86 of 17,065 possessions (0.5%) the cached
   `event_time_arrays.pkl` assigns a different possessing team than `build_possession_arrays` does today (the
   current causal rule falls back to the first actor when the possession-team column is absent, so a possession that
   starts with an opponent event is assigned to the opponent). Rebuilding the test arrays gives the same GRU test AUC
   (0.6636) to four decimals, so the effect is below the reported precision, but the two paths are not identical.
5. **Resolved 2026-11-09: dashboard limitation text was hard-coded.** `limitation_lines()` in
   `src/app/possession_data.py` now reads every number from `repro/frozen_metrics_test.json` and
   `repro/sealed_holdout.json`; a test checks that the text follows the files.
6. **Hosted CI has never run** on the current workflow; see [repro/local_ci.md](repro/local_ci.md) for the local run.
7. **Local `data/processed` split files are stale.** On this machine `data/processed/split_manifest.csv` holds 6
   matches and `train/val/test.parquet` are pass-level tables (written 2026-11-07), while the events and possessions
   cover all 99 matches. `verify_artefacts.py` and `practical_usefulness.py` read that manifest, so they should not be
   rerun until `data/processed` is rebuilt; the sealed-holdout scripts read `configs/research_split_manifest.csv`.
   Data is gitignored, so a fresh clone is not affected.
8. **In-cohort re-score drifted slightly (still open).** With today's arrays builder and inputs the committed models give
   0.6639 / 0.6623 on 43,497 test snapshots, against the documented 0.6649 / 0.6636 on 43,541. The cause has not been
   established. The Euro 2024 comparison uses today's builder on both sides.
