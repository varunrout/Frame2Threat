# Claims ledger

Every quantitative claim the project makes, where its evidence lives, and whether it may still be quoted.
Status: **Current** (reproduced, may be quoted), **Diagnostic** (true but not evidence for the headline task),
**Exploratory** (not validated), **Withdrawn** (do not reuse). Last reviewed 2026-11-04.

## Current

| Claim | Value | Evidence | Note |
|---|---|---|---|
| Event-time XGBoost, all test snapshots | ROC AUC 0.665 [0.647, 0.681] | [repair_retrain/event_time_results.md](repro/repair_retrain/event_time_results.md) | Fresh retrain, 500-resample match bootstrap |
| Event-time GRU, all test snapshots | ROC AUC 0.664 [0.642, 0.687] | same | |
| Ball x/y logistic baseline | ROC AUC 0.642 [0.622, 0.661] | same | The lift of the models over it is about 0.02 |
| One snapshot per possession | XGBoost 0.736, GRU 0.739 | same | Different sampling from the all-snapshot figure |
| Committed event-time models re-score | XGBoost 0.6649, GRU 0.6636 | [repro/artefact_check.md](repro/artefact_check.md) | Earlier training run than the fresh retrain; `make verify` |
| Pass progression, event-only / event+360 | ROC AUC 0.7337 / 0.7380 | [repro/artefact_check.md](repro/artefact_check.md) | 360 lift is small; coverage is partial |
| Alert behaviour at validation thresholds | Precision 0.554 / 0.555, recall 0.520 / 0.519 (XGBoost / GRU) on 2,249 test possessions | [repro/practical_usefulness.json](repro/practical_usefulness.json) | Corrected after a labelling bug; see the open items below |
| Fixed research split | 99 matches: 69 train, 15 validation, 15 test | [configs/research_split_manifest.csv](../configs/research_split_manifest.csv) | Guarded by `tests/test_repo_integrity.py` |

## Diagnostic and exploratory (never headline evidence)

| Claim | Value | Evidence | Why it is not early-warning evidence |
|---|---|---|---|
| Retrospective whole-possession models | XGBoost 0.957, GRU 0.957, ensemble 0.977 | [audit/exp023_v2_corrected_labels.md](audit/exp023_v2_corrected_labels.md) | Use completed possession summaries |
| Graph-only pass GNN | ROC AUC 0.595 | [pass_label/gnn_comparison.md](pass_label/gnn_comparison.md) | Positions only, no pass destination; not like-for-like with XGBoost |
| Player attribution | 623 players with at least 100 possessing-team events | [attribution/player_leaderboard.csv](attribution/player_leaderboard.csv) | Retrospective model explanation, not causal value |

## Withdrawn (do not reuse)

| Old claim | Why withdrawn | Where |
|---|---|---|
| Early forecasting at 50% of events: GRU 0.820, XGBoost 0.847 | The prefix length encodes the possession's total length | [feature_timing_audit.md](feature_timing_audit.md) |
| First event-time results 0.668 / 0.670 | Same-moment leak | [audit/same_moment_before.md](audit/same_moment_before.md) |
| Pass model AUC about 0.88 on `dangerous_progression_k` | The label was self-determining; stale values remain in `models/results_summary.json` | [label_methodology.md](label_methodology.md) |
| GNN near-parity with XGBoost (0.841 vs 0.845) | Old label | [03_methodology.md](03_methodology.md) |
| In-sample v2 attribution leaderboard (453 players) | Not cross-fitted | [attribution/ranking_change_audit.md](attribution/ranking_change_audit.md) |
| Practical-usefulness values precision 0, recall 0, false alerts 89.9%, XGBoost lead time 16 | Labelling bug in `scripts/practical_usefulness.py`, fixed | [audit/final_audit_2026-10.md](audit/final_audit_2026-10.md) |

## Open items: inconsistencies found and not yet resolved

These were found during the 2026-11-04 review. None changes a headline AUC; each should be settled before a
public release.

1. **False-alert rate is quoted three ways.** The retrain report shows 39.4% and 46.9% of safe possessions tipped
   ([repair_retrain/event_time_results.md](repro/repair_retrain/event_time_results.md)); `practical_usefulness.py` gives
   55.9% (XGBoost) and 46.6% (GRU); the dashboard's limitation text says "39-48%". The definitions and thresholds have
   not been reconciled.
2. **Calibration error is quoted two ways.** ECE 0.039 / 0.039 (ten equal-width bins, all snapshots) versus 0.082 / 0.053
   (possession-level alert evaluation). Different units; the dashboard text says "ECE ~0.04".
3. **GNN comparison file disagrees with itself.** [pass_label/gnn_comparison.md](pass_label/gnn_comparison.md) gives
   0.728 for XGBoost event+360 on the 360 subset in its summary line and 0.715 in its table.
4. **Arrays cache differs slightly from the current builder.** For 86 of 17,065 possessions (0.5%) the cached
   `event_time_arrays.pkl` assigns a different possessing team than `build_possession_arrays` does today (the
   current causal rule falls back to the first actor when the possession-team column is absent, so a possession that
   starts with an opponent event is assigned to the opponent). Rebuilding the test arrays gives the same GRU test AUC
   (0.6636) to four decimals, so the effect is below the reported precision, but the two paths are not identical.
5. **Dashboard limitation text is hard-coded.** The numbers in `src/app/enhanced_possession_page.py` are typed in, not
   read from the artefacts, so they can drift from the reports.
6. **Hosted CI has never run** on the current workflow; see [repro/local_ci.md](repro/local_ci.md) for the local run.
