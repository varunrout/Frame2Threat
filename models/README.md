# models/

Model files committed to the repository, what each predicts and how to check it. Hashes are the first 16 hex
characters of SHA-256 (`sha256sum models/<file>`). Numbers are test ROC AUC on the held-out matches; the
check that recomputes them is `make verify` ([latest result](../reports/repro/artefact_check.md)).

`.joblib` and `.pt` files are pickles: only load files from a source you trust.

## Current models

| File | Predicts | Test ROC AUC | Trained by | SHA-256 | Card |
|---|---|---|---|---|---|
| `eventtime_xgboost.joblib` | Event-time danger: P(shot or box entry later in this possession) after each event | 0.6649 | `scripts/run_event_time.py` | `6094119e194fc711` | [event_time](../reports/repro/model_cards/event_time.md) |
| `eventtime_gru.pt` | Same task, sequence model | 0.6636 | `scripts/run_event_time.py` | `76d61575022ed15f` | [event_time](../reports/repro/model_cards/event_time.md) |
| `eventtime_meta.json` | Alert thresholds (XGBoost 0.390, GRU 0.422, chosen on validation), feature schema, tuning | n/a | `scripts/run_event_time.py` | n/a | [event_time](../reports/repro/model_cards/event_time.md) |
| `pass_progression_event_only.joblib` | Whether a pass is followed by progression within 5 events (`progression_beyond_pass_k`), event features | 0.7337 | `scripts/train_pass_models.py` | `7271c0175948b200` | [pass_xgboost](../reports/repro/model_cards/pass_xgboost.md) |
| `pass_progression_event_360.joblib` | Same label, plus 360 freeze-frame geometry | 0.7380 | `scripts/train_pass_models.py` | `96387ba0f6a7b2d2` | [pass_xgboost](../reports/repro/model_cards/pass_xgboost.md) |
| `pass_gnn_progression.pt` | Same label from player positions only (graph, no pass destination); exploratory | 0.595 | `scripts/train_pass_gnn.py` | `eb0457bd819aa884` | [pass_gnn](../reports/repro/model_cards/pass_gnn.md) |
| `v2_corrected_xgboost_poss_dangerous.joblib` | **Retrospective** whole-possession danger (uses completed summaries); diagnostics and attribution only | 0.957 | `scripts/rerun_v2_corrected_labels.py` | `be1df8dff9015a0e` | [possession_retrospective](../reports/repro/model_cards/possession_retrospective.md) |
| `v2_corrected_gru_poss_dangerous.pt` | Same, sequence model | 0.957 | `scripts/rerun_v2_corrected_labels.py` | `e66849e513e17e6d` | [possession_retrospective](../reports/repro/model_cards/possession_retrospective.md) |

Each `pass_*.json` next to a model is its sidecar: label definition, recipe and feature count.

The retrospective models score a possession using information from after the moment of interest. Their high AUC is
**not** early-warning evidence; do not compare it with the event-time rows.

## Stale files in this folder (do not use)

| File | Why it is stale |
|---|---|
| `results_summary.json`, `v1_results_summary.json` | Record the old self-determining label `dangerous_progression_k` (AUC about 0.88). Superseded; see [../reports/audit/](../reports/audit/README.md). |
| `superseded/eventtime_meta.json` | Metadata of the pre-fix event-time model. |

## Local-only files (gitignored, not in the repository)

`gru_poss_dangerous.pt`, `xgboost_poss_dangerous.joblib` (trained before the label fix) and the binaries in
`superseded/` stay on the author's machine as evidence. The dashboard and CLI never load them, but the legacy code in
`src/evaluation/possession_attribution.py` and `src/models/gru_train_script.py` still refers to
`models/gru_poss_dangerous.pt`, so those legacy paths do not work on a fresh clone.
