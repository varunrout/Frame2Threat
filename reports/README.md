# reports/

Everything the project has measured, written up. If you read only three things, read these:

1. [../README.md](../README.md) for the headline results and what they do and do not show.
2. [repro/model_cards/event_time.md](repro/model_cards/event_time.md) for the primary model.
3. [audit/README.md](audit/README.md) for what went wrong and which numbers are withdrawn.

## Folders

| Folder | Contents | Status |
|---|---|---|
| `repro/` | Evidence that can be inspected from a fresh clone: model cards, the November retrain, the demo possession, artefact and local-CI checks. See [repro/README.md](repro/README.md). | **Current** |
| `event_time/` | Event-time results (XGBoost, GRU, baselines, reliability, buckets). | Current. The `superseded_pre_same_moment_fix/` subfolder is **withdrawn**. |
| `pass_label/` | Pass-level results on `progression_beyond_pass_k`, the 360 ablation, the GNN comparison. | Current. The `superseded_pre_possession_length_fix/` subfolder is **withdrawn**. |
| `attribution/` | Cross-fitted retrospective attribution, leaderboards and the ranking-change audit. | Current, exploratory (not causal). |
| `audit/` | Defects found, before/after evidence, the repair report. | Evidence; see [audit/README.md](audit/README.md). |
| `figures/` | Charts, grouped by topic. | Mixed: `demo/`, `practical/`, `interpretability/`, `tactical/`, `attribution/` are current; `superseded_attribution/` and the loose `poss_*`, `v1_*`, `v2_*`, `v3_*` files are historical figures cited by the older reports. |

## Documents in this folder

| File | What it is | Status |
|---|---|---|
| [final_report.md](final_report.md) | The long project report. Opens with a corrections box; the body is the older pass-level and possession-level story. | Historical, with corrections. Read the top box first. |
| [03_methodology.md](03_methodology.md) | Technical approach. Sections on fraction-based forecasting are marked superseded. | Mixed |
| [label_methodology.md](label_methodology.md) | Exact definition of every label, including why the old pass label was self-determining. | Current reference |
| [feature_timing_audit.md](feature_timing_audit.md) | Which features are safe for early prediction; the v3 numbers are withdrawn in its addendum. | Current reference |
| [CLAIMS.md](CLAIMS.md) | Every quantitative claim, its evidence and whether it may be quoted; lists open inconsistencies. | **Current** |
| [dataset_card.md](dataset_card.md) | Source, cohort, split, known biases. | **Current** |
| [data_dictionary.md](data_dictionary.md) | Columns of every table, and the model files of each version. | Reference; lists some superseded models |
| [design_decisions.md](design_decisions.md) | Short note on why the project is built as it is. | Reference |
| [experiment_log.md](experiment_log.md) | Every experiment (`EXP-0xx`) with its status; notebook provenance notes. | Historical log; check the status of each entry |

## How to read a number

A result is current only if it comes from `repro/`, `event_time/`, `pass_label/` or `attribution/` and is not under a
`superseded*` folder. When a document disagrees with [../README.md](../README.md) or [audit/README.md](audit/README.md),
the latter two win.
