# scripts/

Run everything from the repository root (`python scripts/<name>.py`). Scripts are grouped here by
**role**. The files stay in one folder on purpose: committed reports, model cards and CI quote these
paths as the commands that produced their numbers, so moving them would break that evidence trail.

Experiment IDs (`EXP-0xx`) refer to [`reports/experiment_log.md`](../reports/experiment_log.md).

## 1. Maintained route (current results come from these)

Event-time danger prediction, the primary task. Run in this order for a full rebuild.

| Script | What it does | Writes |
|---|---|---|
| `prepare_event_time.py` | Builds inputs from the fixed match manifest (cached StatsBomb events, else public download). Refuses to overwrite existing tables. | `data/processed/` (or `--output-dir`): events, possessions, split manifest, `event_time_arrays.pkl` (what the dashboard and CLI read) |
| `run_event_time.py` | EXP-021/022: trains baselines, event-time XGBoost and GRU, bootstrap CIs, calibration. | `models/eventtime_*`, `reports/event_time/` |
| `evaluate_event_time.py` | Re-evaluates saved event-time models on rebuilt held-out arrays, with calibration and a constant baseline. | `reports/repro/repair_evaluation.json` (`--output`) |
| `train_pass_models.py` | EXP-033: final pass-progression models (event-only and event+360). | `models/pass_progression_*`, `reports/pass_label/pass_model_final.*` |
| `train_pass_gnn.py` | EXP-032: graph-only `PassFrameGNN` and its comparison with XGBoost on the 360 subset. | `models/pass_gnn_progression.*`, `reports/pass_label/` |
| `run_attribution.py` | EXP-031: cross-fitted, retrospective event attribution. Exploratory, not causal. | `data/processed/attribution_events*.parquet`, `reports/attribution/`, `reports/figures/attribution/` |

## 2. Verification and demo (cheap, safe to re-run)

| Script | What it does | Writes |
|---|---|---|
| `verify_artefacts.py` | Reloads the saved models, re-scores the test split and compares with the committed reports (`make verify`). | `reports/repro/artefact_check.md` and `.json` (the JSON adds git SHA, package versions and model SHA-256s) |
| `check_pass_model_repro.py` | Retrains each pass-model recipe on the unchanged split and records the numbers; saves no model. | `reports/pass_label/pass_model_repro_gate.md` |
| `make_demo_evidence.py` | Scores one held-out possession with both event-time models (`make demo`). | `reports/figures/demo/`, `reports/repro/cli_demo.txt`, `demo_possession.json` |
| `make_evidence_pack.py` | 25 test-split possessions scored by all models, retrospective ones clearly labelled. | `reports/repro/scored_possessions_sample.csv` |

## 3. Analysis (reads saved models, produces figures and tables)

| Script | What it does | Writes |
|---|---|---|
| `practical_usefulness.py` | Precision, recall, false-alert rate, lead time and calibration at the validation-chosen thresholds. | `reports/figures/practical/`, `reports/repro/` |
| `interpretability_analysis.py` | SHAP importance for XGBoost and trajectory analysis for the GRU. | `reports/figures/interpretability/`, `reports/repro/interpretability_*.json` |
| `error_analysis.py` | Event-time AUC per segment (competition, possession origin, period, minute band) with match-bootstrap intervals; flags small segments. Re-scores committed models, no retraining. | `reports/event_time/error_analysis.{csv,md}`, `reports/figures/error_analysis/` |
| `tactical_walkthrough.py` | Event-by-event walkthrough of the France demo possession (match 3857266, possession 121). | `reports/figures/tactical_walkthrough.png`, `reports/repro/tactical_walkthrough.json` |

## 4. Audit and repair history (evidence for the October/November 2026 repairs)

Re-runnable, but they describe defects that are already fixed. Read them with
[`reports/audit/`](../reports/audit/).

| Script | Purpose |
|---|---|
| `audit_defects.py` | Reproduces the four defects found in the first self-audit ("before" numbers). |
| `audit_same_moment_leak.py` | Same-moment leak in the event-time model. |
| `audit_shot_columns.py` | Shot xG / outcome parsing check. |
| `compare_event_time_fix.py`, `compare_pass_label_fix.py` | Before/after tables for EXP-026 and EXP-027. |
| `reparse_events_cache.py` | One-off: adds shot columns to the cached parsed events (aborts if existing columns differ). |
| `rerun_exp019_vocab_fix.py` | EXP-020: tipping-point re-run after the event vocabulary fix. |
| `rerun_v2_corrected_labels.py` | EXP-023: retrospective possession models on the corrected labels. |
| `run_pass_label_ablation.py` | EXP-024/025: both pass labels, with and without 360 features. Source of the label comparison. |

## 5. Older v1 pass-level path (kept for the smoke test, not for current results)

| Script | Purpose |
|---|---|
| `run_pipeline.py`, `reproduce_v1.py`, `reproduce_v1.sh` | Raw (or synthetic) data to scored pass outputs with the v1 event-only model. Used by CI for the synthetic smoke test (`make pipeline-smoke`). **Not** the event-time task; do not quote its numbers as current results. |

## Conventions
- Check `--help` before running; several scripts take `--output-dir` so a rerun need not overwrite committed results.
- Anything marked "synthetic" is an engineering smoke test, never football evidence.
