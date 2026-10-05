# Frame2Threat

[![CI](https://github.com/varunrout/Frame2Threat/actions/workflows/ci.yml/badge.svg)](https://github.com/varunrout/Frame2Threat/actions/workflows/ci.yml)

**Breaking the Block: Predicting and Explaining Dangerous Progression from StatsBomb Event Data**

A reproducible football analytics system that predicts whether a pass breaks a defensive line, produces dangerous progression, or leads to final-third / box entries and shots using StatsBomb Open Data.  The project evaluates 360 freeze-frame spatial context directly and finds that, in this dataset, event attributes carry most of the pass-level predictive signal: adding 360 geometry improves ROC AUC by +0.002 (95% CI -0.001 to +0.006) on the original label and +0.0015 (-0.004 to +0.007) on a corrected label that no longer rewards a pass for where it ends.  Extended with possession-level sequential modelling and player-level attribution.

> **Upfront caveats:** This is a StatsBomb Open Data project with partial 360 coverage, not a production tracking-data model.  The reported 360 lift is marginal (about +0.002 ROC AUC on either label, intervals include zero), so the strongest result is the event/possession modelling rather than a claim that freeze frames transform performance.  **Corrections (October 2026):** a self-audit of the project found four defects (event-type decoding, a possession-length leak in the early-forecasting design, opponent coordinates in the wrong frame, and a self-determining pass label).  They are fixed, every affected result was re-run, and the superseded numbers are kept and marked in `reports/`; see `reports/audit/`.  **A second self-audit round** then fixed five further items: a same-moment leak in the event-time model (a pass ending in the box now counts as the danger event), the v1 `possession_length` feature (it mixed matches), shot xG / goal columns that were never parsed, the possession EDA notebook (re-run on corrected data; the recycling claim reversed once length is controlled) and the data description (99 matches; 360 covers 52.3% of passes).  Several higher-level workflows were assembled with agent assistance and then grounded with tests, CI, reproducibility scripts, and explicit limitations.

---

## Quick start

```bash
git clone https://github.com/varunrout/Frame2Threat.git
cd Frame2Threat
pip install -e ".[dev]"
streamlit run src/app/app.py   # app; demo mode labels random placeholder scores
pytest tests/ -q               # run all tests (123 passing)
```

---

## Project overview

Frame2Threat models attacking danger from StatsBomb event data, then tests whether 360 freeze-frame geometry adds meaningful signal beyond those event attributes.  The central finding is deliberately modest: 360 context is useful for interpretability and geometry-dependent labels such as line breaks, but it does not materially improve the pass-level danger model here (about +0.002 ROC AUC on either label; both within noise).

The system operates at **two complementary granularities**:

### v1 — Pass-level prediction

For each individual open-play pass, predict whether it leads to dangerous outcomes.

| Label | Definition |
|-------|-----------|
| `strict_line_break` | Pass crosses ≥ 2 defenders (by x-position in freeze frame) |
| `loose_line_break` | Pass crosses ≥ 1 defender (by x-position) |
| `dangerous_progression_k` | Final-third entry OR box entry OR shot within k=5 same-possession events.  **Self-determining** (the receipt of the pass itself counts; `end_x` alone gives AUC 0.786): kept for comparability only |
| `progression_beyond_pass_k` | **Preferred.** Shot, new box entry, or >= 10 m further upfield into a more advanced zone within k=5 events, ignoring the pass's own receipt and the opponent's events (`reports/label_methodology.md`, Label G) |
| `final_third_entry_k` | Ball reaches x ≥ 80 within k events |
| `box_entry_k` | Ball enters penalty box within k events |
| `shot_within_k` | Shot occurs within k events |
| `threat_gain` | Zone-value delta (xT proxy, range ≈ [−1, 1]) |

### v2 — Possession-level prediction

For each full possession sequence, predict danger and attribute it to individual events and players.

| Label | Definition |
|-------|-----------|
| `poss_dangerous` | Shot or penalty-box entry during the possession (**primary target**) |
| `poss_has_shot` / `poss_entered_box` / `poss_entered_final_third` | Core outcome labels |
| `poss_tempo` / `poss_verticality` / `poss_phase` | Structural characterisation |
| `poss_broke_pressure` / `poss_bypassed_lines` | Defensive disruption labels |

### v3 — Early danger forecasting (rebuilt as an event-time formulation)

Can we anticipate danger *before* it happens?  The original v3 asked for "the first f% of events", which needs the possession's total length: the prefix length alone gives ROC AUC 0.68, and 21-59% of dangerous possessions already contained their shot or box entry inside the prefix.  **Those numbers (GRU 0.820, XGBoost 0.847 at 50%) are withdrawn.**  The replacement is an in-play hazard: at every event `t` of a possession that has not yet produced a shot or box entry, estimate the probability that the possessing team produces one in a later event, from events `0..t` only (real elapsed seconds, current event type, ball position, start context).

Test split (15 matches, unchanged split), match-level bootstrap 95% intervals:

| Model | ROC AUC, all snapshots | ROC AUC, one random snapshot per possession |
|-------|-----------------------|--------------------------------------------|
| Ball x/y only (logistic regression) | 0.642 [0.622, 0.661] | 0.695 [0.674, 0.717] |
| Events so far only (logistic regression) | 0.561 [0.523, 0.603] | 0.608 [0.579, 0.638] |
| Start context only (XGBoost) | 0.592 [0.570, 0.617] | 0.599 [0.572, 0.626] |
| XGBoost, snapshot features | 0.665 [0.647, 0.681] | 0.736 [0.714, 0.759] |
| GRU (1 layer, hidden 64, learned event-type embedding) | 0.664 [0.642, 0.687] | 0.739 [0.710, 0.767] |

Mean ROC AUC over 20 independent one-per-possession draws: ball x/y 0.692, events so far 0.612, start context 0.599, XGBoost 0.729, GRU 0.744.  *Superseded first version of this table (same-moment leak: a pass ending in the box was visible in its own snapshot):* all snapshots XGBoost 0.668, GRU 0.670, ball x/y 0.648; one per possession XGBoost 0.749, GRU 0.756, ball x/y 0.713.

Key finding: danger is only **modestly** forecastable in play.  The models beat a ball-position-only baseline by +0.02 (all snapshots) to +0.04 (one per possession) ROC AUC, and XGBoost and the GRU are indistinguishable (GRU - XGBoost: -0.001 [-0.021, +0.022]).  Most of the signal is where the ball currently is.  Calibration is imperfect (test row prevalence 0.404 vs 0.362 in train; XGBoost under-predicts between 0.3 and 0.6, the GRU below 0.3; see `reports/event_time/event_time_results.md`).  A threshold chosen on validation (predicted-positive rate = validation prevalence) alerts 90-91% of dangerous possessions with a median lead of 9-10 events (about 8-9 s), but also 39% (GRU) to 48% (XGBoost) of non-dangerous ones.  There is no dominant "tipping" event type: passes, carries and ball receipts trigger first alerts roughly in proportion to how often they occur, and the earlier "dribbles are the dominant trigger (64.4%)" was an artefact of a decoding bug (id 3 is Ball Receipt, not Dribble).

Timing audit: the full-possession v2 scores are retrospective upper bounds, not leakage-free early predictions.  The tabular model uses completed-possession summaries such as `max_x_reached` and `territory_gained`, while `poss_dangerous` includes box entry.  See `reports/feature_timing_audit.md` for the feature-level audit and the addendum on the length leak.

### Key results (all versions)

Superseded values are kept in `reports/` with the reason; this table shows current numbers only.

| Model | Level | Task | Test ROC AUC |
|-------|-------|------|-------------|
| XGBoost (event-only, 27 features) | Pass | `dangerous_progression_k` (self-determining label) | 0.881 [0.868, 0.893] |
| XGBoost (event+360, 41 features) | Pass | `dangerous_progression_k` | 0.883 [0.870, 0.895] |
| `end_x` alone (logistic regression) | Pass | `dangerous_progression_k` | 0.800 |
| XGBoost (event-only, 27 features) | Pass | `progression_beyond_pass_k` | **0.728** [0.713, 0.743] |
| XGBoost (event+360, 41 features) | Pass | `progression_beyond_pass_k` | 0.729 [0.715, 0.745] |
| `end_x` alone (logistic regression) | Pass | `progression_beyond_pass_k` | 0.543 |
| XGBoost (41 possession features, retrospective) | Possession | `poss_dangerous` (corrected labels) | 0.957 |
| PossessionGRU (full event sequence, retrospective) | Possession | `poss_dangerous` (corrected labels) | 0.957 |
| Logit-mean XGB + GRU ensemble (retrospective upper bound) | Possession | `poss_dangerous` (corrected labels) | 0.977 |
| XGBoost, event-time snapshots | Possession (in play) | `danger_after_t`, all snapshots | 0.665 |
| GRU, event-time snapshots | Possession (in play) | `danger_after_t`, all snapshots | 0.664 |
| GRU, event-time snapshots | Possession (in play) | one random snapshot per possession | 0.739 |

At pass level the 360 geometry lift is +0.0021 [-0.0010, +0.0055] ROC AUC on the original label and +0.0015 [-0.0036, +0.0070] on the corrected label (on the corrected label average precision +0.0041 [-0.0053, +0.0134] includes zero and only the Brier improvement, -0.0038 [-0.0067, -0.0015], excludes it).  Event attributes carry most of the signal; 360 geometry adds a very small amount that is not clearly distinguishable from zero.  On the corrected label the event model beats the destination-only baselines (`end_x`: +0.185; `(end_x, end_y)`: +0.040 ROC AUC), so the earlier headline 0.881 mostly reflected a label that rewards the pass's own destination.  *Superseded:* the first correction reported +0.0050 on the corrected label ("about 3.5x larger"); that used the broken `possession_length` feature (it mixed matches), see `reports/pass_label/possession_length_fix_before_after.md`.  The PassFrameGNN comparison (0.841 on the 360 subset) could not be re-run in this environment (see Limitations).  The possession retrospective rows come from a like-for-like retrain after the coordinate-frame fix (`reports/audit/exp023_v2_corrected_labels.md`); the shot-column fix does not affect them (no v2 model uses xG or goals as a feature).

### Hard constraints

- Uses **StatsBomb Open Data** only — 99 matches: 64 FIFA World Cup 2022 matches (all with 360 freeze frames) and 35 La Liga 2020/21 Barcelona matches (no 360 frames).  There are no Premier League matches.  360 freeze frames cover **52.3% of passes (26,515 of 50,688)** and 65% of matches.
- **Not** a full tracking-data system. No continuous trajectories, velocities, or pitch control.
- 360 features and geometry-dependent labels are **NaN** for events without freeze-frame coverage.
- 360 geometry adds only about **+0.002 ROC AUC** to the pass-level danger model (intervals include zero on both labels).
- Pass option ranking is restricted to **visible teammates** in the freeze frame.

---

## Repository structure

```
Frame2Threat/
├── README.md
├── pyproject.toml
├── configs/                   # All experiment config (data, labels, features, models, eval)
│   ├── data.yaml
│   ├── labels.yaml
│   ├── features.yaml
│   ├── model_baseline.yaml
│   ├── model_gnn.yaml
│   ├── model_gru.yaml         # v2 GRU architecture + training config
│   ├── model_possession.yaml  # v2 possession XGBoost + (superseded) v3 early-warning config
│   └── eval.yaml
├── data/raw|interim|processed # Data directories (gitignored)
├── models/                    # Trained model artefacts (gitignored)
│   ├── xgboost_dp_event_only.joblib        # v1 pass-level XGBoost (event-only)
│   ├── xgboost_dp_event_360.joblib         # v1 pass-level XGBoost (event+360)
│   ├── xgboost_poss_dangerous.joblib       # v2 possession XGBoost (pre-fix labels)
│   ├── v2_corrected_*                      # v2 retrained on frame-corrected labels
│   ├── gru_poss_dangerous.pt               # v2 PossessionGRU
│   ├── eventtime_gru.pt, eventtime_xgboost.joblib, eventtime_meta.json  # event-time models
│   ├── xgboost_start_only.joblib           # SUPERSEDED fraction-based v3
│   ├── xgboost_cumulative_{25,50,75}pct.joblib  # SUPERSEDED fraction-based v3
│   ├── v1_results_summary.json
│   └── results_summary.json
├── notebooks/                 # 01–10 analysis notebooks
├── reports/                   # Documentation + figures
│   ├── 01_research_motivation.md
│   ├── 02_football_analytics_landscape.md
│   ├── 03_methodology.md
│   ├── data_dictionary.md
│   ├── design_decisions.md
│   ├── label_methodology.md
│   ├── feature_timing_audit.md
│   ├── audit/                  # defect audits (before/after) and re-run results
│   ├── event_time/             # in-play (event-time) results
│   ├── pass_label/             # both pass labels x 360 ablation
│   ├── experiment_log.md
│   ├── final_report.md
│   └── figures/               # 49 PNG figures
├── src/
│   ├── data/     ingest, inventory, parse_events/lineups/360, join_pass_frames,
│   │             parse_possessions, event_types (shared type ids), possession_frame, splits
│   ├── labels/   line_break, dangerous_progression, progression_beyond_pass, downstream_outcomes,
│   │             possession_labels, validation_sampling
│   ├── features/ event_features, geometry_features, graph_builder,
│   │             sequence_context, possession_features, early_features (superseded), event_time
│   ├── models/   baselines, tabular, gnn, hybrid, gru_possession, ranking,
│   │             gru_train_script, train_early_models (superseded), event_time
│   ├── evaluation/ metrics, calibration, ablations, tactical_review,
│   │               possession_attribution, early_evaluation (superseded), event_time_eval
│   ├── visualization/ pitch_plots, freeze_frame_view, explanations
│   └── app/      app.py (Streamlit), cli.py (CLI pipeline), live_scoring.py
└── tests/        conftest, test_ingestion, test_labels, test_features,
                  test_splits, test_models, test_build_store,
                  test_reproduce_pipeline, test_event_types, test_event_time (123 tests)
```

---

## Installation

```bash
poetry install --extras dev         # deterministic install from poetry.lock
poetry run pre-commit install       # optional local git hooks

# Alternative editable install:
pip install -e .                    # core
pip install -e ".[dev,notebooks]"   # + pytest, jupyter
```

PyTorch and `torch-geometric` are optional (required only for GNN and GRU models).

The repository pins the development Python line with `.python-version` (`3.11`)
and commits `poetry.lock` for repeatable dependency resolution. To run the same
local checks as the pre-commit hooks:

```bash
poetry run pre-commit run --all-files
```

---

## Reproduce in one command

### Data store

Build the parquet + SQLite store from configured StatsBomb Open Data:

```bash
make store
```

For a no-network smoke build:

```bash
make store-smoke
```

The direct command is:

```bash
python -m src.data.build_store --verbose
```

The store writes partitioned pass-instance parquet under `data/store/pass_instances/`,
train/validation/test split parquet under `data/store/processed/`, and SQLite
metadata/provenance at `data/store/metadata.sqlite`.

### Tests

CI runs the full pytest suite with coverage and currently enforces a 33%
minimum coverage floor:

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=33
```

### v1 — Pass-level

One-command raw-to-scores reproduction is available for the event-only v1 pass model:

```bash
make pipeline
```

For a quick no-network smoke check:

```bash
make pipeline-smoke
```

The full command fetches/parses configured StatsBomb Open Data when local
caches are absent and writes generated outputs under `data/repro/v1/`,
including:

- `pass_instances.parquet`
- `split_manifest.csv`
- `train.parquet`, `val.parquet`, `test.parquet`
- `v1_event_only_model.joblib`
- `v1_event_only_scored_passes.csv`
- `v1_event_only_summary.json`

The equivalent direct command is:

```bash
python scripts/run_pipeline.py --verbose
```

```python
# 1. Ingest + inventory
from src.data.inventory import build_inventory
inventory = build_inventory(cfg["statsbomb"]["competitions"])

# 2. Parse + canonical table
from src.data.join_pass_frames import build_pass_instances
pass_instances = build_pass_instances(events, frame_summary)

# 3. Labels
from src.labels.line_break import compute_line_break_labels
from src.labels.dangerous_progression import compute_downstream_labels
pass_instances = compute_line_break_labels(pass_instances, frames, label_cfg["line_break"])
pass_instances = compute_downstream_labels(events, pass_instances, label_cfg)

# 4. Split (match-level, no leakage) and materialise pipeline artifacts
from src.data.splits import create_match_level_splits, materialise_split_parquets
train_df, val_df, test_df = create_match_level_splits(pass_instances, seed=42,
    manifest_path="data/processed/split_manifest.csv")
materialise_split_parquets(pass_instances, output_dir="data/processed",
    manifest_path="data/processed/split_manifest.csv")

# 5. Features + train
from src.features.event_features import build_event_features
from src.models.tabular import TabularClassifier
X_train = build_event_features(train_df)
clf = TabularClassifier(model_type="xgboost", task="dangerous_progression_k")
clf.fit(X_train, train_df["dangerous_progression_k"])
```

### v2 — Possession-level

```python
# 1. Build possession sequences (includes labels)
from src.data.parse_possessions import build_possession_sequences, save_possession_sequences
poss = build_possession_sequences(events)
save_possession_sequences(poss)

# 2. Train XGBoost on possession features
# See notebooks/07_possession_features.ipynb

# 3. Train GRU on event sequences
# See notebooks/08_possession_sequence_model.ipynb

# 4. Attribution analysis
from src.evaluation.possession_attribution import attribute_possession, player_attribution_summary
# See notebooks/09_possession_team_analysis.ipynb
```

### Early danger forecasting (event-time) and CLI

```bash
# Train, tune on validation, evaluate (writes reports/event_time/ and models/eventtime_*)
python scripts/run_event_time.py

# Re-run the audits and the re-runs behind the corrections
python scripts/audit_defects.py --mode after --out reports/audit/defect_audit_after.md
python scripts/rerun_exp019_vocab_fix.py
python scripts/rerun_v2_corrected_labels.py
python scripts/run_pass_label_ablation.py
```

```python
from src.features.event_time import build_possession_arrays, build_snapshot_table
arrs = build_possession_arrays(events_df, poss_df[["match_id", "possession_id"]])
snapshots = build_snapshot_table(arrs)     # one row per event t before any shot / box entry
```

The superseded fraction-based v3 code (`src/features/early_features.py`, `src/evaluation/early_evaluation.py`, `src/models/train_early_models.py`, notebook 10) remains runnable for reproducibility but is labelled as superseded.

The `frame2threat` CLI scores a possession event by event with the event-time model:

```bash
# Score a single possession event-by-event (prints danger trajectory and tipping point)
frame2threat score-live possession_events.json            # GRU (default)
frame2threat score-live possession_events.json --model xgb --threshold 0.40
```

The JSON holds a list of event dicts in the `possession_sequences.parquet` `event_sequence` schema, optionally with `own_team` and `elapsed_seconds` per event (or an object with `event_sequence`, `origin`, `period`, `minute`).  Without `elapsed_seconds` the CLI warns and assumes the training-median spacing.  The default alert threshold is the validation-chosen value in `models/eventtime_meta.json`.  `score-batch` and `train-early` are the superseded fraction-based commands.

The CLI is registered as a console entry point in `pyproject.toml`.  Run `frame2threat --help` for full usage.

### App

```bash
streamlit run src/app/app.py
```

---

## Testing

```bash
pytest tests/ -v    # 123 tests, all passing
```

| Test file | Covers |
|-----------|--------|
| `test_ingestion.py` | Parsing correctness, canonical schema |
| `test_labels.py` | Invariants, no future leakage, sanity checks |
| `test_features.py` | Shapes, NaN passthrough, geometry helpers |
| `test_splits.py` | No match in two splits, reproducibility, manifest I/O |
| `test_models.py` | Rule-based and tabular model fit/predict/serialize |
| `test_event_types.py` | One shared event-type vocabulary: frozen ids, round-trip, no duplicate mappings, prefix counts |
| `test_event_time.py` | Possessing-team frame, snapshot eligibility/labels, no future information in features, GRU causality, evaluation helpers |

---

## Notebooks

| Notebook | Purpose | Level |
|----------|---------|-------|
| `01_data_audit.ipynb` | Inventory, parsing, missingness | Data |
| `02_label_validation.ipynb` | Prevalence, sanity checks, zone breakdown | Labels |
| `03_eda.ipynb` | Distributions, heatmaps, 360 coverage | EDA |
| `04_baselines.ipynb` | Rule-based → LogReg → XGBoost + SHAP | v1 models |
| `05_gnn.ipynb` | GNN training and ablations | v1 models |
| `06_error_analysis.ipynb` | FP/FN, explanations, player profiles | v1 eval |
| `07_possession_features.ipynb` | Possession label EDA + XGBoost baseline | v2 models |
| `08_possession_sequence_model.ipynb` | GRU training, ensemble, H1–H4 testing | v2 models |
| `09_possession_team_analysis.ipynb` | Team/player attribution, tactical analysis | v2 eval |
| `10_early_prediction.ipynb` | Start-only, prefix GRU, cumulative XGB, tipping-point analysis (**superseded** fraction-based formulation) | v3 (superseded) |

---

## Reports

| Document | Contents |
|----------|---------|
| `reports/01_research_motivation.md` | Why dangerous progression matters; real-world applications |
| `reports/02_football_analytics_landscape.md` | Football analytics eras; where Frame2Threat fits |
| `reports/03_methodology.md` | Full technical approach: pipeline, features, models, evaluation |
| `reports/data_dictionary.md` | Schema for all canonical tables (v1 + v2) |
| `reports/feature_timing_audit.md` | Possession feature timing audit, length leak, realised-outcome and coordinate-frame findings |
| `reports/audit/` | Defect audit before/after, EXP-020 and EXP-023 re-run outputs |
| `reports/event_time/` | Event-time (in-play) results, buckets, reliability, tipping points |
| `reports/pass_label/` | Both pass labels x 360 ablation with bootstrap intervals |
| `reports/label_methodology.md` | Operational label definitions, invariants, borderline cases |
| `reports/repro/` | Inspectable scored sample, model cards, and repro evidence pack |
| `reports/experiment_log.md` | Experiment registry (EXP-001 to EXP-025; v3 entries marked superseded) |
| `reports/final_report.md` | Comprehensive project report with all results |

---

## Research questions and answers

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | Does 360 context improve prediction? | Barely: +0.0021 [-0.0010, +0.0055] ROC AUC on the original (self-determining) label and +0.0015 [-0.0036, +0.0070] on the corrected `progression_beyond_pass_k`; only the Brier improvement on the corrected label (-0.0038 [-0.0067, -0.0015]) excludes zero.  Event attributes carry most of the measured signal |
| RQ2 | Which geometry features matter? | n_defenders_goal_side, pass_corridor_clear, receiver_between_lines (importance from the original-label model; not re-ranked on the corrected label) |
| RQ3 | How to predict possession danger? | In play, only modestly: ROC AUC 0.66 (all snapshots) to 0.74 (one random snapshot per possession), about +0.02 to +0.04 over a ball-position baseline; XGBoost and GRU are indistinguishable.  Full-possession models are retrospective upper bounds: XGB 0.957, GRU 0.957, logit-mean ensemble 0.977 (corrected labels) |
| RQ4 | Which events matter most? | Cross-fitted LOO attribution (EXP-031): median Gini of |LOO| 0.48 (0.63 dangerous, 0.43 safe); the model leans on shots, goalkeeper/block/clearance events (defending-team evidence) and possessing-team carries/receipts in proportion; the earlier in-sample Gini 0.495 and "dribble 64.4%" are superseded |
| RQ5 | Player-level attribution? | Partly: 623 players have >= 100 possessing-team events; the ranking is reasonably stable (split-half Spearman 0.62, cross-seed 0.89) but mostly reflects player role, so it is for shortlists to inspect, not for ranking (the old 453-player in-sample leaderboard is superseded) |
| RQ6 | GNN vs. tabular? | Unverified: the 0.841 vs 0.845 near-parity used the self-determining label, and the GNN could not be re-run here (no `torch_geometric`, no saved checkpoint) |

---

## Pitch coordinate system

StatsBomb standard: **120 × 80 metres**.  x: own goal (0) → opponent goal (120).  y: left touchline (0) → right touchline (80).  Final third: x ≥ 80.  Penalty box: x ≥ 102, 18 ≤ y ≤ 62.

---

## Limitations

1. 360 coverage is partial — 64 of 99 matches (65%) and 52.3% of passes have freeze frames (the La Liga matches have none), and the measured pass-level lift from 360 geometry is only about +0.002 ROC AUC.
2. Positional snapshots, not tracking — no velocity or trajectory.
3. Partial pitch visibility — not all 22 players are always visible.
4. Open-play only — set-piece passes excluded from v1 modelling.
5. Threat gain is a zone proxy — not a full possession-value model.
6. Limited sample size — 99 matches (64 World Cup 2022, 35 La Liga 2020/21 Barcelona), 15 in the test split.
7. Full-possession v2 scores are retrospective upper bounds because some features summarise the completed possession.
8. Evaluation is held-out, not causal.
9. The test split is 15 matches; intervals are match-level bootstrap and are wide.
10. The GNN results (PassFrameGNN) were not reproduced in this environment, and the `threat_gain` zone-value map still averages over opponent events in their own frame.
11. Event-time models under-predict on the test split (calibration shifts with the base rate) and alert thresholds trade many false alarms for a high detection rate.

---

## Licence

StatsBomb Open Data is used under the [StatsBomb Open Data Licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).  Project code: MIT.
