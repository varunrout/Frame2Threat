# Frame2Threat

[![CI](https://github.com/varunrout/Frame2Threat/actions/workflows/ci.yml/badge.svg)](https://github.com/varunrout/Frame2Threat/actions/workflows/ci.yml)

**Breaking the Block: Predicting and Explaining Dangerous Progression from StatsBomb Event Data**

A reproducible football analytics system that predicts whether a pass breaks a defensive line, produces dangerous progression, or leads to final-third / box entries and shots using StatsBomb Open Data.  The project evaluates 360 freeze-frame spatial context directly and finds that, in this dataset, event attributes carry almost all of the pass-level predictive signal: adding 360 geometry improves ROC AUC by only +0.001.  Extended with possession-level sequential modelling and player-level attribution.

---

## Quick start

```bash
git clone https://github.com/varunrout/Frame2Threat.git
cd Frame2Threat
pip install -e ".[dev]"
streamlit run src/app/app.py   # demo app (works without data via demo mode)
pytest tests/ -q               # run all tests (69 passing)
```

---

## Project overview

Frame2Threat models attacking danger from StatsBomb event data, then tests whether 360 freeze-frame geometry adds meaningful signal beyond those event attributes.  The central finding is deliberately modest: 360 context is useful for interpretability and geometry-dependent labels such as line breaks, but it does not materially improve the headline pass-level danger model here (+0.001 ROC AUC).

The system operates at **two complementary granularities**:

### v1 — Pass-level prediction

For each individual open-play pass, predict whether it leads to dangerous outcomes.

| Label | Definition |
|-------|-----------|
| `strict_line_break` | Pass crosses ≥ 2 defenders (by x-position in freeze frame) |
| `loose_line_break` | Pass crosses ≥ 1 defender (by x-position) |
| `dangerous_progression_k` | Final-third entry OR box entry OR shot within k=5 same-possession events |
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

### v3 — Early danger forecasting

Can we predict danger *before* the possession ends?  v3 evaluates forward-looking models under strict early-information constraints.

| Observation window | Model | Test ROC AUC |
|--------------------|-------|--------------|
| 0% (start context only) | XGBoost start-only | 0.624 |
| 50% of events observed | GRU prefix | 0.820 |
| 50% of events observed | XGBoost cumulative | 0.847 |
| 100% (full possession, retrospective upper bound) | Ensemble (XGB + GRU) | **0.965** |

Key finding: danger is forecastable well before the possession ends — by halfway through, models already exceed 0.82 AUC.  This early-forecasting result is the strongest predictive story in the project.  The GRU tipping point occurs at median fraction 0.619; dribbles are the dominant trigger (64.4%).

Timing audit: the full-possession v2 scores are retrospective upper bounds, not leakage-free early predictions.  The tabular model uses completed-possession summaries such as `max_x_reached` and `territory_gained`, while `poss_dangerous` includes box entry.  See `reports/feature_timing_audit.md` for the feature-level audit.

### Key results (all versions)

| Model | Level | Task | Test ROC AUC |
|-------|-------|------|-------------|
| XGBoost (event-only, 27 features) | Pass | dangerous_progression_k | **0.881** |
| XGBoost (event+360, 41 features) | Pass | dangerous_progression_k | 0.882 |
| PassFrameGNN (graph) | Pass | dangerous_progression_k | 0.841 (val) |
| XGBoost (41 possession features, retrospective) | Possession | poss_dangerous | 0.9505 |
| PossessionGRU (full event sequence, retrospective) | Possession | poss_dangerous | 0.9524 |
| Ensemble (XGB + GRU, retrospective upper bound) | Possession | poss_dangerous | 0.9650 |
| XGBoost cumulative @50% | Possession (early) | poss_dangerous | 0.8472 |
| XGBoost cumulative @75% | Possession (early) | poss_dangerous | 0.8912 |

At pass level, the event-only model (0.881) and event+360 model (0.882) are effectively tied.  Treat the 360 result as an honest negative/marginal finding, not as the main source of model lift.

### Hard constraints

- Uses **StatsBomb Open Data** only — event data plus partial 360 freeze-frame coverage.
- **Not** a full tracking-data system. No continuous trajectories, velocities, or pitch control.
- 360 features and geometry-dependent labels are **NaN** for events without freeze-frame coverage.
- 360 geometry adds only **+0.001 ROC AUC** to the pass-level danger model in the reported ablation.
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
│   ├── model_possession.yaml  # v2/v3 possession XGBoost + early-warning config
│   └── eval.yaml
├── data/raw|interim|processed # Data directories (gitignored)
├── models/                    # Trained model artefacts (gitignored)
│   ├── xgboost_dp_event_only.joblib        # v1 pass-level XGBoost (event-only)
│   ├── xgboost_dp_event_360.joblib         # v1 pass-level XGBoost (event+360)
│   ├── xgboost_poss_dangerous.joblib       # v2 possession XGBoost
│   ├── gru_poss_dangerous.pt               # v2 PossessionGRU
│   ├── xgboost_start_only.joblib           # v3 start-only XGBoost
│   ├── xgboost_cumulative_25pct.joblib     # v3 cumulative XGBoost @25%
│   ├── xgboost_cumulative_50pct.joblib     # v3 cumulative XGBoost @50%
│   ├── xgboost_cumulative_75pct.joblib     # v3 cumulative XGBoost @75%
│   ├── v1_results_summary.json
│   └── results_summary.json
├── notebooks/                 # 01–10 analysis notebooks
├── reports/                   # Documentation + figures
│   ├── 01_research_motivation.md
│   ├── 02_football_analytics_landscape.md
│   ├── 03_methodology.md
│   ├── data_dictionary.md
│   ├── label_methodology.md
│   ├── experiment_log.md
│   ├── final_report.md
│   └── figures/               # 49 PNG figures
├── src/
│   ├── data/     ingest, inventory, parse_events/lineups/360, join_pass_frames,
│   │             parse_possessions, splits
│   ├── labels/   line_break, dangerous_progression, downstream_outcomes,
│   │             possession_labels, validation_sampling
│   ├── features/ event_features, geometry_features, graph_builder,
│   │             sequence_context, possession_features, early_features
│   ├── models/   baselines, tabular, gnn, hybrid, gru_possession, ranking,
│   │             gru_train_script, train_early_models
│   ├── evaluation/ metrics, calibration, ablations, tactical_review,
│   │               possession_attribution, early_evaluation
│   ├── visualization/ pitch_plots, freeze_frame_view, explanations
│   └── app/      app.py (Streamlit), cli.py (CLI pipeline)
└── tests/        conftest, test_ingestion, test_labels, test_features,
                  test_splits, test_models (69 tests)
```

---

## Installation

```bash
pip install -e .                    # core
pip install -e ".[dev,notebooks]"   # + pytest, jupyter
```

PyTorch and `torch-geometric` are optional (required only for GNN and GRU models).

---

## Running the pipeline

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

CI runs the full pytest suite with coverage and currently enforces a 24%
minimum coverage floor:

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=24
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

### v3 — Early danger forecasting

```python
# 1. Build prefix-aware features
from src.features.early_features import build_start_features, build_cumulative_tabular_features
X_start = build_start_features(poss_df)                    # start-only context
X_50pct = build_cumulative_tabular_features(poss_df, 0.50)  # first 50% of events

# 2. Evaluate prefix GRU
from src.evaluation.early_evaluation import evaluate_prefix_gru
results = evaluate_prefix_gru(gru_model, poss_df, fracs=[0.25, 0.50, 0.75, 1.00])

# 3. Train + save cumulative XGBoost models
# python src/models/train_early_models.py
```

### CLI — Early-warning pipeline

The `frame2threat` CLI provides batch and live danger scoring using the saved v2/v3 models:

```bash
# Score all possessions in a parquet file at 50% and 100% observation
frame2threat score-batch data/processed/possession_sequences.parquet \
    -o scores.csv --fracs 0.50,1.00

# Score a single possession event-by-event (prints danger trajectory)
frame2threat score-live possession_events.json

# (Re)train the v3 early-prediction XGBoost models
frame2threat train-early
```

The CLI is registered as a console entry point in `pyproject.toml`.  Run `frame2threat --help` for full usage.

### App

```bash
streamlit run src/app/app.py
```

---

## Testing

```bash
pytest tests/ -v    # 69 tests, all passing
```

| Test file | Covers |
|-----------|--------|
| `test_ingestion.py` | Parsing correctness, canonical schema |
| `test_labels.py` | Invariants, no future leakage, sanity checks |
| `test_features.py` | Shapes, NaN passthrough, geometry helpers |
| `test_splits.py` | No match in two splits, reproducibility, manifest I/O |
| `test_models.py` | Rule-based and tabular model fit/predict/serialize |

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
| `10_early_prediction.ipynb` | Start-only, prefix GRU, cumulative XGB, tipping-point analysis | v3 early forecasting |

---

## Reports

| Document | Contents |
|----------|---------|
| `reports/01_research_motivation.md` | Why dangerous progression matters; real-world applications |
| `reports/02_football_analytics_landscape.md` | Football analytics eras; where Frame2Threat fits |
| `reports/03_methodology.md` | Full technical approach: pipeline, features, models, evaluation |
| `reports/data_dictionary.md` | Schema for all canonical tables (v1 + v2) |
| `reports/feature_timing_audit.md` | Possession feature timing audit and leakage caveats |
| `reports/label_methodology.md` | Operational label definitions, invariants, borderline cases |
| `reports/repro/` | Inspectable scored sample, model cards, and repro evidence pack |
| `reports/experiment_log.md` | 19-experiment registry with results (EXP-001 to EXP-019, v1–v3) |
| `reports/final_report.md` | Comprehensive project report with all results |

---

## Research questions and answers

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | Does 360 context improve prediction? | Barely: +0.001 ROC AUC at pass level, so event attributes carry nearly all measured signal |
| RQ2 | Which geometry features matter? | n_defenders_goal_side, pass_corridor_clear, receiver_between_lines |
| RQ3 | How to predict possession danger? | Leakage-free early forecasting is viable at 0.82+ AUC by 50%; full-possession models report retrospective upper-bound scores of XGB 0.950, GRU 0.952, Ensemble 0.965 |
| RQ4 | Which events matter most? | Moderate concentration (Gini 0.495); dribbles dominate tipping points (64.4%) |
| RQ5 | Player-level attribution? | Yes — 453-player leaderboard, domain-consistent |
| RQ6 | GNN vs. tabular? | Near-parity (0.841 vs 0.845 on 360 subset) |

---

## Pitch coordinate system

StatsBomb standard: **120 × 80 metres**.  x: own goal (0) → opponent goal (120).  y: left touchline (0) → right touchline (80).  Final third: x ≥ 80.  Penalty box: x ≥ 102, 18 ≤ y ≤ 62.

---

## Limitations

1. 360 coverage is partial — not all matches have freeze-frame data, and the measured pass-level lift from 360 geometry is only +0.001 ROC AUC.
2. Positional snapshots, not tracking — no velocity or trajectory.
3. Partial pitch visibility — not all 22 players are always visible.
4. Open-play only — set-piece passes excluded from v1 modelling.
5. Threat gain is a zone proxy — not a full possession-value model.
6. Limited sample size — StatsBomb open data covers ~100 matches.
7. Full-possession v2 scores are retrospective upper bounds because some features summarise the completed possession.
8. Evaluation is held-out, not causal.

---

## Licence

StatsBomb Open Data is used under the [StatsBomb Open Data Licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).  Project code: MIT.
