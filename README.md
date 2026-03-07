# Frame2Threat

**Breaking the Block: Predicting and Explaining Dangerous Progression from StatsBomb Event Data and 360 Freeze Frames**

A reproducible football analytics system that predicts whether a pass breaks a defensive line, produces dangerous progression, or leads to final-third / box entries and shots — using StatsBomb Open Data event attributes combined with 360 freeze-frame spatial context.

---

## Quick start

```bash
git clone https://github.com/varunrout/Frame2Threat.git
cd Frame2Threat
pip install -e ".[dev]"
streamlit run src/app/app.py   # demo app (works without data via demo mode)
pytest tests/ -q               # run all tests
```

---

## Project overview

StatsBomb's 360 data provides a spatial snapshot of all visible players at the moment of each event.  Frame2Threat treats this as *event-conditioned positional intelligence*: for each open-play pass, it combines the event attributes (length, angle, body part, pressure, etc.) with the geometric structure of the visible defensive block to predict downstream attacking danger.

### What the system predicts

| Label | Definition |
|-------|-----------|
| `strict_line_break` | Pass crosses ≥ 2 defenders (by x-position in freeze frame) |
| `loose_line_break` | Pass crosses ≥ 1 defender (by x-position) |
| `dangerous_progression_k` | Final-third entry OR box entry OR shot within k=5 same-possession events |
| `final_third_entry_k` | Ball reaches x ≥ 80 within k events |
| `box_entry_k` | Ball enters penalty box within k events |
| `shot_within_k` | Shot occurs within k events |
| `threat_gain` | Zone-value delta (xT proxy, range ≈ [−1, 1]) |

### Hard constraints

- Uses **StatsBomb Open Data** only — event data plus 360 freeze frames.
- **Not** a full tracking-data system. No continuous trajectories, velocities, or pitch control.
- 360 features and geometry-dependent labels are **NaN** for events without freeze-frame coverage.
- Pass option ranking is restricted to **visible teammates** in the freeze frame.

---

## Repository structure

```
Frame2Threat/
├── README.md
├── pyproject.toml
├── configs/                   # All experiment config (data, labels, features, models, eval)
├── data/raw|interim|processed # Data directories (gitignored)
├── notebooks/                 # 01_data_audit … 06_error_analysis
├── reports/                   # data_dictionary, label_methodology, experiment_log, final_report
├── src/
│   ├── data/     ingest, inventory, parse_events/lineups/360, join_pass_frames, splits
│   ├── labels/   line_break, dangerous_progression, downstream_outcomes, validation_sampling
│   ├── features/ event_features, geometry_features, graph_builder, sequence_context
│   ├── models/   baselines, tabular, gnn, hybrid, ranking
│   ├── evaluation/ metrics, calibration, ablations, tactical_review
│   ├── visualization/ pitch_plots, freeze_frame_view, explanations
│   └── app/      app.py (Streamlit)
└── tests/        conftest, test_ingestion, test_labels, test_features, test_splits, test_models
```

---

## Installation

```bash
pip install -e .                    # core
pip install -e ".[dev,notebooks]"   # + pytest, jupyter
```

PyTorch and `torch-geometric` are optional (required only for GNN models).

---

## Running the pipeline

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

# 4. Split (match-level, no leakage)
from src.data.splits import create_match_level_splits
train_df, val_df, test_df = create_match_level_splits(pass_instances, seed=42,
    manifest_path="data/processed/split_manifest.csv")

# 5. Features + train
from src.features.event_features import build_event_features
from src.models.tabular import TabularClassifier
X_train = build_event_features(train_df)
clf = TabularClassifier(model_type="xgboost", task="dangerous_progression_k")
clf.fit(X_train, train_df["dangerous_progression_k"])

# 6. App
# streamlit run src/app/app.py
```

---

## Testing

```bash
pytest tests/ -v
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

| Notebook | Purpose |
|----------|---------|
| `01_data_audit.ipynb` | Inventory, parsing, missingness |
| `02_label_validation.ipynb` | Prevalence, sanity checks, zone breakdown |
| `03_eda.ipynb` | Distributions, heatmaps, 360 coverage |
| `04_baselines.ipynb` | Rule-based → LogReg → XGBoost + SHAP |
| `05_gnn.ipynb` | GNN training and ablations |
| `06_error_analysis.ipynb` | FP/FN, explanations, player profiles |

---

## Reports

| Document | Contents |
|----------|---------|
| `reports/data_dictionary.md` | Full schema for all canonical tables |
| `reports/label_methodology.md` | Operational definitions, invariants, borderline cases |
| `reports/experiment_log.md` | Experiment registry and results table |
| `reports/final_report.md` | Full research report |

---

## Pitch coordinate system

StatsBomb standard: **120 × 80 metres**.  x: own goal (0) → opponent goal (120).  y: left touchline (0) → right touchline (80).  Final third: x ≥ 80.  Penalty box: x ≥ 102, 18 ≤ y ≤ 62.

---

## Limitations

1. 360 coverage is partial — not all matches have freeze-frame data.
2. Positional snapshots, not tracking — no velocity or trajectory.
3. Partial pitch visibility — not all 22 players are always visible.
4. Open-play only — set-piece passes are excluded.
5. Threat gain is a zone proxy — not a full possession-value model.

---

## Licence

StatsBomb Open Data is used under the [StatsBomb Open Data Licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).  Project code: MIT.
