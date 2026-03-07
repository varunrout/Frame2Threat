# Experiment Log — Frame2Threat

This log records all modelling experiments, hyperparameter choices, and results.  Each entry follows the format: **date | experiment | settings | result | decision**.

---

## Format

```
### EXP-NNN — <short title>
Date: YYYY-MM-DD
Status: [ planned | running | complete | abandoned ]
Author: <initials or "auto">

**Goal:** One sentence.
**Method:** Bullet points of what was done.
**Settings:** Key hyperparameters and config.
**Results:** Table or prose.
**Decision:** What was decided and why.
**Files:** Artefacts produced.
```

---

## Baseline experiments

### EXP-001 — Rule-based line-break benchmark

Date: (to be filled after first run)  
Status: planned  

**Goal:** Establish a deterministic rule-based floor for line-break prediction.

**Method:**
- Apply `RuleBasedLineBreak`: pass length > 15m AND end_x > 75
- Evaluate on test split (match-level)

**Settings:**
- length threshold: 15m
- end_x threshold: 75

**Results:** (to be filled)

**Decision:** This is the no-ML floor.  Any trained model must exceed it.

**Files:** (to be filled)

---

### EXP-002 — Logistic regression, event features only

Date: (to be filled)  
Status: planned  

**Goal:** Establish learnability with a linear model using only event attributes.

**Method:**
- Build event features (`src/features/event_features.py`)
- Train logistic regression with isotonic calibration
- Evaluate on validation split

**Settings:**
- `C=1.0`, `max_iter=1000`, `random_state=42`
- Calibration: isotonic

**Results:** (to be filled)

**Decision:** (to be filled after run)

**Files:** (to be filled)

---

### EXP-003 — XGBoost, event features only

Date: (to be filled)  
Status: planned  

**Goal:** Strong tabular baseline without 360 geometry.

**Method:**
- Build event features only
- Train XGBoost classifier

**Settings:**
- `n_estimators=500`, `max_depth=6`, `lr=0.05`, `subsample=0.8`
- Early stopping: 50 rounds on validation AUC

**Results:** (to be filled)

---

### EXP-004 — XGBoost, event + 360 geometry features

Date: (to be filled)  
Status: planned  

**Goal:** Quantify the additive value of 360 freeze-frame context.

**Method:**
- Build event features + geometry features (`src/features/geometry_features.py`)
- Only use passes with `has_360=True`
- Compare to EXP-003 on same subset

**Results:** (to be filled)

**Decision:** If EXP-004 >> EXP-003: 360 features clearly add value.  
If EXP-004 ≈ EXP-003: Report honestly; the geometry may not carry independent signal beyond event attributes.

---

### EXP-005 — GNN frame model (multitask)

Date: (to be filled)  
Status: planned  

**Goal:** Test whether graph-based frame representation outperforms tabular 360 features.

**Method:**
- Build player graphs for all events with 360 (`src/features/graph_builder.py`)
- Train `PassFrameGNN` with GraphSAGE layers, mean pooling, multitask heads
- Targets: line_break, dangerous_progression_k, final_third_entry_k, box_entry_k, shot_within_k

**Settings:** See `configs/model_gnn.yaml`

**Results:** (to be filled)

---

### EXP-006 — Hybrid GNN + GRU sequence model

Date: (to be filled)  
Status: planned  

**Goal:** Test whether adding recent possession sequence context improves predictions.

**Method:**
- `HybridGNNSeq`: GNN frame embedding + GRU sequence embedding, fused via concatenation
- Targets: same as EXP-005

**Results:** (to be filled)

---

### EXP-007 — Multitask vs. single-task comparison

Date: (to be filled)  
Status: planned  

**Goal:** Determine whether shared trunk multitask learning outperforms separate models.

**Method:**
- Train five separate single-task XGBoost models
- Compare to `MultitaskTabular` and to the GNN multitask head
- Metric: aggregate performance across all tasks

**Results:** (to be filled)

---

### EXP-008 — Ablation study

Date: (to be filled)  
Status: planned  

**Goal:** Quantify the contribution of each feature group.

| Configuration | Features used |
|---------------|---------------|
| event-only | Event attributes only |
| event + 360 | Event + geometry features |
| graph-only | GNN frame embedding only |
| hybrid | GNN + sequence + event |

**Results:** (to be filled)

---

## Results summary table

| EXP | Model | Task | Feature set | ROC-AUC | PR-AUC | Brier | ECE |
|-----|-------|------|-------------|---------|--------|-------|-----|
| 001 | Rule-based | line_break | N/A | — | — | — | — |
| 002 | LogReg | line_break | event-only | — | — | — | — |
| 003 | XGBoost | line_break | event-only | — | — | — | — |
| 004 | XGBoost | line_break | event+360 | — | — | — | — |
| 005 | GNN | multitask | graph | — | — | — | — |
| 006 | Hybrid | multitask | graph+seq | — | — | — | — |

*(Fill in as experiments run.)*

---

## Research question answers

*(To be filled after experiments.)*

1. **Does 360 positional context improve prediction over event-only baselines?** → EXP-003 vs EXP-004
2. **Which geometric properties most strongly relate to line-breaking?** → SHAP analysis of EXP-004
3. **Can graph-based modelling outperform strong tabular baselines?** → EXP-005 vs EXP-004
4. **Does multitask learning help?** → EXP-007
5. **Which players/teams generate the most context-adjusted dangerous progression?** → `src/evaluation/tactical_review.py :: player_progression_profile()`
6. **Can the system rank visible passing options tactically?** → `src/models/ranking.py` evaluation

---

## Configuration notes

All experiment configs are version-controlled in `configs/`.  Each training run should save:
- `configs/` snapshot
- trained model artifact (`.pkl` or `.pt`)
- `metrics.json`
- `predictions.parquet`
- `feature_schema.json`
- `split_manifest.csv`
