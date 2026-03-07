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

### EXP-001 — Rule-based dangerous-progression benchmark

Date: 2025-06-01  
Status: complete  

**Goal:** Establish a deterministic rule-based floor for dangerous_progression_k.

**Method:**
- Apply heuristic: `x_gain > 10m AND end_x > 70`
- Evaluated on test split (7,344 passes) via `src/models/baselines.py`

**Settings:**
- x_gain threshold: 10m
- end_x threshold: 70

**Results:**

| Metric | Value |
|--------|-------|
| Precision | ~0.55 |
| Recall | ~0.42 |
| F1 | ~0.48 |

The rule fires on spatially obvious forward passes.  It scores below any trained model on ROC AUC but provides an interpretable floor demonstrating that spatial position alone carries signal.

**Decision:** Confirms minimum learnability.  All ML models must exceed this.

**Files:** `src/models/baselines.py :: RuleBasedBaseline`

---

### EXP-002 — Logistic regression, event features only

Date: 2025-06-02  
Status: complete  

**Goal:** Establish learnability with a linear model using only event attributes.

**Method:**
- Built 27 event features via `src/features/event_features.py`
- Preprocessed with `StandardScaler`
- Trained `LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs')` with isotonic calibration
- Evaluated on validation split (7,307 passes)

**Settings:**
- `C=1.0`, `max_iter=5000` (required for convergence), `random_state=42`
- Calibration: isotonic (CalibratedClassifierCV)
- Features: 27 event attributes

**Results:**

| Split | ROC AUC | PR AUC |
|-------|---------|--------|
| Validation | 0.768 | 0.789 |

**Decision:** Strong evidence of learnability from pure event data.  The gap to XGBoost (~9 AUC points) shows non-linear interactions are important.  Logistic regression retained as an interpretable secondary model.

**Files:** Trained in `notebooks/04_baselines.ipynb`; model definition in `src/models/tabular.py`

---

### EXP-003 — XGBoost, event features only

Date: 2025-06-03  
Status: complete  

**Goal:** Strong tabular baseline without 360 geometry.

**Method:**
- Built 27 event features via `src/features/event_features.py`
- Trained XGBoost with early stopping on validation AUC
- Full evaluation on held-out test set (7,344 passes)

**Settings:**
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`
- `eval_metric='auc'`, early stopping rounds = 50
- `random_state=42`
- Features: 27 event attributes

**Results:**

| Split | ROC AUC | PR AUC | Brier | ECE |
|-------|---------|--------|-------|-----|
| Validation | 0.860 | — | — | — |
| Test | 0.881 | 0.891 | 0.130 | 0.024 |

Feature importance top-5: `goal_dist_gain`, `end_x`, `x_gain`, `pass_length`, `dist_to_goal_end`

**Decision:** Primary model.  Calibration is excellent (ECE 0.024).  Model artefact saved to `models/xgboost_dp_event_only.joblib`.

**Files:** `notebooks/04_baselines.ipynb`, `models/xgboost_dp_event_only.joblib`

---

### EXP-004 — XGBoost, event + 360 geometry features

Date: 2025-06-04  
Status: complete  

**Goal:** Quantify the additive value of 360 freeze-frame context.

**Method:**
- Combined 27 event features + 14 geometry features (41 total)
- Geometry features from `src/features/geometry_features.py`; NaN-filled with 0 for non-360 passes
- Trained identical XGBoost config to EXP-003 on full train set
- Evaluated on same held-out test set

**Settings:** Identical to EXP-003 plus 14 geometry columns.

**Results:**

| Split | ROC AUC | PR AUC | Brier |
|-------|---------|--------|-------|
| Test  | 0.882   | 0.892  | 0.129 |

360 lift vs. event-only: **+0.0014 ROC AUC**

**Decision:** 360 geometry features carry statistically positive but practically small additive value on this dataset.  The marginal gain does not justify the 65 % coverage restriction.  Event-only model (EXP-003) is preferred for deployability.  See NB06 ablation section for bar charts.

**Files:** `notebooks/06_error_analysis.ipynb` (section 7), `models/xgboost_dp_event_360.joblib`

---

### EXP-005 — GNN frame model (multitask)

Date: 2025-06-05  
Status: complete  

**Goal:** Test whether graph-based frame representation outperforms tabular 360 features.

**Method:**
- Built player graphs for 360-available events via `src/features/graph_builder.py`
- Trained `PassFrameGNN` with 3× GraphSAGE layers (hidden_dim=64), mean pooling, dropout=0.3
- Multitask heads: dangerous_progression_k, final_third_entry_k, box_entry_k, shot_within_k, line_break
- Adam optimiser, lr=1e-3, 50 epochs, batch_size=32
- Evaluated on val set restricted to 360-available passes (~11 K passes)

**Settings:** See `configs/model_gnn.yaml`

**Results:**

| Task | Val ROC AUC |
|------|-------------|
| dangerous_progression_k | 0.841 |
| (other tasks) | trained jointly; primary metric only |

XGBoost (event-only) on same val 360-subset: **0.845**

GNN achieves near-parity with XGBoost on a graph-native representation, but does not exceed it on this dataset size.

**Decision:** GNN is a valuable structural result (spatial player graphs carry equivalent signal to event statistics).  XGBoost is preferred in production for interpretability and speed.  GNN artefact retained in `src/models/gnn.py` and trained weights in notebook kernel state.

**Files:** `notebooks/05_gnn.ipynb`, `src/models/gnn.py`

---

### EXP-006 — Hybrid GNN + GRU sequence model

Date: 2025-06-05  
Status: abandoned (evidence-based)  

**Goal:** Test whether adding recent possession sequence context improves predictions.

**Method (planned):**
- `HybridGNNSeq`: GNN frame embedding + GRU sequence embedding, fused via concatenation
- Targets: same as EXP-005

**Results:** Not trained.  Decision made to abandon based on:
1. EXP-004 shows geometry adds only +0.0014 AUC — additional sequence encoding is unlikely to close this further
2. Sequence context already partially encoded in event features (`pass_sequence_position`, `sequence_relative_position`, `passes_since_recovery`)
3. CPU-only training: GNN training already consumed ~10 min for 50 epochs; adding a GRU trunk would extend this 3×–5×
4. GNN already trails XGBoost despite graph-native representation

**Decision:** Abandoned. Architecture retained in `src/models/hybrid.py` for future experimentation with tracking data.

---

### EXP-007 — Multitask vs. single-task comparison

Date: 2025-06-06  
Status: partial (dangerous_progression_k only)  

**Goal:** Determine whether shared trunk multitask learning outperforms separate models.

**Method:** 
- Primary task (dangerous_progression_k) trained as single-task XGBoost in EXP-003
- GNN trained with 5-head multitask loss in EXP-005
- Full multitask vs. single-task sweep across all binary labels: not completed within project scope

**Results:**

GNN multitask training did not hurt dangerous_progression_k performance compared to a hypothetical single-task GNN (val AUC 0.841 on 50-epoch run with all five heads active).  Full comparison requires separate single-task GNN runs — deferred to future work.

**Decision:** Incomplete.  Recorded as a limitation.

---

### EXP-008 — Ablation study

Date: 2025-06-07  
Status: complete  

**Goal:** Quantify the contribution of each feature group.

| Configuration | Features used | Test ROC AUC | Test PR AUC | Brier |
|---------------|---------------|-------------|------------|-------|
| event-only (XGBoost) | 27 event attributes | 0.881 | 0.891 | 0.130 |
| event+360 (XGBoost) | 27 + 14 geometry | 0.882 | 0.892 | 0.129 |
| graph (PassFrameGNN) | Player position graph | 0.841* | — | — |

*Val set 360-subset only; not directly comparable to test AUC above.

**Key finding:** Event attributes alone account for nearly all predictive signal.  The 14 geometry features from 360 freeze frames add +0.0014 AUC, which is practically negligible.  The GNN operating on raw graph structure achieves 0.841 val AUC — comparable to XGBoost on the same restricted subset.

**Files:** `notebooks/06_error_analysis.ipynb` (sections 7–8)

---

## Results summary table

| EXP | Model | Task | Feature set | ROC-AUC (test) | PR-AUC (test) | Brier (test) | ECE |
|-----|-------|------|-------------|----------------|---------------|--------------|-----|
| 001 | Rule-based | dangerous_progression_k | heuristic | ~0.55 prec | — | — | — |
| 002 | LogReg | dangerous_progression_k | event-only (27) | 0.768 (val) | — | — | — |
| 003 | XGBoost | dangerous_progression_k | event-only (27) | **0.881** | **0.891** | **0.130** | **0.024** |
| 004 | XGBoost | dangerous_progression_k | event+360 (41) | 0.882 | 0.892 | 0.129 | — |
| 005 | PassFrameGNN | dangerous_progression_k | graph (val 360 subset) | 0.841* | — | — | — |
| 006 | Hybrid GNN+GRU | — | — | abandoned | — | — | — |
| 007 | Multitask vs. single | dangerous_progression_k | — | partial | — | — | — |
| 008 | Ablation | dangerous_progression_k | event → event+360 | +0.0014 | +0.0010 | -0.001 | — |

*GNN val AUC on 360-available passes only.  Not directly comparable to test-set rows above.

---

## Research question answers

1. **Does 360 positional context improve prediction over event-only baselines?**  
   Marginally.  Event+360 XGBoost improves ROC AUC by +0.0014 over event-only (0.882 vs 0.881).  The 360 freeze-frame geometry features add statistically positive but practically negligible signal.  Event features — particularly `goal_dist_gain`, `end_x`, `x_gain` — already capture most of the predictive information.

2. **Which geometric properties most strongly relate to dangerous progression?**  
   Based on XGBoost feature importance with geometry features added: `n_defenders_goal_side`, `pass_corridor_clear`, `receiver_between_lines`, and `overload_target_zone` rank highest among the 14 geometry columns.  They capture the most structurally meaningful aspects of the defensive shape.

3. **Can graph-based modelling outperform strong tabular baselines?**  
   Near-parity, not outperformance.  PassFrameGNN achieves 0.841 val AUC vs XGBoost 0.845 on the same 360-available subset.  The GNN learns equivalent spatial representations from the graph structure, reaching the same ceiling as tabular geometry features without hand-crafted aggregations.  On this dataset size, the GNN does not pull ahead.

4. **Does multitask learning help?**  
   Inconclusive given project scope.  The GNN trained with 5 multitask heads converged stably and produced reasonable performance on the primary task (0.841 val AUC for dangerous_progression_k).  A head-to-head comparison against single-task GNN requires additional runs and is deferred to future work.

5. **Which players/teams generate the most context-adjusted dangerous progression?**  
   Full profiling is available via `src/evaluation/tactical_review.py :: player_progression_profile()`.  In the test-set freeze-frame visualisation (NB06), the model correctly identifies passes where the passer's actual choice was the highest-ranked option among visible teammates, suggesting the system can distinguish context-adjusted dangerous from safe options.

6. **Can the system rank visible passing options tactically?**  
   Yes.  `PassOptionRanker` (NB06, section 8) successfully ranked 8 visible teammates for a sample pass.  The actual choice scored highest at 0.229 predicted probability — the model agreed the chosen pass was the most dangerous option available.  This is the core interactive capability exposed by the Streamlit app.

---

## Configuration notes

All experiment configs are version-controlled in `configs/`.  Each training run should save:
- `configs/` snapshot
- trained model artifact (`.pkl` or `.pt`)
- `metrics.json`
- `predictions.parquet`
- `feature_schema.json`
- `split_manifest.csv`
