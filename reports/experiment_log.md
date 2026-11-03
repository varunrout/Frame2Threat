> **Note (2026-11):** the exploratory notebooks cited as provenance below (`notebooks/01`-`10`) were removed from `main`; they remain in git history (last present at commit 15e8ec4).  Their outputs are the figures and tables under `reports/`.

# Experiment Log — Frame2Threat

This log is a reconstructed analysis narrative for modelling experiments, hyperparameter choices, and results.  Entries dated before the repository's real implementation history are sequence markers for the analytical story, not commit dates or proof of when code was written.  Each entry follows the format: **sequence date | experiment | settings | result | decision**.

For the actual executable state of the project, use the Git history, CI runs, `scripts/run_pipeline.py`, and the reproducibility artifacts under `reports/repro/`.

---

## Format

```
### EXP-NNN — <short title>
Date or reconstructed sequence date: YYYY-MM-DD
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

Reconstructed sequence date: 2025-06-01
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

Reconstructed sequence date: 2025-06-02
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

> **Superseded as evidence about event features (EXP-024):** the label `dangerous_progression_k` counts the receipt of the pass itself, so `end_x` alone gives ROC AUC 0.800 on this test split.  On the corrected label `progression_beyond_pass_k` the same model scores 0.718.  The 0.881 reproduces exactly and remains correct *for the original label*.

Reconstructed sequence date: 2025-06-03
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

Reconstructed sequence date: 2025-06-04
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

> **Correction (EXP-024/025):** match-level bootstrap 95% CI for the ROC AUC delta is [-0.0011, +0.0038], so "statistically positive" is withdrawn.  The label `dangerous_progression_k` is also self-determining (see EXP-024); on the corrected label the delta is +0.0050 [-0.0011, +0.0106].

**Decision:** 360 geometry features carry a small additive value on this dataset.  The marginal gain does not justify restricting to the 52.3% of passes (65% of matches) that have 360 coverage.  Event-only model (EXP-003) is preferred for deployability.  See NB06 ablation section for bar charts.

![v1 baseline comparison](figures/v1_baseline_comparison.png)
*Figure: ROC/PR curves across all v1 models (EXP-001 through EXP-004).*

![v1 feature importance](figures/v1_feature_importance.png)
*Figure: SHAP feature importance for XGBoost event-only model (EXP-003).*

![v1 calibration](figures/v1_calibration.png)
*Figure: Calibration diagram for v1 XGBoost (ECE = 0.024).*

**Files:** `notebooks/06_error_analysis.ipynb` (section 7), `models/xgboost_dp_event_360.joblib`

---

### EXP-005 — GNN frame model (multitask)

> **Superseded (EXP-032).**  Original label only (`dangerous_progression_k`, validation, 360 subset); the checkpoint was never committed and the saved notebook output shows 0.834, not 0.841.  The corrected-label retrain is a different, graph-only model: the GNN scores 0.595 [0.578, 0.606] vs 0.715 [0.697, 0.734] for XGBoost event+360 on the identical test subset.

Reconstructed sequence date: 2025-06-05
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

![GNN training curves](figures/v1_gnn_training_curves.png)
*Figure: PassFrameGNN training loss and validation AUC across 50 epochs (EXP-005).*

![XGBoost vs GNN](figures/v1_xgb_vs_gnn.png)
*Figure: XGBoost vs PassFrameGNN on the 360-available validation subset.*

**Files:** `notebooks/05_gnn.ipynb`, `src/models/gnn.py`

---

### EXP-006 — Hybrid GNN + GRU sequence model

Reconstructed sequence date: 2025-06-05
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

Reconstructed sequence date: 2025-06-06
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

Reconstructed sequence date: 2025-06-07
Status: complete

**Goal:** Quantify the contribution of each feature group.

| Configuration | Features used | Test ROC AUC | Test PR AUC | Brier |
|---------------|---------------|-------------|------------|-------|
| event-only (XGBoost) | 27 event attributes | 0.881 | 0.891 | 0.130 |
| event+360 (XGBoost) | 27 + 14 geometry | 0.882 | 0.892 | 0.129 |
| graph (PassFrameGNN) | Player position graph | 0.841* (original label only, superseded; see EXP-032) | — | — |

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
| 005 | PassFrameGNN | dangerous_progression_k | graph (val 360 subset) | 0.841* (original label only, superseded; see EXP-032) | — | — | — |
| 006 | Hybrid GNN+GRU | — | — | abandoned | — | — | — |
| 007 | Multitask vs. single | dangerous_progression_k | — | partial | — | — | — |
| 008 | Ablation | dangerous_progression_k | event → event+360 | +0.0014 | +0.0010 | -0.001 | — |
| 009 | XGBoost | poss_dangerous | possession tabular (41) | **0.9505** | **0.8947** | — | — |
| 010 | PossessionGRU | poss_dangerous | event sequence (8) | **0.9524** | **0.9282** | — | — |
| 011 | Ensemble | poss_dangerous | XGB + GRU | **0.9650** | **0.9358** | — | — |
| 012 | XGBoost ablation | poss_dangerous | origin only | 0.591 | — | — | — |
| 013 | Attribution | poss_dangerous | LOO / Gini | partial | — | — | — |
| 014 | Correlation | v1 vs v2 | team aggregation | r = 0.319 | — | — | — |
| 015 | XGBoost | poss_dangerous | start-only (19) | **0.6241** | **0.5346** | — | — |
| 016 | GRU prefix | poss_dangerous | 25/50/75/100% prefix | 0.7167 / 0.8195 / 0.8793 / **0.9475** | 0.6761 / 0.7732 / 0.8468 / **0.9283** | — | — |
| 017 | XGBoost cumulative | poss_dangerous | 25/50/75/100% prefix-built tabular | 0.8136 / 0.8472 / 0.8912 / **0.9517** | 0.7127 / 0.7624 / 0.8200 / **0.8978** | — | — |
| 019 | Tipping-point analysis | poss_dangerous | GRU cumulative curve | cross-rate 47.5% | dangerous-only 92.0% | — | — |

*GNN val AUC on 360-available passes only.  Not directly comparable to test-set rows above.

![v1 360 ablation](figures/v1_ablation_360.png)
*Figure: Feature-group ablation — event-only vs event+360 (EXP-008).*

---

## V2 — Possession-level experiments

### EXP-009 — Possession-level XGBoost baseline (poss_dangerous)

> **Re-run (EXP-023):** the labels and `max_x_reached` were corrected for the opponent-coordinate defect.  Like-for-like retrain on corrected labels: XGBoost 0.946 -> 0.957 ROC AUC (pre-fix vs corrected), GRU (EXP-010) 0.953 -> 0.957, logit-mean ensemble (EXP-011) 0.974 -> 0.977.  The saved 0.9505 artefact scores 0.945 under the current XGBoost version, so the retrain is the comparable pair.

Reconstructed sequence date: 2025-06-10
Status: complete

**Goal:** Establish a strong tabular baseline at the possession level using aggregated features.

**Method:**
- Built canonical `possession_sequences` table from events via `src/data/parse_possessions.py`
- 13 possession-level labels computed by `src/labels/possession_labels.py`
- 41 features: spatial (start_x/y, end_x/y, max_x_reached, territory_gained), counting (n_events, n_passes, n_carries, n_pressures_faced, duration_seconds), meta (mean_pass_length, has_pressure), and all engineered label-derived features (poss_tempo, poss_verticality, poss_recycled, poss_broke_pressure, poss_bypassed_lines, poss_pressure_index, poss_built_up, phase dummies)
- Target: `poss_dangerous` (= poss_has_shot OR poss_entered_box)
- Same XGBoost hyperparameters as EXP-003; match-level split

**Settings:**
- `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`
- `eval_metric='aucpr'`, early stopping rounds = 30
- `random_state=42`
- 41 possession-level features, 12,092 train / 2,475 test possessions
- Hyperparameters centralised in `configs/model_possession.yaml → xgboost_main`

**Results:**

| Split | ROC AUC | Average Precision |
|-------|---------|-------------------|
| Test  | **0.9505** | **0.8947** |

Feature importance top-5: `max_x_reached`, `territory_gained`, `end_x`, `n_passes`, `mean_pass_length`

**Decision:** Excellent possession-level baseline.  The higher AUC vs pass-level (0.950 vs 0.881) reflects the stronger aggregated signal at the possession granularity.  Model saved to `models/xgboost_poss_dangerous.joblib`.

**Files:** `notebooks/07_possession_features.ipynb`, `models/xgboost_poss_dangerous.joblib`

---

### EXP-010 — PossessionGRU sequence model (poss_dangerous)

Reconstructed sequence date: 2025-06-11
Status: complete

**Goal:** Test whether event-sequence modelling via GRU outperforms aggregated tabular features.

**Method:**
- GRU model takes variable-length event sequences (8 features per timestep) padded to max sequence length
- Architecture: `PossessionGRU` — single-layer GRU (hidden_dim=64), no bidirectional, dropout=0.0
- Input per timestep: `type_id`, `loc_x_norm`, `loc_y_norm`, `end_x_norm`, `end_y_norm`, `under_pressure`, `pass_length_norm`, `minute_norm`
- Binary cross-entropy loss, Adam optimiser, early stopping on val AUC (patience=6)
- Trained for up to 60 epochs (batch_size=512)

**Settings:** See `configs/model_gru.yaml`

**Results:**

| Split | ROC AUC | Average Precision |
|-------|---------|-------------------|
| Test  | **0.9524** | **0.9282** |
| Val   | 0.9363 | — |

The GRU marginally outperforms XGBoost (+0.002 ROC AUC, +0.033 AP on test).  The AP improvement is more notable, suggesting the GRU better separates the positive class at high-confidence thresholds.

**Decision:** GRU is the strongest single model.  Retained as ensemble component.  Weights saved to `models/gru_poss_dangerous.pt`.

**Files:** `notebooks/08_possession_sequence_model.ipynb`, `models/gru_poss_dangerous.pt`, `configs/model_gru.yaml`

---

### EXP-011 — Ensemble (XGBoost + GRU, mean logit fusion)

Reconstructed sequence date: 2025-06-11
Status: complete

**Goal:** Combine the tabular and sequence perspectives into a single prediction.

**Method:**
- Simple mean of logit-space predictions: `p_ensemble = σ(0.5 × logit(p_xgb) + 0.5 × logit(p_gru))`
- No additional training; just averages the two models' outputs at inference time
- Evaluated on the same held-out test set (2,475 possessions)

**Results:**

| Model | Test ROC AUC | Test AP |
|-------|-------------|---------|
| XGBoost alone | 0.9505 | 0.8947 |
| GRU alone | 0.9524 | 0.9282 |
| **Ensemble** | **0.9650** | **0.9358** |

The ensemble gains +1.3 points ROC AUC over the best single model, confirming that the XGBoost and GRU capture complementary information (tabular vs. sequential).

**Decision:** Ensemble is the best-performing configuration.  Both constituent models and the ensemble fusion logic are fully reproducible.

![v2 model comparison curves](figures/v2_model_comparison_curves.png)
*Figure: ROC and PR curves for XGBoost, GRU, and Ensemble (EXP-009/010/011).*

![Possession SHAP summary](figures/poss_shap_summary.png)
*Figure: SHAP feature importance for possession-level XGBoost (EXP-009).*

![Possession calibration](figures/poss_calibration.png)
*Figure: Calibration diagram for possession-level models.*

**Files:** `notebooks/08_possession_sequence_model.ipynb`, `models/results_summary.json`

---

### EXP-012 — H4 ablation: origin-type features only

Reconstructed sequence date: 2025-06-12
Status: complete

**Goal:** Test hypothesis H4 — does possession origin type alone suffice for dangerous-progression prediction?

**Method:**
- Trained XGBoost using ONLY `origin_type` (one-hot encoded) as features
- Same train/test split as EXP-009
- Compared to full-feature XGBoost (EXP-009, AUC 0.9505)

**Results:**

| Configuration | Test ROC AUC |
|---------------|-------------|
| Origin-type only | **0.591** |
| Full features (41) | 0.9505 |
| Full GRU (sequence) | 0.9524 |

**Decision:** H4 REJECTED.  Origin type carries minimal signal (AUC 0.591 vs 0.950+).  Possession outcome is dominated by what happens *during* the possession, not how it started.  This supports the sequential/structural feature engineering approach.

**Files:** `notebooks/08_possession_sequence_model.ipynb`

---

### EXP-013 — H2: Attribution concentration (Gini coefficient)

> **SUPERSEDED by EXP-031.**  The Gini 0.495, the 453-player leaderboard and the high-impact event types were computed in-sample (the GRU had seen the possessions it attributed), on pre-fix labels, with the wrong event-type decoder, and credited defending-team events to players.  Figure moved to `figures/superseded_attribution/`.

Reconstructed sequence date: 2025-06-12
Status: complete

**Goal:** Test hypothesis H2 — is danger concentrated in a few key events within each possession?

**Method:**
- Leave-one-out (LOO) attribution via `src/evaluation/possession_attribution.py`
- For each event in a possession, compute danger-drop = P(dangerous|all events) − P(dangerous|all events minus this one)
- Measure concentration via Gini coefficient across all danger-drops within each possession
- Hypothesis: if danger is concentrated, Gini → 1.0; if evenly spread, Gini → 0.0

**Results:**

| Metric | Value |
|--------|-------|
| Median Gini coefficient | **0.495** |
| Max single-event danger share | **33.2%** |

Interpretation: Moderate concentration — about half the danger variance is attributable to a few pivotal events, but the other half is distributed.  The max-share of 33.2% means even the most important single event typically accounts for about 1/3 of a possession's danger score.

**Decision:** H2 PARTIAL.  Danger is partially concentrated (not fully distributed), but no single event dominates.  This validates the sequential approach: both individual actions and their context matter.

![H2 attribution concentration](figures/superseded_attribution/v2_h2_attribution_concentration.png)
*Figure: Distribution of Gini coefficients and max-event attribution shares (EXP-013).*

**Files:** `notebooks/08_possession_sequence_model.ipynb`

---

### EXP-014 — H1: Pass-level vs possession-level correlation

Reconstructed sequence date: 2025-06-12
Status: complete

**Goal:** Test hypothesis H1 — does possession-level danger signal correlate with pass-level dangerous progression at the team level?

**Method:**
- Aggregated per-team mean possession danger scores (from possession-level models) and per-team mean pass-level dangerous_progression_k rates (from v1 pass-level models)
- Computed Pearson and Spearman correlations across all teams in the dataset

**Results:**

Both v1 (pass-level) and v2 (possession-level) danger metrics rank teams similarly.  Teams with high pass-level dangerous progression rates also tend to have high possession-level danger scores.  Detailed correlation coefficients are reported in NB08.

**Decision:** H1 SUPPORTED.  The two levels of analysis capture overlapping but not identical constructs of team attacking quality.  The possession-level model captures additional sequential context not available at the pass level.

![H1 score correlation](figures/v2_h1_score_correlation.png)
*Figure: v1 pass danger score vs v2 possession danger score (r = 0.319) — EXP-014.*

**Files:** `notebooks/08_possession_sequence_model.ipynb`

---

### EXP-015 — Start-only XGBoost baseline

> **SUPERSEDED (fraction-based formulation, see EXP-021):** the prefix length `ceil(n*f)` encodes the possession length (length alone: ROC AUC 0.683) and many prefixes already contain the outcome (20.8% / 36.9% / 58.7% of dangerous possessions at f = 0.25 / 0.5 / 0.75).  Numbers below are not leakage-free.

Date: 2026-03-09
Status: complete

**Goal:** Test whether dangerous possessions can be forecast from information available at the instant the possession begins.

**Method:**
- Built a strict start-only feature set from `src/features/early_features.py`
- Included only possession-start context: start location, zone flags, period, and `origin_type` one-hot columns
- Explicitly excluded completion-dependent aggregates such as `max_x_reached`, `territory_gained`, `n_events`, phase dummies, and label-derived trajectory summaries
- Trained XGBoost on the same match-level train/val/test split as EXP-009/010

**Results:**

| Split | ROC AUC | Average Precision |
|-------|---------|-------------------|
| Test  | **0.6241** | **0.5346** |

Top features: `origin_from_throw_in`, `origin_from_free_kick`, `start_x`, `start_x_norm`, `origin_regular_play`

**Decision:** There is genuine pre-possession signal, but it is weak relative to in-possession information.  Start context alone is not sufficient for high-quality forecasting.

![EXP-015 start-only importances](figures/v3_exp015_start_only_importance.png)
*Figure: Top feature importances for the start-only XGBoost model (EXP-015).*

**Files:** `notebooks/10_early_prediction.ipynb`, `src/features/early_features.py`

---

### EXP-016 — Prefix GRU evaluation

> **SUPERSEDED (fraction-based formulation, see EXP-021):** the prefix length `ceil(n*f)` encodes the possession length (length alone: ROC AUC 0.683) and many prefixes already contain the outcome (20.8% / 36.9% / 58.7% of dangerous possessions at f = 0.25 / 0.5 / 0.75).  Numbers below are not leakage-free.

Date: 2026-03-09
Status: complete

**Goal:** Evaluate whether the existing possession GRU can recognise danger before the full possession has unfolded.

**Method:**
- Reused the saved `PossessionGRU` from EXP-010
- Scored only the first 25%, 50%, 75%, and 100% of each possession on the held-out test set
- Restricted evaluation to possessions with at least 4 events so each prefix is meaningful
- Used `src/evaluation/early_evaluation.py` to compute ROC-AUC and PR-AUC by prefix fraction

**Results:**

| Observed possession fraction | ROC AUC | Average Precision |
|-----------------------------|---------|-------------------|
| 25% | 0.7167 | 0.6761 |
| 50% | 0.8195 | 0.7732 |
| 75% | 0.8793 | 0.8468 |
| 100% | **0.9475** | **0.9283** |

**Decision:** Strong evidence that the GRU contains forward-looking signal.  By halfway through a possession, the model already reaches 0.820 ROC-AUC on unseen matches.

![EXP-016 prefix GRU curve](figures/v3_exp016_prefix_gru_curve.png)
*Figure: GRU ROC-AUC and PR-AUC as a function of observed possession fraction (EXP-016).*

**Files:** `notebooks/10_early_prediction.ipynb`, `src/evaluation/early_evaluation.py`

---

### EXP-017 — Cumulative-features XGBoost

> **SUPERSEDED (fraction-based formulation, see EXP-021):** the prefix length `ceil(n*f)` encodes the possession length (length alone: ROC AUC 0.683) and many prefixes already contain the outcome (20.8% / 36.9% / 58.7% of dangerous possessions at f = 0.25 / 0.5 / 0.75).  Numbers below are not leakage-free.

Date: 2026-03-09
Status: complete

**Goal:** Test how much predictive power remains when tabular features are rebuilt from only the early part of each possession.

**Method:**
- Created prefix-aware aggregate features via `build_cumulative_tabular_features()`
- Recomputed counting, territory, duration, and pressure aggregates using only the first 25%, 50%, 75%, or 100% of each event sequence
- Zeroed label-derived possession summary columns for partial prefixes to avoid retrospective leakage
- Trained a fresh XGBoost model at each prefix fraction using the same match-level split

**Results:**

| Observed possession fraction | ROC AUC | Average Precision |
|-----------------------------|---------|-------------------|
| 25% | 0.8136 | 0.7127 |
| 50% | 0.8472 | 0.7624 |
| 75% | 0.8912 | 0.8200 |
| 100% | **0.9517** | **0.8978** |

At every partial prefix, cumulative XGBoost outperforms prefix GRU; the gap is largest at 25% (+0.097 ROC-AUC) and narrows by 100% (+0.004).

**Decision:** Early aggregated evidence is highly predictive even without completed-possession summaries.  This is the strongest rebuttal to the reviewer critique that v2 performance is purely retrospective.

![EXP-016 vs EXP-017 comparison](figures/v3_exp016_exp017_auc_comparison.png)
*Figure: Prefix GRU vs cumulative-feature XGBoost ROC-AUC across observed possession fractions (EXP-016/017).*

**Files:** `notebooks/10_early_prediction.ipynb`, `src/features/early_features.py`

**Saved models:** Cumulative XGBoost models are trained and persisted by `src/models/train_early_models.py`:
- `models/superseded/xgboost_start_only.joblib` (EXP-015)
- `models/superseded/xgboost_cumulative_25pct.joblib` (EXP-017 @25%)
- `models/superseded/xgboost_cumulative_50pct.joblib` (EXP-017 @50%)
- `models/superseded/xgboost_cumulative_75pct.joblib` (EXP-017 @75%)

All hyperparameters are loaded from `configs/model_possession.yaml → xgboost_early`.

---

### EXP-019 — GRU tipping-point analysis

Date: 2026-03-09
Status: complete

**Goal:** Identify when a possession first becomes recognisably dangerous to the GRU.

**Method:**
- Computed cumulative GRU danger scores after each event in every test-set possession
- Defined the tipping point as the first event where $P(\text{dangerous}) \ge 0.50$
- Summarised threshold-crossing rates, median crossing time, and event-type distributions at the tipping point and at the maximum single-step jump

**Results:**

| Metric | Value |
|--------|-------|
| All possessions crossing 0.50 | **47.5%** |
| Dangerous possessions crossing 0.50 | **92.0%** |
| Median tipping fraction | **0.619** |
| Most common tipping event | ~~Dribble (64.4%)~~ **Ball Receipt (64.4%)** (corrected, EXP-020) |

Interpretation: ~~the GRU usually commits to a dangerous forecast around the final 40% of the possession, and dribbles are the dominant trigger for the transition into the dangerous regime.~~

> **Corrected (EXP-020):** the "Dribble 64.4%" row is **Ball Receipt**: `EVENT_TYPE_LABELS` decoded id 3 as Dribble while `TYPE_VOCAB` encodes id 3 as Ball Receipt (dribble = 4).  The percentage itself is unchanged because the model and split are unchanged.  Rate-normalised (share / base frequency): ball receipt 2.34x, goalkeeper 3.6x, shot 2.95x, dribble 1.5x (1.2% of crossings), pressure 0.62x, carry 0.44x, pass 0.33x.  The median tipping *fraction* (0.619) is a fraction-of-n quantity and is superseded by EXP-021.

![EXP-019 tipping events](figures/v3_exp019_tipping_events.png)
*Figure: Event-type distributions at the tipping point and at the largest single-step GRU danger increase (EXP-019).*

**Files:** `notebooks/10_early_prediction.ipynb`, `src/evaluation/early_evaluation.py`

---

## Corrections after the self-audit (EXP-020 to EXP-025)

These entries are dated by the repository history, not reconstructed.  They follow a self-audit whose four defects were reproduced by `scripts/audit_defects.py` (`reports/audit/defect_audit_before.md`) before any fix.  Notes on the reproduction: items 1, 2 and 4 matched the initial audit exactly; for item 3 the initial audit counted pass/carry *end* locations in the box test (27% / 42% / 64%), whereas the stored label uses event location only (20.8% / 36.9% / 58.7%); both are recorded.

### EXP-020 — Event-type vocabulary fix; tipping points and LOO re-run

Status: complete.  Code: `src/data/event_types.py` (one `IntEnum`, `encode`, `decode`, `label`; ids unchanged so the trained GRU's inputs stay valid), `tests/test_event_types.py`, `scripts/rerun_exp019_vocab_fix.py`.  Output: `reports/audit/exp020_vocab_fix.md`.

**Defect.**  `TYPE_VOCAB` (pass 1, carry 2, ball receipt 3, dribble 4, shot 5, ...), `EVENT_TYPE_LABELS` (0 Pass, 1 Carry, 2 Shot, 3 Dribble, ...) and `early_features` (pass 0, carry 1) disagreed.  Consequences fixed: the "dribbles 64.4%" claim (it was ball receipts), prefix `n_passes` (counted unknown events) and `n_carries` (counted passes), prefix `mean_pass_length` (also a metres/normalised unit mismatch), CLI/app event names.  `possession_labels.py`'s comment that its ids "match parse_possessions.py" was true (shot 5, pressure 6).

**Re-run (test split, existing GRU, threshold 0.50).**  Crossing rates and median fraction are unchanged (47.5%, 92.0%, 0.619).  Corrected first-crossing distribution with rate normalisation: ball receipt 64.4% (base 27.6%, rate 2.34), carry 10.3% (0.44), pass 9.8% (0.33), pressure 4.6% (0.62), goalkeeper 2.7% (3.61), shot 1.9% (2.95), foul committed 1.2% (1.72), dribble 1.2% (1.54).  LOO: Shot (+0.188), Goalkeeper (+0.137), Clearance, Foul Committed and Block have the largest mean attribution; as argmax 'unlock', ball receipt 33.8% (1.23x), pressure 17.7% (2.37x), shot 5.3% (8.4x).

**Known modelling weakness recorded (not changed here).**  The v2 GRU receives `type_id` as one raw numeric feature, imposing a false order on a categorical variable.  The event-time GRU (EXP-021) replaces it with a learned embedding.

### EXP-021 — Event-time (in-play) formulation replaces fraction-based forecasting

> **Numbers below superseded by EXP-026 (same-moment leak fix).**  The formulation stands; the values changed slightly: all snapshots XGBoost 0.668 -> 0.665, GRU 0.670 -> 0.664, ball x/y 0.648 -> 0.642; one per possession XGBoost 0.749 -> 0.736, GRU 0.756 -> 0.739, ball x/y 0.713 -> 0.695.  Current values: `reports/event_time/event_time_results.md`.

Status: complete.  Code: `src/features/event_time.py`, `src/models/event_time.py`, `src/evaluation/event_time_eval.py`, `scripts/run_event_time.py`; tests `tests/test_event_time.py`.  Output: `reports/event_time/`.

**Why.**  Length of the possession alone gives ROC AUC 0.683; the prefix length `ceil(n*f)` gives 0.682-0.683; many prefixes already contain the outcome (above).  The v3 numbers (GRU 0.820, XGBoost 0.847 at 50%) are withdrawn.

**Method.**  Snapshot rows for each event index `t < n-1` with no shot / box entry by the possessing team at or before `t`; label = shot / box entry in `t+1..n-1`.  Features from events `0..t` only (ball x/y, max x, counts so far, elapsed seconds from timestamps, current event type, start context); a test appends future events and checks the row for `t` is unchanged.  Train matches only; hyper-parameters chosen on validation log loss (XGBoost: depth 5, 59 trees; GRU: lr 0.003).  Rows: 215,185 / 45,300 / 44,914 (train / val / test), 305,399 in total.

**Results (test, match-level bootstrap 95%).**

| Model | ROC AUC, all snapshots | ROC AUC, one snapshot per possession |
|---|---|---|
| Baseline A, ball x/y | 0.648 [0.630, 0.664] | 0.713 [0.688, 0.735] |
| Baseline B, events so far | 0.564 [0.529, 0.601] | 0.530 [0.496, 0.567] |
| Start context only | 0.586 [0.563, 0.610] | 0.643 [0.612, 0.668] |
| XGBoost | 0.668 [0.653, 0.681] | 0.749 [0.727, 0.772] |
| GRU | 0.670 [0.650, 0.689] | 0.756 [0.739, 0.777] |

XGBoost minus ball-position baseline: +0.021 [+0.008, +0.034] (all), +0.036 [+0.023, +0.052] (one per possession).  GRU minus XGBoost: +0.001 [-0.014, +0.018].  Calibration under-predicts on the test split (test prevalence 0.371 vs train 0.325, val 0.280).  Sensitivity including terminal snapshots changes ROC AUC by under 0.01.

**Decision.**  Report early forecasting as modest; the dominant signal is ball position.  `score-live` uses the event-time model.  The old v3 code, models and notebook remain runnable and are labelled superseded.

### EXP-022 — Tipping points in event time

> **Numbers below superseded by EXP-026.**  After the fix: threshold XGBoost 0.390 / GRU 0.422; dangerous alerted 90.6% / 89.6% (was 94% / 93.5%); non-dangerous alerted 47.7% / 39.4% (was 41.9% / 40.6%); median lead 9 / 10 events, 8.4 / 8.9 s (was 6 / 7 events, 6.2 / 7.3 s).  First alerts: XGBoost pass 50.4% (rate 1.65), ball receipt 30.8% (1.07), carry 12.1% (0.47); GRU carry 44.3% (1.74), ball receipt 34.4% (1.19), pass 20.3% (0.66).  Still no dominant event type.

Status: complete.  Threshold chosen on **validation** only: the score quantile whose predicted-positive rate equals validation prevalence (0.280): XGBoost 0.372, GRU 0.353.

**Results (test).**  94% of dangerous possessions are alerted (median lead 6-7 events, about 6-7 s before the shot / box entry); 41-42% of non-dangerous possessions are alerted too.  First alerts: XGBoost pass 60.5% (rate 1.93), ball receipt 27.3% (0.96), carry 6.3% (0.25); GRU pass 40.7% (1.30), carry 34.0% (1.36), ball receipt 22.5% (0.79).  Largest one-step jumps: ball receipt / carry / pass in proportion to their frequency, and foul won (rate 15-17x, 55-61 events).  **There is no dominant tipping event type.**

### EXP-023 — Possession table coordinate-frame fix; v2 re-run

Status: complete.  Code: `src/data/possession_frame.py`, `parse_possessions.py`, `possession_labels.py`; script `scripts/rerun_v2_corrected_labels.py`.  Output: `reports/audit/exp023_v2_corrected_labels.md`.

**Defect.**  15.7% of events in a possession belong to the defending team and are recorded in the opposite frame; `max_x_reached`, box-entry and final-third labels used them unflipped.

**Effect on the 17,065 possessions.**  `poss_dangerous` 32.8% -> 28.3% (765 true -> false, 0 false -> true); `poss_entered_box` 30.0% -> 25.5% (762); `poss_has_shot` 12.4% -> 12.2% (30); `poss_entered_final_third` 75.4% -> 61.5% (2,381); `max_x_reached` changed for 3,893 and `team_name` for 86.  The GRU's input fields are byte-identical for all possessions.

**Like-for-like retrain, test possessions (unchanged v2 configuration).**

| Model | pre-fix labels | corrected labels |
|---|---|---|
| XGBoost | 0.9461 | 0.9572 |
| GRU | 0.9534 | 0.9570 |
| Logit-mean ensemble | 0.9736 | 0.9769 |

The saved XGBoost artefact deserialises under the current XGBoost version at 0.9449 (recorded 0.9505), and the recorded 0.9650 ensemble is not reproduced (0.9732 from the saved models with the documented equal-weight logit average), so neither is used as a comparison point.

### EXP-024 — Non-tautological pass label

Status: complete.  Code: `src/labels/progression_beyond_pass.py`; definition and worked examples in `reports/label_methodology.md` (Label G); invariants in `tests/test_labels.py`.

**Defect.**  The event after a completed pass is usually the receipt of that pass at the pass end, so `dangerous_progression_k` is 94.3% positive for `end_x >= 80` vs 24.8% below; 58% of positives have `end_x >= 80`; `end_x` alone gives ROC AUC 0.786 (0.800 on the test split).  The same look-ahead also reads defending-team events unflipped; the label is kept unchanged for comparability.

**New label `progression_beyond_pass_k`** (k=5): ignore the pass's own receipt and the opponent's events; positive for a shot, a new box entry (pass end not already in the box), or a gain of >= 10 m beyond `end_x` into a more advanced zone.  Prevalence 21.6% (old 43.3%); P(label | end_x >= 80) 0.174; `end_x` alone 0.447 on all passes (0.457 on the test split) when higher `end_x` is scored as more likely positive, i.e. 0.553 / 0.543 with the sign flipped (the logistic baseline learns the negative sign and scores 0.543): a modest inverse relationship, not a free ride.

### EXP-025 — v1 models and the 360 ablation on both labels

> **Numbers below superseded by EXP-027 (`possession_length` fix).**  The "about 3.5x larger" 360 delta (+0.0050) and the 0.718 event-only AUC on the corrected label came from a feature that mixed matches.  Current: event-only 0.728, event+360 0.729, 360 delta +0.0015 [-0.0036, +0.0070] on the corrected label and +0.0021 [-0.0010, +0.0055] on the original; the answer to the key question becomes **no**.  The original entry is kept below unchanged.

Status: complete.  Script: `scripts/run_pass_label_ablation.py`; output `reports/pass_label/pass_label_results.md`.  Same split (36,037 / 7,307 / 7,344 passes; 15 test matches), configuration from `configs/model_baseline.yaml`, 1000 match-level bootstrap resamples.

| Model | `dangerous_progression_k` ROC AUC / AP / Brier | `progression_beyond_pass_k` ROC AUC / AP / Brier |
|---|---|---|
| LR, 27 event features | 0.803 / 0.808 / 0.173 | 0.611 / 0.296 / 0.170 |
| XGBoost, 27 event | 0.881 / 0.891 / 0.130 | 0.718 / 0.411 / 0.189 |
| XGBoost, 41 event+360 | 0.882 / 0.892 / 0.129 | 0.723 / 0.424 / 0.184 |
| LR on `end_x` only | 0.800 / 0.806 / 0.176 | 0.543 / 0.266 / 0.173 |
| XGBoost on (`end_x`, `end_y`) | 0.857 / 0.865 / 0.140 | 0.687 / 0.359 / 0.225 |

360 delta (event+360 minus event-only XGBoost): original label ROC AUC +0.0014 [-0.0011, +0.0038], AP +0.0014 [-0.0005, +0.0033], Brier -0.0011 [-0.0026, +0.0004]; corrected label ROC AUC +0.0050 [-0.0011, +0.0106], AP +0.0129 [+0.0015, +0.0250], Brier -0.0051 [-0.0100, -0.0006].  On the 360-covered passes only: +0.0036 [-0.0012, +0.0083] and +0.0029 [-0.0057, +0.0111].

**Answer.**  Does 360 geometry matter more once the label stops rewarding the pass's own destination?  Slightly: the ROC AUC point estimate is about 3.5x larger and the AP and Brier gains exclude zero, but the ROC AUC interval still includes zero and the absolute gain is +0.005.  The event model's lead over destination-only baselines is large on the corrected label (+0.175 over `end_x`; +0.031 over (`end_x`, `end_y`)).  Brier values of XGBoost are inflated by `scale_pos_weight` and not comparable with the unweighted logistic baselines.  The PassFrameGNN could not be re-run (no `torch_geometric`, no checkpoint).

### EXP-026 — Same-moment leak in the event-time model (self-audit, second round)

Status: complete.  Code: `src/data/possession_frame.py::is_danger_event` (single definition), `src/features/event_time.py`; tests `tests/test_event_time.py`; audit `scripts/audit_same_moment_leak.py`; before/after `reports/event_time/same_moment_fix_before_after.md`; superseded outputs `reports/event_time/superseded_pre_same_moment_fix/`, `models/superseded/`.

**Defect.**  `max_x_so_far` and the GRU's end-x/end-y inputs include the current pass's END location, but danger was judged on event location only.  At a possessing-team pass ending in the box the snapshot therefore already contained the entry that the next event (the receipt) records.  Measured before the fix: 3,973 of 305,399 snapshots (1.30%) were at such a pass; label rate 80.4% there vs 31.9% elsewhere; 2,628 of 4,640 dangerous possessions (56.6%) had such a pass as their last eligible snapshot.

**Fix.**  A possessing-team pass whose end is in the box (x >= 102, 18 <= y <= 62) is itself a danger event; snapshots at or after it are not eligible.  After the fix: 0 snapshots at such a pass; eligible rows 305,399 -> 293,639.  `poss_dangerous` and the other possession labels were **not** changed (the label uses event location, and the receipt normally records the same entry); the event-time definition now disagrees with the stored label on 925 possessions, all of them incomplete or out-of-play passes ending in the box.

**Re-run (EXP-021/022, all five models, both views, buckets, alerts, bootstrap; test, 15 matches).**

| Model | all snapshots AUC before -> after | one per possession AUC before -> after | 20-draw mean before -> after |
|---|---|---|---|
| Ball x/y (LR) | 0.648 -> 0.642 | 0.713 -> 0.695 | 0.724 -> 0.692 |
| Events so far (LR) | 0.564 -> 0.561 | 0.530 -> 0.608 | 0.539 -> 0.612 |
| Start context (XGB) | 0.586 -> 0.592 | 0.643 -> 0.599 | 0.643 -> 0.599 |
| XGBoost | 0.668 -> 0.665 | 0.749 -> 0.736 | 0.756 -> 0.729 |
| GRU | 0.670 -> 0.664 | 0.756 -> 0.739 | 0.768 -> 0.744 |

XGBoost - ball x/y: +0.021 -> +0.022 (all), +0.036 -> +0.041 (one per possession); GRU - XGBoost stays indistinguishable from zero.  Alerts (validation-chosen threshold): dangerous 93.9% / 93.5% -> 90.6% / 89.6% (XGB / GRU), non-dangerous 41.9% / 40.6% -> 47.7% / 39.4%, median lead 6 / 7 -> 9 / 10 events.

**Reading.**  The leak was real but mild for ROC AUC (-0.003 to -0.006 over all snapshots; -0.013 to -0.017 with one snapshot per possession, where the removed rows are the high-label ones).  The events-so-far baseline rises in the one-per-possession view (0.530 -> 0.608) because eligibility now ends at the first pass into the box, so dangerous possessions have shorter pre-danger histories than safe ones; this is hazard behaviour, not a leak of the possession length.  The headline stays modest.

### EXP-027 — `possession_length` in the v1 pass features (self-audit, second round)

> **Recipe superseded by EXP-033.**  The XGBoost numbers below used the EXP-027 recipe (`scale_pos_weight`, 500 trees); the final models use early stopping without weighting (event-only 0.7337, event+360 0.7380 on all test passes) and the 360 gain is then measurable (EXP-033).  Kept as reported.

Status: complete.  Code: `src/features/event_features.py`; tests `tests/test_features.py`; before/after `reports/pass_label/possession_length_fix_before_after.md`; superseded outputs `reports/pass_label/superseded_pre_possession_length_fix/`.

**Defect.**  `possession_length` used `groupby("possession_id").cumcount()`; `possession_id` restarts every match, so different matches' possessions were mixed and the value depended on row order and on which split was being built (mean 145.4, maximum 836, correlation 0.06 with the correct count; changed for 49,216 of 50,688 passes).  **Fix.**  Count earlier passes of the same `(match_id, possession_id)` in event order (period, minute, second); mean 6.6, maximum 82.

**Re-run of EXP-025 (test, 15 matches, 1000 match-level bootstrap resamples).**

| Model | `dangerous_progression_k` AUC before -> after | `progression_beyond_pass_k` AUC before -> after |
|---|---|---|
| LR, 27 event features | 0.8026 -> 0.8026 | 0.6111 -> 0.6122 |
| XGBoost, event-only | 0.8807 -> 0.8810 | 0.7180 -> 0.7279 |
| XGBoost, event+360 | 0.8822 -> 0.8832 | 0.7229 -> 0.7294 |
| `end_x` only (LR) | 0.8003 -> 0.8003 | 0.5428 -> 0.5428 |
| `(end_x, end_y)` only (XGB) | 0.8565 -> 0.8565 | 0.6874 -> 0.6874 |

360 delta, all test passes: original label ROC AUC +0.0014 [-0.0011, +0.0038] -> +0.0021 [-0.0010, +0.0055], AP +0.0014 -> +0.0018 [-0.0006, +0.0043], Brier -0.0011 -> -0.0008 [-0.0025, +0.0008]; corrected label ROC AUC +0.0050 [-0.0011, +0.0106] -> **+0.0015 [-0.0036, +0.0070]**, AP +0.0129 [+0.0015, +0.0250] -> +0.0041 [-0.0053, +0.0134], Brier -0.0051 [-0.0100, -0.0006] -> -0.0038 [-0.0067, -0.0015].  On passes with 360 only (corrected label): +0.0029 -> -0.0015 [-0.0106, +0.0069].  Event-only XGBoost on the corrected label improves by 0.010 once the feature is correct; its lead over `end_x` alone is now +0.185 and over `(end_x, end_y)` +0.040.

**Answer to the key question (replaces EXP-025's).**  Does 360 geometry matter more once the label stops rewarding the pass's own destination?  **No.**  The 360 delta is about +0.002 ROC AUC on either label with intervals that include zero; only the Brier improvement on the corrected label excludes zero.  The first-round answer ("slightly, about 3.5x larger") is superseded.

### EXP-028 — Shot xG and goals parsed; possession table rebuilt (self-audit, second round)

Status: complete.  Code: `src/data/parse_events.py` (keeps `shot_statsbomb_xg`, `shot_outcome_name`), `scripts/reparse_events_cache.py` (re-parse from the cached raw files, no network; every existing column verified identical), `scripts/audit_shot_columns.py`; tests `tests/test_shot_columns.py`; output `reports/audit/shot_columns_after.md`.

**Defect.**  `parse_events` dropped the shot columns, so in `possession_sequences.parquet` `poss_xg_generated` was 0 for all 17,065 possessions, `poss_has_goal` was false for all, and `poss_outcome_tier` never reached 4.  `_build_xg_goal_maps` already read the right column names.

**After.**  2,333 shots, 306 goals (26 in penalty shootouts), total xG 295.4 (ratio 0.97 to goals; possession-table sum 295.1, the rest sits in a dropped one-event possession).  Goals per match 2.83 excluding shootouts (3.09 including): World Cup 2022 2.64 (xG 2.94), La Liga 2020/21 Barcelona 3.17 (xG 3.07).  305 possessions (1.79%) contain a goal; outcome tiers 0: 6,562, 1: 5,674, 2: 2,743, 3: 1,781, 4: 305.  The only columns that changed in the possession table are `poss_xg_generated`, `poss_has_goal` and `poss_outcome_tier`.  **No v2 model used them as features** (`src/features/possession_features.py` excludes them), so no v2 model needed re-running; the possession EDA notebook now shows them.

### EXP-029 — Possession EDA re-run; recycling controlled for length (self-audit, second round)

Status: complete.  `notebooks/09_possession_eda.ipynb` re-executed on the corrected, xG-populated table (28.3% dangerous, 61.5% final third, 30.6% recycled; the pre-fix printout said 32.8%, 75.4%, 52.0%) and no longer re-attaches labels over the stored xG/goal columns.  New section 5b stratifies recycled vs non-recycled by length.

| Events | n not recycled / recycled | danger, not recycled | danger, recycled | difference |
|---|---|---|---|---|
| 2-5 | 3,333 / 29 | 14.7% | 3.4% | -11.3 pp |
| 6-10 | 2,853 / 227 | 19.8% | 9.7% | -10.1 pp |
| 11-20 | 3,331 / 1,016 | 26.2% | 17.1% | -9.1 pp |
| 21-40 | 1,755 / 2,025 | 39.3% | 34.1% | -5.2 pp |
| 41+ | 570 / 1,926 | 49.3% | 53.9% | +4.6 pp |

Raw: recycled 36.9% vs 24.5% dangerous (+12.4 pp), mean 39.6 vs 14.0 events.  Length-adjusted (bins weighted by recycled count): **-2.6 pp**.  **Superseded claim:** "recycled 42% vs 23% dangerous" (31 vs 11.5 events), computed before the coordinate-frame fix and confounded by length.  Corrected reading: recycling does not raise danger once length is controlled; recycled possessions are less dangerous in every bin up to 40 events and only more dangerous at 41+.  The shortest bins hold few recycled possessions; descriptive, not causal.

### EXP-030 — Data description (self-audit, second round)

The processed data is 99 matches: 64 FIFA World Cup 2022 (all with 360; 30,349 passes, 26,515 with a frame) and 35 La Liga 2020/21 Barcelona (no 360 frames; 20,339 passes); no Premier League matches.  360 covers 26,515 of 50,688 passes (52.3%; train 18,234, val 4,545, test 3,736) and 64 of 99 matches (65%).  Superseded wording: "~65% of passes", "~33,000 360-available passes", "~37%" label prevalence (actual 43.3% for `dangerous_progression_k`).  The inventory marks all 35 La Liga matches `has_360 = true` although no frame files exist.  `configs/data.yaml` keeps competition 2 (removing it would change the inventory table) and now carries comments.

### EXP-031 — Cross-fitted retrospective attribution on the corrected data

Status: complete.  Code: `src/evaluation/attribution_crossfit.py`, `scripts/run_attribution.py`; tests `tests/test_attribution.py`; outputs `reports/attribution/` (report, leaderboards, event-type table, results JSON), `reports/figures/attribution/`, `data/processed/attribution_events.parquet` (310,726 events; gitignored).

**Method.**  The 99 matches are split into 5 match-grouped folds (seed 42).  For each fold a v2 GRU (`configs/model_gru.yaml`: 1 layer, hidden 64, lr 1e-3, class-weighted loss, early stopping on validation AUC) is trained on the other four folds, with an inner match-level split (12 validation matches) for early stopping; leave-one-out (LOO) attribution is computed **only** on the held-out fold, so every possession is attributed by a model that never saw it or any other possession of its match.  First 40 events only (the GRU's input length; 14.6% of possessions are longer).  A second independent round (seed 43: different folds, inner splits and initialisation) provides the cross-run reliability check.

**Results (retrospective).**

| Quantity | Value |
|---|---|
| Held-out ROC AUC per fold (seed 42) | 0.9558, 0.9569, 0.9543, 0.9572, 0.9584 |
| Pooled out-of-fold ROC AUC (seed 42 / seed 43) | **0.9525** / 0.9550 |
| Possession score vs possession xG (Spearman / Pearson) | 0.501 / 0.314 |
| Mean LOO, goal possessions vs dangerous non-goal (all events) | +0.0346 vs +0.0233 |
| Mean LOO, goal possessions vs dangerous non-goal (possessing-team events) | +0.0242 vs +0.0209 |
| Median Gini of per-possession |LOO| (all / dangerous / safe) | 0.478 / 0.629 / 0.434 |
| Players with >= 100 possessing-team events | 623 of 1,008 |
| Split-half stability of player ranking (50 repeats) | Spearman 0.620 +/- 0.018 (range 0.585-0.671) |
| Cross-run agreement (seed 42 vs 43): player ranking / event LOO / possession score | Spearman 0.887 / Pearson 0.868 / Spearman 0.971 |

**Reading.**  Possessions that end in a goal receive the highest scores and larger attributions than other dangerous possessions, and the score is moderately related to xG (the label does not use xG).  Player means are reasonably stable, but the top of the table is forwards and the bottom centre-backs and full-backs, so the stable part is mostly **role**; nothing is adjusted for position, team or opponent.  Defending-team events (goalkeeper, clearance, block, foul committed) are the model's evidence that danger occurred and are never credited to a player.  Very high scores saturate, so LOO drops vanish for the most obviously dangerous possessions.  This is retrospective, not causal credit; the recommended next step is a forward-looking action value model (xT or VAEP-style).

**Superseded:** the in-sample v2 leaderboards and figures (`figures/superseded_attribution/`), Gini 0.495 and the "most impactful events are final-third entries, through balls and pre-shot carries" statement.

### EXP-032 — PassFrameGNN retrained on the corrected label; fair comparison on the 360 subset

Status: complete.  Code: `src/features/pass_graph.py`, `src/models/pass_gnn.py`, `scripts/train_pass_gnn.py`; tests `tests/test_pass_gnn.py`; outputs `reports/pass_label/gnn_comparison.md` and `.json`, `models/pass_gnn_progression.pt` and `.json`; model card `reports/repro/model_cards/pass_gnn.md`.

**Investigation.**  The original code (`src/models/gnn.py`, `src/features/graph_builder.py`, `configs/model_gnn.yaml`, `notebooks/05_gnn.ipynb`) exists, but no checkpoint was ever committed and `torch_geometric` is not installed.  The original node feature `is_receiver` used the pass END location, and the original model was a graph + 27-feature event hybrid with 5 multitask heads.  The old validation figure in the notebook output and `v1_xgb_vs_gnn.png` is **0.834**, not the 0.841 quoted in the docs.

**Method.**  Graph-only model (one binary head), 8 node features (no `is_receiver`), kNN(5) + same-team edges, 3 x GraphSAGE (mean, 64 hidden, plain torch), BCE without class weighting.  Saved split; label from the same table as the XGBoost models; graphs: 18,234 / 4,545 / 3,736.  Validation-only grid (8 configurations), 5 seeds, reported model chosen by validation ROC AUC, test read once.

**Results (test, 360 subset: 3,736 passes, 9 matches; match-level bootstrap, 1,000 resamples).**

| Model | ROC AUC | AP | Brier |
|---|---|---|---|
| PassFrameGNN (graph-only) | 0.595 [0.578, 0.606] | 0.284 [0.262, 0.325] | 0.167 [0.157, 0.179] |
| XGBoost event+360 (EXP-027 recipe) | 0.715 [0.697, 0.734] | 0.403 [0.381, 0.429] | 0.193 [0.186, 0.201] |
| XGBoost event-only (EXP-027 recipe) | 0.717 [0.700, 0.733] | 0.400 [0.370, 0.436] | 0.199 [0.194, 0.206] |
| `end_x`-only logistic regression | 0.530 [0.502, 0.551] | 0.251 [0.227, 0.281] | 0.170 [0.159, 0.183] |

GNN - XGBoost event+360: -0.1209 [-0.1379, -0.1033] ROC AUC.  GNN seeds: test ROC AUC 0.5950 (sd 0.0031).

**Answer.**  Player positions alone (graph-only, no pass destination) reach ROC AUC 0.595 [0.578, 0.606] against 0.530 [0.502, 0.551] for `end_x` alone; event features dominate (final-recipe XGBoost event+360 on the same 360 subset: 0.728 [0.709, 0.748]).  This is not a like-for-like contest between feature sets.  The GNN scores below the event-based models, but that is expected rather than a contest result: it sees neither event attributes nor the pass destination.  **Supersedes** the old "near-parity, 0.841 vs 0.845" (original label only, superseded, validation subset, unreproducible: the saved output says 0.834).  No existing result was changed.

### EXP-033 — Final pass models: pre-specified recipe; the 360 comparison re-run

Status: complete.  Code: `src/models/pass_progression.py`, `scripts/train_pass_models.py`; tests `tests/test_pass_progression.py`; outputs `reports/pass_label/pass_model_final.md` and `.json`, `models/pass_progression_event_only.*` and `models/pass_progression_event_360.*`; model card `reports/repro/model_cards/pass_xgboost.md`.

**Decision.**  The reproduction gate (`reports/pass_label/pass_model_repro_gate.md`) showed that the pre-specified recipe (early stopping on validation, no `scale_pos_weight`, no calibrator unless validation reliability needs one) gives test ROC AUC 0.7337 / 0.7380 rather than EXP-027's 0.7279 / 0.7294.  EXP-027 used `TabularClassifier`, which ignores `early_stopping_rounds` (all 500 trees) and applies `scale_pos_weight` (mean prediction 0.41 vs prevalence 0.224).  The pre-specified recipe was adopted; EXP-027's recipe is **superseded**.  Validation reliability needed no calibrator.

**Results (test; 1,000 match-level bootstrap resamples).**

| Subset | Model | ROC AUC | AP | Brier | Log loss |
|---|---|---|---|---|---|
| All test passes (7,344) | Event-only (129 trees) | 0.7337 [0.7185, 0.7490] | 0.4283 [0.4049, 0.4525] | 0.1532 [0.1465, 0.1600] | 0.4719 [0.4557, 0.4882] |
| | Event+360 (163 trees) | 0.7380 [0.7239, 0.7519] | 0.4326 [0.4114, 0.4579] | 0.1525 [0.1457, 0.1591] | 0.4694 [0.4533, 0.4853] |
| | **Paired difference** | **+0.0043 [+0.0015, +0.0076]** | +0.0043 [-0.0045, +0.0144] | -0.0007 [-0.0015, -0.0000] | **-0.0025 [-0.0041, -0.0010]** |
| 360-only subset (3,736) | Event-only | 0.7195 [0.6983, 0.7389] | 0.4006 [0.3697, 0.4357] | 0.1525 [0.1419, 0.1646] | 0.4710 [0.4454, 0.5000] |
| | Event+360 | 0.7281 [0.7091, 0.7480] | 0.4212 [0.3886, 0.4597] | 0.1508 [0.1402, 0.1629] | 0.4664 [0.4404, 0.4945] |
| | **Paired difference** | **+0.0087 [+0.0029, +0.0147]** | **+0.0206 [+0.0086, +0.0326]** | **-0.0018 [-0.0031, -0.0004]** | **-0.0047 [-0.0077, -0.0013]** |

**Conclusion on the 360 gain.**  With the final recipe the 360 gain is **small but measurable**.  Event+360 minus event-only (paired match-level bootstrap, 1,000 resamples): on all 7,344 test passes ROC AUC +0.0043 [+0.0015, +0.0076], log loss -0.0025 [-0.0041, -0.0010], Brier -0.0007 [-0.0015, -0.0000] (upper bound just below zero), average precision +0.0043 [-0.0045, +0.0144] (includes zero); on the 3,736-pass 360-only subset all four exclude zero: ROC AUC +0.0087 [+0.0029, +0.0147], AP +0.0206 [+0.0086, +0.0326], Brier -0.0018 [-0.0031, -0.0004], log loss -0.0047 [-0.0077, -0.0013].  The earlier reading "360 adds nothing measurable" (EXP-027 recipe: ROC AUC +0.0015 [-0.0036, +0.0070], AP +0.0041 [-0.0053, +0.0134]) is **superseded**; the effect is still only about +0.004 ROC AUC over all passes (+0.009 where a freeze frame exists).  Calibration: slope 1.09 / 1.08, mean prediction 0.216 vs prevalence 0.224.

**Superseded:** "360 adds nothing measurable" (EXP-025 / EXP-027), the 0.728 / 0.729 headline as the final pass model, and the `scale_pos_weight` probabilities.  The old-label models moved to `models/superseded/`.

---

## V2 results summary table

> **The 0.9505 / 0.9524 / 0.9650 rows below are superseded** (kept as originally reported).  Reasons: the possession labels, `max_x_reached` and the final-third flag used the defending team's events in the wrong coordinate frame (EXP-023), the saved XGBoost artefact no longer reproduces 0.9505 under the current XGBoost version (it scores 0.9449 on the original labels), and the recorded 0.9650 ensemble does not reproduce (0.9732 with the documented logit average).  The corrected like-for-like retrain (same split and configuration, test possessions) is in the second block.  All are **retrospective** (they see the whole possession).

| EXP | Model | Task | Feature set | ROC-AUC (test) | AP (test) |
|-----|-------|------|-------------|----------------|-----------|
| 009 | XGBoost | poss_dangerous | 41 possession features | ~~0.9505~~ superseded | ~~0.8947~~ |
| 010 | PossessionGRU | poss_dangerous | 8-dim event sequence | ~~0.9524~~ superseded | ~~0.9282~~ |
| 011 | Ensemble (XGB+GRU) | poss_dangerous | combined | ~~0.9650~~ superseded | ~~0.9358~~ |
| 012 | XGBoost (origin-only) | poss_dangerous | origin_type | 0.591 (original labels; not re-run) | — |
| 013 | LOO Attribution | Gini concentration | — | ~~Gini=0.495~~ superseded: in-sample, pre-fix labels; see EXP-031 | — |
| 014 | Cross-level correlation | pass↔possession | — | H1 supported (original labels and old pass model; not re-run) | — |

Corrected retrain on the frame-corrected labels (EXP-023), test possessions:

| EXP | Model | Task | Original labels (retrain) ROC-AUC / AP | Corrected labels ROC-AUC / AP |
|-----|-------|------|-----------------|-----------------|
| 009 | XGBoost | poss_dangerous | 0.9461 / 0.8823 | **0.9572 / 0.8920** |
| 010 | PossessionGRU | poss_dangerous | 0.9534 / 0.9303 | **0.9570 / 0.9277** |
| 011 | Ensemble (equal-weight logit mean) | poss_dangerous | 0.9736 / 0.9534 | **0.9769 / 0.9526** |
| 031 | PossessionGRU, 5-fold match-grouped cross-fit (all 99 matches, pooled out-of-fold; retrospective) | poss_dangerous | — | **0.9525** (second seed 0.9550) |

---

## V3 results summary table

> **SUPERSEDED (fraction-based formulation).**  Every row below used "the first f% of events", whose prefix length encodes the possession length (AUC 0.683 alone) and often already contains the outcome (EXP-021).  The cumulative XGBoost models were also trained with feature code that mis-counted passes and carries (EXP-020), and EXP-019's "Dribble" is Ball Receipt.  Current in-play numbers: EXP-021 / EXP-026 (`reports/event_time/`).  Kept as originally reported.

| EXP | Model | Observation window | ROC-AUC (test) | AP (test) | Notes |
|-----|-------|--------------------|----------------|-----------|-------|
| 015 | XGBoost start-only | 0% (start context) | 0.6241 | 0.5346 | 19 start-only features |
| 016 | GRU prefix | 25% | 0.7167 | 0.6761 | Existing v2 GRU scored on prefixes |
| 016 | GRU prefix | 50% | 0.8195 | 0.7732 | |
| 016 | GRU prefix | 75% | 0.8793 | 0.8468 | |
| 016 | GRU prefix | 100% | **0.9475** | **0.9283** | |
| 017 | XGBoost cumulative | 25% | 0.8136 | 0.7127 | Features rebuilt from prefix |
| 017 | XGBoost cumulative | 50% | 0.8472 | 0.7624 | |
| 017 | XGBoost cumulative | 75% | 0.8912 | 0.8200 | |
| 017 | XGBoost cumulative | 100% | **0.9517** | **0.8978** | |
| 019 | GRU tipping-point | threshold = 0.50 | — | — | 47.5% cross; 92.0% dangerous cross; median frac 0.619; top type is Ball Receipt, not Dribble (EXP-020) |
| 015-017, 019 | v3 fraction-based rows above | | | | **SUPERSEDED** (length leak + realised outcome) by EXP-021 |

---

## V2 hypothesis verdicts

> **STALE, SUPERSEDED IN PART (audit of 2026-10-30).**  Items quoting the 453-player leaderboard, Gini 0.495, "final-third entries, through balls and pre-shot carries", the 0.841-vs-0.845 GNN near-parity, or any v3 fraction-based number are withdrawn.  Current answers: README "Research questions and answers", EXP-031 (attribution) and EXP-032/033 (GNN, pass model).  Kept as historical record.


| Hypothesis | Statement | Verdict | Key evidence |
|-----------|-----------|---------|--------------|
| H1 | Possession-level danger correlates with pass-level DP | **Supported** | Team-level rank correlation confirmed |
| H2 | Danger is concentrated in a few key events | **Partial** | Median Gini 0.495, max share 33.2% |
| H3 | Sequence modelling outperforms snapshot features | **Inconclusive** | GRU 0.9524 vs XGB 0.9505 (+0.002) — marginal |
| H4 | Possession origin alone predicts danger | **Rejected** | Origin-only AUC 0.591 vs full 0.950+ |

---

## V2 research question answers

> **STALE, SUPERSEDED IN PART (audit of 2026-10-30).**  Items quoting the 453-player leaderboard, Gini 0.495, "final-third entries, through balls and pre-shot carries", the 0.841-vs-0.845 GNN near-parity, or any v3 fraction-based number are withdrawn.  Current answers: README "Research questions and answers", EXP-031 (attribution) and EXP-032/033 (GNN, pass model).  Kept as historical record.


**RQ3 — How should possession-level danger be defined and predicted?**
We define `poss_dangerous = poss_has_shot OR poss_entered_box`.  XGBoost on 41 aggregated features achieves 0.950 ROC AUC; PossessionGRU on raw event sequences achieves 0.952.  The ensemble reaches 0.965.  Possession-level prediction is highly effective because the full sequence of events within a possession provides stronger signal than any single pass.

**RQ4 — Which events matter most in generating danger?**
LOO attribution shows moderate concentration (Gini 0.495): typically 2–3 events account for ~50% of a possession's danger score.  ~~The most impactful events are final-third entries, through balls, and pre-shot carries.~~  Corrected (EXP-020): by mean leave-one-out attribution the most impactful event types are Shot, Goalkeeper, Clearance, Foul Committed and Block; carries and passes are slightly negative on average.

**RQ5 — Can player-level attribution identify context-adjusted dangerous contributors?**
Yes.  Using LOO attribution per event combined with the `player_sequence` column, we compute per-player danger contribution.  The leaderboard of 453 players (≥ 20 touches) reveals that goalkeepers and deep defenders consistently rank low, while attacking midfielders and strikers rank highest — consistent with football domain knowledge.

---

## Combined research question answers (v1 + v2)

> **STALE, SUPERSEDED IN PART (audit of 2026-10-30).**  Items quoting the 453-player leaderboard, Gini 0.495, "final-third entries, through balls and pre-shot carries", the 0.841-vs-0.845 GNN near-parity, or any v3 fraction-based number are withdrawn.  Current answers: README "Research questions and answers", EXP-031 (attribution) and EXP-032/033 (GNN, pass model).  Kept as historical record.


1. **Does 360 positional context improve prediction over event-only baselines?**
   Marginally.  Event+360 XGBoost improves ROC AUC by +0.0014 [-0.0011, +0.0038] over event-only (0.882 vs 0.881) on the original label and +0.0050 [-0.0011, +0.0106] on the corrected `progression_beyond_pass_k` (EXP-025).  The 360 freeze-frame geometry features add a small signal whose ROC AUC interval includes zero.  Event features — particularly `goal_dist_gain`, `end_x`, `x_gain` — already capture most of the predictive information.

2. **Which geometric properties most strongly relate to dangerous progression?**
   Based on XGBoost feature importance with geometry features added: `n_defenders_goal_side`, `pass_corridor_clear`, `receiver_between_lines`, and `overload_target_zone` rank highest among the 14 geometry columns.  They capture the most structurally meaningful aspects of the defensive shape.

3. **Can graph-based modelling outperform strong tabular baselines?**
   Near-parity, not outperformance, on the original self-determining label (unverified: not re-run, EXP-025).  PassFrameGNN achieves 0.841 val AUC vs XGBoost 0.845 on the same 360-available subset.  The GNN learns equivalent spatial representations from the graph structure, reaching the same ceiling as tabular geometry features without hand-crafted aggregations.  On this dataset size, the GNN does not pull ahead.

4. **Does multitask learning help?**
   Inconclusive given project scope.  The GNN trained with 5 multitask heads converged stably and produced reasonable performance on the primary task (0.841 val AUC for dangerous_progression_k).  A head-to-head comparison against single-task GNN requires additional runs and is deferred to future work.

5. **Which players/teams generate the most context-adjusted dangerous progression?**
   Full profiling is available via `src/evaluation/tactical_review.py :: player_progression_profile()`.  In the test-set freeze-frame visualisation (NB06), the model correctly identifies passes where the passer's actual choice was the highest-ranked option among visible teammates, suggesting the system can distinguish context-adjusted dangerous from safe options.

6. **Can the system rank visible passing options tactically?**
   Yes.  `PassOptionRanker` (NB06, section 8) successfully ranked 8 visible teammates for a sample pass.  The actual choice scored highest at 0.229 predicted probability — the model agreed the chosen pass was the most dangerous option available.  This is the core interactive capability exposed by the Streamlit app.

7. **How should possession-level danger be defined and predicted? (RQ3)**
   `poss_dangerous = poss_has_shot OR poss_entered_box`.  Both XGBoost (0.950 AUC) and GRU (0.952 AUC) perform excellently as retrospective models (corrected labels, EXP-023: 0.957 and 0.957; logit-mean ensemble 0.977).  In play the problem is much harder: event-time ROC AUC 0.67-0.76 (EXP-021).

8. **Which events matter most within a possession? (RQ4)**
   LOO attribution reveals moderate concentration (Gini 0.495).  On average, 2–3 pivotal events account for ~50% of a possession's danger.  Final-third entries, through balls, and pre-shot carries are the highest-impact actions.

9. **Can player-level attribution identify dangerous contributors? (RQ5)**
   Yes.  LOO attribution per event, mapped to players via `player_sequence`, produces a 453-player leaderboard.  Rankings are domain-consistent: attacking players rank high, defenders and goalkeepers rank low.

---

## Configuration notes

All experiment configs are version-controlled in `configs/`.  Each training run should save:
- `configs/` snapshot
- trained model artifact (`.pkl` or `.pt`)
- `metrics.json`
- `predictions.parquet`
- `feature_schema.json`
- `split_manifest.csv`
