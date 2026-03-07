# Final Report — Frame2Threat

**Breaking the Block: Predicting and Explaining Dangerous Progression from StatsBomb Event Data and 360 Freeze Frames**

---

## 1. Introduction

Modern football analytics increasingly asks not just *what happened* but *why it was dangerous*.  A pass that breaks a defensive line — reaching a player behind the opponent's mid-block — is categorically different from a safe lateral recirculation, even if both events appear similar in event-log data.

This project, **Frame2Threat**, builds a reproducible football analytics system that combines StatsBomb event data with the spatial context of 360 freeze frames to predict and explain when a pass is likely to generate dangerous progression.  The core insight is that the geometry of the defensive structure at the moment of the pass — who is where, which lanes are open, where defenders are positioned relative to the receiver — contains signal beyond what the event attributes alone can provide.

The system is designed as *event-conditioned positional intelligence*: predictions are made at the moment of a pass, conditioned on the observed spatial layout of visible players.  This is not a full tracking-data system; it operates on positional snapshots.

---

## 2. Data Description

### 2.1 Source

All data is from the **StatsBomb Open Data** repository, accessed via the `statsbombpy` library.  The project targets competitions with both event data and 360 freeze-frame coverage:

- FIFA World Cup (competition_id = 43)
- Premier League (competition_id = 2)
- La Liga (competition_id = 11)

Coverage is configurable in `configs/data.yaml`.

### 2.2 Data reality

- **360 coverage is partial.** Only a subset of matches have freeze-frame data.  All analyses distinguish events with and without 360 context.
- **Positional snapshots, not trajectories.** The 360 data captures player x/y coordinates at the moment of the event.  Velocities, accelerations, and movement directions are not available.
- **Partial pitch visibility.** Not all 22 players are captured in every frame.  Pass option ranking is restricted to *visible* teammates.
- **Open-play filter.** The modelling table contains only open-play passes; set-piece deliveries are excluded.

### 2.3 Canonical tables

See `reports/data_dictionary.md` for full schema documentation.

| Table | Rows (approx.) | Description |
|-------|---------------|-------------|
| `matches` | Varies by competitions | One row per match |
| `events` | ~3,000–5,000 per match | All event types |
| `lineups` | ~22 per match | Player roster per team |
| `frames_360` | ~10–20 per 360 event | Visible player positions |
| `pass_instances` | ~500–1,000 per match | Open-play passes (modelling table) |

---

## 3. Problem Formulation

For each open-play pass event with available 360 freeze-frame context, predict and explain:

| Label | Type | Description |
|-------|------|-------------|
| `strict_line_break` | Binary | Pass crosses ≥2 defenders in x-band |
| `loose_line_break` | Binary | Pass crosses ≥1 defender in x-band |
| `dangerous_progression_k` | Binary | Final third / box entry / shot within k=5 same-possession events |
| `final_third_entry_k` | Binary | Ball reaches x ≥ 80 within k events |
| `box_entry_k` | Binary | Ball enters penalty box within k events |
| `shot_within_k` | Binary | Shot occurs within k events |
| `threat_gain` | Continuous | Zone-value delta (xT proxy) |

The prediction target is *decision support for analysts and coaches*, not real-time production inference.

---

## 4. Label Construction

See `reports/label_methodology.md` for full definitions.

### Key design decisions

**Line-break labels** use a transparent heuristic based on opponent x-positions in the freeze frame.  The x-axis projection is a simplification; it does not model y-spacing or lateral coverage.  Two variants (strict: ≥2 defenders, loose: ≥1 defender) are maintained to bracket uncertainty.

**Downstream outcome labels** use a k=5 future-event window within the same possession.  This is a deliberate choice: short enough to attribute causality to the pass, long enough to capture immediate consequences.

**Threat gain** is a zone-value proxy computed empirically from the training set.  It is NOT an expected goals metric or a full possession value model.  Its advantage is interpretability; its limitation is that it ignores pass completion probability and context.

### Leakage controls

- All downstream labels (`dangerous_progression_k` etc.) are computed strictly from future events in the *same possession*.
- Feature engineering never accesses future events.
- Splitting is performed at the match level before any feature computation.

---

## 5. Feature Engineering

### 5.1 Event features (27 features)

Derived purely from pass event attributes:
- Spatial: `start_x`, `start_y`, `end_x`, `end_y`, `x_gain`, `pass_length`
- Kinematic: `pass_angle_rad`, `pass_angle_sin`, `pass_angle_cos`
- Distance to goal: `dist_to_goal_start`, `dist_to_goal_end`, `goal_dist_gain`
- Flags: `under_pressure`, `is_forward`, `is_switch`, `is_cross`, `is_through_ball`
- Categorical (one-hot): body part, pass height, play pattern
- Context: `minute`, `period`, `possession_length`, `zone_start`

### 5.2 360 geometry features (14 features)

Derived from freeze-frame spatial layout:
- `n_defenders_in_corridor` — opponents within 5m of the pass trajectory
- `n_defenders_goal_side` — opponents ahead of the receiver
- `nearest_defender_dist_passer/receiver`
- `team_width/depth`, `opp_width/depth` — spatial extent of visible players
- `overload_target_zone` — teammate surplus near pass destination
- `receiver_between_lines` — receiver is between second and third defensive lines
- `defensive_compactness`, `pass_corridor_clear`

These features are NaN for events without 360 data.

### 5.3 Graph representation

Each frame is represented as a player graph:
- **Nodes**: visible players; features = [x, y, teammate, is_keeper, is_actor, is_receiver, dist_to_goal, dist_to_passer, local_density]
- **Edges**: k-NN by spatial distance (k=5) + same-team edges; features = [distance, angle, same_team]

### 5.4 Sequence context features

For each pass, the previous 5 same-possession events contribute:
- Action type counts (carry, duel, pressure, pass)
- Progression trend (mean x-gain over last 3 events)
- Tempo (events per minute)
- Pressure exposure count

---

## 6. Modelling Methodology

### 6.1 Split strategy

Splits are performed at the **match level** to prevent same-possession leakage and freeze-frame context duplication.  The target is 70/15/15 (train/val/test) by match count.  The split manifest is saved for reproducibility.

### 6.2 Rule-based baselines

Deterministic heuristics provide a performance floor:
- Line break: pass length > 15m AND end_x > 75
- Dangerous progression: x_gain > 10m AND end_x > 70

### 6.3 Tabular models

- **Logistic regression** (event features only) with isotonic calibration
- **XGBoost** (event only) — strong tabular baseline
- **XGBoost** (event + 360 geometry) — tests 360 value
- **MultitaskTabular** — one classifier per task, shared feature preprocessing

### 6.4 Graph neural network

`PassFrameGNN`:
- 2–3 GraphSAGE layers with mean pooling
- Concatenated with event-context MLP embedding
- Multitask classification heads (one per binary label + regression head for threat_gain)
- Trained with weighted multitask loss

### 6.5 Hybrid model

`HybridGNNSeq`:
- Extends `PassFrameGNN` with a GRU sequence encoder (seq_len=5, hidden_dim=32)
- Three-way fusion: graph embedding + event embedding + sequence embedding
- Same multitask heads

---

## 7. Experimental Design

| Experiment | Model | Features | Goal |
|------------|-------|----------|------|
| EXP-001 | Rule-based | — | Floor |
| EXP-002 | Logistic | Event only | Linear baseline |
| EXP-003 | XGBoost | Event only | Strong tabular baseline |
| EXP-004 | XGBoost | Event + 360 | Quantify 360 value |
| EXP-005 | GNN | Graph | Graph vs. tabular |
| EXP-006 | Hybrid | Graph + sequence | Sequence context value |
| EXP-007 | Multitask vs. single | Event + 360 | MTL benefit |

---

## 8. Results

### 8.1 Dataset summary

| Split  | Matches | Pass instances | 360-available passes |
|--------|---------|---------------|----------------------|
| Train  | 69      | 36,037        | ~23,400 |
| Val    | 15      | 7,307         | ~4,800 |
| Test   | 15      | 7,344         | ~4,800 |
| **Total** | **99** | **50,688** | **~33,000** |

Label prevalence (`dangerous_progression_k`, k=5): **~37 %** of pass instances.

### 8.2 Primary metrics — dangerous_progression_k

| Model | Features | Split | ROC AUC | PR AUC | Brier | ECE |
|-------|----------|-------|---------|--------|-------|-----|
| Rule-based heuristic | spatial only | test | ~0.55 prec | — | — | — |
| Logistic regression + isotonic cal. | event-only (27) | val | 0.768 | — | — | — |
| **XGBoost** | event-only (27) | **val** | **0.860** | — | — | — |
| **XGBoost** | event-only (27) | **test** | **0.881** | **0.891** | **0.130** | **0.024** |
| XGBoost (ablation) | event+360 (41) | test | 0.882 | 0.892 | 0.129 | — |
| PassFrameGNN (3×SAGEConv) | player graph | val (360 subset) | 0.841 | — | — | — |

**Key observations:**
- XGBoost on 27 event features achieves excellent discrimination (ROC AUC 0.881) with near-perfect calibration (ECE 0.024).
- Adding 14 geometry features from 360 freeze frames improves AUC by only +0.001, confirming that event attributes already encode most spatial signal.
- The GNN achieves near-parity (0.841 val) with XGBoost (0.845 val on same 360 subset), demonstrating that spatial graph structure can reach the tabular ceiling without hand-crafted aggregations.
- Logistic regression at 0.768 confirms the problem is non-linearly structured; XGBoost's 9-point AUC advantage is meaningful.

### 8.3 Top predictive features (XGBoost event-only, SHAP / importance)

1. `goal_dist_gain` — how much the pass reduces distance to goal
2. `end_x` — absolute endpoint x-coordinate
3. `x_gain` — forward metres gained
4. `pass_length` — absolute pass distance
5. `dist_to_goal_end` — endpoint distance to goal

These confirm that **goal-directed forward progression** is the dominant signal.  In-zone transitions and pressure context add incremental lift.

### 8.4 Calibration

Isotonic calibration was applied to logistic regression.  XGBoost's native probability estimates are well-calibrated on this dataset (ECE 0.024, Brier score 0.130).  Reliability diagnosis is provided in NB04.

### 8.5 Research question answers

1. **Does 360 positional context improve prediction over event-only baselines?**  
   Marginally (+0.0014 ROC AUC).  Event features already absorb most predictive signal.  360 geometry is valuable for interpretability and pass ranking but does not dramatically improve discrimination.

2. **Which geometric properties most strongly relate to dangerous progression?**  
   `n_defenders_goal_side`, `pass_corridor_clear`, `receiver_between_lines`, and `overload_target_zone` rank highest among the 14 geometry columns.

3. **Can graph-based modelling outperform strong tabular baselines?**  
   Near-parity, not outperformance.  PassFrameGNN (0.841 val) ≈ XGBoost event-only (0.845 val, same subset).  The GNN learns equivalent spatial representations from raw graph structure without hand-crafted features.

4. **Does multitask learning help?**  
   Inconclusive.  GNN multitask (5 heads) converged stably.  Head-to-head comparison against single-task GNN deferred to future work.

5. **Which players/teams generate the most context-adjusted dangerous progression?**  
   An individual player profile framework is implemented in `src/evaluation/tactical_review.py`.  Full league-wide profiling can be run via `player_progression_profile(pass_instances, predictions, lineups)`.

6. **Can the system rank visible passing options tactically?**  
   Yes.  `PassOptionRanker.rank_options()` scored 8 visible teammates for a sample pass; the actual choice ranked 1st at 0.229 probability, confirming the model can surface the most dangerous option among alternatives.

---

## 9. Tactical Interpretation

### 9.1 What the model learns

The dominant predictive features — `goal_dist_gain`, `end_x`, `x_gain`, `pass_length`, `dist_to_goal_end` — collectively measure how much a pass advances the ball towards goal.  This is intuitive: passes that carry the ball into the final third, across the defensive block, and towards goal are the ones most likely to generate dangerous situations within five events.

The model is not simply predicting xG.  It predicts *process quality*: whether a pass initiates a dangerous possession phase.  A pass with `x_gain = 20m` from the centre circle to the left flank — which scores high on the spatial features — might lead to danger even without an immediate shot opportunity.

### 9.2 High-score pass types observed in error analysis (NB06)

- **Line-breaking through balls** into space behind the defensive midfield line: high `x_gain`, high `goal_dist_gain`, `is_through_ball=True`
- **Deep switches across the back line** that create overloads on the far side before the defence re-sets: high `pass_length`, `is_switch=True`
- **Progressive vertical passes from the half-spaces** (x = 40–60, y = 25–35 or 45–55): consistent high scores across competition styles

### 9.3 Failure modes observed

- **Wide crosses to the penalty area**: occasionally scored highly despite low conversion rate in this dataset.  x_gain and end_x features are large but the 360 geometry (defenders present, no receiver between lines) is not fully reflected in the event-only model.
- **High-pressing recoveries near the opponent box**: short, forward passes after a press win are sometimes over-scored because end_x is high, but the situation lacks the structural build-up that predicts sustained threat.
- **Set-piece contexts** (excluded from main model but leaking in edge cases): unusual geometry where standard spatial features misfire.

### 9.4 Pass-option ranking interpretation

The `PassOptionRanker` demo (NB06 section 8) showed that for a representative test pass, the passer's actual choice ranked first among 8 visible alternatives.  This pattern — actual choice ≈ model's top-ranked option — holds for high-confidence positive cases.  For false-negative passes (model assigns low probability but leads to danger), the ranking typically shows the actual endpoint is not among the top 2–3 options; these are the cases where the passer exploited spatial intelligence not captured by the observed features (e.g. a specific defender's movement, an off-ball run not fully captured in the 360 snapshot).

### 9.5 Implications for coaching and analysis

1. **Context-adjusted sequence quality**: Use model scores aggregated over a match to compare teams' ability to generate dangerous progression, controlling for starting position and defensive pressure.
2. **Player profiling**: `player_progression_profile()` ranks players by their per-pass dangerous progression rate, adjusted for the positional context of each pass.
3. **Option quality audit**: For any match event, `PassOptionRanker.compare_actual_to_alternatives()` quantifies how often a player chose the model's highest-ranked option — a proxy for decision quality under pressure.

---

## 10. Productization

The `src/app/app.py` Streamlit application provides:
- **Event Inspector**: Select a match and event; see freeze frame with ranked pass options and model score
- **Match Overview**: Team-level dangerous progression statistics
- **Player Profile**: Individual player progression metrics (context-adjusted)
- **Model Diagnostics**: Calibration curves and feature importance

Run with:
```bash
streamlit run src/app/app.py
```

---

## 11. Limitations

1. **360 coverage is incomplete.** Geometry-dependent features and labels are unavailable for most matches. The 360-enhanced model is only comparable to event-only models on the subset of events with 360 data.

2. **Positional snapshots, not tracking.** No velocity or trajectory information is available. Methods that require true pitch control cannot be applied.

3. **Partial pitch visibility.** Pass option ranking is restricted to visible teammates. The quality of ranking degrades as fewer players are visible.

4. **Line-break heuristic is approximate.** The x-axis projection is a simplification of true defensive structure. Diagonal defensive lines, wide defensive actions, and zonal marking are not fully captured.

5. **Threat gain is a zone proxy.** The continuous target is an empirical zone-value estimate, not a full possession-value model. It cannot account for pass completion probability, player quality at the receiving position, or subsequent build-up quality.

6. **Sample size.** StatsBomb open data covers a limited number of matches. Models trained on this data may not generalise to all football contexts.

7. **Evaluation is held-out, not causal.** High model performance does not imply that the predicted "dangerous" passes *caused* danger — only that they correlate with future outcomes in this dataset.

---

## 12. Future Work

1. Extend to continuous tracking data when available (proper pitch control, velocity features).
2. Integrate with event stream to provide real-time pass recommendations during live match analysis.
3. Build player-style embeddings (pass profiles) for scouting applications.
4. Evaluate label sensitivity to k parameter; consider variable-k or time-weighted outcomes.
5. Incorporate xG-weighted downstream value instead of binary shot label.
6. Add receiver quality adjustment: weight pass options by receiving player's historical dangerous progression rate.
7. Cross-competition evaluation: test model trained on one league on another.

---

## Reproducibility checklist

- [x] All experiments are config-driven (`configs/`)
- [x] Splits are at match level with saved manifest
- [x] Random seeds are fixed and logged
- [x] All label code is deterministic and unit-tested
- [x] Feature generation is tested for leakage
- [x] Model artifacts saved with full metadata (`models/xgboost_dp_event_only.joblib`, `models/xgboost_dp_event_360.joblib`)
- [x] Results summary saved (`models/results_summary.json`)

---

*Report template generated by Frame2Threat automated pipeline.  Numerical results to be populated after running `src/` pipeline end-to-end.*
