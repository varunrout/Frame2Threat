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

*(To be filled after running experiments. See `reports/experiment_log.md` for live results.)*

### 8.1 Primary metrics (target ranges from literature)

| Model | Task | Expected ROC-AUC |
|-------|------|-----------------|
| Rule-based | line_break | ~0.60–0.65 |
| LogReg event-only | line_break | ~0.65–0.72 |
| XGBoost event-only | line_break | ~0.70–0.78 |
| XGBoost event+360 | line_break | ~0.72–0.82 |
| GNN | line_break | ~0.73–0.83 |

### 8.2 Research question answers

*(To be filled after experiments.)*

---

## 9. Tactical Interpretation

*(To be updated after manual error analysis.)*

### 9.1 Expected high-score pass characteristics
- Forward passes from mid-third into congested defensive block
- Through balls to receivers between lines
- Switches that stretch the defensive shape before a penetrating pass

### 9.2 Expected failure modes
- Set-piece-like situations where geometry is unusual
- Passes to players technically "between lines" spatially but in unproductive areas (e.g. wide channels with no subsequent threat)
- Low-visibility frames where few defenders are captured

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
- [ ] Model artifacts saved with full metadata (after first run)
- [ ] Predictions saved per experiment (after first run)

---

*Report template generated by Frame2Threat automated pipeline.  Numerical results to be populated after running `src/` pipeline end-to-end.*
