# Label Methodology — Frame2Threat

This document provides the precise operational definitions for every label in the Frame2Threat modelling dataset.  Each label was designed to answer a specific football analytics question, to be reproducibly computable from StatsBomb data, and to avoid data leakage.

---

## Analytical units

### v1 — Pass level

**One open-play pass event with available StatsBomb 360 freeze-frame context.**

All v1 labels are computed per pass.  Set-piece deliveries (corners, free kicks, goal kicks, throw-ins) are excluded.

### v2 — Possession level

**One continuous possession sequence** (same `match_id` + `possession_id`), parsed by `src/data/parse_possessions.py :: build_possession_sequences()`.

All v2 labels are computed per possession.  Each row aggregates every event in that possession into a fixed-length structural summary *plus* a variable-length `event_sequence` used by sequence models.  Set-piece deliveries can start a possession but no special exclusion is applied — all possessions are labelled.

---

## Label A — `strict_line_break` / `loose_line_break`

### Question
Did this pass break through a defensive layer?

### Definition

A pass is labelled as a **line break** if ALL of the following hold:

1. **Open play** — `pass_type` is NULL or not a set piece (configurable: `open_play_only=True`).
2. **Meaningful forward progress** — `end_x − start_x ≥ min_forward_gain_m` (default: 5.0 m).
3. **Defenders between passer and receiver** — at least N opponent players (from the 360 freeze frame) have x-coordinates strictly between `start_x` and `end_x`.

| Variant | N opponents required | Config key |
|---------|----------------------|------------|
| `strict_line_break` | ≥ 2 | `strict_threshold: 2` |
| `loose_line_break` | ≥ 1 | `loose_threshold: 1` |

### Implementation
`src/labels/line_break.py :: compute_line_break_labels()`

### Data requirements
Requires 360 freeze-frame data.  Rows without 360 context receive **NaN** (not False).

### Invariant
By construction: `strict_line_break ≤ loose_line_break` (every strict positive is also a loose positive).

### Limitations
- The heuristic uses the x-axis projection of defender positions.  It does not account for y-axis spacing (e.g. a wide pass that passes around rather than through a central block may be mislabelled).
- Defenders not visible in the freeze frame are not counted, which can cause false negatives.
- No temporal component: the definition is snapshot-based at the moment of the pass.

### Borderline cases
- A pass that clips one edge of a defensive line but reaches a player only marginally ahead: labelled positive if the 1/2 defender criterion is met.
- A lateral pass to a player behind the defensive line: labelled negative (x-gain < threshold).

---

## Label B — `dangerous_progression_k`

### Question
Did this pass lead to a dangerous attacking outcome within the next *k* events of the same possession?

### Definition
Binary (True/False, never NaN).  True if **any** of the following occur within the next *k* events of the **same possession** (same `match_id` AND `possession_id`):

- Ball reaches the **final third** (x ≥ 80.0): → `final_third_entry_k = True`
- Ball enters the **penalty box** (x ≥ 102.0, 18 ≤ y ≤ 62): → `box_entry_k = True`
- A **shot** event occurs: → `shot_within_k = True`

`dangerous_progression_k = final_third_entry_k OR box_entry_k OR shot_within_k`

Default k = 5 (configurable in `configs/labels.yaml`).

### Implementation
`src/labels/dangerous_progression.py :: compute_downstream_labels()`

### Leakage controls
- **Only same-possession future events** are inspected (no cross-possession lookahead).
- **Feature generation** never accesses these future events.
- The possession window is defined by `(match_id, possession_id)` tuples, ordered by event `index`.

### Why k=5?
A window of 5 events captures roughly 1–3 passes and carry actions — enough to see the immediate downstream consequence of a pass without attributing outcomes from much later in the possession to the original pass.  This is a design choice; sensitivity to k should be evaluated in ablations.

---

## Label C — `final_third_entry_k`

### Question
Did the ball reach the final third within the next *k* events?

### Definition
True if any event within the next k same-possession events has a recorded `location_x ≥ 80.0`.

The final third boundary (x = 80 on a 120m pitch) corresponds to the last 33.3 metres toward goal.

### Implementation
Part of `src/labels/dangerous_progression.py :: compute_downstream_labels()`.

---

## Label D — `box_entry_k`

### Question
Did the ball enter the penalty box within the next *k* events?

### Definition
True if any event within the next k same-possession events satisfies:
- `location_x ≥ 102.0` AND
- `18.0 ≤ location_y ≤ 62.0`

These coordinates approximate the 18-yard box on a 120×80 pitch.

### Implementation
Part of `src/labels/dangerous_progression.py :: compute_downstream_labels()`.

---

## Label E — `shot_within_k`

### Question
Did the team take a shot within the next *k* events?

### Definition
True if any event within the next k same-possession events has `type_name = "Shot"`.

### Implementation
Part of `src/labels/dangerous_progression.py :: compute_downstream_labels()`.

---

## Label F — `threat_gain`

### Question
How much did this pass improve the team's attacking position?

### Definition
A continuous target computed as a zone-value delta:

```
threat_gain = zone_value(end_zone) − zone_value(start_zone)
```

where:
- The pitch is divided into a `zone_grid_x × zone_grid_y` grid (default: 12 × 8 = 96 zones).
- `zone_value(z)` is the empirical proportion of possessions passing through zone z that contain a shot, final-third entry, or box entry, normalised to [0, 1].
  
Specifically:
```
raw_zone_value(z) = (n_shots_from_z + 0.3×n_box_entries_from_z + 0.1×n_ft_entries_from_z) / n_possessions_through_z
zone_value(z) = raw_zone_value(z) / max(raw_zone_value)
```

`threat_gain` is in the range approximately [−1, 1].

### Documented limitations
- This is a **transparent xT proxy**, not a full possession-value model.
- Zone values are computed empirically from the training dataset (no external xT table).
- Sparse zones (few possessions) have noisy estimates.
- The metric captures location-based value change, not conditional probability of completion.
- It is NOT a shot-on-target probability or an expected goals metric.

### Implementation
`src/labels/downstream_outcomes.py :: compute_threat_gain()`

---

## Label prevalence (expected ranges)

Based on typical StatsBomb open data (World Cup / La Liga):

| Label | Expected prevalence | Notes |
|-------|----------------------|-------|
| `strict_line_break` | ~10–20% | Higher in possession-dominant teams |
| `loose_line_break` | ~25–40% | By construction ≥ strict |
| `dangerous_progression_k` (k=5) | ~15–30% | Varies by pitch zone |
| `final_third_entry_k` | ~10–25% | Higher from mid-third |
| `box_entry_k` | ~5–12% | Lower; only deep entries count |
| `shot_within_k` | ~3–8% | Relatively rare |
| `threat_gain` | Mean ≈ 0.02–0.08 | Right-skewed; forward passes dominate |

![v1 label correlations](figures/v1_label_correlations.png)
*Figure: Correlation matrix of v1 pass-level labels — illustrates inter-label dependencies.*

---

# v2 — Possession-level labels

All possession labels are computed by `src/labels/possession_labels.py :: attach_possession_labels()`.
The function receives the raw structural possession table (from `build_possession_sequences`) and returns a copy with all 13 label columns appended.

---

## Group A — Core outcome

### `poss_has_shot`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the possession contain at least one shot? |
| Definition | True if any event in `event_sequence` has `type_id = 5` (Shot). |
| Implementation | Derived from `_parse_sequence()` in `possession_labels.py`. |
| Leakage note | Uses only events *within* the possession — no future lookahead. |

### `poss_entered_final_third`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the ball ever reach the final third during this possession? |
| Definition | True if `max_x_reached ≥ 80.0` (metres, on a 120 m pitch). |
| Implementation | Computed from the scalar `max_x_reached` column. |

### `poss_entered_box`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the ball enter the penalty area at any point? |
| Definition | True if any event in `event_sequence` has `loc_x_norm ≥ 0.85` AND `0.225 ≤ loc_y_norm ≤ 0.775` (denormalised: x ≥ 102, 18 ≤ y ≤ 62). |
| Implementation | Derived from `_parse_sequence()` in `possession_labels.py`. |

### `poss_dangerous` (primary binary target)

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Was this possession *dangerous*? |
| Definition | `poss_has_shot OR poss_entered_box` |
| Invariant | `poss_dangerous = True` whenever `poss_has_shot = True` or `poss_entered_box = True`. |
| Rationale | Combines the two strongest indicators of attacking threat into a single binary target for model training.  This is the default target for v2 XGBoost and GRU models. |

---

## Group B — Richer outcome

### `poss_xg_generated`

| Property | Value |
|----------|-------|
| Type | float |
| Question | How much expected threat (xG) did this possession generate? |
| Definition | Sum of `shot_statsbomb_xg` across all shot events in the possession.  NaN if `events_df` was not provided. |
| Implementation | Built from `_build_xg_goal_maps()` using the flat events table. |
| Data requirement | Requires the full `events_df` DataFrame to be passed to `attach_possession_labels()`. |

### `poss_has_goal`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the possession end with a goal? |
| Definition | True if any shot's `shot_outcome_name` contains "goal" (case-insensitive).  False if `events_df` was not provided. |
| Implementation | Built from `_build_xg_goal_maps()`. |

### `poss_outcome_tier`

| Property | Value |
|----------|-------|
| Type | int8 (ordinal) |
| Question | What was the highest outcome this possession achieved? |
| Definition | Ordinal encoding: **0** = nothing, **1** = final-third entry, **2** = box entry, **3** = shot, **4** = goal. |
| Invariant | If `poss_has_goal = True` → tier = 4; if `poss_has_shot = True` → tier ≥ 3; if `poss_entered_box = True` → tier ≥ 2; if `poss_entered_final_third = True` → tier ≥ 1. |
| Implementation | Computed by `_outcome_tier()` in `possession_labels.py`. |

---

## Group C — Tempo / structural

### `poss_tempo`

| Property | Value |
|----------|-------|
| Type | float32 |
| Question | How fast was this possession? |
| Definition | `n_events / max(duration_seconds, 1.0)` — events per second of possession time. |
| Usage | Also used to classify `poss_phase` (counter-attacks have tempo ≥ 2.0). |

### `poss_verticality`

| Property | Value |
|----------|-------|
| Type | float32 |
| Question | How direct was this possession? |
| Definition | `territory_gained / (n_events × mean_pass_length + ε)` — a ratio of net forward progress to total pass volume.  High values indicate direct, vertical play; low values indicate circulatory build-up. |
| Edge case | Possessions with `territory_gained = 0` or negative (backwards movement) get values near or below 0. |

### `poss_recycled`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the possession go backwards and then recover? |
| Definition | True if x-coordinate fell by ≥ 15 m from a peak, then recovered by ≥ 15 m from the subsequent trough. |
| Implementation | Single-pass scan of denormalised x-coordinates in `_parse_sequence()`. |

### `poss_phase`

| Property | Value |
|----------|-------|
| Type | category (`counter`, `build_up`, `progression`, `final_third`) |
| Question | What tactical phase does this possession represent? |
| Definition | Rule-based classification (evaluated in priority order): |

| Phase | Criterion |
|-------|-----------|
| `counter` | `poss_tempo ≥ 2.0` events/s OR `origin_type` contains "counter" |
| `final_third` | Not counter AND `start_x ≥ 80.0` |
| `build_up` | Not counter, not final-third AND `start_x ≤ 40.0` |
| `progression` | Everything else (midfield play) |

Implementation: `_assign_phase()` in `possession_labels.py`.

---

## Group D — Defensive disruption

### `poss_broke_pressure`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the team play through defensive pressure? |
| Definition | True if the possession contained at least one event with `under_pressure = 1.0` AND at least 3 more events occurred *after* the last pressure event. |
| Rationale | Captures possessions where the team survived a pressing action and maintained meaningful control. |
| Implementation | Derived from `_parse_sequence()`. |

### `poss_bypassed_lines`

| Property | Value |
|----------|-------|
| Type | bool |
| Question | Did the team rapidly break through the midfield? |
| Definition | True if *both*: (a) `max_x_reached ≥ 80.0` was reached within the first 4 events of the possession, AND (b) the possession started in the team's own half (`start_x ≤ 50.0`). |
| Rationale | Captures long balls, counter-attacks, and quick transitions that bypass defensive lines. |
| Implementation | Combined from `_parse_sequence()` (first 4 events check) and a scalar filter on `start_x`. |

---

## Possession label prevalence (expected ranges)

| Label | Expected prevalence | Notes |
|-------|---------------------|-------|
| `poss_has_shot` | ~8–15% | Higher in high-press teams |
| `poss_entered_final_third` | ~35–50% | Common for strong possession teams |
| `poss_entered_box` | ~15–25% | Stricter than final-third entry |
| `poss_dangerous` | ~18–28% | Union of shot + box entry |
| `poss_xg_generated` | Mean ≈ 0.02–0.05 | Right-skewed; most possessions generate 0 xG |
| `poss_has_goal` | ~1–3% | Rare event |
| `poss_outcome_tier` | Mode = 0 | ~55–70% of possessions reach nothing |
| `poss_tempo` | Mean ≈ 0.8–1.2 | Counter-attacks may exceed 2.0 |
| `poss_verticality` | Mean ≈ 0.3–0.6 | Varies widely by team style |
| `poss_recycled` | ~5–15% | More common in slow build-up play |
| `poss_phase` | Mode = `progression` | Build-up and counter phases each ~15–25% |
| `poss_broke_pressure` | ~10–20% | Higher for press-resistant teams |
| `poss_bypassed_lines` | ~5–10% | More common in counter-pressing styles |

![Possession label prevalence](figures/poss_eda_label_prevalence.png)
*Figure: Prevalence rates for all 13 possession-level labels.*

![Possession label correlation](figures/poss_eda_label_corr.png)
*Figure: Inter-label correlation matrix for possession-level labels.*

---

## Sanity checks

### v1 pass-level checks

The following checks are implemented in `src/labels/validation_sampling.py :: label_sanity_checks()`:

1. `strict_line_break` prevalence ≤ `loose_line_break` prevalence
2. `dangerous_progression_k` = `final_third_entry_k` OR `box_entry_k` OR `shot_within_k` (exact union)
3. No NaN in `dangerous_progression_k`, `final_third_entry_k`, `box_entry_k`, `shot_within_k`
4. `strict_line_break` is NaN exactly where `has_360 = False`
5. `threat_gain` ∈ [−1, 1]
6. `threat_gain` has no NaN where `start_x` and `end_x` are valid
7. All labels are boolean or float (no string contamination)
8. `box_entry_k` prevalence ≤ `final_third_entry_k` prevalence (box is a subset of final third)
9. No duplicate `event_uuid` values in the label table

### v2 possession-level checks

The following invariants should hold for every labelled possession table:

1. `poss_dangerous = poss_has_shot | poss_entered_box` (exact definition)
2. `poss_outcome_tier ≥ 3` whenever `poss_has_shot = True`
3. `poss_outcome_tier ≥ 2` whenever `poss_entered_box = True`
4. `poss_outcome_tier ≥ 1` whenever `poss_entered_final_third = True`
5. `poss_outcome_tier = 4` implies `poss_has_goal = True`
6. `poss_tempo ≥ 0` (non-negative by construction)
7. `poss_phase` ∈ {`counter`, `build_up`, `progression`, `final_third`} — no other values
8. `poss_xg_generated ≥ 0` where not NaN
9. All boolean labels have dtype `bool` (enforced by `_set_label_dtypes()`)
10. `poss_bypassed_lines = True` only where `start_x ≤ 50` (own-half filter)
11. No duplicate `(match_id, possession_id)` pairs in the possession table

---

## Manual validation protocol

### v1 pass-level validation
For each label, at least 20 positive and 20 negative examples should be inspected with `src/labels/validation_sampling.py :: sample_positives()` / `sample_negatives()`.

### v2 possession-level validation
For each possession label, inspect at least 20 positive and 20 negative possessions.  For structural labels (`poss_phase`, `poss_recycled`, `poss_bypassed_lines`), additionally verify the `event_sequence` timeline against the operational definition.

### Inspection criteria
Inspectors should verify:
- **True positives**: The labelled case genuinely matches the operational definition.
- **False positives**: Identify systematic over-labelling patterns and refine thresholds if needed.
- **Edge cases**: Document borderline cases that the rule handles in a potentially controversial way.

Results of manual validation should be appended to `reports/experiment_log.md`.
