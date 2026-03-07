# Label Methodology — Frame2Threat

This document provides the precise operational definitions for every label in the Frame2Threat modelling dataset.  Each label was designed to answer a specific football analytics question, to be reproducibly computable from StatsBomb data, and to avoid data leakage.

---

## Analytical unit

**One open-play pass event with available StatsBomb 360 freeze-frame context.**

All labels are computed per pass.  Set-piece deliveries (corners, free kicks, goal kicks, throw-ins) are excluded.

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

---

## Sanity checks

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

---

## Manual validation protocol

For each label, at least 20 positive and 20 negative examples should be inspected with `src/labels/validation_sampling.py :: sample_positives()` / `sample_negatives()`.

Inspectors should verify:
- **True positives**: The labelled case genuinely matches the operational definition.
- **False positives**: Identify systematic over-labelling patterns and refine thresholds if needed.
- **Edge cases**: Document borderline cases that the rule handles in a potentially controversial way.

Results of manual validation should be appended to `reports/experiment_log.md`.
