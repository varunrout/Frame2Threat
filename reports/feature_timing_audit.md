# Feature Timing Audit

This audit records which possession-level features are safe for early prediction
and which features depend on the completed possession. It exists because the
headline full-possession model predicts `poss_dangerous`, where:

```text
poss_dangerous = poss_has_shot OR poss_entered_box
```

`poss_entered_box` is determined by the maximum x-coordinate reached during the
possession. Any model that receives full-possession spatial summaries can
therefore receive information that is at, or downstream of, the target event.

## Current Full-Possession Features

`src.features.possession_features.build_tabular_features()` currently includes
the following completion-dependent columns:

| Feature | Why It Is Completion-Dependent |
|---|---|
| `max_x_reached` | Computed from every event in the possession; directly overlaps with box/final-third entry labels. |
| `territory_gained` | Derived from `max_x_reached - start_x`; inherits the same timing problem. |
| `progression_speed` | Uses completed `territory_gained` and completed duration. |
| `n_events`, `n_passes`, `n_carries` | Known only after the possession has unfolded. |
| `n_pressures_faced`, `pressure_rate`, `has_pressure` | Summarise pressure across the whole possession. |
| `duration_seconds` | Known only at the end of the possession. |
| `mean_pass_length`, `pass_rate`, `carry_rate` | Summaries over the completed sequence. |
| `poss_tempo`, `poss_verticality`, `poss_recycled`, `poss_broke_pressure`, `poss_bypassed_lines`, `poss_phase` | Derived possession summaries/labels, not start-time features. |

These features are acceptable for retrospective possession description, but
they should not be presented as leakage-free early prediction features.

## Leakage-Free Feature Sets

> **SUPERSEDED (see the addendum below).** The v3 fraction-based prefix path described in this section is *not* leakage-free: the prefix length `ceil(n*f)` encodes the possession's total length, and many prefixes already contain the outcome. Kept for the record; the event-time formulation replaces it.

The v3 prefix path was originally described as leakage-free:

- `build_start_features()` keeps only features known when the possession begins.
- `build_cumulative_tabular_features(poss_df, frac)` recomputes aggregate
  features from only the first observed prefix of each possession.
- For `frac < 1.0`, label-derived possession summaries are zeroed because they
  cannot be derived safely from a partial sequence.

~~The README should therefore lead with v3 early-forecasting results until the full-possession v2 model is rebuilt with a strict timing contract.~~ *Withdrawn: see the addendum.*

## Current Claim Status

The reported full-possession metrics remain useful as retrospective diagnostics:

- XGBoost full possession: ROC AUC 0.9505
- PossessionGRU full sequence: ROC AUC 0.9524
- XGB + GRU ensemble: ROC AUC 0.9650

They are not valid as a leakage-free predictive result.  (After the coordinate-frame correction below they are, like-for-like, XGBoost 0.946 -> 0.957, GRU 0.953 -> 0.957, logit-mean ensemble 0.974 -> 0.977, `reports/audit/exp023_v2_corrected_labels.md`.)

## Required Follow-Up

To make a full-possession model headline-safe, one of these paths is required:

1. Rebuild the target as a retrospective classification task and label the
   model honestly as retrospective.
2. Rebuild features around an event-time cutoff: no feature computed at or
   after the first shot/box-entry event may enter the prediction row.
3. Prefer the v3 prefix setting as the headline task and report full-possession
   scores only as upper-bound or retrospective references.

---

## Addendum: defects found in the self-audit (EXP-020 onwards)

Measured by `scripts/audit_defects.py` (`reports/audit/defect_audit_before.md`, `defect_audit_after.md`).

### 1. Possession-length leak in the fraction-based prefixes

"The first f% of events" requires the possession's total length n.  The prefix length
`ceil(n*f)` is therefore a monotone function of n, and n alone predicts `poss_dangerous`:

| Quantity | Value |
|---|---|
| ROC AUC of total length n (pre-fix labels) | 0.683 |
| ROC AUC of prefix length `ceil(n*f)`, f = 0.25 / 0.5 / 0.75 | 0.683 / 0.682 / 0.682 |
| Median length, dangerous vs safe | 23 vs 12 events |
| Start-only XGBoost (EXP-015), for comparison | 0.624 |

Length alone beats the start-context model, so the v3 gain over start-only was partly length.
In the event-time formulation, events-so-far (the only length information a snapshot has)
gives ROC AUC 0.558 over all eligible rows and 0.606 with one snapshot per possession (0.557 / 0.544 before the same-moment fix in addendum 2 changed which snapshots are eligible; the one-per-possession value is higher because dangerous possessions now have shorter pre-danger histories, which is hazard behaviour, not a leak of the possession length).

### 2. The outcome is often already inside the prefix

Share of `poss_dangerous` possessions whose first `ceil(n*f)` events already contain a shot or a
box entry (event location, the label's own definition; pre-fix data):

| f | 0.25 | 0.5 | 0.75 |
|---|---:|---:|---:|
| event location | 20.8% | 36.9% | 58.7% |
| incl. pass/carry end locations (the initial audit's definition) | 27.4% | 42.0% | 64.0% |

(Under the initial audit's definition about 3% of *safe* possessions also reach the box through a
pass/carry end that the label ignores.)  A model scored on those prefixes is partly reading the answer.
The v3 numbers (GRU 0.820, XGBoost 0.847 at 50%) are withdrawn.  The event-time formulation only
creates a row when no shot or box entry has occurred at or before `t`, so this share is 0% by construction.

### 3. Opponent events used in the wrong coordinate frame

StatsBomb records `location` from the acting team's perspective, and a possession's event stream
contains the defending team's events (15.7% of events).  `max_x_reached`, box-entry labels and prefix
spatial features used those coordinates unflipped, crediting the possessing team with positions that
are really near its own goal.  The shared code path (`src/data/possession_frame.py`,
`parse_possessions.py`, `possession_labels.py`) now scores progression and danger on the possessing
team's events only.  Effect on the stored possession table (17,065 possessions; nothing changed in the
other direction):

| Label | Old rate | Corrected rate | Changed (true -> false) |
|---|---:|---:|---:|
| `poss_dangerous` | 32.8% | 28.3% | 765 |
| `poss_entered_box` | 30.0% | 25.5% | 762 |
| `poss_has_shot` | 12.4% | 12.2% | 30 |
| `poss_entered_final_third` | 75.4% | 61.5% | 2,381 |

`max_x_reached` changed for 3,893 possessions, `team_name` for 86 (first event was the opponent's).
The GRU's input fields are byte-identical for every possession; sequences gained an `own_team`
flag that is not a GRU input.  The pass-level `dangerous_progression_k` look-ahead has the same
problem (it reads defending-team events unflipped); that label is deliberately left unchanged for
comparability and superseded by `progression_beyond_pass_k`.  The zone-value map behind `threat_gain`
also averages over all events and has not been corrected (see "Not fixed" in the final summary).

### Current claim status

| Claim | Status |
|---|---|
| Full-possession (retrospective) scores | Hold as retrospective diagnostics; numbers above |
| v3 early forecasting 0.82-0.85 at 50% | Withdrawn (length leak + realised outcome) |
| Early forecasting in event time | ROC AUC 0.66 (all snapshots) to 0.74 (one per possession) after the same-moment fix (first version 0.67 / 0.75-0.76); mostly ball position (`reports/event_time/event_time_results.md`) |

---

## Addendum 2: second self-audit round

### 4. Same-moment leak (event-time model)

`max_x_so_far` and the GRU's end-x / end-y inputs include the end location of the *current* pass, while danger was judged on event location only.  At a possessing-team pass that ends in the box the snapshot therefore already contained the box entry that the next event (the receipt, at the pass end) records.  Measured before the fix (`reports/audit/same_moment_before.md`): 1.30% of snapshots (3,973 of 305,399), label rate 80.4% there vs 31.9% elsewhere, and 56.6% of dangerous possessions (2,628 of 4,640) had such a pass as their last eligible snapshot.  Fix: a possessing-team pass ending in the box is itself a danger event (`possession_frame.is_danger_event`), so snapshots at or after it are not eligible (`reports/audit/same_moment_after.md`: 0 such snapshots, 293,639 eligible rows).  Effect on the event-time results (`reports/event_time/same_moment_fix_before_after.md`): ROC AUC 0.668 -> 0.665 (XGBoost) and 0.670 -> 0.664 (GRU) over all snapshots; 0.749 -> 0.736 and 0.756 -> 0.739 with one snapshot per possession.  The possession labels (`poss_dangerous`, ...) were **not** changed; they judge danger on event location, and the event-time definition now differs on 925 possessions (all incomplete or out-of-play passes ending in the box).

Updated causality rule for any snapshot feature: nothing computed from events `0..t` may reveal a danger event, so end locations of events at `t` are subject to the same definition of danger as the label.  (A carry's end is not stored, so carries cannot leak this way.)

### 5. `possession_length` in the v1 pass features

The v1 feature grouped by `possession_id` alone; `possession_id` restarts every match, so the value mixed matches and depended on row order and split.  It was a running match-level counter (mean 145, maximum 836) rather than a possession length, was correlated 0.06 with the correct count, and is fixed (`reports/pass_label/possession_length_fix_before_after.md`).  It is a causal feature (earlier passes only), so this was a correctness problem, not a future leak.

### 6. Shot xG and goals

`poss_xg_generated`, `poss_has_goal` and the tier-4 outcome were constant because the shot columns were never parsed; they are not features of any v2 model (excluded in `possession_features.py`), so no timing or leakage consequence for v2.  They are now populated (`reports/audit/shot_columns_after.md`) and, being outcomes, must stay out of any feature set.

### Current claim status (second round)

| Claim | Status |
|---|---|
| Event-time ROC AUC 0.668 / 0.670 (all), 0.749 / 0.756 (one per possession) | Superseded: 0.665 / 0.664 and 0.736 / 0.739 |
| 360 delta on the corrected label +0.0050, "about 3.5x larger" | Superseded: +0.0015 [-0.0036, +0.0070]; no more than on the original label |
| Recycled possessions 42% vs 23% dangerous | Superseded: confounded by length; -2.6 pp once controlled |
