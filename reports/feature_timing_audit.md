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

The current leakage-free possession prediction path is v3:

- `build_start_features()` keeps only features known when the possession begins.
- `build_cumulative_tabular_features(poss_df, frac)` recomputes aggregate
  features from only the first observed prefix of each possession.
- For `frac < 1.0`, label-derived possession summaries are zeroed because they
  cannot be derived safely from a partial sequence.

The README should therefore lead with v3 early-forecasting results until the
full-possession v2 model is rebuilt with a strict timing contract.

## Current Claim Status

The reported full-possession metrics remain useful as retrospective diagnostics:

- XGBoost full possession: ROC AUC 0.9505
- PossessionGRU full sequence: ROC AUC 0.9524
- XGB + GRU ensemble: ROC AUC 0.9650

They are not currently valid as the headline leakage-free predictive result.

## Required Follow-Up

To make a full-possession model headline-safe, one of these paths is required:

1. Rebuild the target as a retrospective classification task and label the
   model honestly as retrospective.
2. Rebuild features around an event-time cutoff: no feature computed at or
   after the first shot/box-entry event may enter the prediction row.
3. Prefer the v3 prefix setting as the headline task and report full-possession
   scores only as upper-bound or retrospective references.
