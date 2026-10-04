# Model Card: Event-Time (In-Play) Possession Danger Models

## Summary

The event-time models estimate, at every event `t` of a possession that has not yet produced a
shot or a box entry, the probability that the possessing team produces one in events
`t+1 ... n-1` of the same possession (`danger_after_t`). They replace the superseded
fraction-based v3 models (`early_xgboost.md`), whose "first f% of events" prefix needed the
possession's total length and often already contained the outcome.

Artefacts: `models/eventtime_xgboost.joblib`, `models/eventtime_gru.pt`,
`models/eventtime_meta.json` (alert thresholds, tuned hyper-parameters). Code:
`src/features/event_time.py`, `src/models/event_time.py`, `src/evaluation/event_time_eval.py`,
`scripts/run_event_time.py`. Full results: `reports/event_time/event_time_results.md`.

## Intended Use

- Quantify how much of a possession's eventual danger can be anticipated from what has already
  happened, without knowing how long the possession will last.
- Drive `frame2threat score-live`, which prints a per-event danger trajectory.
- Not for decisions about individual players, and not a calibrated risk score (see below).

## Timing Contract

- One snapshot row per event index `t < n-1` with **no shot or box entry by the possessing team at
  or before `t`**. Label: a shot or box entry by the possessing team occurs in events `t+1..n-1`.
- Features use events `0..t` only: current ball x/y (last possessing-team event), maximum x so
  far, events / passes / carries / pressures so far, elapsed **seconds** since the first event
  (real timestamps), the current event's type (one-hot), whether it was the possessing team's,
  and start context (origin, start x/y, period, minute).
- No feature uses the possession length, a fraction of it, or anything after `t`.
  `tests/test_event_time.py` appends arbitrary future events and asserts that the row for `t` is
  unchanged, and that the GRU output at step `t` equals the output on the truncated prefix.
- Coordinates: ball progression and danger use the possessing team's events only; the defending
  team's events (recorded in the opposite frame) are flipped when they are model inputs.

## Models

| Model | Description |
|---|---|
| Baseline A | Logistic regression on current ball x/y |
| Baseline B | Logistic regression on events-so-far (how much length alone explains) |
| Reference | XGBoost on start context only |
| XGBoost | Full snapshot feature set (59 trees, depth 5, chosen on validation log loss) |
| GRU | Causal, 1 layer, hidden 64, learned 8-dim event-type embedding (replaces the raw numeric `type_id` of v2); chosen on validation log loss |

Training: train matches only; every hyper-parameter chosen on validation. Split unchanged
(`data/processed/split_manifest.csv`, seed 42).

## Results (test split, 15 matches; match-level bootstrap 95% intervals)

| Model | ROC AUC, all eligible snapshots (44,914 rows) | ROC AUC, one random snapshot per possession (2,452) |
|---|---:|---:|
| Baseline A: ball x/y | 0.648 [0.630, 0.664] | 0.713 [0.688, 0.735] |
| Baseline B: events so far | 0.564 [0.529, 0.601] | 0.530 [0.496, 0.567] |
| Start context only | 0.586 [0.563, 0.610] | 0.643 [0.612, 0.668] |
| XGBoost | 0.668 [0.653, 0.681] | 0.749 [0.727, 0.772] |
| GRU | 0.670 [0.650, 0.689] | 0.756 [0.739, 0.777] |

- XGBoost and the GRU are statistically indistinguishable (GRU - XGB, all rows: +0.007
  [-0.005, +0.022]).
- Both beat the ball-position baseline by only +0.02 (all rows) to +0.04 (one per possession).
  Most of the signal is where the ball currently is.
- Length alone is close to uninformative once the length leak is removed (0.53-0.56).
- Replacing the old fraction headline (GRU 0.820, XGBoost 0.847 at 50% of events) with this
  formulation lowers the headline to roughly 0.67-0.76 depending on how snapshots are weighted.

## Known Limitations

- **Calibration is imperfect.** Both models under-predict across the range on the test split
  (e.g. GRU bin 0.3-0.4: mean prediction 0.35, observed 0.41). The test split has 15 matches and a
  higher row prevalence (0.371) than train (0.325) and validation (0.280); there is no recalibration.
- Snapshots exist only where the possession continues (`t < n-1`), so the final event of a
  non-dangerous possession is not scored. A sensitivity run that adds those terminal snapshots
  (label 0) changes ROC AUC by under 0.01 (XGBoost 0.664, GRU 0.670).
- Long possessions contribute many rows; the one-per-possession numbers are reported for that reason.
- Tipping-point alerts (threshold = validation score quantile with predicted-positive rate equal to
  validation prevalence) fire on 94% of dangerous possessions but also on about 41% of non-dangerous
  ones: the threshold trades a high detection rate for many false alarms.
- StatsBomb Open Data only; labels and features come from events, not tracking.
- `score-live` needs per-event `elapsed_seconds`; without it, events are assumed to be spaced by
  the training median and the CLI prints a warning.
