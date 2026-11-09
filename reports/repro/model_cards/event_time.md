# Model Card: Event-Time (In-Play) Possession Danger Models

## November 2026 reproduction

Full retraining completed on the same 99-match manifest, with dependency versions,
input hashes and feature schema recorded in `reports/repro/repair_retrain/eventtime_meta.json`.
Fresh XGBoost / GRU all-snapshot AUC: **0.6653 / 0.6636**; one-per-possession:
**0.7361 / 0.7390**. Equal-width ten-bin ECE, snapshot-weighted, in absolute probability: **0.0394 / 0.0390**.
The fresh XGBoost selected 55 trees; the 52-tree specification below describes the
preserved earlier model. Its cross-version reloaded probabilities have different
calibration despite near-identical ranking. Use fresh-run metrics and dependencies
together: `reports/repro/repair_retrain/event_time_results.md` and `reports/audit/REPAIR_REPORT.md`.
Fresh binaries remain local under `data/repro/repair_event_time/models`; regenerate
with `scripts/run_event_time.py`, or set `FRAME2THREAT_ROOT` to that run directory.
The remaining sections record the previous experiment and its limitations.

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
  or before `t`**.  A possessing-team pass whose end location is in the box counts as the box entry itself
  (second self-audit round; before this, `max_x_so_far` and the GRU's end-x/end-y inputs already contained
  the entry that the next event records). Label: a shot or box entry by the possessing team occurs in events `t+1..n-1`.
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
| XGBoost | Full snapshot feature set (52 trees, depth 5, min child weight 20, chosen on validation log loss) |
| GRU | Causal, 1 layer, hidden 64, learned 8-dim event-type embedding (replaces the raw numeric `type_id` of v2); chosen on validation log loss |

Training: train matches only; every hyper-parameter chosen on validation. Split unchanged
(`data/processed/split_manifest.csv`, seed 42).

## Results (test split, 15 matches; match-level bootstrap 95% intervals)

| Model | ROC AUC, all eligible snapshots (43,541 rows) | ROC AUC, one random snapshot per possession (2,286) | 20-draw mean (sd) |
|---|---:|---:|---:|
| Baseline A: ball x/y | 0.642 [0.622, 0.661] | 0.695 [0.674, 0.717] | 0.692 (0.006) |
| Baseline B: events so far | 0.561 [0.523, 0.603] | 0.608 [0.579, 0.638] | 0.612 (0.008) |
| Start context only | 0.592 [0.570, 0.617] | 0.599 [0.572, 0.626] | 0.599 (0.000) |
| XGBoost | 0.665 [0.647, 0.681] | 0.736 [0.714, 0.759] | 0.729 (0.006) |
| GRU | 0.664 [0.642, 0.687] | 0.739 [0.710, 0.767] | 0.744 (0.007) |

*Superseded first version (same-moment leak, 44,914 rows):* all snapshots 0.648 / 0.564 / 0.586 / 0.668 / 0.670; one per possession 0.713 / 0.530 / 0.643 / 0.749 / 0.756 (same model order).  Before/after tables: `reports/event_time/same_moment_fix_before_after.md`.

- XGBoost and the GRU are statistically indistinguishable (GRU - XGB, all rows: -0.001
  [-0.021, +0.022]).
- Both beat the ball-position baseline by only +0.02 (all rows) to +0.04 (one per possession); GRU - ball x/y is +0.021 [-0.000, +0.040] over all rows.
  Most of the signal is where the ball currently is.
- Length so far is weak (0.56 over all rows, 0.61 with one snapshot per possession, where dangerous possessions have shorter pre-danger histories by construction).
- Replacing the old fraction headline (GRU 0.820, XGBoost 0.847 at 50% of events) with this
  formulation lowers the headline to roughly 0.66-0.74 depending on how snapshots are weighted.

## Known Limitations

- **Calibration is imperfect.** XGBoost under-predicts between 0.3 and 0.6 (bin 0.5-0.6: mean 0.54,
  observed 0.63); the GRU is close in the 0.3-0.7 bins but under-predicts below 0.3 (bin 0.0-0.1: mean
  0.08, observed 0.19). The test split has 15 matches and a higher row prevalence (0.404) than train
  (0.362) and validation (0.329); there is no recalibration.
- Snapshots exist only where the possession continues (`t < n-1`), so the final event of a
  non-dangerous possession is not scored. A sensitivity run that adds those terminal snapshots
  (label 0) changes ROC AUC by under 0.01 (XGBoost 0.661, GRU 0.667).
- Long possessions contribute many rows; the one-per-possession numbers are reported for that reason.
- Tipping-point alerts (threshold = validation score quantile with predicted-positive rate equal to
  validation prevalence) fire on 90.6% (XGBoost) / 89.6% (GRU) of dangerous possessions with a median lead of 9 / 10 events
  (8.4 / 8.9 s), but also on 47.7% / 39.4% of non-dangerous ones: the threshold trades a high detection
  rate for many false alarms.  (Superseded first version: 94% / 93.5%, 41.9% / 40.6%, lead 6 / 7 events.)
- StatsBomb Open Data only; labels and features come from events, not tracking.
- `score-live` needs per-event `elapsed_seconds`; without it, events are assumed to be spaced by
  the training median and the CLI prints a warning.
