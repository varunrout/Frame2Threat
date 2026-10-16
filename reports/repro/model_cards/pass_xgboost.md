# Model Card: Pass-Level XGBoost

## Summary

The v1 pass-level XGBoost models predict `dangerous_progression_k` for each
open-play pass: whether the pass leads to final-third entry, box entry, or a
shot within the next five same-possession events.

## Intended Use

- Inspect pass-level dangerous progression from event data.
- Compare event-only features against event+360 geometry.
- Support tactical review and feature attribution, not automated player
  recruitment decisions.

## Data And Features

| Item | Value |
|---|---|
| Unit | Open-play pass |
| Label | `dangerous_progression_k` |
| Event-only features | 27 |
| Event+360 features | 41 |
| Train rows | 36,037 |
| Validation rows | 7,307 |
| Test rows | 7,344 |

## Reported Metrics

| Model | Test ROC AUC | Test PR AUC | Brier |
|---|---:|---:|---:|
| XGBoost event-only | 0.8807 | 0.8911 | 0.1299 |
| XGBoost event+360 | 0.8822 | 0.8925 | 0.1288 |

The event+360 lift is only +0.0014 ROC AUC. Treat this as a marginal/negative
finding for 360 additive value in this dataset.

> **Correction (October 2026).** These metrics are for `dangerous_progression_k`, which is self-determining: the receipt of the pass itself sits inside the look-ahead, so `end_x` alone gives ROC AUC 0.800 on this test split.  With a match-level bootstrap the 360 delta is +0.0014 [-0.0011, +0.0038] (not "statistically positive"); after the `possession_length` fix (the v1 feature used to mix matches) it is +0.0021 [-0.0010, +0.0055] and the metrics are event-only 0.8810, event+360 0.8832.  On the corrected label `progression_beyond_pass_k` (Label G), after the fix: event-only 0.728 [0.713, 0.743], event+360 0.729 [0.715, 0.745], delta +0.0015 [-0.0036, +0.0070], `end_x` alone 0.543 (`reports/pass_label/pass_label_results.md`; before/after in `possession_length_fix_before_after.md`).  *Superseded:* 0.718 / 0.723 / +0.0050 (broken `possession_length`).

## Known Limitations

- 360 coverage is partial.
- The pass-level model sees only an event snapshot, not player trajectories.
- GNN comparison: the old validation-only near-parity (0.841 vs 0.845) is original label only and superseded.  The retrained graph-only PassFrameGNN on the corrected label does not beat XGBoost (0.595 [0.578, 0.606] vs 0.715 [0.697, 0.734] ROC AUC on the identical 360 test subset); see `pass_gnn.md`.
- Bootstrap intervals are in `reports/pass_label/pass_label_results.md` (15 test matches).
- XGBoost uses `scale_pos_weight`, so its Brier score is not comparable with unweighted baselines.

## Artefacts

- Results: `models/v1_results_summary.json`
- Calibration figure: `reports/figures/v1_calibration.png`
- Ablation figure: `reports/figures/v1_ablation_360.png`
