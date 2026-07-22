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

## Known Limitations

- 360 coverage is partial.
- The pass-level model sees only an event snapshot, not player trajectories.
- The GNN comparison is validation-only in the current report.
- Metrics are reported without bootstrap confidence intervals.

## Artefacts

- Results: `models/v1_results_summary.json`
- Calibration figure: `reports/figures/v1_calibration.png`
- Ablation figure: `reports/figures/v1_ablation_360.png`
