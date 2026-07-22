# Model Card: Retrospective Possession Models

## Summary

The v2 possession models predict `poss_dangerous` after observing the completed
possession. These models are useful retrospective diagnostics, but their scores
must not be presented as leakage-free early predictions.

## Intended Use

- Retrospective possession review.
- Attribution and tactical analysis after a sequence has unfolded.
- Upper-bound comparison against early-forecasting models.

## Data And Features

| Model | Unit | Test ROC AUC | Test AP |
|---|---|---:|---:|
| XGBoost possession tabular | Full possession | 0.9505 | 0.8947 |
| PossessionGRU | Full event sequence | 0.9524 | 0.9282 |
| Equal-weight XGB+GRU ensemble | Full possession/sequence | 0.9650 | 0.9358 |

## Timing Caveat

The tabular model includes completed-possession summaries such as
`max_x_reached` and `territory_gained`. The target `poss_dangerous` includes
box entry, which is itself determined by completed spatial progression. See
`reports/feature_timing_audit.md`.

## Known Limitations

- Retrospective feature timing makes these metrics upper bounds.
- Ensemble weighting is fixed rather than validation-tuned.
- Calibration and confidence intervals are not fully reported for every
  possession model.
- The Streamlit demo may fall back to synthetic scores when artefacts/data are
  absent.

## Artefacts

- Results: `models/results_summary.json`
- Timing audit: `reports/feature_timing_audit.md`
- Calibration figure: `reports/figures/poss_calibration.png`
