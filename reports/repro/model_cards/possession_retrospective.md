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

## Correction (October 2026)

The defending team's events were scored in the opposite coordinate frame, so 765 `poss_dangerous` labels (13.7% of positives) and 2,381 final-third flags were spurious; the possession table was regenerated with the possessing team's events only (`src/data/possession_frame.py`).  Like-for-like retrain with the unchanged configuration, same split (`reports/audit/exp023_v2_corrected_labels.md`):

| Model | Pre-fix labels | Corrected labels |
|---|---:|---:|
| XGBoost possession tabular | 0.9461 | 0.9572 |
| PossessionGRU | 0.9534 | 0.9570 |
| Equal-weight logit-mean ensemble | 0.9736 | 0.9769 |

The retrospective scores hold.  The saved XGBoost artefact deserialises under the current XGBoost version at 0.9449 (recorded 0.9505), and the recorded 0.9650 ensemble is not reproduced (0.9732 from the saved models), so the retrain is the comparable pair.  Retrained artefacts: `models/v2_corrected_*`.

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
