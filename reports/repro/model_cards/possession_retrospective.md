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

## Cross-fitted attribution use (EXP-031, October 2026)

The retrospective GRU is used to attribute a possession's score to its events by leave-one-out (LOO).  The earlier
attribution (Gini 0.495, a 453-player leaderboard, "final-third entries and through balls matter most") was
**in-sample** (the GRU had been trained on the possessions it attributed), used pre-fix labels and the wrong event-type
decoder, and credited defending-team events to players; it is superseded (figures kept in
`reports/figures/superseded_attribution/`).

Current method (`reports/attribution/ATTRIBUTION_REPORT.md`): 5 match-grouped folds over all 99 matches; a GRU per fold
trained on the other folds (inner match-level split for early stopping); LOO only on the held-out fold, first 40 events.

| Quantity | Value |
|---|---|
| Pooled out-of-fold ROC AUC (corrected labels; retrospective) | 0.952 (folds 0.9558, 0.9569, 0.9543, 0.9572, 0.9584); second seed 0.955 |
| Possession score vs xG, Spearman | 0.50 |
| Median Gini of |LOO| (dangerous / safe) | 0.63 / 0.43 |
| Player ranking, split-half Spearman / cross-seed Spearman | 0.62 / 0.89 |

**What it is for:** tagging key moments for video review and producing scouting shortlists to inspect.  **What it is not:**
causal credit (zeroed-event sequences are unrealistic), a measure of defenders' quality (defending-team events are
evidence of danger), or a basis for ranking players on its own (the stable part of the ranking is mostly role; no
adjustment for position, team or opponent).  Scores are class-weighted GRU outputs, not calibrated probabilities, and
very high scores saturate so LOO drops vanish.  Next step: a forward-looking action value model (xT or VAEP-style).
