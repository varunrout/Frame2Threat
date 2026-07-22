# Model Card: Early XGBoost Possession Forecasts

## Summary

The v3 early XGBoost models predict `poss_dangerous` from possession-start or
prefix-observed features. This is the cleaner predictive setting for the
project because features are restricted to information available at the
declared observation fraction.

## Intended Use

- Estimate danger before the possession has fully unfolded.
- Compare start-only context against 25%, 50%, and 75% cumulative observation.
- Provide a leakage-aware headline result for portfolio review.

## Data And Features

| Model | Feature Set | Train Rows | Test ROC AUC | Test AP |
|---|---|---:|---:|---:|
| `xgboost_start_only` | Possession-start context | 12,092 | 0.6241 | 0.5346 |
| `xgboost_cumulative_25pct` | First 25% of events | 12,092 | 0.8136 | 0.7127 |
| `xgboost_cumulative_50pct` | First 50% of events | 12,092 | 0.8472 | 0.7624 |
| `xgboost_cumulative_75pct` | First 75% of events | 12,092 | 0.8912 | 0.8200 |

## Timing Contract

`build_cumulative_tabular_features()` recomputes aggregate columns from only
the observed prefix. Label-derived possession summaries are zeroed for partial
prefixes because they cannot be safely derived mid-possession.

## Known Limitations

- Prefix models still use engineered aggregates from the observed prefix, so
  they are an offline approximation of live scoring.
- No confidence intervals are reported yet.
- The current committed sample contains XGBoost early scores only; GRU prefix
  scores require loading the PyTorch artefact.

## Artefacts

- Results: `models/results_summary.json`
- Scored sample: `reports/repro/scored_possessions_sample.csv`
- Comparison figure: `reports/figures/v3_exp016_exp017_auc_comparison.png`
