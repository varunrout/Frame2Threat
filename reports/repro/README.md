# Reproducible Evidence Pack

This directory contains the committed evidence that can be inspected from a
fresh clone without downloading StatsBomb data or retraining models.

## Files

| File | Purpose |
|---|---|
| `scored_possessions_sample.csv` | A 25-row possession sample scored with the local XGBoost early/prefix models and the retrospective full-possession XGBoost model. |
| `scored_possessions_manifest.json` | Source paths, column list, and generation notes for the scored sample. |
| `model_cards/pass_xgboost.md` | Model card for the v1 pass-level XGBoost models. |
| `model_cards/early_xgboost.md` | Model card for v3 start-only and cumulative XGBoost models. |
| `model_cards/possession_retrospective.md` | Model card for the v2 retrospective possession models and ensemble. |

## Calibration And Figures

The calibration plots and diagnostic figures are already committed under
`reports/figures/`:

| Figure | Meaning |
|---|---|
| `reports/figures/v1_calibration.png` | Reliability diagram for the v1 pass-level XGBoost model. |
| `reports/figures/poss_calibration.png` | Possession-level calibration diagram. Interpret with the retrospective caveat in `reports/feature_timing_audit.md`. |
| `reports/figures/v3_exp016_exp017_auc_comparison.png` | Prefix GRU vs cumulative XGBoost early-forecasting comparison. |
| `reports/figures/v3_exp016_prefix_gru_curve.png` | GRU prefix ROC-AUC and PR-AUC across observation fractions. |

## Scored Sample Notes

The CSV uses the first 25 possessions by `(match_id, possession_id)` from the
local processed possession table available in this workspace. It includes:

- start-only XGBoost probability
- 50% cumulative XGBoost probability
- 75% cumulative XGBoost probability
- full-possession XGBoost probability, labelled as retrospective

The full raw and processed datasets remain gitignored. This sample is not a
substitute for a one-command rebuild; it is a small, inspectable artefact that
lets a reviewer see concrete scored output and model-card context immediately.

## Hashes

```text
scored_possessions_sample.csv    SHA256 510AFADEC5209442258E6E3A652B2BA05C4BD7B1F14912F55B7E4DF88E2F59FA
scored_possessions_manifest.json SHA256 2C177E7E9043BB8752A1F9DC27FA7EB8986672610E43B08EA9C90666C3A3C75C
```
