# Reproducible Evidence Pack

This directory contains the committed evidence that can be inspected from a
fresh clone without downloading StatsBomb data or retraining models.

The November 2026 repair adds `repair_retrain/` (full fresh training, 500-resample
bootstrap, model metadata, reliability tables, genuine replay CSVs and the verified
clean environment) and `repair_evaluation.json` (independent re-scoring of the prior
models, including a constant baseline and ECE). These are distinct from the earlier
25-possession pack below. See `../audit/REPAIR_REPORT.md` for the current validation status.

## Files

| File | Purpose |
|---|---|
| `scored_possessions_sample.csv` | 25 test-split possessions (fixed seed 42) scored with the **event-time** XGBoost and GRU (max score, first alert, lead before the shot / box entry) plus the **retrospective** XGBoost and GRU (columns prefixed `RETROSPECTIVE_`, not early predictions).  Regenerate with `python scripts/make_evidence_pack.py`. |
| `scored_possessions_manifest.json` | Source paths, column list, and generation notes for the scored sample. |
| `model_cards/pass_xgboost.md` | Model card for the final pass-level XGBoost models on `progression_beyond_pass_k` (early stopping, no `scale_pos_weight`); the old-label models are superseded. |
| `model_cards/early_xgboost.md` | Model card for v3 start-only and cumulative XGBoost models (**superseded**: fraction-based, length leak). |
| `model_cards/pass_gnn.md` | Model card for the PassFrameGNN retrained on the corrected label (graph-only; does not beat XGBoost). |
| `model_cards/event_time.md` | Model card for the event-time (in-play) XGBoost and GRU models that replace them. |
| `model_cards/possession_retrospective.md` | Model card for the retrospective possession models, their corrected-label retrain and the cross-fitted attribution use (`reports/attribution/ATTRIBUTION_REPORT.md`). |

## Calibration And Figures

The calibration plots and diagnostic figures are already committed under
`reports/figures/`:

| Figure | Meaning |
|---|---|
| `reports/figures/v1_calibration.png` | Reliability diagram for the v1 pass-level XGBoost model. |
| `reports/figures/poss_calibration.png` | Possession-level calibration diagram. Interpret with the retrospective caveat in `reports/feature_timing_audit.md`. |
| `reports/figures/v3_exp016_exp017_auc_comparison.png` | **Superseded** (fraction-based, length leak): prefix GRU vs cumulative XGBoost comparison. |
| `reports/figures/v3_exp016_prefix_gru_curve.png` | **Superseded** (fraction-based): GRU prefix ROC-AUC and PR-AUC across observation fractions. |
| `reports/event_time/` | Current early-forecasting evidence: results, buckets, reliability tables, before/after of the same-moment-leak fix. |

## Scored Sample Notes

The CSV draws 25 possessions with at least 4 events at random (seed 42) from the **test-split** matches of
the local processed possession table.  Columns:

- `eventtime_xgb_max_p`, `eventtime_gru_max_p`: highest in-play score before the shot / box entry
  (step `t` uses events `0..t` only; scores exist while no shot or box entry, including a possessing-team
  pass ending in the box, has occurred);
- `eventtime_*_first_alert_idx` and `eventtime_*_lead_events`: first event at which the score crosses the
  validation-chosen threshold (`models/eventtime_meta.json`) and how many events before
  `first_danger_event_idx` that was;
- `RETROSPECTIVE_xgb_prob`, `RETROSPECTIVE_gru_prob`: full-possession models retrained on the corrected labels
  (`models/v2_corrected_*`).  They see the whole possession and are **not** early predictions.

The earlier fraction-based columns (`xgb_start_prob`, `xgb_50pct_prob`, `xgb_75pct_prob`) were withdrawn: the
prefix length encodes the possession length (see `reports/feature_timing_audit.md`).  The models behind them are
kept in `models/superseded/`.

The full raw and processed datasets remain gitignored. This sample is not a
substitute for a one-command rebuild; it is a small, inspectable artefact that
lets a reviewer see concrete scored output and model-card context immediately.

## Hashes

```text
scored_possessions_sample.csv    SHA256 A7312BAB53514BCC712726A265A9098CDE336FE4C35F0D610AC8F34139B6094B
scored_possessions_manifest.json SHA256 86F95D090F22BA30116F2CB03ED2DD688415A85E11887E3C3228FFE1D461C3B7
```
