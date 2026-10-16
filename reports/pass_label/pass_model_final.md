# Final pass models (EXP-033): XGBoost, early stopping on validation, no `scale_pos_weight`

Unchanged match-level split; test read once.  Supersedes the EXP-027 recipe (`scale_pos_weight`, all 500 trees), whose probabilities were inflated (mean prediction 0.41 vs prevalence 0.224).

Trees: event-only 129, event+360 163.  Calibrator fitted on validation: event-only False, event+360 False (rule in the script docstring).  Validation calibration (raw): event-only intercept +0.025 / slope 1.022; event+360 -0.004 / 0.997.

## all test passes: 7,344 passes, 15 matches, positive rate 0.224

| model | n | roc_auc | avg_precision | brier | log_loss | cal. intercept | cal. slope | mean pred |
|---|---|---|---|---|---|---|---|---|
| event_only | 7344 | 0.7337 [0.7185, 0.7490] | 0.4283 [0.4049, 0.4525] | 0.1532 [0.1465, 0.1600] | 0.4719 [0.4557, 0.4882] | +0.15 | 1.09 | 0.216 |
| event_360 | 7344 | 0.7380 [0.7239, 0.7519] | 0.4326 [0.4114, 0.4579] | 0.1525 [0.1457, 0.1591] | 0.4694 [0.4533, 0.4853] | +0.15 | 1.08 | 0.216 |


Paired match-level bootstrap (1,000 resamples), event+360 minus event-only:

| statistic | event+360 - event-only [95% CI] | interval excludes zero |
|---|---|---|
| roc_auc | +0.0043 [+0.0015, +0.0076] | yes |
| avg_precision | +0.0043 [-0.0045, +0.0144] | no |
| brier | -0.0007 [-0.0015, -0.0000] | yes |
| log_loss | -0.0025 [-0.0041, -0.0010] | yes |


## 360-only subset: 3,736 passes, 9 matches, positive rate 0.217

| model | n | roc_auc | avg_precision | brier | log_loss | cal. intercept | cal. slope | mean pred |
|---|---|---|---|---|---|---|---|---|
| event_only | 3736 | 0.7195 [0.6983, 0.7389] | 0.4006 [0.3697, 0.4357] | 0.1525 [0.1419, 0.1646] | 0.4710 [0.4454, 0.5000] | -0.02 | 0.99 | 0.219 |
| event_360 | 3736 | 0.7281 [0.7091, 0.7480] | 0.4212 [0.3886, 0.4597] | 0.1508 [0.1402, 0.1629] | 0.4664 [0.4404, 0.4945] | -0.07 | 0.99 | 0.226 |


Paired match-level bootstrap (1,000 resamples), event+360 minus event-only:

| statistic | event+360 - event-only [95% CI] | interval excludes zero |
|---|---|---|
| roc_auc | +0.0087 [+0.0029, +0.0147] | yes |
| avg_precision | +0.0206 [+0.0086, +0.0326] | yes |
| brier | -0.0018 [-0.0031, -0.0004] | yes |
| log_loss | -0.0047 [-0.0077, -0.0013] | yes |


## Conclusion on the 360 gain

**At least one interval excludes zero** (statistic: view, estimate [95% CI]): roc_auc: all test passes, +0.0043 [+0.0015, +0.0076]; brier: all test passes, -0.0007 [-0.0015, -0.0000]; log_loss: all test passes, -0.0025 [-0.0041, -0.0010]; roc_auc: 360-only subset, +0.0087 [+0.0029, +0.0147]; avg_precision: 360-only subset, +0.0206 [+0.0086, +0.0326]; brier: 360-only subset, -0.0018 [-0.0031, -0.0004]; log_loss: 360-only subset, -0.0047 [-0.0077, -0.0013].  "360 adds nothing measurable" no longer holds for those statistics; see the documentation for the updated reading.

### Reliability, event_only (all test passes, 10 equal-width bins)

| bin | n | mean_pred | observed |
|---|---|---|---|
| 0.0-0.1 | 1750 | 0.070 | 0.061 |
| 0.1-0.2 | 2096 | 0.147 | 0.151 |
| 0.2-0.3 | 1516 | 0.249 | 0.250 |
| 0.3-0.4 | 1191 | 0.346 | 0.367 |
| 0.4-0.5 | 587 | 0.440 | 0.503 |
| 0.5-0.6 | 170 | 0.534 | 0.518 |
| 0.6-0.7 | 33 | 0.625 | 0.758 |
| 0.7-0.8 | 1 | 0.723 | 0.000 |
| 0.8-0.9 | 0 | nan | nan |
| 0.9-1.0 | 0 | nan | nan |


### Reliability, event_360 (all test passes, 10 equal-width bins)

| bin | n | mean_pred | observed |
|---|---|---|---|
| 0.0-0.1 | 1775 | 0.068 | 0.061 |
| 0.1-0.2 | 2123 | 0.146 | 0.149 |
| 0.2-0.3 | 1510 | 0.249 | 0.256 |
| 0.3-0.4 | 1113 | 0.346 | 0.380 |
| 0.4-0.5 | 566 | 0.442 | 0.468 |
| 0.5-0.6 | 209 | 0.543 | 0.555 |
| 0.6-0.7 | 45 | 0.633 | 0.711 |
| 0.7-0.8 | 3 | 0.726 | 0.333 |
| 0.8-0.9 | 0 | nan | nan |
| 0.9-1.0 | 0 | nan | nan |


