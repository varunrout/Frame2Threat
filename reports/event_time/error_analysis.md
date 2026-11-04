# Error analysis: event-time models by segment

Committed event-time XGBoost and GRU re-scored on the held-out test snapshots (15 matches, no retraining).
ROC AUC with a 300-resample match-level bootstrap 95% interval. **Descriptive only**: segments are not
corrected for multiple comparisons and the test split is small. Rows marked `small` have fewer than 100 snapshots in the rarer class or fewer than 5 matches; do not read anything into them.
Competition is derived from the match-id range and is confounded with 360 coverage and with club.

Overall: XGBoost 0.6649, GRU 0.6636 on 43,541 snapshots.

## By competition

| Segment | Snapshots | Matches | Prevalence | XGBoost AUC [95% CI] | GRU AUC [95% CI] | Note |
|---|---|---|---|---|---|---|
| La Liga 2020/21 (Barcelona) | 17,849 | 6 | 0.462 | 0.663 [0.637, 0.681] | 0.660 [0.632, 0.689] |  |
| World Cup 2022 | 25,692 | 9 | 0.364 | 0.659 [0.643, 0.679] | 0.661 [0.635, 0.692] |  |

## By origin

| Segment | Snapshots | Matches | Prevalence | XGBoost AUC [95% CI] | GRU AUC [95% CI] | Note |
|---|---|---|---|---|---|---|
| From Corner | 260 | 12 | 0.942 | 0.513 [0.185, 0.982] | 0.548 [0.399, 0.698] | small |
| From Counter | 414 | 15 | 0.336 | 0.634 [0.524, 0.759] | 0.670 [0.587, 0.782] |  |
| From Free Kick | 6,874 | 15 | 0.427 | 0.634 [0.596, 0.667] | 0.664 [0.630, 0.700] |  |
| From Goal Kick | 4,131 | 15 | 0.404 | 0.634 [0.576, 0.697] | 0.614 [0.538, 0.689] |  |
| From Keeper | 1,526 | 15 | 0.366 | 0.694 [0.589, 0.766] | 0.664 [0.566, 0.749] |  |
| From Kick Off | 1,630 | 15 | 0.327 | 0.727 [0.640, 0.815] | 0.661 [0.534, 0.748] |  |
| From Throw In | 10,507 | 15 | 0.438 | 0.652 [0.608, 0.697] | 0.666 [0.637, 0.692] |  |
| Other | 92 | 3 | 0.000 | n/a | n/a | small |
| Regular Play | 18,107 | 15 | 0.383 | 0.668 [0.641, 0.696] | 0.661 [0.627, 0.700] |  |

## By period

| Segment | Snapshots | Matches | Prevalence | XGBoost AUC [95% CI] | GRU AUC [95% CI] | Note |
|---|---|---|---|---|---|---|
| period 1 | 22,365 | 15 | 0.360 | 0.665 [0.617, 0.703] | 0.660 [0.621, 0.692] |  |
| period 2 | 21,176 | 15 | 0.451 | 0.653 [0.627, 0.676] | 0.658 [0.630, 0.685] |  |

## By minute band

| Segment | Snapshots | Matches | Prevalence | XGBoost AUC [95% CI] | GRU AUC [95% CI] | Note |
|---|---|---|---|---|---|---|
| 0-30 | 15,217 | 15 | 0.348 | 0.660 [0.606, 0.714] | 0.665 [0.619, 0.700] |  |
| 30-60 | 14,510 | 15 | 0.437 | 0.649 [0.614, 0.677] | 0.641 [0.606, 0.671] |  |
| 60-90 | 12,093 | 15 | 0.431 | 0.662 [0.632, 0.691] | 0.667 [0.621, 0.707] |  |
| 90+ | 1,721 | 15 | 0.444 | 0.748 [0.675, 0.811] | 0.743 [0.667, 0.804] |  |
