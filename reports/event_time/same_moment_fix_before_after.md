# Same-moment leak fix: before / after (EXP-026)

*Before* = `superseded_pre_same_moment_fix/` (danger judged on event location only); *after* = a possessing-team pass ending in the box is itself the danger event.  Same split, same hyper-parameter search on validation, same seeds.

Eligible test rows: 44,914 -> 43,541; test possessions with rows: 2,452 -> 2,286; row prevalence (train/val/test): 0.325->0.362 / 0.280->0.329 / 0.371->0.404

## All eligible snapshots

| model | AUC before | AUC after | AP before | AP after | Brier before | Brier after |
|---|---|---|---|---|---|---|
| LR_xy | 0.648 | 0.642 | 0.497 | 0.527 | 0.221 | 0.229 |
| LR_len | 0.564 | 0.561 | 0.428 | 0.452 | 0.233 | 0.240 |
| XGB_start | 0.586 | 0.592 | 0.433 | 0.482 | 0.234 | 0.240 |
| XGB | 0.668 | 0.665 | 0.539 | 0.567 | 0.216 | 0.223 |
| GRU | 0.670 | 0.664 | 0.542 | 0.572 | 0.221 | 0.224 |

## One random snapshot per possession (seed 42)

| model | AUC before | AUC after | AP before | AP after | Brier before | Brier after |
|---|---|---|---|---|---|---|
| LR_xy | 0.713 | 0.695 | 0.533 | 0.487 | 0.190 | 0.197 |
| LR_len | 0.530 | 0.608 | 0.356 | 0.402 | 0.215 | 0.211 |
| XGB_start | 0.643 | 0.599 | 0.465 | 0.417 | 0.211 | 0.215 |
| XGB | 0.749 | 0.736 | 0.606 | 0.581 | 0.181 | 0.188 |
| GRU | 0.756 | 0.739 | 0.597 | 0.581 | 0.180 | 0.182 |

## Mean (sd) ROC AUC over 20 one-per-possession draws

| model | before | after |
|---|---|---|
| LR_xy | 0.724 (0.006) | 0.692 (0.006) |
| LR_len | 0.539 (0.007) | 0.612 (0.008) |
| XGB_start | 0.643 (0.000) | 0.599 (0.000) |
| XGB | 0.756 (0.005) | 0.729 (0.006) |
| GRU | 0.768 (0.006) | 0.744 (0.007) |

## Match-level bootstrap 95% intervals, ROC AUC (all snapshots)

| model / difference | before | after |
|---|---|---|
| LR_xy | 0.648 [0.630, 0.664] | 0.642 [0.622, 0.661] |
| LR_len | 0.564 [0.529, 0.601] | 0.561 [0.523, 0.603] |
| XGB_start | 0.586 [0.563, 0.610] | 0.592 [0.570, 0.617] |
| XGB | 0.668 [0.653, 0.681] | 0.665 [0.647, 0.681] |
| GRU | 0.670 [0.650, 0.689] | 0.664 [0.642, 0.687] |
| XGB - LR_xy | 0.021 [0.008, 0.034] | 0.022 [0.009, 0.034] |
| XGB - LR_len | 0.104 [0.066, 0.136] | 0.104 [0.060, 0.143] |
| GRU - LR_xy | 0.022 [0.004, 0.038] | 0.021 [-0.000, 0.040] |
| GRU - LR_len | 0.106 [0.082, 0.125] | 0.103 [0.080, 0.124] |
| GRU - XGB | 0.001 [-0.014, 0.018] | -0.001 [-0.021, 0.022] |
| XGB - XGB_start | 0.082 [0.056, 0.105] | 0.073 [0.041, 0.102] |

## Match-level bootstrap 95% intervals, ROC AUC (one snapshot per possession)

| model / difference | before | after |
|---|---|---|
| LR_xy | 0.713 [0.688, 0.735] | 0.695 [0.674, 0.717] |
| LR_len | 0.530 [0.496, 0.567] | 0.608 [0.579, 0.638] |
| XGB_start | 0.643 [0.612, 0.668] | 0.599 [0.572, 0.626] |
| XGB | 0.749 [0.727, 0.772] | 0.736 [0.714, 0.759] |
| GRU | 0.756 [0.739, 0.777] | 0.739 [0.710, 0.767] |
| XGB - LR_xy | 0.036 [0.023, 0.052] | 0.041 [0.029, 0.052] |
| XGB - LR_len | 0.220 [0.174, 0.263] | 0.128 [0.105, 0.155] |
| GRU - LR_xy | 0.043 [0.023, 0.066] | 0.044 [0.022, 0.063] |
| GRU - LR_len | 0.226 [0.189, 0.263] | 0.131 [0.114, 0.153] |
| GRU - XGB | 0.007 [-0.005, 0.022] | 0.003 [-0.014, 0.017] |
| XGB - XGB_start | 0.107 [0.085, 0.136] | 0.137 [0.114, 0.162] |

## Alerts, lead time and thresholds

| model | threshold before -> after | dangerous alerted | safe alerted (false alarm) | median lead (events) | median lead (s) |
|---|---|---|---|---|---|
| XGB | 0.372 -> 0.390 | 93.9% -> 90.6% | 41.9% -> 47.7% | 6 -> 9 | 6.2 -> 8.4 |
| GRU | 0.353 -> 0.422 | 93.5% -> 89.6% | 40.6% -> 39.4% | 7 -> 10 | 7.3 -> 8.9 |

## ROC AUC by events-so-far bucket

| bucket | model | n before | n after | AUC before | AUC after |
|---|---|---|---|---|---|
| 1-3 | LR_xy | 6835 | 6594 | 0.618 | 0.609 |
| 1-3 | XGB | 6835 | 6594 | 0.641 | 0.632 |
| 1-3 | GRU | 6835 | 6594 | 0.648 | 0.639 |
| 4-6 | LR_xy | 5713 | 5603 | 0.613 | 0.619 |
| 4-6 | XGB | 5713 | 5603 | 0.641 | 0.651 |
| 4-6 | GRU | 5713 | 5603 | 0.645 | 0.651 |
| 7-10 | LR_xy | 6295 | 6153 | 0.637 | 0.646 |
| 7-10 | XGB | 6295 | 6153 | 0.662 | 0.672 |
| 7-10 | GRU | 6295 | 6153 | 0.665 | 0.672 |
| 11+ | LR_xy | 26071 | 25191 | 0.651 | 0.640 |
| 11+ | XGB | 26071 | 25191 | 0.667 | 0.657 |
| 11+ | GRU | 26071 | 25191 | 0.671 | 0.660 |


## ROC AUC by elapsed-seconds bucket

| bucket | model | n before | n after | AUC before | AUC after |
|---|---|---|---|---|---|
| 0-5 | LR_xy | 12377 | 12079 | 0.617 | 0.619 |
| 0-5 | XGB | 12377 | 12079 | 0.641 | 0.644 |
| 0-5 | GRU | 12377 | 12079 | 0.642 | 0.644 |
| 5-10 | LR_xy | 7114 | 6947 | 0.628 | 0.635 |
| 5-10 | XGB | 7114 | 6947 | 0.660 | 0.669 |
| 5-10 | GRU | 7114 | 6947 | 0.662 | 0.670 |
| 10-20 | LR_xy | 9925 | 9666 | 0.656 | 0.656 |
| 10-20 | XGB | 9925 | 9666 | 0.667 | 0.673 |
| 10-20 | GRU | 9925 | 9666 | 0.670 | 0.669 |
| 20+ | LR_xy | 15498 | 14849 | 0.649 | 0.630 |
| 20+ | XGB | 15498 | 14849 | 0.667 | 0.643 |
| 20+ | GRU | 15498 | 14849 | 0.673 | 0.659 |

