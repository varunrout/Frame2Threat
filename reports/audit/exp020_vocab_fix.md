# EXP-020 — EXP-019 and LOO summary re-run with the corrected event-type vocabulary

Test split (unchanged, 2,475 possessions, 2,377 with >=3 events); existing GRU `models/gru_poss_dangerous.pt`; threshold 0.50 (as in EXP-019).

## Tipping point (first event where P(dangerous) >= 0.50)

- All possessions crossing 0.50: **47.5%**
- Dangerous possessions crossing 0.50: **92.0%**
- Median tipping fraction (fraction-of-n; superseded formulation, see EXP-021): **0.619**

Old decoder applied to the same tipping events (reproduces the published figure):

| old (wrong) label | share |
|---|---:|
| Dribble | 64.4% |
| Shot | 10.3% |
| Carry | 9.8% |
| Other | 5.5% |
| Duel | 4.6% |

Id 3 is *Ball Receipt* in the encoder, which the old decoder printed as 'Dribble'. The published '64.4% dribbles' is therefore the ball-receipt share.

### Corrected tipping-event distribution (with rate normalisation)

| event type | n | share | base frequency | rate (share / base) |
|---|---:|---:|---:|---:|
| Ball Receipt | 728 | 64.4% | 27.6% | 2.34 |
| Carry | 116 | 10.3% | 23.1% | 0.44 |
| Pass | 111 | 9.8% | 29.4% | 0.33 |
| Pressure | 52 | 4.6% | 7.5% | 0.62 |
| Goalkeeper | 30 | 2.7% | 0.7% | 3.61 |
| Shot | 21 | 1.9% | 0.6% | 2.95 |
| Block | 16 | 1.4% | 1.0% | 1.40 |
| Foul Committed | 14 | 1.2% | 0.7% | 1.72 |
| Dribble | 13 | 1.2% | 0.7% | 1.54 |
| Duel | 12 | 1.1% | 1.7% | 0.64 |
| Unknown | 8 | 0.7% | 4.8% | 0.15 |
| Clearance | 7 | 0.6% | 0.9% | 0.66 |
| Foul Won | 1 | 0.1% | 0.7% | 0.13 |
| Interception | 1 | 0.1% | 0.6% | 0.15 |

### Corrected largest-single-step-jump event distribution

| event type | n | share | base frequency | rate (share / base) |
|---|---:|---:|---:|---:|
| Ball Receipt | 1,451 | 61.0% | 29.1% | 2.10 |
| Pressure | 222 | 9.3% | 7.9% | 1.19 |
| Carry | 146 | 6.1% | 24.4% | 0.25 |
| Pass | 145 | 6.1% | 26.9% | 0.23 |
| Foul Committed | 139 | 5.8% | 0.8% | 7.69 |
| Unknown | 66 | 2.8% | 4.2% | 0.66 |
| Goalkeeper | 49 | 2.1% | 0.7% | 3.03 |
| Block | 41 | 1.7% | 1.1% | 1.63 |
| Duel | 36 | 1.5% | 1.5% | 1.02 |
| Dribble | 32 | 1.3% | 0.8% | 1.71 |
| Shot | 15 | 0.6% | 0.6% | 0.98 |
| Foul Won | 14 | 0.6% | 0.7% | 0.81 |
| Clearance | 11 | 0.5% | 1.0% | 0.47 |
| Interception | 10 | 0.4% | 0.5% | 0.87 |

### LOO 'unlock' event (argmax attribution) by type

| event type | n | share | base frequency | rate (share / base) |
|---|---:|---:|---:|---:|
| Ball Receipt | 796 | 33.8% | 27.6% | 1.23 |
| Pressure | 416 | 17.7% | 7.5% | 2.37 |
| Pass | 281 | 11.9% | 29.4% | 0.41 |
| Carry | 165 | 7.0% | 23.1% | 0.30 |
| Shot | 125 | 5.3% | 0.6% | 8.42 |
| Unknown | 115 | 4.9% | 4.8% | 1.02 |
| Goalkeeper | 114 | 4.8% | 0.7% | 6.58 |
| Foul Committed | 101 | 4.3% | 0.7% | 5.96 |
| Duel | 78 | 3.3% | 1.7% | 1.98 |
| Dribble | 65 | 2.8% | 0.7% | 3.70 |
| Block | 41 | 1.7% | 1.0% | 1.73 |
| Clearance | 23 | 1.0% | 0.9% | 1.04 |
| Interception | 17 | 0.7% | 0.6% | 1.24 |
| Foul Won | 16 | 0.7% | 0.7% | 0.99 |

### LOO mean attribution by event type

| event type | n events | mean LOO attribution | mean |LOO| |
|---|---:|---:|---:|
| Shot | 288 | +0.1884 | 0.1886 |
| Goalkeeper | 336 | +0.1373 | 0.1403 |
| Clearance | 430 | +0.0424 | 0.0656 |
| Foul Committed | 329 | +0.0389 | 0.0562 |
| Block | 461 | +0.0329 | 0.0698 |
| Dribble | 341 | +0.0146 | 0.0485 |
| Interception | 266 | +0.0120 | 0.0658 |
| Duel | 763 | +0.0056 | 0.0438 |
| Foul Won | 314 | +0.0012 | 0.0519 |
| Pressure | 3,402 | +0.0012 | 0.0662 |
| Ball Receipt | 12,588 | -0.0017 | 0.0369 |
| Unknown | 2,190 | -0.0029 | 0.0164 |
| Pass | 13,403 | -0.0072 | 0.0272 |
| Carry | 10,545 | -0.0161 | 0.0260 |

## Known modelling weakness (not changed in this phase)

`type_id` is fed to the GRU as one raw numeric feature, which imposes a false ordering on a categorical variable (e.g. Pass=1 < Carry=2 < Ball Receipt=3). Phase 2 replaces it with a learned embedding in the new event-time model. The v2 GRU itself is unchanged.
