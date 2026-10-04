# possession_length fix: before / after (EXP-027)

*Before* = `superseded_pre_possession_length_fix/` (`groupby('possession_id').cumcount()`, mixing matches); *after* = grouped by `(match_id, possession_id)` in event order.  Same split, configuration and seeds.

## `dangerous_progression_k`

| model | ROC AUC before | after | AP before | after | Brier before | after |
|---|---|---|---|---|---|---|
| LR_event | 0.8026 | 0.8026 | 0.8079 | 0.8080 | 0.1731 | 0.1731 |
| XGB_event | 0.8807 | 0.8810 | 0.8911 | 0.8916 | 0.1299 | 0.1298 |
| XGB_event360 | 0.8822 | 0.8832 | 0.8925 | 0.8935 | 0.1288 | 0.1290 |
| LR_end_x | 0.8003 | 0.8003 | 0.8059 | 0.8059 | 0.1761 | 0.1761 |
| XGB_end_xy | 0.8565 | 0.8565 | 0.8649 | 0.8649 | 0.1400 | 0.1400 |

360 delta (event+360 minus event-only XGBoost), 1000 match-level bootstrap resamples:

| subset | statistic | before | after |
|---|---|---|---|
| all test passes | ROC AUC | +0.0014 [-0.0011, +0.0038] | +0.0021 [-0.0010, +0.0055] |
| all test passes | average precision | +0.0014 [-0.0005, +0.0033] | +0.0018 [-0.0006, +0.0043] |
| all test passes | Brier | -0.0011 [-0.0026, +0.0004] | -0.0008 [-0.0025, +0.0008] |
| passes with 360 only | ROC AUC | +0.0036 [-0.0012, +0.0083] | +0.0030 [-0.0024, +0.0083] |
| passes with 360 only | average precision | +0.0025 [-0.0022, +0.0067] | +0.0024 [-0.0021, +0.0063] |
| passes with 360 only | Brier | -0.0015 [-0.0041, +0.0014] | -0.0005 [-0.0029, +0.0021] |

## `progression_beyond_pass_k`

| model | ROC AUC before | after | AP before | after | Brier before | after |
|---|---|---|---|---|---|---|
| LR_event | 0.6111 | 0.6122 | 0.2961 | 0.2962 | 0.1699 | 0.1698 |
| XGB_event | 0.7180 | 0.7279 | 0.4110 | 0.4222 | 0.1889 | 0.1948 |
| XGB_event360 | 0.7229 | 0.7294 | 0.4239 | 0.4263 | 0.1838 | 0.1910 |
| LR_end_x | 0.5428 | 0.5428 | 0.2656 | 0.2656 | 0.1733 | 0.1733 |
| XGB_end_xy | 0.6874 | 0.6874 | 0.3586 | 0.3586 | 0.2247 | 0.2247 |

360 delta (event+360 minus event-only XGBoost), 1000 match-level bootstrap resamples:

| subset | statistic | before | after |
|---|---|---|---|
| all test passes | ROC AUC | +0.0050 [-0.0011, +0.0106] | +0.0015 [-0.0036, +0.0070] |
| all test passes | average precision | +0.0129 [+0.0015, +0.0250] | +0.0041 [-0.0053, +0.0134] |
| all test passes | Brier | -0.0051 [-0.0100, -0.0006] | -0.0038 [-0.0067, -0.0015] |
| passes with 360 only | ROC AUC | +0.0029 [-0.0057, +0.0111] | -0.0015 [-0.0106, +0.0069] |
| passes with 360 only | average precision | +0.0138 [-0.0037, +0.0323] | +0.0027 [-0.0111, +0.0114] |
| passes with 360 only | Brier | -0.0132 [-0.0184, -0.0080] | -0.0061 [-0.0107, -0.0007] |

