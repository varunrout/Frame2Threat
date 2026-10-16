# Model Card: Pass-Level Progression XGBoost (final recipe)

## Summary

Two XGBoost models predict, for each open-play pass, whether the passing team makes *further* dangerous progress after
the pass (`progression_beyond_pass_k`).  Recipe (pre-specified): `configs/model_baseline.yaml` settings, **early stopping
on validation, no `scale_pos_weight`**, no post-hoc calibrator unless validation reliability shows one is needed (it did not).
Artefacts: `models/pass_progression_event_only.joblib` (27 event features, 129 trees) and
`models/pass_progression_event_360.joblib` (41 features, 163 trees), each with a `.json` metadata file next to it
(label definition, ordered features, match counts, git commit, test metrics with intervals, reliability table, calibration).
Code: `src/models/pass_progression.py`, `scripts/train_pass_models.py`; results: `reports/pass_label/pass_model_final.md`.

> **Superseded:** the EXP-027 recipe (`scale_pos_weight`, all 500 trees; test ROC AUC 0.728 / 0.729) inflated probabilities (mean
> prediction 0.41 vs prevalence 0.224) and is replaced by this recipe.  The old-label models `xgboost_dp_event_only.joblib` and
> `xgboost_dp_event_360.joblib` (`dangerous_progression_k`) are in `models/superseded/` and are for comparison only: that label is
> largely decided by where the pass ends (`end_x` alone: ROC AUC 0.800), so its 0.881 / 0.882 is not evidence about event or 360 information.

## Target

`progression_beyond_pass_k` (k = 5; 21.6% positive overall, 22.4% in the test split): within the next 5 events of the same
possession, ignoring the receipt of this pass and the opponent's events, a shot, a new box entry, or a gain of at least 10 m beyond the pass end
into a more advanced zone (`reports/label_methodology.md`, Label G).

## Features, training data and split

- 27 engineered event features (`src/features/event_features.py`, including the corrected `possession_length`); event+360 adds 14 freeze-frame
  geometry features (zero where there is no frame, as in training).
- Data: 64 FIFA World Cup 2022 matches (360 on 87% of their passes) and 35 La Liga 2020/21 Barcelona matches (no 360 frames); 50,688 open-play passes,
  of which 26,515 (52.3%) have a freeze frame.  360 therefore exists only for World Cup matches.
- Split: the saved match-level 70/15/15 split (seed 42; `data/processed/split_manifest.csv`): 69 / 15 / 15 matches,
  36,037 / 7,307 / 7,344 passes.  Early stopping on validation; test read once.

## Metrics (test; match-level bootstrap 95% intervals, 1,000 resamples)

| Model | Test passes | ROC AUC | AP | Brier | Log loss | Calibration intercept / slope |
|---|---|---|---|---|---|---|
| Event-only | 7,344 | 0.734 [0.718, 0.749] | 0.428 [0.405, 0.452] | 0.153 [0.147, 0.160] | 0.472 [0.456, 0.488] | +0.15 / 1.09 |
| Event+360 | 7,344 | 0.738 [0.724, 0.752] | 0.433 [0.411, 0.458] | 0.152 [0.146, 0.159] | 0.469 [0.453, 0.485] | +0.15 / 1.08 |
| Event-only, 360 subset | 3,736 | 0.719 [0.698, 0.739] | 0.401 [0.370, 0.436] | 0.153 [0.142, 0.165] | 0.471 [0.445, 0.500] | -0.02 / 0.99 |
| Event+360, 360 subset | 3,736 | 0.728 [0.709, 0.748] | 0.421 [0.389, 0.460] | 0.151 [0.140, 0.163] | 0.466 [0.440, 0.495] | -0.07 / 0.99 |

For scale: `end_x` alone gives ROC AUC 0.543 on this label.

### Does 360 add anything?  (event+360 minus event-only, paired bootstrap)

| Subset | ROC AUC | AP | Brier | Log loss |
|---|---|---|---|---|
| All test passes | +0.0043 [+0.0015, +0.0076] | +0.0043 [-0.0045, +0.0144] | -0.0007 [-0.0015, -0.0000] | -0.0025 [-0.0041, -0.0010] |
| 360-only subset | +0.0087 [+0.0029, +0.0147] | +0.0206 [+0.0086, +0.0326] | -0.0018 [-0.0031, -0.0004] | -0.0047 [-0.0077, -0.0013] |

With the final recipe the 360 gain is **small but measurable**.  Event+360 minus event-only (paired match-level bootstrap, 1,000 resamples): on all 7,344 test passes ROC AUC +0.0043 [+0.0015, +0.0076], log loss -0.0025 [-0.0041, -0.0010], Brier -0.0007 [-0.0015, -0.0000] (upper bound just below zero), average precision +0.0043 [-0.0045, +0.0144] (includes zero); on the 3,736-pass 360-only subset all four exclude zero: ROC AUC +0.0087 [+0.0029, +0.0147], AP +0.0206 [+0.0086, +0.0326], Brier -0.0018 [-0.0031, -0.0004], log loss -0.0047 [-0.0077, -0.0013].  The earlier reading "360 adds nothing measurable" (EXP-027 recipe: ROC AUC +0.0015 [-0.0036, +0.0070], AP +0.0041 [-0.0053, +0.0134]) is **superseded**; the effect is still only about +0.004 ROC AUC over all passes (+0.009 where a freeze frame exists).

*Superseded reading:* "no measurable 360 gain" (EXP-025/027 recipe).

### Calibration

Validation reliability needed no calibrator (event-only intercept +0.025, slope 1.022; event+360 -0.004, 0.997).
On test the mean prediction is 0.216 against a prevalence of 0.224; the 10-bin reliability tables (`pass_model_final.md`, `pass_model_reliability_*.csv`) are close to the diagonal below 0.5
and thin above (33 and 45 passes in the 0.6-0.7 bin, 1 and 3 above 0.7).  Calibration intercept +0.15 on all test passes reflects a test prevalence slightly above validation.

## Intended use

Analyst review of pass progression: finding passes the model rates as likely to lead to further progress, and passes whose actual outcome differed from the
expectation.  The app (`src/app/app.py`) shows the event-only score, the event+360 score where a freeze frame exists, a player table (passes, mean predicted,
actual rate, actual minus predicted; players under 50 passes hidden by default) and the saved diagnostics.

## Out of scope

- Player recruitment or ranking decisions on its own: player means mix role, team and opponent, and 'actual minus predicted' is noisy for small samples.
- Scoring passes outside open play, or La Liga passes with the 360 model (no freeze frames exist).
- Causal claims about what a pass 'caused'.

## Known limitations

- Static freeze frames: one snapshot at the moment of the pass; no velocities or trajectories; players outside the camera view are missing.
- 360 coverage is partial (52.3% of passes; World Cup 2022 only), and the geometry features are zero for passes without a frame.
- Open-play passes only.
- The test split has 15 matches (9 with 360), so intervals are wide.
- Carries contribute only their start point to the label, so progress made carrying appears one event later.
- GNN context: graph-only player positions alone reach ROC AUC 0.595 [0.578, 0.606] on the 360 subset against 0.530 [0.502, 0.551] for `end_x` alone; event features dominate
  (see `pass_gnn.md`; not a like-for-like comparison).
