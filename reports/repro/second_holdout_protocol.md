# Second sealed holdout and position-only boosted baseline: protocol

Written and committed on 2026-11-09, **before** the new baseline was fitted or anything below was scored. It
follows [sealed_holdout_protocol.md](sealed_holdout_protocol.md) (the Euro 2024 protocol) and reuses its frozen
definitions unchanged. The machine-readable version is
[configs/second_holdout.json](../../configs/second_holdout.json); its SHA-256 goes into the result file. The result
is reported **whatever it is**. Nothing is scored twice; a plumbing fix that forces a rerun is reported with both
runs.

**Why this exists.** An external review found the Euro 2024 result sound but the comparison baseline weak: a
linear model on ball x/y cannot bend around the goal, so a nonlinear model on position alone may close the gap that
the event-time models show. This protocol adds that stronger baseline and a second unseen competition.

## 1. What is fixed and not touched

- **Committed models, byte-identical** (`models/eventtime_xgboost.joblib`, `eventtime_gru.pt`,
  `eventtime_meta.json`; SHA-256 values in the config, checked before scoring). No retraining, refitting, retuning or
  recalibration.
- **Validation-prevalence thresholds as committed:** XGBoost 0.3903678810626692, GRU 0.4221348512993151.
- **Euro 2024 stays as scored** in [sealed_holdout.json](sealed_holdout.json). The models are re-run on the same
  Euro 2024 inputs (hashes pinned) **only** to pair them with the new baseline. Their AUCs must reproduce the
  committed values; if they do not, both are reported and the difference is flagged.
- **Frozen definitions** of eligible snapshot, possession population, alert, precision, recall, false-alert rate
  and ECE: section 3 of [sealed_holdout_protocol.md](sealed_holdout_protocol.md), code in
  `src/evaluation/frozen_metrics.py`.

## 2. New position-only boosted baseline (`GBM_pos`)

- **Features (position only, no event history):** at each eligible snapshot, from the same causal `cur_x`,
  `cur_y` the models use (the last possessing-team event location, possessing-team frame, StatsBomb 120 x 80
  units, goal centre at (120, 40)):
  - `cur_x`, `cur_y`;
  - `dist_goal` = sqrt((120 - cur_x)^2 + (40 - cur_y)^2);
  - `angle_goal` = atan2(|40 - cur_y|, max(120 - cur_x, 0)), radians from the goal axis;
  - `zone` = 3 * min(floor(cur_x / 20), 5) + min(floor(cur_y / (80/3)), 2), an integer 0..17 (six length bands,
    three lanes);
  - `attack_dir` = +1. Included as specified, but in the possessing-team frame every possession attacks towards
    x = 120, so it is constant and carries no information. Disclosed, not hidden.
- **Same budget as the event-time XGBoost** (`fit_xgb` in `scripts/run_event_time.py`): XGBoost, learning rate
  0.05, up to 800 trees with early stopping after 30 rounds on validation log loss, subsample 0.8, column subsample
  0.8, grid max_depth {3, 5} x min_child_weight {1, 20}, configuration chosen by validation log loss, seed 42,
  2 threads.
- **Data:** fitted on the eligible snapshots of the **original 69 training matches only**, early-stopped and
  selected on the **15 validation matches only** (`configs/research_split_manifest.csv`, inputs in
  `data/processed`, hashes pinned). It never sees a test, Euro 2024 or Euro 2020 match.
- **Labelled as added after the Euro 2024 scoring.** It is frozen here before it is scored on anything.
- The old ball x/y logistic baseline (`LR_xy`, C = 0.01, refit on the same 69 training matches) is kept unchanged.

## 3. Second sealed holdout: UEFA Euro 2020

- StatsBomb competition 55, season 43, men, all 51 matches; data revision
  `3bfbffe1de5750ebd47d770be0bb924a10cde54f` (the pinned revision). All 51 have 360 data and none is in the
  research cohort ([holdout_rule_check.json](holdout_rule_check.json)). Never scored before.
- Inputs built with the same pipeline functions as Euro 2024 (`build_holdout_inputs` in
  `scripts/sealed_holdout.py`) into `data/holdout/euro2020/`. The run refuses to start if that folder already holds
  inputs.

## 4. Metrics

| Target | What is scored |
|---|---|
| **Euro 2020** | Everything below |
| **15 in-cohort test matches** | `GBM_pos` added; paired differences against it; precision operating point |
| **Euro 2024** | `GBM_pos` added and paired differences against it only (no new operating point) |

1. ROC AUC over all eligible snapshots for XGBoost, GRU, `LR_xy` and `GBM_pos`, with 95% percentile intervals
   from **500 match-level bootstrap resamples** (seed 42), and paired differences XGBoost minus `LR_xy`, XGBoost
   minus `GBM_pos`, GRU minus `LR_xy`, GRU minus `GBM_pos`, GRU minus XGBoost on the same resamples. The same for
   one snapshot per possession (seed 42).
2. ECE (frozen definition) and Brier score for all four.
3. At the **validation-prevalence thresholds** (XGBoost and GRU): precision, recall, false-alert rate,
   dangerous-possession alert rate, median lead time in events.
4. **Precision-oriented operating point** (XGBoost and GRU), threshold chosen on the 15 **validation** matches
   only: the lowest threshold at which validation snapshot precision is at least **0.65** and at least 1% of
   validation snapshots alert. If no threshold meets this, the operating point is reported as not reachable.
   Reported: the threshold, its validation precision and recall, and precision, recall and false-alert rate on the
   target.

## 5. Reading rules (fixed now)

- **AUC generalises** to Euro 2020 for a model if 0.665 lies inside or below its 95% interval; it does **not** if
  the whole interval lies below 0.665 (as for Euro 2024).
- **Lift over the strong baseline holds** only if the 95% interval of model minus `GBM_pos` lies entirely above
  zero. Otherwise the lift is **not shown**, and the README headline must say so.
- The same rule is applied to the in-cohort test split and to Euro 2024.
- Two men's international tournaments are two external tests, not evidence for club leagues, other seasons or
  women's football.

## 6. Run once

```bash
python scripts/second_holdout.py
```

The script refuses to run if `reports/repro/second_holdout.json` exists, if the git working tree is dirty, or if
any pinned hash differs. It records git SHA, data revision, command, config SHA-256, seed, package versions and model
SHA-256s. `scripts/render_second_holdout.py` writes `reports/repro/second_holdout.md` from the JSON.
