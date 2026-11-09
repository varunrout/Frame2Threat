# Sealed-holdout protocol: UEFA Euro 2024

Written and committed on 2026-11-09, **before** any holdout input was built or scored. The machine-readable
version is [configs/sealed_holdout.json](../../configs/sealed_holdout.json); its SHA-256 is recorded in the
result file. The result will be reported **whatever it is**: no rerun to improve a number, and if a plumbing
fix forces a rerun, both runs are reported.

## 1. Holdout competition and selection rule

- **Competition:** UEFA Euro 2024, men (StatsBomb `competition_id` 55, `season_id` 282), all 51 matches.
- **Data version:** StatsBomb Open Data at commit `3bfbffe1de5750ebd47d770be0bb924a10cde54f`, the same pinned
  revision as the research cohort ([configs/research_data.json](../../configs/research_data.json)).
- **Rule:** among men's competition-seasons, the one with the most matches that have 360 data and are outside
  the 99-match research cohort; ties broken by the lowest competition id, then by the **latest season**.
  Women's competitions are excluded because the models were never trained on women's football.
- **Amendment, recorded openly:** the first version of the rule broke ties by competition id only. Euro 2024
  and Euro 2020 both have 51 such matches and share competition id 55, so that rule did not pick a single
  competition-season. The "latest season" tie-break was added on 2026-11-09, before anything was scored.
- **Check:** [holdout_rule_check.json](holdout_rule_check.json), written by `scripts/check_holdout_rule.py`.
  Euro 2024 (51) and Euro 2020 (51) lead, then 1. Bundesliga 2023/24 (34). Copa America 2024 and AFCON 2023
  have no 360 matches at this revision. No Euro 2024 match is in the research cohort.

- **Amendment 2, recorded openly (2026-11-09, before any holdout input was built or scored):** the first
  config pinned the SHA-256 of `data/processed/split_manifest.csv`. That local file turned out to hold only 6
  matches (a stale smoke-size file written on 2026-11-07), while this protocol names the committed 99-match
  manifest. The config now pins `configs/research_split_manifest.csv` instead, and the script reads the split
  from it. Nothing else changed.

## 2. What is scored

- **Models, exactly as committed** in `models/` (no retraining, no refitting, no recalibration):
  `eventtime_xgboost.joblib`, `eventtime_gru.pt`, `eventtime_meta.json`. SHA-256 values are fixed in the
  config, and the script refuses to run if any differs.
- **Alert thresholds, as they are now** (chosen on validation, stored in `models/eventtime_meta.json`):
  XGBoost 0.3903678810626692, GRU 0.4221348512993151.
- **Baseline:** ball x/y logistic regression (`cur_x`, `cur_y`, standardised, `C = 0.01`, the value chosen on
  validation in `eventtime_meta.json`) refit on the **original 69 training matches only** and scored on the
  holdout. It never sees a holdout match.
- **Inputs:** built with the existing pipeline functions, unchanged and in the same order as
  `scripts/prepare_event_time.py`: `get_events(revision=...)`, `parse_events`, `build_possession_sequences`,
  `build_possession_arrays`. Written to `data/holdout/euro2024/` (gitignored). Plumbing is fixed only if this
  breaks, and any change is reported.

## 3. Definitions (frozen)

- **Eligible snapshot:** event `t` of a possession, scored from events `0..t`, before the possessing team's
  first shot or box entry, excluding the last event of a possession with no such outcome (nothing follows it).
  This is `build_snapshot_table(...)` with `terminal == False`, as in `run_event_time.py`,
  `evaluate_event_time.py` and `verify_artefacts.py`. Label `y = 1` if the possession later produces a shot or
  box entry.
- **Possession population:** every possession with at least one eligible snapshot. No minimum length.
- **Alert:** an eligible snapshot whose score is at or above the model's threshold.
- **Precision and recall:** at the snapshot level, over all eligible snapshots, alert versus `y`.
- **False-alert rate** (the one definition used everywhere from now on): the share of **safe possessions**
  (label 0) in the population above that raise **at least one alert** at an eligible snapshot.
  Also reported: the share of dangerous possessions alerted before the outcome, and median lead time in events
  from the first alert to the outcome.
- **ECE** (the one definition used everywhere from now on): ten equal-width probability bins on [0, 1],
  **weighted by snapshot count**: `sum_b n_b * |mean predicted_b - observed rate_b| / N`, over all eligible
  snapshots. Units: absolute probability (0.04 means the predicted probability is off by 4 percentage points on
  average per snapshot). This is `ece_10_equal_width` in `evaluate_event_time.py`.

## 4. Metrics reported

For XGBoost, GRU and the ball x/y baseline, on the holdout:

1. ROC AUC over all eligible snapshots, with a 95% percentile interval from **500 match-level bootstrap
   resamples** (seed 42), and paired differences XGBoost minus baseline, GRU minus baseline, GRU minus XGBoost.
2. ROC AUC on one eligible snapshot per possession (seed 42), with the same bootstrap.
3. Precision, recall, false-alert rate, dangerous-possession alert rate, median lead time (XGBoost and GRU, at
   the thresholds above).
4. ECE as defined above, and Brier score.

The same code also scores the original 15 test matches, so that every holdout number has an in-cohort
counterpart computed with identical definitions.

## 5. Pre-specified reading

- **Reference:** the in-cohort ROC AUC of 0.665 (fresh retrain; the committed models re-score at 0.6649 and
  0.6636).
- A model's AUC is said to **generalise** if 0.665 lies inside or below its holdout 95% interval, and **not to
  generalise** if the whole interval lies below 0.665.
- The **lift over ball position** holds on the holdout only if the 95% interval of model minus baseline lies
  above zero.
- One competition of 51 international matches is one external test, not proof of generalisation to other
  leagues, seasons or women's football.

## 6. Run once

```bash
python scripts/sealed_holdout.py --target holdout
```

The script refuses to run if `reports/repro/sealed_holdout.json` already exists, and records the git SHA, data
revision, command, config SHA-256, seed and model SHA-256s in that file.
