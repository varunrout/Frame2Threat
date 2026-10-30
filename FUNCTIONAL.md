> **Historical (pre-correction) review.**  Its statements about the 0.847 v3 number, 453 players and GNN near-parity are withdrawn; see `reports/audit/final_audit_2026-10.md` for the current state.

# Frame2Threat - Functional Completeness Plan

## 1. Coherence verdict

The built code broadly matches its own claims, and where it does not, the mismatch is disclosed rather than hidden. That is the standout feature. The README states a marginal +0.001 ROC AUC from 360 and this is backed by `models/results_summary.json` (event-only 0.8807 vs event+360 0.8822). The three modelling tiers (pass, possession, early) all exist as real modules and configs. Labels are decomposed and auditable as claimed.

The one substantive coherence problem is that the README still surfaces the possession ensemble (0.965) and GRU/XGB possession scores (0.95+) prominently in a "Key results" table, while `reports/feature_timing_audit.md` correctly explains those are retrospective upper bounds contaminated by completion-dependent features (max_x_reached, territory_gained, box entry in the target). The audit says the README "should lead with v3 early-forecasting results until the full-possession v2 model is rebuilt"; the README caveats this but does not fully lead with it. A reader skimming the results table could still take 0.965 as a predictive achievement. So: coherent and honest at the level of caveats, not yet fully coherent at the level of emphasis.

## 2. Functional gaps

| ID | Priority | Type | Claim/Expectation | What is actually there | Fix | Acceptance criteria |
|----|----------|------|-------------------|------------------------|-----|---------------------|
| F1 | High | Leakage | Possession model predicts poss_dangerous | Features summarise the completed possession, and the target includes box entry (a max-x event); this is target leakage, documented in the timing audit | Rebuild with an event-time cutoff feature set; report leakage-free number | A full-possession metric with no at/after-target features exists |
| F2 | High | Emphasis | Headline results table implies 0.95+ possession prediction | Those are retrospective upper bounds per the audit | Restructure the results table so v3 leakage-aware curve is the headline and full-possession is clearly a diagnostic | README top result is the leakage-free v3 number |
| F3 | Medium | Reproducibility | "Reproduce in one command" | `make pipeline` exists but needs network/data and gitignored artefacts to hit headline numbers | Add a from-clone path that reproduces at least one metric from committed inputs | One metric reproducible offline from clone |
| F4 | Medium | Evaluation | Calibration reported via figures and Brier | Brier is in the JSON; calibration is a committed PNG, but no numeric ECE/calibration metric | Compute and commit ECE per model | ECE numbers present in results_summary.json |
| F5 | Medium | Coverage claim | "74 tests, all passing" | 74 test functions exist, but coverage floor is 24% and torch paths untested in CI | Raise floor and test GRU/GNN | Coverage floor 45%+, GRU/GNN covered |
| F6 | Low | App | Streamlit app | Runs on synthetic data with placeholder scores (now labelled) | Load a real scored fixture | App shows real model output |
| F7 | Low | GNN result | RQ6 claims GNN vs tabular near-parity (0.841 vs 0.845 on 360 subset) | GNN AUC reported on val, not test; comparison is on a subset | Report GNN on the same test protocol or state the subset clearly in the table | GNN and tabular compared on identical held-out protocol |

## 3. What analytically complete looks like

- A single, clearly leakage-free headline predictive task (pass-level dangerous progression and/or v3 prefix forecasting), with 360 ablation reported as the marginal finding it is.
- The full-possession model either rebuilt leakage-free or explicitly demoted to a labelled retrospective diagnostic everywhere it appears, including the top results table.
- Every reported metric traceable from clone to number via committed inputs or a documented rebuild command, with the leakage-aware v3 curve as the primary evidence.
- Calibration reported numerically (ECE) not just visually, alongside Brier and PR AUC.
- Attribution outputs (player/team leaderboards) accompanied by at least a face-validity check against known players, which the reports gesture at (domain-consistent, 453 players).

## 4. Evaluation gaps

- Baselines: rule-based and logistic baselines exist for v1; good. v2/v3 lack an explicit trivial baseline (for example, prevalence or start-zone-only) reported alongside every early-forecasting number so the 0.62 start-only AUC has a floor to beat.
- Holdout: match-level splits with a persisted manifest are in place and are the right design; no leakage across folds. The weakness is feature-time leakage within the possession task, not fold leakage.
- Calibration: Brier scores are reported (event-only 0.1299, event+360 0.1288) and reliability diagrams are committed, but no Expected Calibration Error is computed. Name the metric: report ECE (10-bin) per model in results_summary.json.
- Metric to lead with: for the honest predictive claim, lead with test ROC AUC and PR AUC (average precision) of the v3 cumulative XGBoost at 50% (0.847 ROC / 0.762 AP), because it is the strongest leakage-free forward-looking number.

## 5. Definition of functionally done

- No reported predictive metric depends on at-or-after-target features; any that does is labelled retrospective in every location including the top table.
- v3 leakage-aware forecasting is the single headline; 360 ablation framed as the marginal finding.
- ECE computed and committed per model alongside Brier and average precision.
- At least one headline metric reproduces from a fresh clone offline.
- GNN and tabular compared on an identical held-out test protocol.