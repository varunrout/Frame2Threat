#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# create-functional-issues.sh
#
# Purpose: create GitHub issues for the FUNCTIONAL completeness gaps in
#          Frame2Threat only (analytical/product substance). Engineering
#          hygiene (CI, packaging, secrets, tests-as-hygiene) is out of scope
#          and lives in FIXES.md.
#
# Requires: gh CLI authenticated with repo scope (run: gh auth login).
# Run:      bash create-functional-issues.sh
# Idempotent labels via --force; issues are appended (re-running creates
# duplicates, so run once).
# ---------------------------------------------------------------------------

REPO="varunrout/Frame2Threat"

# --- labels (idempotent) ---------------------------------------------------
gh label create functional        --repo "$REPO" --color 5319e7 --description "Functional completeness gap" --force
gh label create claim-vs-reality  --repo "$REPO" --color b60205 --description "README/story vs built reality" --force
gh label create model-validation  --repo "$REPO" --color d93f0b --description "Baseline/holdout/calibration gap" --force
gh label create end-to-end        --repo "$REPO" --color 0e8a16 --description "Raw-to-decision pipeline gap" --force
gh label create so-what           --repo "$REPO" --color fbca04 --description "Output does not end in a decision" --force
gh label create scope             --repo "$REPO" --color c5def5 --description "Scope coherence" --force
gh label create consistency       --repo "$REPO" --color bfd4f2 --description "Internal consistency" --force
gh label create P0                --repo "$REPO" --color b60205 --description "Priority 0" --force
gh label create P1                --repo "$REPO" --color d93f0b --description "Priority 1" --force
gh label create P2                --repo "$REPO" --color fbca04 --description "Priority 2" --force

# --- issues ----------------------------------------------------------------

gh issue create --repo "$REPO" \
  --title "Remove target leakage from the headline possession model and re-report AUC" \
  --label "model-validation,P0,functional" \
  --body "Claim: Ensemble 0.965 and XGBoost 0.9505 ROC AUC honestly predict possession danger (poss_dangerous).

What is actually there: build_tabular_features feeds max_x_reached and territory_gained, both computed over the full possession, into the model. The primary label poss_dangerous = poss_has_shot OR poss_entered_box, and poss_entered_box is defined by how far up the pitch the ball reached. The label is leaked into the features, so 0.95+ is near-tautological rather than predictive skill.

Fix: either restrict the headline claim to the leakage-free early-prediction (prefix) setting, or rebuild features so nothing computed at or after the danger event enters the model, then re-report. State the leakage explicitly in the README.

Acceptance criteria: full-possession model re-run with a documented feature-timing audit showing no post-outcome feature; headline AUC replaced by the leakage-free figure; SHAP no longer dominated by max_x_reached."

gh issue create --repo "$REPO" \
  --title "Reframe the 360 claim: it adds +0.001 AUC and is absent from the headline models" \
  --label "claim-vs-reality,P0,functional" \
  --body "Claim: the project predicts danger 'from StatsBomb Event Data and 360 Freeze Frames' - 360 spatial context is the differentiator.

What is actually there: RQ1 answer is +0.001 AUC (0.881 vs 0.882) at pass level, and the entire v2/v3 possession headline (0.95-0.965) uses zero 360 features. The selling point contributes nothing to the promoted results.

Fix: reframe the README so the honest finding leads - 360 geometry does not measurably improve prediction here, with an explanation (partial coverage, snapshot not tracking). Stop presenting 360 as the load-bearing contribution.

Acceptance criteria: README headline and abstract no longer imply 360 drives the results; the +0.001 finding is stated as a headline result, not buried in an RQ table."

gh issue create --repo "$REPO" \
  --title "Evaluate the GNN on the held-out test set, not validation" \
  --label "model-validation,P1,functional" \
  --body "Claim: PassFrameGNN 0.841 and RQ6 'GNN vs tabular near-parity (0.841 vs 0.845)'.

What is actually there: notebook 05 prints 'GNN validation metrics' (0.834-0.864); the 3,736 test graphs are built but never scored. RQ6 compares GNN validation AUC against XGBoost test AUC.

Fix: evaluate the GNN on the held-out test split and compare like-for-like against XGBoost test AUC on the same 360 subset.

Acceptance criteria: GNN test-set ROC AUC reported; RQ6 uses test AUC for both models; the '(val)' caveat is removed by making it a real holdout number."

gh issue create --repo "$REPO" \
  --title "Report calibration (ECE) and reliability for the possession danger scores" \
  --label "model-validation,P1,functional" \
  --body "Claim: models produce a usable danger score for tactical/scouting decisions and an early-warning alert.

What is actually there: calibrate_model and calibration_metrics exist, but no ECE or reliability result is reported for the headline poss_dangerous XGB, GRU or ensemble. A score used to trigger alerts must be calibrated.

Fix: report ECE and a reliability diagram for each headline possession model on the test split; apply isotonic/Platt calibration if ECE is material, before ensembling.

Acceptance criteria: ECE reported for XGB, GRU and ensemble on poss_dangerous; ensemble probabilities calibrated (ECE stated) rather than a raw average of two uncalibrated models."

gh issue create --repo "$REPO" \
  --title "Provide a single raw-to-scores pipeline command" \
  --label "end-to-end,P1,functional" \
  --body "Claim: 'Running the pipeline' implies going from raw StatsBomb data to prediction.

What is actually there: no single orchestrator (no Makefile/DVC/main). The pipeline is loose notebook snippets; the frame2threat CLI only scores an already-built possession_sequences.parquet using gitignored .joblib/.pt artefacts. A fresh clone cannot run raw to decision.

Fix: provide one command that runs ingest to parse to labels to features to train to score on a small competition, producing the parquet and models it needs.

Acceptance criteria: a single command runs on a clean checkout with only StatsBomb open data and produces scores end to end; documented and tested on at least one match."

gh issue create --repo "$REPO" \
  --title "Validate the player attribution leaderboard against an external metric" \
  --label "so-what,P1,functional" \
  --body "Claim: RQ5 'Player-level attribution? Yes - 453-player leaderboard, domain-consistent'.

What is actually there: player_attribution_summary produces leave-one-out attribution and a leaderboard, but 'domain-consistent' is asserted with no quantitative check (no correlation to xG/assists/minutes, no ground truth, no significance). The output ends at a table.

Fix: validate the leaderboard against an external signal (per-player xG contribution, chances created per 90) and report the correlation; state which decision the leaderboard drives (e.g. shortlist top-N).

Acceptance criteria: leaderboard correlated to an independent player-value metric with a reported coefficient; each leaderboard framed as a concrete scouting/selection decision."

gh issue create --repo "$REPO" \
  --title "Add majority-class and rule baselines to the possession task" \
  --label "model-validation,P1,functional" \
  --body "Claim: 0.95 possession AUC is a strong model result.

What is actually there: the only comparison for poss_dangerous is XGB vs GRU, both fed the leaky features. There is no majority-class baseline and no trivial spatial rule baseline. A one-line rule max_x >= 102 would score near-perfect, which is what exposes the leakage.

Fix: add a majority-class baseline and a trivial rule baseline (max_x_reached >= 102) to the possession task; show that once leakage is removed the trained model beats them by a meaningful margin.

Acceptance criteria: baseline table for poss_dangerous including majority-class and rule-based rows; trained model's lift over the honest baseline reported on the leakage-free feature set."

gh issue create --repo "$REPO" \
  --title "Validate threat_gain against realised shots and goals" \
  --label "so-what,P2,functional" \
  --body "Claim: threat_gain is an xT proxy label and modelling target.

What is actually there: compute_threat_gain builds a zone-value delta from finite data with low-data zones defaulting to 0, but it is never validated against actual possession value, shot generation or an established xT grid.

Fix: validate the zone-value grid against downstream shots/goals (or a public xT surface) and report the agreement; otherwise demote it to an exploratory feature.

Acceptance criteria: threat_gain zone values correlated with realised shot/goal rate per zone, coefficient reported; or the label is explicitly marked unvalidated and dropped from headline claims."

gh issue create --repo "$REPO" \
  --title "Fix README notebook names and pipeline snippets to match real files" \
  --label "consistency,P2,functional" \
  --body "Claim: the README notebook table and v2 pipeline instructions reference specific notebooks.

What is actually there: README lists and imports notebooks/07_possession_features.ipynb and notebooks/09_possession_team_analysis.ipynb; the repo actually contains 07_possession_baseline.ipynb and 09_possession_eda.ipynb. The documented pipeline points at files that do not exist.

Fix: align README notebook names and the 'Running the pipeline' snippets with the actual filenames.

Acceptance criteria: every notebook referenced in README exists at that path; the v2 snippet block points at real files."

gh issue create --repo "$REPO" \
  --title "Make the Streamlit demo show real predictions or flag random outputs" \
  --label "claim-vs-reality,P1,functional" \
  --body "Claim: Quick start says 'streamlit run src/app/app.py' works without data via demo mode, presented as a demo of the system.

What is actually there: in demo mode the app fabricates pass instances, freeze frames and ranked candidate danger scores with np.random; a reviewer following Quick start sees synthetic numbers, not model output, with only a warning banner.

Fix: make demo mode load the real saved model over a small bundled sample, or clearly label demo outputs as random and not representative of model behaviour.

Acceptance criteria: the app in its default reviewer path shows real model predictions on bundled sample data, or the demo explicitly states the scores are random placeholders."

gh issue create --repo "$REPO" \
  --title "Evaluate pass-option ranking against realised outcomes" \
  --label "so-what,P2,functional" \
  --body "Claim: pass option ranking over visible teammates is a usable output.

What is actually there: ranking.py builds synthetic pass rows to alternative teammates and scores them, but there is no evaluation that recommended options are actually better than the pass played (no realised-outcome comparison, no top-k accuracy). It ends at a ranked list.

Fix: evaluate ranking quality against realised outcomes (did higher-ranked available options correspond to better downstream danger when chosen) and report a decision metric.

Acceptance criteria: ranking evaluated with a stated metric (e.g. top-1 agreement with best realised outcome, or lift over the pass played) on held-out passes."

gh issue create --repo "$REPO" \
  --title "Cut or demote unvalidated branches to a single validated headline task" \
  --label "scope,P2,functional" \
  --body "Claim: three model families across v1/v2/v3 with a 19-experiment registry read as a complete research programme.

What is actually there: the differentiators (360 geometry, GNN) either underperform or are unevaluated on test, while the promoted headline is the leaky possession model. Breadth outruns the depth of validation.

Fix: cut or clearly demote the unvalidated branches; make the leakage-free early-prediction story the spine, fully evaluated, and present the rest as exploratory.

Acceptance criteria: one clearly-scoped, fully-validated headline task (leakage-free early danger forecasting) with baselines, holdout and calibration; secondary branches explicitly marked exploratory."

gh issue create --repo "$REPO" \
  --title "Add confidence intervals and justify the ensemble weighting" \
  --label "model-validation,P2,functional" \
  --body "Claim: single-point AUCs (0.9505, 0.9524, 0.965) support ranking of models.

What is actually there: the ensemble is an equal-weight average of an uncalibrated XGB and GRU; the weight is never tuned or justified, and no confidence interval is given for any AUC on the 2,475-possession test set, so 0.9505 vs 0.9524 vs 0.965 differences are within noise.

Fix: bootstrap confidence intervals for each headline AUC and justify the ensemble weighting; only claim a model is better if the CI supports it.

Acceptance criteria: each headline AUC reported with a bootstrap CI; ensemble weighting selected on validation and shown to beat both members outside CI overlap, or claims softened to comparable."
