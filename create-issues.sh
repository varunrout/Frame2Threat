#!/usr/bin/env bash
set -euo pipefail

# create-issues.sh
# Creates the Frame2Threat finish-off issues (and their labels) on GitHub.
#
# What it does:
#   1. Creates an idempotent label set (--force, so re-running is safe).
#   2. Creates one issue per fix in the FIXES.md priority table.
#
# Prerequisites:
#   - GitHub CLI installed: https://cli.github.com/
#   - Authenticated: run 'gh auth login' first.
#   - Write access to varunrout/Frame2Threat.
#
# Run:
#   chmod +x create-issues.sh
#   ./create-issues.sh

REPO="varunrout/Frame2Threat"

echo "Creating labels on $REPO..."
for label in ci data security testing docs packaging feature P0 P1 P2; do
  gh label create "$label" --repo "$REPO" --force
done

echo "Creating issues on $REPO..."

gh issue create --repo "$REPO" \
  --title "Add GitHub Actions CI (ruff, black, mypy, pytest, build)" \
  --label "ci,P0" \
  --body "Problem: the repo has no CI, so the '69 tests pass' and 'it runs' claims are unproven on any clone but the author's. Fix: add .github/workflows/ci.yml running ruff, black --check, mypy, and pytest with coverage under Poetry on push and PR, plus a package build check, installing from the lock file. Acceptance criteria: green CI on main and on a PR; failing lint or a failing test blocks the run; coverage is reported."

gh issue create --repo "$REPO" \
  --title "Commit a reproducible slice: scored possessions, model cards, calibration plots" \
  --label "feature,P0" \
  --body "Problem: model artefacts (.joblib, .pt) and all data are gitignored, so no headline metric reproduces from a fresh clone. Fix: commit a small slice under reports/repro/ containing sample scored possessions, per-model model cards, and one calibration plot per model. Acceptance criteria: a cloner can inspect scored output and calibration in reports/repro/ without fetching StatsBomb or training anything."

gh issue create --repo "$REPO" \
  --title "Add one-command fetch-and-reproduce script for v1" \
  --label "feature,P0" \
  --body "Problem: the headline AUCs are unverifiable end to end and there is no single command that rebuilds v1. Fix: add scripts/reproduce_v1.sh (and a make reproduce target) that fetches StatsBomb data via statsbombpy, builds the store, trains v1 and prints the pass-level AUC; wire a reduced smoke version into CI. Acceptance criteria: one command from a clean venv reproduces the v1 pass-level number within tolerance, and CI runs a reduced smoke reproduce."

gh issue create --repo "$REPO" \
  --title "Persist data to a parquet + SQLite store with an ingestion entry point" \
  --label "data,P1" \
  --body "Problem: StatsBomb JSON is fetched and processed into transient in-memory tables with no persisted, queryable store, which is why nothing reproduces without a full rebuild. Fix: persist wide feature/label matrices to parquet partitioned by competition and season, and match/competition inventory plus run provenance to SQLite; add src/data/build_store.py as the single ingestion entry point; keep raw and derived data gitignored. Acceptance criteria: python -m src.data.build_store populates parquet and SQLite, downstream code reads the store rather than ad hoc frames, and raw data stays untracked."

gh issue create --repo "$REPO" \
  --title "Defend possession AUC with a per-competition holdout and split diagram" \
  --label "feature,P1" \
  --body "Problem: 0.95+ possession AUC on ~100 matches invites leakage suspicion that the README does not rebut. Fix: add a per-competition (or per-season) holdout alongside the match-level split, commit an explicit split diagram, and report the holdout AUC next to the random-split one. Acceptance criteria: README shows the split diagram and a per-competition holdout AUC, and the gap between random-split and competition-holdout is stated plainly."

gh issue create --repo "$REPO" \
  --title "Reframe the 360 freeze-frame contribution honestly" \
  --label "docs,P1" \
  --body "Problem: 360 features add only +0.001 AUC yet 360 is sold as the central premise. Fix: state the marginal lift up front, isolate where 360 helps on the covered subset (line-break geometry), and stop implying 360 is the main driver. Acceptance criteria: README and final report state the +0.001 headline lift and show the covered-subset breakdown, with no claim implying 360 is the primary signal."

gh issue create --repo "$REPO" \
  --title "Remove backdated log and add a first-person design-decisions note" \
  --label "docs,P1" \
  --body "Problem: the June-2025 experiment log is backdated (commits are March 2026) and Copilot-authored commit messages make the work look agent-generated and hard to defend. Fix: remove or clearly relabel the backdated log and add reports/design_decisions.md, a first-person note defending label choices, split strategy, model selection and the 360 finding. Acceptance criteria: no backdated dates are presented as a real timeline, and a first-person design-decisions note covers the five key choices."

gh issue create --repo "$REPO" \
  --title "Deploy the Streamlit app and add a live link" \
  --label "feature,P1" \
  --body "Problem: the Streamlit demo has no live link, so the interactive story is invisible without local setup. Fix: deploy the app in demo mode (no data required) to a public host and link it from the README. Acceptance criteria: the README carries a working live demo URL that loads in demo mode without local data."

gh issue create --repo "$REPO" \
  --title "Pin dependencies, commit lock file, add pre-commit config" \
  --label "packaging,P1" \
  --body "Problem: reproducibility depends on unpinned or unverified dependency resolution. Fix: commit poetry.lock, pin the Python version, add a pre-commit config (ruff, black, end-of-file, trailing-whitespace) and document install. Acceptance criteria: poetry install from the lock file is deterministic, pre-commit run --all-files passes, and install steps are in the README."

gh issue create --repo "$REPO" \
  --title "Enforce a coverage floor and test the store and reproduce path" \
  --label "testing,P1" \
  --body "Problem: 69 tests exist but there is no enforced coverage floor and no proof they gate merges. Fix: run pytest with --cov in CI, set a coverage threshold, and add tests for the new store ingestion and reproduce path. Acceptance criteria: CI fails below the coverage floor, and store ingestion and the reproduce script have tests."

gh issue create --repo "$REPO" \
  --title "Update README: CI badge, reproduce section, honest limitations" \
  --label "docs,P1" \
  --body "Problem: the README states metrics but lacks a CI badge, a reproduce section and an upfront caveat block. Fix: add a CI badge, a 'Reproduce in one command' section, and a short honest-limitations block covering sample size, the 360 lift and the agent-assisted build. Acceptance criteria: the README top shows the CI badge and the reproduce and limitations sections are present and accurate."

gh issue create --repo "$REPO" \
  --title "Write per-model model cards" \
  --label "docs,P2" \
  --body "Problem: there are no model cards, so model scope, training data and known failure modes are undocumented. Fix: write a one-page model card per model family (pass XGB, possession GRU, ensemble, early-forecast) under reports/model_cards/. Acceptance criteria: each shipped model has a card covering data, metrics, intended use and limitations."

gh issue create --repo "$REPO" \
  --title "Add a Dockerfile for reproducible runs" \
  --label "packaging,P2" \
  --body "Problem: there is no Dockerfile, so the 'it runs' claim depends on the author's environment. Fix: add a Dockerfile that installs from the lock file and runs the CLI/demo, and document docker run. Acceptance criteria: docker build succeeds and the container runs the CLI on the sample slice."

echo "Done."