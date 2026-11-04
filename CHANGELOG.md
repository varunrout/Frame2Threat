# Changelog

Dates are commit dates. There are no tagged releases yet.

## 2026-11-04: documentation and engineering pass

- README rewritten as a product front page: status, quotable results, quickstart, repository map, roadmap.
- Added indexes: `scripts/README.md`, `models/README.md` (with hashes and AUCs), `reports/README.md`, `docs/README.md`,
  `reports/audit/README.md`; added `reports/CLAIMS.md` and `reports/dataset_card.md`.
- Superseded folders now carry notes saying why they must not be cited.
- Removed the unused Possession page; its pure helpers moved to `src/app/possession_data.py`.
- Makefile reduced to targets that work (`make help`, `dashboard`, `check`, `verify`, `demo`).
- `scripts/prepare_event_time.py` now also writes `event_time_arrays.pkl`, so a prepared directory can be inspected
  in the dashboard without retraining.
- Fixed three mypy errors in `src/app/app.py`. Hosted CI is unavailable (billing); the CI steps were run locally
  and recorded in `reports/repro/local_ci.md`.
- `scripts/verify_artefacts.py` now also writes `reports/repro/artefact_check.json` with git SHA, package versions and
  model hashes (`src/evaluation/provenance.py`); added seeded-determinism tests for the GRU and XGBoost training paths
  (`tests/test_reproducibility.py`).
- Added `scripts/error_analysis.py` and its report `reports/event_time/error_analysis.md`: AUC by competition, possession
  origin, period and minute band (committed models re-scored, no retraining).
- Removed orphan code (no importers, 0% coverage): `src/models/{gnn,hybrid,ranking,train_early_models,gru_train_script}.py`,
  `src/features/sequence_context.py`, `src/evaluation/ablations.py`; the README row for pass-option ranking went with
  `ranking.py`. Deleted three stale JSON files in `models/`. Moved `ARCHITECTURE.md` to
  `reports/audit/ARCHITECTURE_REVIEW.md`. All remain in git history.
- Added `tests/test_repo_integrity.py` (committed split, markdown links, model hashes, documented AUCs).

## 2026-11-03

- Dashboard polish: opens on the Possession Inspector, cached scoring, cleaner sidebar, enhanced possession page.
- Repository cleanup: notebooks, issue-creation scripts and process notes removed (still in git history up to `15e8ec4`).
- Seven current model artefacts committed (2.5 MB).

## 2026-11-02

- CI gates fixed (formatting, type checking, coverage); wheel install test added.
- November repair report and fresh retrain of the event-time models.

## 2026-10-30

- Repository audit: hardened CLI with input validation, Possession Inspector rebuilt on event-time models,
  artefact verification added, stale claims marked.

## 2026-10-16

- Pass models trained on `progression_beyond_pass_k`; PassFrameGNN retrained with a fair comparison.

## 2026-10-05

- Fraction-based early forecasting retired in favour of the event-time formulation. Earlier results withdrawn;
  see `reports/audit/README.md`.
