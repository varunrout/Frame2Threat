# Final repository audit (2026-10-30)

Starting point: `main` at e85b163, 154 tests passing.  Earlier rounds had already corrected the label, leak, coordinate and
decoder defects, retrained the GNN and rebuilt attribution; this audit checked those claims against the code and artefacts.

## Verified (executed)

| Check | Evidence |
|---|---|
| Saved models reproduce documented test ROC AUC (pass event-only 0.7337, event+360 0.7380; event-time XGB 0.6649, GRU 0.6636) | `scripts/verify_artefacts.py`, `reports/repro/artefact_check.md` |
| CLI scores real held-out possessions, exports CSV/Parquet, fails clearly | `tests/test_cli.py` (22 tests), `reports/repro/cli_demo.txt` |
| All five app pages render on real data without errors; demo mode is explicit | `tests/test_app_smoke.py` |
| Full suite | 184 passed (154 before) |

## Defects found and fixed in this audit

1. **App Possession Inspector was misleading**: it plotted the retrospective, pre-fix, in-sample possession GRU
   (`gru_poss_dangerous.pt`) as a "danger trajectory" with a 0.5 threshold, and a "player leaderboard" re-attributed in-sample
   on 300 random possessions (superseded method).  Rebuilt on the event-time models, held-out matches by default,
   cross-fitted attribution labelled retrospective, leaderboard with minimum-sample slider and intervals (`src/app/possession_page.py`).
2. `streamlit run src/app/app.py` failed with `No module named 'src'` without an editable install; path bootstrap added.
3. CLI crashed with tracebacks on missing/invalid input, had no schema validation, CSV-only output and no real-data sample loader.
4. 19 undefined-name (F821) and 29 other pyflakes findings (`np` undefined in `gru_possession.py` annotations etc.); CI lint gate
   only checked `E9,F63,F7`.  Now `E9,F` over `src tests scripts`.
5. CI installed no torch (torch tests silently skipped), omitted pyarrow/streamlit; `pyproject.toml` lacked pyarrow and forced
   torch-geometric (unused by the retrained GNN).  `poetry.lock` regenerated.
6. Stale superseded claims left unmarked in `experiment_log.md` (combined RQ answers), `FUNCTIONAL.md`, `FIXES.md`: banners added.

## Status against the acceptance criteria

| Criterion | Status |
|---|---|
| Documented environment installs | PARTIAL: lock regenerated and imports verified in the dev env; clean-venv install see REPRODUCIBILITY note below |
| Unit/integration tests pass | PASS (184) |
| Target and leakage checks pass | PASS (`test_labels`, `test_event_time`, `test_pass_progression`) |
| GNN reproducible, comparable sample/split | PASS from earlier round (EXP-032/033); graph-only GNN 0.595 vs XGB 0.715 on identical 360 subset; GNN not preferred.  Checkpoint is gitignored; regenerate with `scripts/train_pass_gnn.py` |
| Attribution recomputed and audited | PASS (EXP-031); ranking-change audit PARTIAL/BLOCKED for a full count (old table never committed): `reports/attribution/ranking_change_audit.md` |
| Event-time re-evaluated, calibration and false alerts reported | PASS (`reports/event_time/`) |
| CLI on real inputs | PASS |
| App shows real predictions, no misleading placeholders | PASS |
| Model outputs match documented artefacts | PASS for the four checked artefacts; GNN and possession-retrospective models not re-scored here |
| CI passes | NOT VERIFIED: workflow rewritten, not run on GitHub (nothing pushed) |
| Public deployment | NOT DONE (optional) |
| Browser screenshots | NOT DONE: replaced by reproducible `reports/figures/demo/possession_demo.png` |

## Remaining limitations
Open-data scale (99 matches, 15 test), 360 on 52% of passes, event-time AUC about 0.66, alerts fire on 39-48% of safe possessions,
coverage floor 40% on a bare checkout, black/mypy advisory only.
