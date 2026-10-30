> **Historical fix list.**  Current status of every item: `reports/audit/final_audit_2026-10.md`.

# Frame2Threat - Finish-Off Plan

Football | Substantial (80% complete) | Python, XGBoost, PyTorch, scikit-learn, pandas, SQLite, GitHub Actions

## Where it stands

Engineering hygiene is unusually good for a portfolio analytics repo. CI exists and runs lint, format, type-check, a 74-test suite with a coverage floor, two smoke builds and a package build. A parquet+SQLite data store, a one-command pipeline, a CLI entry point, pre-commit hooks, and a committed evidence pack with model cards and SHA256 hashes are all present. The remaining work is closing the reproducibility gap (gitignored artefacts, data download required), tightening CI (low coverage floor, torch not installed so GRU/GNN untested), and finishing the honesty-driven remediation the timing audit already identified.

## Priority fixes

| ID | Priority | Area | Problem | Fix | Acceptance criteria | Effort |
|----|----------|------|---------|-----|---------------------|--------|
| H1 | High | Feature completion | Full-possession v2 model leaks target via max_x/box features; the 0.965 score is retrospective, disclosed but still headlined | Build an event-time-cutoff feature path so no feature at/after the first shot or box-entry enters the row; re-report | A leakage-free full-possession metric exists and README leads with it or with v3 | L |
| H2 | High | Reproducibility | Trained artefacts gitignored; headline numbers not rebuildable from a clone | Commit small artefacts or a cached fixture, or a `make reproduce-headline` that regenerates results_summary.json from committed inputs | Fresh clone reproduces at least one headline metric without external download | M |
| H3 | High | CI coverage | Torch not installed in CI, so GRU/GNN and early_evaluation paths are never exercised | Add a CI job (or extra install) with torch + torch-geometric, or mark and test lightweight stubs | CI runs at least one test touching GRU and GNN code | M |
| M1 | Medium | Tests | Coverage floor is only 24%; large modules (evaluation, visualization, app) likely untested | Raise floor incrementally toward 50%, add tests for possession_attribution, early_evaluation, calibration | `--cov-fail-under=45` passes | M |
| M2 | Medium | Data layer | SQLite store holds only metadata/partitions; analytical tables are parquet-only (correct), but there is no SQLite-preferred query path for the small tables | Keep partitioned parquet for pass instances (right choice); add SQLite tables for splits manifest and run provenance queries | A reviewer can query provenance and split assignment from SQLite alone | S |
| M3 | Medium | Demo app | Streamlit app scores random synthetic data with placeholder labels | Ship a small real scored fixture and load it in demo mode; keep synthetic only as fallback | App shows genuine model output on committed sample | M |
| M4 | Medium | Docs | README lists 74 tests and specific AUCs inline; drift risk as code changes | Generate the results table and test count from artefacts in CI; fail on drift | README numbers are generated, not hand-maintained | S |
| L1 | Low | Secrets | No secrets in repo (StatsBomb open data needs none); no secret-scanning guard | Add a gitleaks/pre-commit secret scan for future-proofing | Secret-scan hook runs in pre-commit and CI | S |
| L2 | Low | Packaging | Package builds but console script imports `src.app.cli`; `src` is a non-namespaced top-level package | Rename to a `frame2threat` package or configure src-layout properly | `pip install .` then `frame2threat --help` works outside repo root | M |
| L3 | Low | CI deps | CI installs a hand-maintained pip list duplicating pyproject deps | Install from `.[dev]` or lock file instead of a manual list | CI dependency list is single-sourced | S |
| D1 | Done | CI/CD | CI pipeline with lint, tests, coverage floor, smoke builds and package build | Implemented in `.github/workflows/ci.yml` | - | - |
| D2 | Done | Data layer | Parquet + SQLite store builder with synthetic smoke mode | Implemented in `src/data/build_store.py` | - | - |
| D3 | Done | Reproducibility | One-command raw-to-scores pipeline and Makefile targets | Implemented in `scripts/run_pipeline.py`, `Makefile` | - | - |
| D4 | Done | Evidence | Committed scored sample, model cards and SHA256 hashes | Present under `reports/repro/` | - | - |
| D5 | Done | Docs honesty | Feature timing/leakage audit and design-decisions note | `reports/feature_timing_audit.md`, `reports/design_decisions.md` | - | - |

## Data layer design

Keep the current split: large analytical tables (pass instances, possession sequences, split parquets) stay as partitioned parquet under `data/store/` and `data/processed/`; small query and provenance data (inventory, parquet partition index, run rows, and ideally the split manifest) live in SQLite at `data/store/metadata.sqlite`. This is the SQLite-preferred-else-parquet pattern done correctly: SQLite for indexable metadata a reviewer can query without loading dataframes, parquet for the columnar bulk. The gap is only that the split manifest and provenance are not yet fully queryable from SQLite; add those tables.

## CI/CD design

Current pipeline is sound. Target additions: (1) single-source dependencies from pyproject/lock rather than a manual pip list; (2) a second matrix leg or job that installs torch and runs the GRU/GNN tests; (3) raise the coverage floor on a schedule; (4) a docs-drift check that regenerates the README results table and test count from `models/results_summary.json` and pytest collection, failing on mismatch; (5) add gitleaks. Keep the smoke store build, smoke pipeline and package build steps.

## Definition of done

- One headline metric reproduces from a fresh clone with no external download.
- The full-possession model is either leakage-free or the README leads unambiguously with v3.
- CI installs torch and exercises GRU and GNN; coverage floor at least 45%.
- README numbers and test count are generated from artefacts, not hand-typed.
- Demo app shows real scored output on a committed fixture.