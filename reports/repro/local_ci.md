# Local CI run

GitHub Actions cannot run for this repository at the moment (account billing issue), so the steps
of `.github/workflows/ci.yml` were executed by hand on a developer machine. This is **not** a
substitute for a hosted run: it covers one Python version (Python 3.12.7, Windows) rather than the 3.11/3.12
Linux matrix.

Run date: 2026-11-04

| Step | Command | Result |
|---|---|---|
| Lint | `ruff check src tests scripts` | pass |
| Format | `black --check src tests scripts` | pass (107 files) |
| Types | `mypy src` | pass (64 files) after fixing 3 errors in `src/app/app.py` found by the first run |
| Tests | `pytest --cov=src --cov-fail-under=50` | 219 passed, coverage 63.5% (re-run after the documentation pass, commit `ad903b1`) |
| Clean checkout | `git archive HEAD`, then `pytest` in the extracted folder (tracked files only, no data) | 215 passed, 4 skipped (real-data tests) |
| CLI smoke | `frame2threat --help`, synthetic store, synthetic pipeline smoke | pass |
| Wheel | `pip wheel . --no-deps`, install in a fresh venv outside the checkout, `frame2threat --help` | pass |
| Lock | `poetry check --lock` | pass (one deprecation warning for `project.license`) |

To repeat: run the commands in the table from the repository root.
