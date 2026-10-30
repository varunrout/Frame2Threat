# Reproducibility guide

Python 3.11 or 3.12 (`.python-version` pins 3.11).  Data: StatsBomb Open Data only (public, licence in README).
Everything under `data/` and the `*.joblib` / `*.pt` model files is gitignored and regenerated.

## 1. Install

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch (any torch >= 2 works)
python -m pip install -e ".[dev]"
# or, locked:  poetry install --extras dev
```

`torch-geometric` is **not** needed (only the superseded `src/models/gnn.py`: `pip install -e ".[legacy-gnn]"`).

## 2. Test and lint

```bash
ruff check src tests scripts      # pyflakes gate used by CI
pytest -q                          # 184 tests locally; tests needing gitignored data/models skip on a bare checkout
```

## 3. Acquire data and rebuild features (needs network)

```bash
python scripts/run_pipeline.py --verbose      # StatsBomb Open Data -> data/repro/v1 (pass-level, event-only)
python -m src.data.build_store --verbose      # parquet + SQLite store
```

The processed tables used by the reports (`data/processed/`: events, pass instances, labels, possessions, splits) were built
by the `src/data` modules (see `ARCHITECTURE.md`); the split is match-level, seed 42, stored in `split_manifest.csv`.
**Not re-executed end to end in the 2026-10-30 audit** (it needs the full download); the audit re-scored saved models on the
existing processed data instead.

## 4. Train and reproduce results

```bash
python scripts/train_pass_models.py        # final pass models (progression_beyond_pass_k), refuses to overwrite
python scripts/train_pass_gnn.py           # graph-only PassFrameGNN + paired comparison vs XGBoost
python scripts/run_event_time.py           # event-time LR / XGBoost / GRU, thresholds, reliability
python scripts/run_attribution.py          # cross-fitted retrospective attribution
python scripts/make_evidence_pack.py       # reports/repro scored sample
```

## 5. Verify the saved artefacts match the reports

```bash
python scripts/verify_artefacts.py         # writes reports/repro/artefact_check.md, exit 1 on mismatch
```

## 6. Run the CLI and the app

```bash
frame2threat sample-possession -o poss.json --dangerous 1     # real held-out test possession
frame2threat score-live poss.json --model gru -o scores.csv   # event-time, step t uses events 0..t only
frame2threat score-batch data/processed/events_parsed.parquet --limit 200 -o scores.parquet
python scripts/make_demo_evidence.py                          # figure + CLI transcript in reports/
streamlit run src/app/app.py
```

Errors (bad JSON, schema violations, missing model files) print `error: ...` and exit with status 2.
Without `data/processed` the app shows an explicit **DEMO MODE** banner and random placeholder scores; with data it never does.
