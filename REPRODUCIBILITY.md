# Reproducibility guide

Python 3.11 or 3.12 (`.python-version` pins 3.11).  Data: StatsBomb Open Data only (public, licence in README).
Research data (`data/`) is gitignored and regenerated. Seven current model files are committed (see
[models/README.md](models/README.md)); other `*.joblib` / `*.pt` / `*.ubj` files are gitignored and regenerated.

## 1. Install

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
# or, locked:  poetry install --extras dev
```

`torch-geometric` is **not** needed (only the superseded `src/models/gnn.py`: `pip install -e ".[legacy-gnn]"`).

## 2. Test and lint

```bash
make check                         # ruff, black, mypy, pytest: the gates CI runs (hosted CI is unavailable)
pytest -q                          # tests needing gitignored data skip on a bare checkout
```

## 3. Acquire data and rebuild features (needs network)

For the maintained event-time task, use the fixed research manifest:

```bash
python scripts/prepare_event_time.py --output-dir data/processed
python scripts/run_event_time.py --data-dir data/processed --threads 2
python scripts/evaluate_event_time.py --data-dir data/processed
```

`prepare_event_time.py --small --output-dir data/repro/small` builds nine genuine matches
for execution checks. It is not a headline reproduction. Preparation uses cached raw
events when available; otherwise StatsBomb downloads require network access. Choose a
new output directory if research inputs already exist. Training can be isolated using
`--output-dir data/repro/new_run`; set `FRAME2THREAT_ROOT` to that run root for CLI scoring.
Model metadata records input hashes, package versions, seed and schema. Fixed seeds
do not guarantee identical numbers across different numerical-library versions.
New XGBoost runs write native `eventtime_xgboost.ubj` for portable loading; the CLI
prefers it and retains `.joblib` compatibility for old runs. The repair verified
predictions across XGBoost 3.0.5 and 3.4.1. A clean CPU installation was also tested;
its complete dependency list is `reports/repro/repair_retrain/fresh_environment.txt`.

The following older commands build the separate pass-level research store:

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
