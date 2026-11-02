# Frame2Threat repair report — 2 November 2026

Starting revision: `9aa2f05`, branch `main`, clean working tree.
No remote changes authorised or made. Local data and model binaries exist but are gitignored.

## Repair checklist

- [x] Inspect repository, previous audits, working tree and open issues.
- [x] Confirm remote CI: latest run 37188379809 failed on 4 October 2026,
  at remote revision 49094fc; no job steps or runner allocated. Local changes have not run remotely.
- [x] Reproduce tests, lint, formatting, packaging and synthetic pipelines.
- [x] Repair input validation, sample fidelity and training/inference contracts.
- [x] Rebuild event-time training with isolated outputs and provenance.
- [x] Verify genuine held-out predictions, uncertainty and calibration.
- [x] Provide reproducible data preparation and fresh-install checks (dependency install blocked; wheel verified).
- [x] Reframe README around maintained event-time research; classify other branches.
- [ ] Review final diff, record results and exact outstanding blockers.

## Repairs implemented

1. **Inference fidelity:** replaced rounded possession-sequence export and cached timing
   with direct parsed-event export. Starting minute, timestamps, actor-frame coordinates,
   ownership, pressure and missing values now round-trip through the training features.
   Missing pass ends no longer become artificial zero coordinates. Pass lengths divided
   by 60 may exceed one (a long pass); validation no longer rejects those genuine values.
2. **Input failures:** reject non-integral event IDs, non-finite/negative times, fractional
   ownership/pressure, invalid thresholds and missing batch feature columns. Validate
   XGBoost feature order and GRU numeric width before inference. Reject invalid stream
   probabilities; empty batch selections return a correctly shaped empty result.
3. **Reproducibility:** added the 99-match manifest and `prepare_event_time.py`, including
   a nine-match genuine-data mode. Added separate data/output paths to training, two-thread
   resource control, input SHA256 hashes, dependency versions, seed and schema metadata.
   Training rebuilds arrays instead of trusting an unversioned pickle; split manifests
   must assign each match exactly once. Fallback timing now uses training matches only.
4. **Evaluation:** added `evaluate_event_time.py` to re-score saved models from rebuilt
   arrays, report a training-prevalence baseline and equal-width ten-bin ECE in addition
   to ROC AUC, AP, Brier and log loss. No calibration was fitted on the test set.
5. **Engineering:** added input round-trip and prediction-prefix regression tests;
   formatted the Python tree (13 previously nonconforming files). Formatting is now a
   CI gate; added wheel installation and outside-checkout CLI checks to CI. Raised the
   coverage floor from 40% to 50% after measuring 53.16% in a data-free checkout.
6. **Presentation:** rewrote README around the maintained event-time question, corrected
   dependency/coverage/notebook instructions and marked the architecture review historical.
   Prior README is retained under `reports/audit/README_before_2026-11_repair.md`.

## Executed checks and results

| Command/check | Actual result |
|---|---|
| Baseline `pytest --cov=src --cov-fail-under=40 -q` | 184 passed, 17 warnings; 61.56% coverage; 519.25 s |
| Repaired full suite (plugin autoload disabled, pytest-cov explicitly enabled) | 198 passed, 17 warnings; 61.50% coverage; 575.48 s |
| Final full suite after type/loader/parser repairs, `--cov-fail-under=50` | 202 passed, 17 warnings; 62.76% coverage; 287.49 s |
| Data-free checkout, same installed dependencies | 192 passed, 6 skipped, 7 warnings; 53.16% coverage; 283.94 s |
| Input/schema/leakage regression module | 16 passed; also included in the final 202-test full suite |
| `ruff check src tests scripts` | PASS |
| `black --check src tests scripts` | PASS, 102 files conform |
| `pip wheel . --no-deps --no-build-isolation` | PASS, wheel produced |
| Installed wheel outside checkout, isolated site-packages with shared dependencies | PASS: imported installed `src`, CLI help, both genuine model scoring paths |
| `build_store --synthetic --output-dir data/store/repair_smoke` | Completed parquet/SQLite build |
| `run_pipeline.py --smoke --synthetic --output-dir data/repro/repair_smoke` | Completed synthetic training/scoring; not football evidence |
| `prepare_event_time.py --small --output-dir data/repro/repair_small` | 9 genuine cached matches; 36,314 events; 1,479 possessions |
| Full `prepare_event_time.py --output-dir data/repro/repair_full_inputs` | 99 genuine cached matches; 373,682 events; 17,065 possessions |
| Rebuilt inputs versus training inputs | All parsed event values match by event UUID; all possession keys and split assignments match |
| `verify_artefacts.py` | Four saved models reproduce documented AUC within 0.0005 |
| `evaluate_event_time.py` | Rebuilt held-out arrays; finite valid predictions; results below |
| Both `score-live` models on regenerated genuine example | PASS; CSVs in reports/repro/repair_demo_*.csv |
| Fresh venv `pip install -e '.[dev]'` | BLOCKED: initial DNS failure; network-enabled retry exhausted disk space during dependency installation. Partial environment removed; not a successful fresh install |
| `mypy src` after repairs and YAML stubs | PASS: no issues in 61 source files; initially 20 errors in 14 files |
| `poetry check --lock` | PASS; only existing SPDX-format deprecation warning |

The available interpreter is Python 3.13.3. CI targets 3.11/3.12, which were not installed
on this machine. Tests using local models are a different verification from a clean clone.
The 33% threshold in the previous README was stale: CI already required 40% at the start.

Type-check repairs include explicitly narrowing missing pass-end zones, removing stale
ignores, using the public matplotlib colormap accessor, handling malformed/infinite
boolean input and declaring YAML type stubs in dev dependencies. Possession-GRU epoch
loss now divides by observed batch rows (not dataset length when rows are dropped);
empty loaders fail clearly. Regression tests cover both cases. Formatting and type
checking are now mandatory CI steps rather than advisory steps.

## Independently re-evaluated existing models

15 test matches, 43,541 eligible snapshots. Training snapshot prevalence 0.36198;
test prevalence 0.40438. No test fitting or threshold selection was performed.

| Model | All-snapshot AUC | AP | Brier | Log loss | ECE (10 bins) | One-per-possession AUC |
|---|---:|---:|---:|---:|---:|---:|
| Training-prevalence constant | 0.5000 | 0.4044 | 0.2427 | 0.6786 | 0.0424 | 0.5000 |
| XGBoost | 0.6649 | 0.5669 | 0.2301 | 0.6523 | 0.0926 | 0.7362 |
| GRU | 0.6636 | 0.5721 | 0.2239 | 0.6423 | 0.0390 | 0.7390 |

Machine-readable results: `reports/repro/repair_evaluation.json`. Pass artefact AUCs:
event-only 0.7337, event+360 0.7380. These are **re-scored saved models**, not fresh
training results. Historical 0.668/0.670 and 0.749/0.756 are superseded and were not targets.

## Genuine football walkthrough

France, match 3857266, possession 121, minute 68, second-half throw-in: 27 recorded
events. Regenerated `reports/repro/demo_possession.json` is from a held-out match.
XGBoost first alerts at step 1 (Pass); GRU at step 8 (Carry). The final pre-outcome
step is 18, with probabilities 0.6540 / 0.5953. Step 19 realises the operational
danger target and later steps have no forecast. This illustrates a model disagreement
and an early alert, not proof of tactical usefulness or a live deployment.

## Full retraining completed

Executed `python -u scripts/run_event_time.py --output-dir data/repro/repair_event_time
--n-boot 500 --threads 2` on all 99 matches. Two GRU learning rates were compared on
validation only: 0.003 (10 epochs, best validation log loss 0.6127) and 0.001 (9 epochs,
0.6146). XGBoost selected depth 5, child weight 20, 55 trees on validation. No test-driven
retuning was performed. The first attempt was restarted to enforce the CPU limit for
XGBoost as well as torch.

| Freshly trained model | All-snapshot AUC (95% match-bootstrap CI) | One-per-possession AUC | AP | Brier | Log loss | ECE |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | 0.6653 [0.6471, 0.6810] | 0.7361 | 0.5675 | 0.2233 | 0.6372 | 0.0394 |
| GRU | 0.6636 [0.6421, 0.6865] | 0.7390 | 0.5721 | 0.2239 | 0.6423 | 0.0390 |

Fresh outputs, calibration tables, bootstrap intervals and metadata are retained in
`reports/repro/repair_retrain/`. Trained binaries are in the ignored directory
`data/repro/repair_event_time/models/` (GRU 77,919 bytes; XGBoost 223,192 bytes).
The reproducible trainer is provided instead of committing pickle-based binaries.
Set `FRAME2THREAT_ROOT` to `data/repro/repair_event_time` to load this run.
The original root models and their historical reports are intentionally preserved.

Environment: numpy 2.3.2, pandas 2.3.2, torch 2.9.0 CPU, XGBoost 3.0.5,
scikit-learn 1.7.1. Input hashes and feature names are in the run metadata. The earlier
serialized XGBoost emitted a cross-version loading warning; its ranking reproduced but
its calibration differs substantially from fresh training. Do not reuse an AUC match
as proof of probability compatibility across dependency versions.

The fresh XGBoost also successfully scored the genuine example: first alert at step 6,
last pre-outcome probability 0.5240. Its CSV is separate from the original model replay.
Fresh GRU replay also passed: first alert at step 8, final pre-outcome probability 0.5953.

## Scientific boundaries

- Match splits and existing prefix/coordinate/label regression tests pass. Training
  selection uses validation log loss; thresholds use validation prevalence quantiles.
- No causal claim follows from predictive attribution. Pass endpoints are known only
  after the pass, and line breaks are geometric labels, not independently annotated truth.
- Event-time danger includes incomplete passes ending in the box. Possession labels
  use a different event-location definition. Reports must not conflate the two.
- Long possessions contribute more snapshot rows. Safe terminal events are omitted
  from the primary estimand and included in the existing sensitivity analysis.
- Existing confidence intervals resample matches, not independent events. They do not
  capture repeated test inspection, cohort selection or future-league generalisation.
- No external competition/season holdout, production feed validation, latency study,
  market-linked betting evaluation or independent tactical usefulness study was added.

## External blockers

Latest remote run: https://github.com/varunrout/Frame2Threat/actions/runs/37188379809
at SHA `49094fc7b36e920f4c5324cd28a83cc65e70d465`. The GitHub check annotation says:
"The job was not started because your account is locked due to a billing issue."
No runner or job steps ran. This is an account-level blocker, not evidence of a
test failure. Local revision `9aa2f05` and this repair have not run remotely. No push
was made, no issue was closed, and CI is **not green** for this repair.

## Initial findings

The previous absence of event-time binaries does not describe this working directory:
both files exist locally, but a fresh clone still requires training. The latest local
workflow already requires torch and 40% coverage, whereas README still says 33% and
calls torch optional. Historical 0.668/0.670 event-time AUCs were superseded by the
same-moment correction (reported 0.665/0.664). These are not new reproduction results.

Concrete defects repaired: sample JSON omitted starting minute; omitted pass ends
become zero instead of missing; validation admits infinite/negative timestamps,
fractional event IDs and invalid ownership/pressure values; batch schema validation
omits columns used by feature generation; training accepts overlapping split manifests
and an unversioned cached feature pickle; fallback timing median includes held-out data.

## Remote issue review

Open issues inspected: #6, #9, #13, #14, #17, #18, #20, #21, #22, #23, #25, #26, #27.
No issues closed. #9 deployment and #14 Docker remain separate deliverables; #6
competition holdout remains scientifically relevant. #13 model cards and #17 corrected
GNN evaluation have local evidence, pending verification. #20 attribution and #25 option
ranking lack independent outcome validation. #22 threat gain must remain unvalidated.
#18/#21/#27 require careful separation of retrospective and event-time evaluation.
#23 notebook paths and #26 headline scope are addressed locally in README. #22 is
addressed by explicitly demoting threat gain. These changes are not on GitHub yet,
so their remote acceptance criteria have not been marked complete.
