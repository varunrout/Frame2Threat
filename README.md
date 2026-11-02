# Frame2Threat

**Can the events of an unfolding possession warn an analyst of a later shot or penalty-area entry?**

Frame2Threat studies this question using StatsBomb Open Data. Its maintained task is event-time danger prediction: after each observed event, estimate whether the possessing team will produce a shot or box entry later in that possession. Current held-out discrimination is modest (ROC AUC about 0.66); much of the signal comes from ball position. This is a research prototype for tactical review, not a deployed live feed or a betting model.

## Football problem

A sequence of passes and carries can move an attack toward goal while pressure and possession origin provide context. The model asks whether the sequence so far is likely to develop into danger. An analyst can replay the changing score alongside video and inspect where the model first raises an alert. Scores are neither causal measures of player quality nor recommendations to place bets.

## What is predicted

At completed event `t`, before any qualifying danger event, predict:

`P(shot or box entry in later events of this possession | observed events 0..t)`.

A possessing-team shot, event located in the box, or pass ending in the box realises the outcome; scoring then stops. **Even an incomplete pass ending in the box counts under this operational definition.** This is an attempted-entry proxy, not exclusively controlled entry. Pass end locations are available only after the event completes. The task does not predict pass choice before the pass is played.

Full-possession XGBoost/GRU models use completed summaries and are **retrospective classification only**. Their high AUCs are not early-warning evidence. Historical batch scoring, causal event-by-event replay, and production live deployment are distinct: only the first two are implemented.

## Data and architecture

The fixed research cohort contains 99 matches: 64 World Cup 2022 matches and 35 Barcelona La Liga 2020/21 matches. There are 15 held-out test matches. Partial 360 freeze frames cover 26,515 of 50,688 passes (52.3%); they are not continuous tracking. The committed [match manifest](configs/research_split_manifest.csv) fixes train/validation/test membership.

`StatsBomb events -> parse -> possession arrays -> causal features -> XGBoost / GRU -> event probabilities`

The CLI and training share `src/features/event_time.py`. Opponent coordinates are transformed into the possession team's frame for the GRU; opponent events cannot realise attacking danger. JSON coordinates are in the **acting team's** frame and `own_team` identifies which events require transformation. Missing coordinates are JSON `null`, not zero. Model files are generated locally and ignored by Git.

New training runs save native XGBoost `.ubj` files for portable inference, alongside
legacy `.joblib` files. The loader prefers the native format. Both models' genuine
replay probabilities were verified across the research and fresh-install environments
to within 0.000001; this check matters because matching AUC alone does not establish
probability compatibility.

## Evaluation

The saved pre-repair models were independently reloaded and re-scored in this repair: XGBoost **0.6649**, GRU **0.6636**, all eligible held-out snapshots. See [artefact verification](reports/repro/artefact_check.md). Full retraining results and environment differences are recorded in [REPAIR_REPORT.md](REPAIR_REPORT.md); do not equate re-scoring with retraining.

The full November repair retrain reports (500 match-bootstrap resamples; complete outputs in
[repair retraining results](reports/repro/repair_retrain/event_time_results.md)):

| Model | All-snapshot ROC AUC (match-bootstrap 95% CI) | One snapshot per possession |
|---|---|---|
| Ball x/y logistic baseline | 0.642 [0.622, 0.661] | 0.695 |
| Events-so-far logistic baseline | 0.561 [0.523, 0.603] | 0.608 |
| Start-context XGBoost | 0.590 | 0.598 |
| Event-time XGBoost | 0.665 [0.647, 0.681] | 0.736 |
| Event-time GRU | 0.664 [0.642, 0.687] | 0.739 |

Hyperparameters and alert thresholds use validation only. Match-level bootstrap accounts for clustered observations. All-snapshot results weight long possessions more heavily; the one-per-possession estimate and 20 repeat draws expose this sensitivity. Terminal safe snapshots are excluded from primary training/evaluation, with their inclusion reported as a sensitivity analysis. At deployment the final safe event is not known in advance, so this sampling distinction matters.

The fresh retrain gives XGBoost/GRU AUC 0.6653/0.6636, Brier 0.2233/0.2239 and ten-bin ECE 0.0394/0.0390. Calibration remains imperfect. These scores are not validated betting probabilities. See [fresh results, log loss, reliability and false-alert analysis](reports/repro/repair_retrain/event_time_results.md); the [earlier experiment](reports/event_time/event_time_results.md) remains archived separately. Random match holdout does not establish future-season or unseen-competition generalisation.

## Reproduce

Use Python 3.11 or 3.12 for the CI configuration. Installation needs network access; torch is a declared dependency. The legacy graph implementation alone requires the `legacy-gnn` extra.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=50
python -m ruff check src tests scripts
frame2threat --help
```

Offline **synthetic engineering smoke tests** (not football model evidence):

```bash
python -m src.data.build_store --synthetic --output-dir data/store/smoke
python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke
```

Build a small **genuine football** dataset (nine fixed matches; downloads missing raw event files):

```bash
python scripts/prepare_event_time.py --small --output-dir data/repro/small
python scripts/run_event_time.py --data-dir data/repro/small --output-dir data/repro/small_run --n-boot 100 --threads 2
```

That small run validates execution, not the headline metrics. For the full experiment, use the full fixed cohort:

```bash
python scripts/prepare_event_time.py --output-dir data/processed
python scripts/run_event_time.py --data-dir data/processed --threads 2
frame2threat score-live reports/repro/demo_possession.json --model xgb -o scores.csv
frame2threat score-live reports/repro/demo_possession.json --model gru -o scores_gru.csv
```

Preparation refuses to overwrite research inputs. If `data/processed` already exists, choose a new directory and pass it to training. Training writes models and reports beneath `--output-dir` (default repository root); use a separate output directory to preserve an existing experiment. Run the CLI from that run directory or set `FRAME2THREAT_ROOT` to it. Full training includes validation model selection and bootstrap evaluation and takes materially longer than the smoke tests. Dependency versions, input hashes and feature schema are saved in model metadata.

The CLI also supports `sample-possession` (requires processed football data) and `score-batch` (requires parsed events and both models). [Reproducibility details](REPRODUCIBILITY.md) describe optional research branches. Pickle/joblib checkpoints should only be loaded from trusted sources.

## Genuine match example

[demo_possession.json](reports/repro/demo_possession.json) is a held-out France possession, match 3857266, possession 121, beginning with a throw-in in the second half. It preserves actor-frame coordinates, elapsed times and starting minute. Replaying it shows how probabilities change as passes, receipts and carries arrive, then stops at the qualifying box-entry/shot event. The exported CSV records each probability and first threshold crossing; the repair report gives the executed results. This is historical replay, not a deployed live service.

## Research branches and limitations

| Branch | Status and scope |
|---|---|
| Event-time XGBoost / GRU | Primary research task; causal features, match holdout, baselines and uncertainty |
| Event-only pass progression | Retain as secondary post-pass prediction on `progression_beyond_pass_k`; saved model re-scored at AUC 0.7337 |
| 360 spatial augmentation | Secondary ablation; saved model AUC 0.7380, small lift, incomplete and competition-dependent coverage |
| Line-break detection | Geometric heuristic counting defenders crossed along x; not independently validated tactical ground truth |
| Possession XGBoost / GRU / ensemble | Retrospective diagnostics; completion features overlap with outcomes |
| GNN | Exploratory graph-only diagnostic; corrected-label held-out results in reports, not a like-for-like feature comparison |
| Pass-option ranking | Exploratory; unplayed counterfactual outcomes are unavailable, no recommendation-quality validation |
| Player attribution | Exploratory retrospective model explanation, not causal player value or a validated scouting ranking |
| Threat gain | Unvalidated zone-value proxy; opponent-frame and sparse-zone limitations; excluded from headline claims |

No latency, feed revision, clock synchronisation, odds linkage, market calibration or operational monitoring study has been completed. The cohort is small and selective, observations within matches overlap, and the test set has been inspected repeatedly during development. Confidence intervals do not account for that research selection. Further external and temporal validation is necessary before commercial use.

## Repository and audit history

- `src/data`, `src/features`, `src/labels`: parsing, coordinates, causal features and targets.
- `src/models`, `src/evaluation`: training, metrics, bootstrap, calibration and exploratory models.
- `src/app`: CLI and Streamlit research interface.
- `scripts/prepare_event_time.py`, `scripts/run_event_time.py`: maintained data-to-model route.
- `tests/`: synthetic regression/integration tests plus optional local-data checks.
- `reports/event_time`, `reports/pass_label`, `reports/repro`: research results and model cards.
- `notebooks/07_possession_baseline.ipynb`, `notebooks/09_possession_eda.ipynb`: retrospective analysis.

The October 2026 self-audit found event decoding, future possession-length leakage, opponent-coordinate and self-determining-label defects. A later round found same-moment leakage. Fraction-prefix results (including AUC 0.82–0.85) are withdrawn; the original event-time 0.668/0.670 results are superseded. See [audit history](reports/audit/), [timing audit](reports/feature_timing_audit.md), [label methodology](reports/label_methodology.md), and [current repair report](REPAIR_REPORT.md). Earlier documentation is retained as historical material, not current guidance.

Data attribution: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Follow its licence and attribution requirements when distributing data or derived examples.
