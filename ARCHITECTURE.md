# Frame2Threat - Architectural & Additional Functional Fixes

Football | Substantial (80% complete) | Architectural debt: moderate

## 1. Architecture as-is

The repo is organised as a clean layered Python package under `src/` with configs externalised to YAML, a test suite, CI, and a reproducibility layer. Layers already exist and are mostly respected:

```
configs/*.yaml                          declarative data/label/feature/model/eval config
        |
src/data/     ingest -> inventory -> parse_events/lineups/360 -> join_pass_frames
              -> parse_possessions -> splits -> build_store (parquet + SQLite)
        |
src/labels/   line_break, dangerous_progression, downstream_outcomes,
              possession_labels, validation_sampling
        |
src/features/ event_features, geometry_features, graph_builder,
              sequence_context, possession_features, early_features
        |
src/models/   baselines, tabular, gnn, hybrid, gru_possession, ranking,
              train_early_models
        |
src/evaluation/ metrics, calibration, ablations, possession_attribution,
                early_evaluation, tactical_review
        |
src/visualization/ pitch_plots, freeze_frame_view, explanations
        |
src/app/      cli.py (frame2threat entry point), app.py (Streamlit demo)

scripts/run_pipeline.py  ->  data/repro/v1/ artefacts + summary json
reports/repro/           ->  committed scored sample, model cards, hashes
```

Around 13k lines of source across ~50 modules. Data is gitignored; a synthetic generator backs CI smoke and the demo app. SQLite holds metadata/provenance and the parquet partition index; parquet holds analytical tables. This is a genuinely layered design, not a notebook dump.

Debt is moderate, not low, because: the top-level package is literally named `src` (import path `src.app.cli`), which breaks clean installation outside the repo root; trained artefacts are gitignored so the model layer cannot be reproduced from clone; CI does not install torch so the GRU/GNN branches of the model layer are untested; and the possession feature layer has a known timing-leakage boundary that the architecture does not yet enforce in code (only in a document).

## 2. Target architecture

Keep the layering; make three boundaries explicit and enforced.

```
            +------------------- configs (YAML) -------------------+
            v                                                      v
[ data ] -> [ store: parquet(bulk) + SQLite(metadata/manifest) ]
            |
            v
[ labels ] --+                          leakage contract enforced here
             v                          (event-time cutoff registry)
[ features: build_row(cutoff_time) ] -> [ v1 pass | v2 retro | v3 prefix ]
             |
             v
[ models (persisted, versioned artefacts) ]
             |
             v
[ evaluation: metrics + calibration(ECE) + attribution ]
             |
             v
[ serving: CLI (real fixtures) | Streamlit (real scored sample) ]
```

Layers: data/store, labels, features (with an explicit cutoff-time contract), models (with committed small artefacts or a rebuild command), evaluation, and serving. The rename from `src` to a `frame2threat` package makes the whole thing installable and importable outside the repo.

## 3. Architectural fixes

| ID | Priority | Area | Problem | Proposed change | Why it matters | Effort |
|----|----------|------|---------|-----------------|-----------------|--------|
| A1 | High | Feature contract | Timing-leakage boundary lives only in a markdown audit, not in code | Add a feature registry marking each feature start-safe / prefix-safe / completion-only; features builder refuses completion-only features when a cutoff is set | Prevents the leakage from silently returning in future edits | M |
| A2 | High | Packaging | Top-level package is `src`; entry point is `src.app.cli:main` | Rename to `frame2threat/` src-layout package | Installable and importable outside repo root; unblocks CLI packaging | M |
| A3 | High | Model artefacts | Trained models gitignored; model layer not reproducible from clone | Commit small artefacts or add a deterministic rebuild-from-committed-fixture command | Reproducibility of the headline layer | M |
| A4 | Medium | CI matrix | Torch not installed; GRU/GNN layer untested | Add a torch CI job exercising gru_possession and gnn | Coverage of the deep-model branch | M |
| A5 | Medium | Store | SQLite holds partition index and provenance but not the split manifest | Load split_manifest and run provenance into SQLite tables | Reviewer can query splits/provenance without dataframes | S |
| A6 | Low | Config drift | README results and test count are hand-maintained | Generate them from results_summary.json and pytest collection in CI | Docs stay truthful as code changes | S |

## 4. Additional functional fixes

- Compute ECE (10-bin) per model and store it in results_summary.json next to Brier and average precision.
- Report the GNN on the same test protocol as the tabular model, or annotate the val/subset caveat directly in the results table.
- Add a trivial baseline (prevalence or start-zone-only) reported beside every v3 early-forecasting number.
- Wire the Streamlit demo and CLI score-live path to a committed real scored fixture so the serving layer demonstrates true output.

## 5. Sequencing

1. A1 feature-cutoff contract and F1 leakage rebuild (removes the biggest correctness risk and lets v3 headline honestly).
2. A2 package rename, then A3 committed artefacts / rebuild command (makes the model and serving layers reproducible).
3. A4 torch CI job and coverage-floor raise (tests the deep-model branch).
4. A5 SQLite manifest/provenance, F4 ECE, baselines, and A6 docs generation (polish and audit trail).
5. Serving fixtures for CLI and Streamlit last.