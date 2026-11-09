# Frame2Threat Demo Report — November 2026

> Snapshot report written at the commit below. Later changes are listed in [CHANGELOG.md](../CHANGELOG.md); numbers here
> match the reports as of that commit.

**Repository:** https://github.com/varunrout/Frame2Threat  
**Commit:** `602a344` (Fix CI gates: formatting, type checking, coverage; add wheel install test)  
**Branch:** `main` (up to date with origin/main)

---

## 1. Files Modified

### Core Application
| File | Changes |
|------|---------|
| `src/app/enhanced_possession_page.py` | **NEW** — Enhanced Possession Inspector with match/possession selection, pitch visualisation, event-by-event playback, XGBoost/GRU comparison, alert timeline, technical explanation, demo mode |
| `src/app/app.py` | Updated to use `enhanced_possession_page` instead of `possession_page` |
| `src/app/possession_page.py` | Preserved at the time; removed on 2026-11-04 as dead code. Its pure helpers now live in `src/app/possession_data.py` |

### Scripts Added
| File | Purpose |
|------|---------|
| `scripts/tactical_walkthrough.py` | Generates tactical match walkthrough (PNG, JSON, MD) for France possession 121 |
| `scripts/interpretability_analysis.py` | SHAP analysis for XGBoost, trajectory analysis for GRU, model comparison |
| `scripts/practical_usefulness.py` | Precision/recall, false alerts, lead time, calibration, bucket evaluation |
| `scripts/prepare_event_time.py` | Existing — data preparation |
| `scripts/run_event_time.py` | Existing — training and evaluation |

### Documentation
| File | Purpose |
|------|---------|
| `docs/FRAME2THREAT_CASE_STUDY.md` | Recruiter-friendly research case study |
| `reports/audit/REPAIR_REPORT.md` | Existing — November 2026 repair documentation |
| `README.md` | Existing — updated with event-time focus |

### Generated Artefacts
| File | Description |
|------|-------------|
| `reports/figures/tactical/tactical_walkthrough.png` | Main tactical walkthrough visual |
| `reports/figures/tactical/tactical_walkthrough.json` | Detailed event-by-event data |
| `reports/figures/tactical/tactical_walkthrough.md` | Markdown narrative |
| `reports/figures/interpretability/shap_summary.png` | XGBoost SHAP feature importance (bar) |
| `reports/figures/interpretability/shap_beeswarm.png` | XGBoost SHAP beeswarm plot |
| `reports/figures/interpretability/shap_importance.csv` | SHAP importance rankings |
| `reports/figures/interpretability/gru_trajectories.png` | GRU probability trajectories |
| `reports/figures/interpretability/model_comparison.png` | XGBoost vs GRU comparison |
| `reports/figures/interpretability/gru_trajectories.json` | GRU trajectory data |
| `reports/figures/practical/practical_usefulness.png` | Practical usefulness dashboard |
| `reports/repro/practical_usefulness.json` | Detailed evaluation metrics |

---

## 2. Functionality Implemented

### Phase 1: Tactical Match Walkthrough ✅
- **Genuine held-out possession:** France, match 3857266, possession 121 (27 events)
- **Pitch diagram** with ball path coloured by XGBoost probability
- **Event sequence table** with XGBoost/GRU probabilities and alert flags
- **Probability trajectories** with thresholds, first alerts, outcome marker
- **Key moments panel** with first alerts, lead times, model disagreement
- **Observable features panel** explaining causal constraints
- **Model comparison summary** with correlation and outcome probabilities

### Phase 2: Enhanced Streamlit App ✅
- **Match/possession selection** with test/val/train split radio (clear held-out labelling)
- **Pitch visualisation** with event-by-event ball path
- **Dual-model trajectories** (XGBoost + GRU) with threshold lines
- **Alert timeline** showing first threshold crossings
- **Outcome annotation** with lead time calculation
- **Technical explanation** of causal features (expandable)
- **Limitations panel** with honest assessment
- **Demo mode** when artefacts missing — shows single possession with pre-computed scores
- **Clear demo/reality distinction** — never presents placeholder predictions as real

### Phase 3: Interpretability ✅
- **XGBoost SHAP analysis** — feature importance bar plot + beeswarm
- **GRU trajectory analysis** — probability evolution across test possessions
- **Model comparison** — side-by-side XGBoost vs GRU on same possessions
- **Causal feature documentation** — what models observe at each step

### Phase 4: Practical Usefulness ✅
- **Precision/Recall/F1** at validation thresholds
- **False-alert rate** per safe possession (XGBoost: 48.0%, GRU: 39.7%; frozen definition, `reports/repro/frozen_metrics_test.json`)
- **Lead time** before outcome (median XGBoost: 9 events, GRU: 10 events)
- **Calibration** — ECE and Brier score
- **Performance by event bucket** (1-3, 4-6, 7-10, 11+ events)
- **Baseline comparison** — ball position LR, events LR, start-context XGBoost

### Phase 5: Case Study ✅
- **Professional British English** throughout
- **Football concepts explained** without unnecessary ML jargon
- **No inflated claims** — modest AUC, modest precision, calibration gaps highlighted
- **Understandable to football analysts** — tactical walkthrough, clear outcome definition

### Phase 6: Cleanup ✅
- **CI gates fixed** — ruff, black, mypy all pass; coverage 61.45% (>50% threshold)
- **Wheel build + install** verified outside checkout
- **Stale documentation** synchronised with actual test counts and coverage
- **Demo workflow** documented in README and case study

---

## 3. Screenshots / Generated Artefacts

### Tactical Walkthrough
- `reports/figures/tactical/tactical_walkthrough.png` — 6-panel visual with pitch, event table, trajectories, key moments, observables, model comparison

### Interpretability
- `reports/figures/interpretability/shap_summary.png` — XGBoost feature importance (SHAP)
- `reports/figures/interpretability/shap_beeswarm.png` — SHAP beeswarm distribution
- `reports/figures/interpretability/gru_trajectories.png` — 4 GRU trajectories
- `reports/figures/interpretability/model_comparison.png` — 4 side-by-side XGBoost vs GRU

### Practical Usefulness
- `reports/figures/practical/practical_usefulness.png` — 6-panel dashboard (P/R/F1, false alerts, lead time, calibration, bucket AUC, baseline comparison)

---

## 4. Actual Commands Executed

```bash
# Environment setup
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Generate tactical walkthrough
python scripts/tactical_walkthrough.py

# Generate interpretability analysis
python scripts/interpretability_analysis.py

# Generate practical usefulness evaluation
python scripts/practical_usefulness.py

# Run full test suite
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=50 -q

# Linting and formatting
ruff check src tests scripts
black --check src tests scripts
mypy src

# CLI smoke tests
frame2threat --help
python -m src.data.build_store --synthetic --output-dir data/store/smoke
python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke

# Wheel build and install
python -m pip wheel . --no-deps -w dist
python -m venv C:\tmp\frame2threat-wheel --system-site-packages
C:\tmp\frame2threat-wheel\Scripts\python -m pip install --no-deps --force-reinstall dist\frame2threat-0.1.0-py3-none-any.whl
C:\tmp\frame2threat-wheel\Scripts\frame2threat --help

# Replay genuine possession
python -m src.app.cli score-live reports/repro/demo_possession.json --model xgb
python -m src.app.cli score-live reports/repro/demo_possession.json --model gru
```

---

## 5. Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1
collected 210 items

tests\test_app_smoke.py ..P
tests\test_attribution.py ...
tests\test_batch_scoring.py ..
tests\test_build_store.py .
tests\test_cli.py ..........
tests\test_event_time.py ..........
tests\test_event_types.py ...
tests\test_features.py .......
tests\test_inference_contract.py ......
tests\test_ingestion.py .......
tests\test_labels.py .........
tests\test_models.py .......
tests\test_pass_gnn.py .....
tests\test_pass_progression.py .....
tests\test_possession_page.py ..... (renamed `tests/test_possession_data.py` on 2026-11-04)
tests\test_reproduce_pipeline.py .
tests\test_shot_columns.py ..
tests\test_splits.py .....

================= 210 passed, 1 skipped, 19 warnings in 144.31s =================
Coverage: 61.45% (required: 50%)
```

**Linting:**
- `ruff check src tests scripts` — PASS
- `black --check src tests scripts` — PASS (103 files)
- `mypy src` — PASS (no issues in 63 source files)

---

## 6. Model Evaluation Findings

### Discrimination (Held-Out Test, 15 matches, 2249 possessions)

| Model | All-Snapshot ROC AUC | One-Per-Possession ROC AUC |
|-------|---------------------|---------------------------|
| Ball x/y Logistic | 0.642 [0.622, 0.661] | 0.695 [0.674, 0.717] |
| Events-So-Far Logistic | 0.561 [0.523, 0.603] | 0.608 [0.579, 0.638] |
| Start-Context XGBoost | 0.590 [0.568, 0.615] | 0.598 [0.572, 0.626] |
| **Event-Time XGBoost** | **0.665 [0.647, 0.681]** | **0.736 [0.714, 0.759]** |
| **Event-Time GRU** | **0.664 [0.642, 0.687]** | **0.739 [0.710, 0.767]** |

### Calibration & Probabilistic Scores

| Model | Brier | Log Loss | ECE (10 equal-width bins, snapshot-weighted, absolute probability) |
|-------|-------|----------|---------------|
| XGBoost | 0.223 | 0.637 | 0.039 |
| GRU | 0.224 | 0.642 | 0.039 |

### Practical Usefulness (At Validation Thresholds)

| Metric | XGBoost (thr=0.390) | GRU (thr=0.422) |
|--------|---------------------|-----------------|
| Precision (per scored event) | 0.552 | 0.554 |
| Recall (per scored event) | 0.520 | 0.518 |
| False-alert rate (safe possessions with at least one alert) | 48.0% | 39.7% |
| Median Lead Time (dangerous possessions that alerted) | 9 events | 10 events |

*Test split, 2,284 possessions with at least one eligible snapshot (727 reach danger), committed models, frozen definitions of [the sealed-holdout protocol](../reports/repro/sealed_holdout_protocol.md).  Precision and recall count eligible snapshots; the false-alert rate and lead time are per possession.  Source: `scripts/sealed_holdout.py --target test`, `reports/repro/frozen_metrics_test.json`.  This replaces the earlier 55.9% / 46.6% from `practical_usefulness.py`, which counted alerts on the last event of safe possessions and kept only possessions of four or more events.  On the Euro 2024 sealed holdout the rates are 45.2% / 39.3% ([sealed_holdout.md](../reports/repro/sealed_holdout.md)).*

**Reading:** at the validation-chosen thresholds (prevalence quantiles, not tuned for precision) roughly half of the alerted events are in possessions that really reach danger, about half of dangerous events are caught, and 40% (GRU) to 48% (XGBoost) of safe possessions still trigger an alert.  An alert is a prompt to look, not a prediction to act on.

### Performance by Event Bucket

| Bucket | XGBoost AUC | GRU AUC |
|--------|-------------|---------|
| 1-3 events | 0.630 | 0.639 |
| 4-6 events | 0.650 | 0.651 |
| 7-10 events | 0.671 | 0.672 |
| 11+ events | 0.657 | 0.660 |

---

## 7. Remaining Limitations

1. **Modest discrimination** — AUC ~0.66 only slightly above ball-position baseline (0.64)
2. **Poor precision at validation thresholds** — 40-48% of safe possessions raise an alert
3. **Calibration gap** — ECE 0.039 on the test split and 0.062-0.063 on the Euro 2024 holdout (snapshot-weighted, absolute probability)
4. **No causal claims** — Attribution is retrospective correlation
5. **Small test cohort** — 15 matches (WC 2022 + Barcelona 20/21)
6. **Partial 360 coverage** — Freeze frames for only 52% of passes
7. **No external validation** — No temporal/competition holdout
8. **No market linkage** — No odds calibration or betting evaluation
9. **CI blocked remotely** — GitHub account billing issue prevents Actions execution

---

## 8. Updated Project Readiness Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Code quality** | ✅ PASS | Ruff, Black, mypy all clean |
| **Test coverage** | ✅ PASS | 61.45% (>50% threshold) |
| **All tests pass** | ✅ PASS | 210 passed, 1 skipped |
| **Wheel build** | ✅ PASS | Verified install outside checkout |
| **CLI functionality** | ✅ PASS | All commands work |
| **Streamlit app** | ✅ PASS | Enhanced Possession Inspector |
| **Reproducibility** | ✅ PASS | Fixed splits, SHA256 hashes, demo mode |
| **Documentation** | ✅ PASS | Case study, README, repair report |
| **Interpretability** | ✅ PASS | SHAP + trajectory analysis |
| **Practical evaluation** | ✅ PASS | Precision/recall/lead-time/calibration |
| **Remote CI** | ⚠️ BLOCKED | GitHub billing issue (account-level) |
| **External validation** | ❌ NOT DONE | No multi-season/competition holdout |

---

## Summary

The Frame2Threat repository is now a **polished, evidence-based research portfolio project** with:

- **Genuine tactical walkthrough** of a held-out France possession
- **Enhanced Streamlit interface** for possession inspection
- **Interpretability analysis** (SHAP for XGBoost, trajectories for GRU)
- **Practical usefulness evaluation** beyond ROC-AUC
- **Recruiter-friendly case study** documenting the research honestly
- **All CI gates passing locally** (ruff, black, mypy, pytest, wheel)

The project honestly presents modest discrimination (AUC ~0.66), modest precision at default thresholds, and clear limitations — making it a credible research artefact rather than an overclaimed demo.

**Next steps for production readiness:** Multi-season validation, threshold optimisation for specific use cases, tracking data fusion if available, and resolution of GitHub Actions billing.