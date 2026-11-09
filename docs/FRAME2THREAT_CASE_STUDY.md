# Frame2Threat: Event-Time Danger Forecasting — A Research Case Study

**Version:** November 2026  
**Repository:** https://github.com/varunrout/Frame2Threat  
**Status:** Research prototype — not a deployed system

---

## 1. Football Research Question

**Can the events of an unfolding possession warn an analyst of a later shot or penalty-area entry?**

This is the central question. In football, possessions evolve through sequences of passes, carries, and receipts. An analyst watching a match — or reviewing it afterwards — wants to know: *given what has happened so far, how likely is this possession to become dangerous?*

Frame2Threat addresses this as an **event-time prediction** problem: after each completed event *t*, estimate the probability that the possessing team will produce a shot or box entry in a *later* event of the same possession.

---

## 2. Why Event-Time Danger Prediction Matters

Traditional football analytics often uses **full-possession summaries** (e.g., "this possession had 8 passes and ended in a shot"). But these are retrospective — they describe what *did* happen, not what *was likely* at each moment.

Event-time prediction is different:
- **Prospective:** At event *t*, only events 0..*t* are known
- **Actionable:** An analyst can see when the model first raises an alert
- **Tactically interpretable:** The changing probability traces the possession's narrative

This matters for:
- **Pre-match preparation:** Understanding which possession patterns lead to danger
- **Live tactical review:** Identifying moments where risk escalated
- **Player profiling:** Which players are involved when danger probability rises
- **Model auditing:** Seeing *why* the model thinks a moment is dangerous

---

## 3. Data and Observational Constraints

### Data Source
- **StatsBomb Open Data** — 99 matches (64 World Cup 2022, 35 Barcelona La Liga 2020/21)
- **Fixed cohort:** Train/validation/test split fixed by match (15 held-out test matches)
- **No data leakage:** Split manifest committed to `configs/research_split_manifest.csv`

### What the Models Observe (Causal Features Only)
At each event *t*, features are computed from events 0..*t* only:
- Ball position (x, y) in possession frame — opponent events flipped
- Pass end position — only after pass completes
- Event type (Pass, Carry, Receipt, Pressure, Duel, etc.)
- Pressure indicator (binary, from StatsBomb)
- Elapsed time since possession start
- Cumulative counts: passes, carries, pressures so far
- Max x-reached — furthest territorial advance
- Possession context: origin, period, starting minute, start coordinates

### What Is NOT Observed (By Design)
- Future events
- Total possession length
- Whether danger will occur
- Opponent player positions (except partial 360 freeze frames for 52% of passes)
- Continuous tracking data — this is **event data**, not tracking data

### Outcome Definition
A possession is "dangerous" if it contains a **shot** or **penalty-box entry** by the possessing team.  
**Crucially:** An incomplete pass *ending in the box* counts as a box entry. This is an attempted-entry proxy, not exclusively controlled entry.

---

## 4. Model Approach

Two models, both trained on the same causal features:

| Model | Architecture | Key Characteristics |
|-------|--------------|---------------------|
| **XGBoost** | Gradient-boosted trees (depth 5, 55 trees) | Snapshot classifier — each event scored independently |
| **GRU** | 1-layer GRU (hidden 64, learned type embedding) | Causal sequence model — one forward pass scores all prefixes |

**Training:** Binary cross-entropy on eligible snapshots (events before any danger event).  
**Validation:** Hyperparameters and alert thresholds chosen on validation log loss / prevalence quantile only.  
**Test:** 15 held-out matches, never seen during training or validation.

---

## 5. Genuine Match Walkthrough

**France vs Opponent — Match 3857266, Possession 121**  
Period 2, Minute 68, From Throw-In — 27 events, outcome realised at Event 19 (Dribble into box)

### Event-by-Event Probability Evolution

| Event | Type | Team | XGBoost P | GRU P | Flags |
|-------|------|------|-----------|-------|-------|
| 1 | Pass | France | 0.460 | 0.233 | [XGB TIP] |
| 2 | Ball Receipt | France | 0.444 | 0.236 | [XGB] |
| 3 | Pass | France | 0.493 | 0.283 | [XGB] |
| 4 | Ball Receipt | France | 0.473 | 0.299 | [XGB] |
| 5 | Carry | France | 0.498 | 0.345 | [XGB] |
| 6 | Pass | France | 0.549 | 0.375 | [XGB] |
| 7 | Ball Receipt | France | 0.513 | 0.408 | [XGB] |
| 8 | Carry | France | 0.528 | 0.437 | [XGB] [GRU TIP] |
| 9 | Pass | France | 0.530 | 0.446 | [XGB] [GRU] |
| 10 | Ball Receipt | France | 0.560 | 0.492 | [XGB] [GRU] |
| 11 | Carry | France | 0.579 | 0.519 | [XGB] [GRU] |
| 12 | Pass | France | 0.568 | 0.502 | [XGB] [GRU] |
| 13 | Ball Receipt | France | 0.588 | 0.531 | [XGB] [GRU] |
| 14 | Carry | France | 0.602 | 0.423 | [XGB] [GRU] |
| 15 | Pressure | Opponent | 0.520 | 0.368 | [XGB] |
| 16 | Pass | France | 0.612 | 0.494 | [XGB] [GRU] |
| 17 | Ball Receipt | France | 0.632 | 0.559 | [XGB] [GRU] |
| 18 | Carry | France | 0.654 | 0.595 | [XGB] [GRU] |
| 19 | Dribble | France | — | — | [OUTCOME] |

### Key Moments

**First Alerts:**
- **XGBoost:** Event 1 (P = 0.460 >= 0.390) — immediate response to first pass toward goal
- **GRU:** Event 8 (P = 0.437 >= 0.422) — requires sequence context before raising probability

**Lead Time Before Outcome:**
- **XGBoost:** 18 events lead time
- **GRU:** 11 events lead time

**Outcome Realised:**
- Event 19 (Dribble) — Shot/box entry by possessing team
- Final XGBoost P(danger) = 0.654
- Final GRU P(danger) = 0.595

### Model Disagreement Analysis
- **XGBoost threshold:** 0.390 (validation prevalence quantile)
- **GRU threshold:** 0.422 (validation prevalence quantile)
- **First alert difference:** 7 events — XGBoost alerts earlier (more sensitive to early territorial gain)
- **Trajectory correlation:** 0.920 — models agree on overall trend but differ in sensitivity

### What Was Observable at Each Step
Only causal features from events 0..*t*:
1. Ball position in possession frame
2. Pass end position (after completion)
3. Event type and pressure
4. Elapsed time and cumulative counts
5. Max x-reached and possession context

**Not observed:** Future events, possession length, outcome, opponent positions (except partial 360 frames).

---

## 6. Held-Out Results and Baselines

### Discrimination (ROC AUC)

| Model | All Snapshots | One Per Possession |
|-------|---------------|-------------------|
| Ball x/y Logistic | 0.642 [0.622, 0.661] | 0.695 [0.674, 0.717] |
| Events-So-Far Logistic | 0.561 [0.523, 0.603] | 0.608 [0.579, 0.638] |
| Start-Context XGBoost | 0.590 [0.568, 0.615] | 0.598 [0.572, 0.626] |
| **Event-Time XGBoost** | **0.665 [0.647, 0.681]** | **0.736 [0.714, 0.759]** |
| **Event-Time GRU** | **0.664 [0.642, 0.687]** | **0.739 [0.710, 0.767]** |

*95% match-level bootstrap CIs (500 resamples). All-snapshot weights long possessions more heavily.*

### Calibration and Probabilistic Scores

| Model | Brier | Log Loss | ECE (10 equal-width bins, snapshot-weighted, absolute probability) |
|-------|-------|----------|---------------|
| XGBoost | 0.223 | 0.637 | 0.039 |
| GRU | 0.224 | 0.642 | 0.039 |

**Calibration is imperfect** — models under-predict on held-out test. Probabilities should be used to *rank moments within a possession*, not as absolute probabilities.

### Practical Usefulness (At Validation Thresholds)

| Metric | XGBoost (thr=0.390) | GRU (thr=0.422) |
|--------|---------------------|-----------------|
| Precision (per scored event) | 0.552 | 0.554 |
| Recall (per scored event) | 0.520 | 0.518 |
| False-alert rate (safe possessions with at least one alert) | 48.0% | 39.7% |
| Median Lead Time (dangerous possessions that alerted) | 9 events | 10 events |

*Test split, 2,284 possessions with at least one eligible snapshot (727 reach danger), committed models, frozen definitions of [the sealed-holdout protocol](../reports/repro/sealed_holdout_protocol.md).  Precision and recall count eligible snapshots; the false-alert rate and lead time are per possession.  Source: `scripts/sealed_holdout.py --target test`, `reports/repro/frozen_metrics_test.json`.  This replaces the earlier 55.9% / 46.6% from `practical_usefulness.py`, which counted alerts on the last event of safe possessions and kept only possessions of four or more events.  On the Euro 2024 sealed holdout the rates are 45.2% / 39.3% ([sealed_holdout.md](../reports/repro/sealed_holdout.md)).*

**Reading:** at the validation-chosen thresholds (prevalence quantiles, not tuned for precision) roughly half of the alerted events are in possessions that really reach danger, about half of dangerous events are caught, and 40% (GRU) to 48% (XGBoost) of safe possessions still trigger an alert.  An alert is a prompt to look, not a prediction to act on.

---

## 7. What Worked

1. **Causal event-time formulation** — No leakage from future events or possession length
2. **Rigorous validation** — Match-level bootstrap CIs, thresholds fixed on validation only
3. **Reproducible pipeline** — `prepare_event_time.py` -> `run_event_time.py` with SHA256 input hashes
4. **Honest demo** — Single held-out possession replayable via CLI (`frame2threat score-live`)
5. **Interpretability** — SHAP for XGBoost, trajectory comparison for GRU
6. **Audit trail** — October 2026 self-audit found and fixed multiple leakage bugs

---

## 8. What Did Not Work

1. **Modest discrimination** — ROC AUC ~0.66 is only slightly above ball-position baseline (0.64)
2. **Modest precision at validation thresholds** — about 0.55 precision and a 40-48% false-alert rate on safe possessions make alerts noisy
3. **Calibration gap** — ECE 0.039 on the test split and 0.062-0.063 on the Euro 2024 holdout (snapshot-weighted, absolute probability)
4. **No causal claims** — Attribution is retrospective correlation, not causal contribution
5. **Small test cohort** — 15 matches (WC 2022 + Barcelona 20/21) limits generalisation
6. **Partial 360 coverage** — Freeze frames for only 52% of passes; not continuous tracking
7. **No external validation** — Results may not transfer to other competitions/seasons

---

## 9. Research Limitations

| Limitation | Impact |
|------------|--------|
| **Clustered observations** | Events within a possession are highly correlated; bootstrap CIs account for match clustering but not repeated test inspection |
| **Cohort selection** | World Cup + one club season; no temporal or competition holdout |
| **Outcome proxy** | Incomplete pass in box = box entry; not independently validated tactical ground truth |
| **No market linkage** | No odds calibration, no betting-market evaluation |
| **No latency study** | Event-time scoring is fast (<1ms/event) but feed revision/clock sync not studied |
| **Attribution != causation** | SHAP/LOO drops show what model leans on, not what *causes* danger |

---

## 10. Possible Future Applications

1. **Tactical video review tool** — Sync probability trajectory with video for analyst review
2. **Possession pattern mining** — Cluster possessions by probability trajectory shape
3. **Player involvement profiling** — Which players are present when P(danger) rises?
4. **Threshold optimisation** — Cost-sensitive thresholds for specific use cases (e.g., low false-alert for live alerts)
5. **Multi-season validation** — Test on 2023/24, 2024/25 seasons when available
6. **Tracking data fusion** — If tracking data becomes available, test if off-ball features add signal

---

## Appendix: Reproduction Commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Synthetic smoke test
python -m src.data.build_store --synthetic --output-dir data/store/smoke
python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke

# Genuine data (9 matches)
python scripts/prepare_event_time.py --small --output-dir data/repro/small
python scripts/run_event_time.py --data-dir data/repro/small --output-dir data/repro/small_run

# Full cohort (99 matches)
python scripts/prepare_event_time.py --output-dir data/processed
python scripts/run_event_time.py --data-dir data/processed --threads 2

# Replay held-out possession
frame2threat score-live reports/repro/demo_possession.json --model xgb -o scores.csv
frame2threat score-live reports/repro/demo_possession.json --model gru -o scores_gru.csv

# Streamlit app
streamlit run src/app/app.py
```

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `scripts/prepare_event_time.py` | Data preparation (causal features, labels, splits) |
| `scripts/run_event_time.py` | Training, validation selection, bootstrap evaluation |
| `scripts/tactical_walkthrough.py` | Generate match walkthrough visualisation |
| `scripts/interpretability_analysis.py` | SHAP (XGBoost) and trajectory (GRU) analysis |
| `scripts/practical_usefulness.py` | Precision/recall/lead-time/calibration evaluation |
| `src/app/enhanced_possession_page.py` | Streamlit Possession Inspector |
| `src/features/event_time.py` | Causal feature engineering |
| `src/models/event_time.py` | GRU architecture and training |
| `reports/repro/repair_retrain/event_time_results.md` | Full retraining results |

---

*This case study documents the Frame2Threat research prototype as of November 2026. It is a research artefact, not a production system. All results are from held-out test matches with validation-fixed thresholds. No claims of operational or betting utility are made.*