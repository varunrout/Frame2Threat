# Model Card: PassFrameGNN on the corrected pass label (EXP-032)

## Summary

A graph neural network over the 360 freeze frame at the moment of the pass, trained on
`progression_beyond_pass_k` (the corrected label; `dangerous_progression_k` is kept for comparison only).
It answers one question: **how much of pass progression can the freeze-frame geometry alone explain**, with no
event attributes and no information about where the pass ends?  On the identical 360 test subset (3,736 passes, 9 World Cup 2022 matches) the graph-only PassFrameGNN scores ROC AUC 0.595 [0.578, 0.606] against 0.715 [0.697, 0.734] for XGBoost event+360; the paired difference is -0.1209 [-0.1379, -0.1033], so the GNN is clearly worse.  It beats the `end_x`-only baseline (0.530 [0.502, 0.551]; paired difference +0.0646 [+0.0501, +0.0845]).

Artefacts: `models/pass_gnn_progression.pt` and `models/pass_gnn_progression.json` (metadata, seeds, git commit
`ea07f0d`).  Code: `src/features/pass_graph.py`, `src/models/pass_gnn.py`; training and comparison:
`scripts/train_pass_gnn.py`; full results: `reports/pass_label/gnn_comparison.md` and `.json`.

## Label

`progression_beyond_pass_k` (k = 5; prevalence 21.6% overall, 21.7% in the test subset): within the next 5 events of
the same possession, ignoring the receipt of this pass and the opponent's events, a shot, a new box entry, or a gain of
at least 10 m beyond the pass end into a more advanced zone (`reports/label_methodology.md`, Label G).  The label comes
from `compute_progression_beyond_pass`, the same table the XGBoost models use; alignment is asserted by `event_uuid`
and against `data/processed/pass_labels_beyond.parquet`.

## Graph definition

One graph per open-play pass with 360 data (train 18,234 / val 4,545 / test 3,736; the
saved match-level 70/15/15 split, seed 42, is reused, not regenerated).

- **Nodes:** every visible player in the freeze frame.  **8 features**, scaled to about [0, 1]: x, y, teammate,
  is_keeper, is_actor, distance to goal, distance to the passer (pass **start** location), local density (other
  players within 10 m).
- **Edges:** each node to its 5 nearest neighbours, plus same-team links in both directions (as the original
  `graph_builder.py`).  No edge features are used (the original GraphSAGE ignored them as well).
- **Leakage contract (tested):** a graph is a function of that event's freeze-frame rows and the pass start only.  The
  builder accepts no pass end location, label or later-event information (`tests/test_pass_gnn.py` perturbs the end
  location, labels, possession and other events' frames and checks the graph is unchanged, and scans the module for any
  such reference).

### Deviations from the original PassFrameGNN (`src/models/gnn.py`, `src/features/graph_builder.py`)

| Original | Here | Why |
|---|---|---|
| 9 node features including `is_receiver` (nearest teammate to the pass **end**) | 8 features, `is_receiver` dropped | It encodes the pass destination, which the corrected label is designed not to reward |
| Graph embedding fused with a 27-feature event vector (includes the destination) and 5 multitask heads | Graph only, one binary head | The question here is what geometry alone explains; the original was a graph + event hybrid |
| `torch_geometric` SAGEConv (silently an MLP without message passing if the library was missing) | Plain-torch GraphSAGE mean aggregation, unit-tested against the formula | The library is not installed; no silent fallback |
| Raw coordinates | Coordinates and distances scaled | Stable optimisation |

The original checkpoint was never committed, so the old 0.841 / 0.834 numbers cannot be reproduced; this is a new
model, not a reproduction.

## Architecture and training

Node projection (8 to 64, ReLU), 3 x GraphSAGE (mean aggregation, 64 hidden, dropout after each layer), mean pooling,
fusion block (Linear 64 to 64, ReLU, dropout), one Linear(64, 1) head; plain BCE with no class weighting; Adam, batch
size 64, early stopping on validation ROC AUC (patience 6, at most 40 epochs).

- **Tuning (validation only):** a fixed 8-configuration grid over learning rate {1e-3, 3e-3}, dropout {0.1, 0.3}, weight
  decay {0, 1e-4} with seed 0 (25 epochs at most); validation ROC AUC 0.565 to 0.575.  Chosen: lr 0.001, dropout
  0.3, weight decay 0.0.
- **Seeds:** 5 (0 to 4) with the chosen configuration.  Validation ROC AUC 0.5782 (sd 0.0028);
  **test** ROC AUC 0.5950 (sd 0.0031), AP 0.2826 (0.0025), Brier
  0.1671 (0.0003).  The reported model is seed 0, chosen by **validation** ROC
  AUC; test was read once, after selection.

## Results (test, identical 360 subset, 1,000 match-level bootstrap resamples, 95% intervals)

| Model | ROC AUC | Average precision | Brier | Calibration intercept / slope |
|---|---|---|---|---|
| PassFrameGNN (graph-only) | 0.595 [0.578, 0.606] | 0.284 [0.262, 0.325] | 0.167 [0.157, 0.179] | -0.09 / 0.96 |
| XGBoost event+360 (EXP-027 recipe) | 0.715 [0.697, 0.734] | 0.403 [0.381, 0.429] | 0.193 [0.186, 0.201] | -1.05 / 0.73 |
| XGBoost event-only (EXP-027 recipe) | 0.717 [0.700, 0.733] | 0.400 [0.370, 0.436] | 0.199 [0.194, 0.206] | -1.11 / 0.78 |
| `end_x`-only logistic regression | 0.530 [0.502, 0.551] | 0.251 [0.227, 0.281] | 0.170 [0.159, 0.183] | -0.64 / 0.49 |
| *Reference:* XGBoost event+360, early-stopped, unweighted | 0.728 [0.709, 0.748] | 0.421 [0.389, 0.460] | 0.151 [0.140, 0.163] | -0.07 / 0.99 |

Paired differences: GNN - XGBoost event+360 -0.1209 [-0.1379, -0.1033] ROC AUC, -0.1194 [-0.1428, -0.0898] AP;
GNN - `end_x`-only +0.0646 [+0.0501, +0.0845] ROC AUC; XGBoost event+360 - event-only -0.0015 [-0.0106, +0.0069] (no measurable 360 gain
on this subset either).

**The GNN does not beat XGBoost.**  It carries real signal beyond the destination (above the `end_x` baseline) but about
0.12 ROC AUC less than the event-based models, which also see the pass destination, length, angle and pressure.
Calibration: the GNN is well calibrated (slope 0.96, mean prediction 0.225 vs prevalence 0.217)
because it uses plain BCE; the EXP-027 XGBoost recipe uses `scale_pos_weight` and is inflated (mean prediction
0.39), which is why its Brier score is worse than the GNN's despite much higher discrimination.  The
unweighted reference rows show the fair calibration comparison.  Reliability tables: `reports/pass_label/gnn_comparison.md`.

## Intended use and scope

- A diagnostic of how much progression the static freeze-frame geometry explains **on its own**.  Not a replacement
  for the event model and not for production scoring.
- It is **not** a like-for-like contest between feature sets: the GNN has no event attributes and no destination, XGBoost
  has both.  A graph + event hybrid (the original design, minus `is_receiver`) was not trained here.

## Limitations

- Static freeze frames: one snapshot at the moment of the pass, no velocities or trajectories; players outside the
  camera view are missing.
- 360 data exists only for the 64 World Cup 2022 matches (52.3% of passes; the La Liga matches have none), so this
  model never sees La Liga and cannot score passes without a frame.
- The test subset is 3,736 passes from 9 matches (the full test split has 15 matches); the bootstrap
  intervals are match-level and wide.
- Small, fixed tuning budget; several runs reached their best validation epoch close to the epoch cap, so longer
  training might add a little; it is unlikely to close a 0.12 gap.
- GraphSAGE here is a plain-torch implementation, tested against the SAGE mean formula, not `torch_geometric`.
- Single label and a single split.
