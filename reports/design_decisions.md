# Design Decisions

This note explains the main choices I would defend if asked why Frame2Threat is built this way.  It is intentionally written in the first person because the project should be accountable for its assumptions, not just present model outputs.

## 1. I used StatsBomb Open Data rather than private tracking data

I chose StatsBomb Open Data because it makes the project reproducible and inspectable.  The trade-off is that 360 freeze-frame coverage is partial and snapshot-based: it gives visible player locations at event moments, not continuous trajectories, velocities, or pitch-control surfaces.  That constraint matters, so I treat this as an event-data and freeze-frame project rather than a full tracking-data system.

## 2. I made the labels outcome-oriented and auditable

I defined `dangerous_progression_k` as a short-horizon downstream outcome: final-third entry, box entry, or shot within the same possession over the next five events.  I chose this because it is closer to a football decision than raw completion or x/y gain alone.  I kept separate component labels (`final_third_entry_k`, `box_entry_k`, `shot_within_k`) so the aggregate can be audited rather than acting as a black box.

For line-break labels, I accepted that 360 availability is required.  When freeze-frame data is missing, the label should be missing rather than silently guessed.

## 3. I split by match to avoid leakage

I use match-level train/validation/test splits because row-level splits would put highly related events from the same match, possession patterns, teams, and tactical context into multiple folds.  That would make results look cleaner than they are.  The split manifest is persisted so later runs can reproduce the exact assignment.

## 4. I used tabular baselines before complex sequence and graph models

I started with rule-based and tabular models because they are easier to inspect and harder to over-sell.  XGBoost on event features is a strong baseline for this data: it captures location, pass geometry, pressure, and play-context effects without needing raw player graphs.  The GRU and GNN models are still useful experiments, but I do not treat them as automatically better just because they are more complex.

## 5. I report the 360 result as a marginal finding, not a headline victory

The event-only pass model reaches 0.881 ROC AUC, while the event+360 version reaches 0.882.  I read that as a practical tie.  The honest conclusion is that 360 geometry is useful for interpretation and line-break logic, but it does not materially improve the reported pass-level danger model in this dataset.  The stronger predictive story is the possession and early-danger work, especially the prefix models that forecast danger before a possession is complete.

## Agent assistance

Some implementation and documentation work was assembled with agent assistance.  I do not want that to obscure the engineering standard: the defensible artifacts are the code, tests, CI, reproducibility commands, stored manifests, and the explicit caveats above.  Agent assistance is not evidence by itself; reproducible outputs and honest limitations are.
