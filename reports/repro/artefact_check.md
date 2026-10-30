# Artefact check

Saved models re-scored on the held-out test matches versus the value recorded in `reports/`.

| artefact | recomputed ROC AUC | documented | within tolerance |
|---|---|---|---|
| pass model event_only, all test passes | 0.7337 | 0.7337 | PASS |
| pass model event_360, all test passes | 0.7380 | 0.7380 | PASS |
| event-time XGB, all test snapshots | 0.6649 | 0.6649 | PASS |
| event-time GRU, all test snapshots | 0.6636 | 0.6636 | PASS |
