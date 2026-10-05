# Pass model reproduction gate (`progression_beyond_pass_k`)

Target (EXP-027): event-only ~0.728, event+360 ~0.729 test ROC AUC.  Test prevalence 0.224.  Unchanged split; no artefact saved.

| features | recipe | trees | scale_pos_weight | test ROC AUC | test AP | test Brier | mean prediction |
|---|---|---|---|---|---|---|---|
| event-only (27) | EXP-027 recipe (spw, 500 trees) | 500 | 3.67 | 0.7279 | 0.4222 | 0.1948 | 0.4076 |
| event-only (27) | ES on validation, no spw | 129 | 1.00 | 0.7337 | 0.4283 | 0.1532 | 0.2164 |
| event-only (27) | no ES, no spw (500 trees) | 500 | 1.00 | 0.7286 | 0.4246 | 0.1540 | 0.2135 |
| event+360 (41) | EXP-027 recipe (spw, 500 trees) | 500 | 3.67 | 0.7294 | 0.4263 | 0.1910 | 0.3989 |
| event+360 (41) | ES on validation, no spw | 163 | 1.00 | 0.7380 | 0.4326 | 0.1525 | 0.2160 |
| event+360 (41) | no ES, no spw (500 trees) | 500 | 1.00 | 0.7345 | 0.4221 | 0.1536 | 0.2119 |
