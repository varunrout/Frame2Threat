# EXP-023: v2 possession-level models on frame-corrected labels

Test possessions: 2,475. Test prevalence old 0.360 -> corrected 0.320.

| models | test labels | model | ROC AUC | AP | Brier |
|---|---|---|---:|---:|---:|
| saved v2 (trained on old labels) | old labels (as published) | XGBoost | 0.9449 | 0.8804 | 0.0865 |
| saved v2 (trained on old labels) | old labels (as published) | GRU | 0.9524 | 0.9282 | 0.0910 |
| saved v2 (trained on old labels) | old labels (as published) | Ensemble (mean) | 0.9704 | 0.9476 | 0.0720 |
| saved v2 (trained on old labels) | corrected labels | XGBoost | 0.9575 | 0.8950 | 0.0775 |
| saved v2 (trained on old labels) | corrected labels | GRU | 0.9358 | 0.8678 | 0.1117 |
| saved v2 (trained on old labels) | corrected labels | Ensemble (mean) | 0.9736 | 0.9436 | 0.0730 |
| retrained on old labels | old labels | XGBoost | 0.9461 | 0.8823 | 0.0895 |
| retrained on old labels | old labels | GRU | 0.9534 | 0.9303 | 0.1027 |
| retrained on old labels | old labels | Ensemble (mean) | 0.9709 | 0.9468 | 0.0800 |
| retrained on corrected labels | corrected labels | XGBoost | 0.9572 | 0.8920 | 0.0786 |
| retrained on corrected labels | corrected labels | GRU | 0.9570 | 0.9277 | 0.0860 |
| retrained on corrected labels | corrected labels | Ensemble (mean) | 0.9750 | 0.9470 | 0.0677 |
