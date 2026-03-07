"""
src.evaluation – Metrics computation, calibration, and ablation utilities.

Provides:
- Ranking metrics  : nDCG@k, MRR, top-1 hit-rate.
- Classification   : ROC-AUC, PR-AUC, Brier score, log-loss.
- Calibration      : Expected Calibration Error (ECE), reliability diagrams.
- Ablation runner  : orchestrates the experiment matrix defined in eval.yaml.
"""
