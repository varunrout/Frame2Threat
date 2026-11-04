"""
src.models – Model definitions and training entry points.

Models available:
- baseline  : Logistic Regression, XGBoost, LightGBM (tabular features only).
- pass_gnn  : graph-only GraphSAGE pass classifier (360 freeze-frame graph), plain torch.
- multitask : Shared encoder with per-label heads.

All trainable models follow a common ``fit`` / ``predict_proba`` interface so
that they can be swapped transparently in the evaluation harness.
"""
