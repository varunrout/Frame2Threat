"""
src.models – Model definitions and training entry points.

Models available:
- baseline  : Logistic Regression, XGBoost, LightGBM (tabular features only).
- gnn       : GraphSAGE-based pass threat classifier (360 freeze-frame graph).
- hybrid    : GNN + GRU sequence fusion model.
- multitask : Shared encoder with per-label heads.

All trainable models follow a common ``fit`` / ``predict_proba`` interface so
that they can be swapped transparently in the evaluation harness.
"""
