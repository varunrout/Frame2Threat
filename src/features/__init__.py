"""
src.features – Feature engineering pipeline for pass threat modelling.

Sub-modules:
- event_features   : scalar features derived from StatsBomb event attributes.
- geometry_features: spatial features computed from 360 freeze-frame positions.
- graph_builder    : constructs PyG ``Data`` objects (nodes = players, edges = spatial relations).
- pipeline         : sklearn-compatible transformer that combines all feature groups.
"""
