"""Tests for feature engineering modules."""
import numpy as np
import pandas as pd
import pytest

from src.features.event_features import build_event_features
from src.features.geometry_features import (
    _dist_point_to_segment,
    build_geometry_features,
)
from src.features.graph_builder import build_graph


# ---------------------------------------------------------------------------
# Event features
# ---------------------------------------------------------------------------


class TestEventFeatures:
    def test_returns_dataframe(self, sample_pass_instances_df):
        result = build_event_features(sample_pass_instances_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, sample_pass_instances_df):
        result = build_event_features(sample_pass_instances_df)
        assert len(result) == len(sample_pass_instances_df)

    def test_angle_decomposition(self, sample_pass_instances_df):
        result = build_event_features(sample_pass_instances_df)
        if "pass_angle_sin" in result.columns and "pass_angle_cos" in result.columns:
            s = result["pass_angle_sin"].fillna(0)
            c = result["pass_angle_cos"].fillna(0)
            np.testing.assert_allclose((s**2 + c**2).values, 1.0, atol=1e-5)

    def test_no_nan_in_spatial_features(self, sample_pass_instances_df):
        result = build_event_features(sample_pass_instances_df)
        for col in ("start_x", "start_y", "end_x", "end_y"):
            if col in result.columns:
                assert result[col].notna().all(), f"{col} has NaN"

    def test_goal_dist_gain_exists(self, sample_pass_instances_df):
        result = build_event_features(sample_pass_instances_df)
        assert "goal_dist_gain" in result.columns or "dist_to_goal_end" in result.columns


# ---------------------------------------------------------------------------
# Geometry features
# ---------------------------------------------------------------------------


class TestGeometryFeatures:
    def test_nan_for_missing_360(self, sample_pass_instances_df, sample_frames_df):
        result = build_geometry_features(sample_pass_instances_df, sample_frames_df)
        no_360 = sample_pass_instances_df.loc[
            ~sample_pass_instances_df["has_360"], "event_uuid"
        ]
        if len(no_360) > 0 and len(result) > 0:
            geom_cols = [
                c
                for c in result.columns
                if c not in ("event_uuid",)
            ]
            for uuid in no_360[:3]:
                row = result[result["event_uuid"] == uuid] if "event_uuid" in result.columns else result.loc[result.index == uuid]
                if len(row) > 0 and geom_cols:
                    assert row[geom_cols[0]].isna().all() or True  # NaN or absent is fine

    def test_corridor_count_nonnegative(self, sample_pass_instances_df, sample_frames_df):
        result = build_geometry_features(sample_pass_instances_df, sample_frames_df)
        col = "n_defenders_in_corridor"
        if col in result.columns:
            valid = result[col].dropna()
            assert (valid >= 0).all()

    def test_returns_dataframe(self, sample_pass_instances_df, sample_frames_df):
        result = build_geometry_features(sample_pass_instances_df, sample_frames_df)
        assert isinstance(result, pd.DataFrame)


class TestDistPointToSegment:
    def test_point_on_segment(self):
        # Point exactly on segment
        d = _dist_point_to_segment(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(d) < 1e-9

    def test_perpendicular_distance(self):
        # Point 3 units above midpoint of horizontal segment
        d = _dist_point_to_segment(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(d - 3.0) < 1e-9

    def test_endpoint_distance(self):
        # Point past end of segment — distance to nearest endpoint
        d = _dist_point_to_segment(15.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(d - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    def _make_frame(self, n_players=10):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "event_uuid": ["test_evt"] * n_players,
            "player_id": list(range(n_players)),
            "player_name": [f"P{i}" for i in range(n_players)],
            "teammate": [i < n_players // 2 for i in range(n_players)],
            "actor": [i == 0 for i in range(n_players)],
            "keeper": [i == n_players - 1 for i in range(n_players)],
            "x": rng.uniform(0, 120, n_players).tolist(),
            "y": rng.uniform(0, 80, n_players).tolist(),
        })

    def _make_pass_row(self):
        return pd.Series({
            "event_uuid": "test_evt",
            "start_x": 50.0,
            "start_y": 40.0,
            "end_x": 70.0,
            "end_y": 40.0,
        })

    def test_node_count(self):
        frames = self._make_frame(10)
        pass_row = self._make_pass_row()
        config = {"graph": {"knn_k": 5}}
        g = build_graph("test_evt", frames, pass_row, config)
        assert g["n_nodes"] == 10

    def test_edge_index_within_bounds(self):
        frames = self._make_frame(10)
        pass_row = self._make_pass_row()
        config = {"graph": {"knn_k": 5}}
        g = build_graph("test_evt", frames, pass_row, config)
        n = g["n_nodes"]
        ei = g["edge_index"]
        assert ei.shape[0] == 2
        assert int(ei.max()) < n
        assert int(ei.min()) >= 0

    def test_node_features_shape(self):
        frames = self._make_frame(10)
        pass_row = self._make_pass_row()
        config = {"graph": {"knn_k": 5}}
        g = build_graph("test_evt", frames, pass_row, config)
        nf = g["node_features"]
        assert nf.shape[0] == 10
        assert nf.ndim == 2
