"""Position-only baseline features and the precision operating point (second_holdout_protocol.md)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.position_baseline import (
    POSITION_FEATURES,
    position_features,
    precision_threshold,
)


def test_position_features_geometry():
    rows = pd.DataFrame({"cur_x": [120.0, 60.0, 0.0], "cur_y": [40.0, 0.0, 79.9]})
    f = position_features(rows)
    assert list(f.columns) == POSITION_FEATURES
    assert f.dist_goal.iloc[0] == pytest.approx(0.0)
    assert f.dist_goal.iloc[1] == pytest.approx(np.hypot(60, 40))
    assert f.angle_goal.iloc[1] == pytest.approx(np.arctan2(40, 60))
    assert f.zone.tolist() == [3 * 5 + 1, 3 * 3 + 0, 0 + 2]
    assert (f.attack_dir == 1).all()


def test_precision_threshold_picks_lowest_threshold_meeting_target():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    out = precision_threshold(y, p, 0.75, 0.0)
    # top 4 -> 3/4 = 0.75 precision; top 5 -> 0.6
    assert out["reachable"] and out["threshold"] == pytest.approx(0.6)
    assert out["precision"] == pytest.approx(0.75) and out["recall"] == pytest.approx(1.0)


def test_precision_threshold_respects_ties_and_support():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.2, 0.1])
    assert precision_threshold(y, p, 0.9, 0.0)["reachable"] is False
    assert (
        precision_threshold(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.1, 0.1]), 0.9, 0.5)[
            "reachable"
        ]
        is False
    )
