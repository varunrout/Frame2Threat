"""Position-only features and the precision operating point fixed in ``second_holdout_protocol.md``.

Coordinates are the causal ``cur_x`` / ``cur_y`` snapshot features: the last possessing-team event location in the
possessing team's frame (StatsBomb 120 x 80 units, attacking towards x = 120, goal centre at (120, 40)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.possession_frame import PITCH_LENGTH, PITCH_WIDTH

GOAL_X, GOAL_Y = PITCH_LENGTH, PITCH_WIDTH / 2
POSITION_FEATURES = ["cur_x", "cur_y", "dist_goal", "angle_goal", "zone", "attack_dir"]


def position_features(rows: pd.DataFrame) -> pd.DataFrame:
    """The six position-only features of the boosted baseline; no event history."""
    x = rows["cur_x"].to_numpy(dtype=float)
    y = rows["cur_y"].to_numpy(dtype=float)
    dx = GOAL_X - x
    dy = GOAL_Y - y
    zone_x = np.clip(np.floor(x / 20.0), 0, 5)
    zone_y = np.clip(np.floor(y / (PITCH_WIDTH / 3.0)), 0, 2)
    return pd.DataFrame(
        {
            "cur_x": x,
            "cur_y": y,
            "dist_goal": np.hypot(dx, dy),
            "angle_goal": np.arctan2(np.abs(dy), np.maximum(dx, 0.0)),
            "zone": 3 * zone_x + zone_y,
            # The possessing-team frame always attacks towards x = 120: constant by construction.
            "attack_dir": np.ones(len(x)),
        },
        index=rows.index,
    )


def precision_threshold(
    y: np.ndarray, p: np.ndarray, target: float, min_alert_share: float
) -> dict[str, float | bool | None]:
    """Lowest threshold whose alerts (p >= threshold) reach ``target`` precision on these rows.

    Requires at least ``min_alert_share`` of rows to alert. Returns the threshold with its precision and recall on
    the same rows, or ``reachable = False``.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    # Only cut between distinct scores, so ties are alerted together.
    last_of_value = np.r_[ps[1:] != ps[:-1], True]
    tp = np.cumsum(ys)
    n_alert = np.arange(1, len(ys) + 1)
    precision = tp / n_alert
    ok = last_of_value & (precision >= target) & (n_alert >= min_alert_share * len(ys))
    if not ok.any():
        return {"reachable": False, "threshold": None, "precision": None, "recall": None}
    k = int(np.nonzero(ok)[0].max())
    return {
        "reachable": True,
        "threshold": float(ps[k]),
        "precision": float(precision[k]),
        "recall": float(tp[k] / max(ys.sum(), 1)),
        "alert_share": float(n_alert[k] / len(ys)),
    }
