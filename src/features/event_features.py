"""Event-context feature engineering for pass instances."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Goal centre on StatsBomb pitch
_GOAL_X: float = 120.0
_GOAL_Y: float = 40.0

# Pitch dimensions
_PITCH_X_MAX: float = 120.0
_PITCH_Y_MAX: float = 80.0

# Zone boundaries (x-axis cuts)
_THIRD_X1: float = 40.0   # own third / mid third boundary
_THIRD_X2: float = 80.0   # mid third / final third boundary
_BOX_X: float = 102.0     # approximate edge of penalty area
_CENTRAL_Y_LO: float = 26.0
_CENTRAL_Y_HI: float = 54.0

# Canonical body-part and height categories
_BODY_PARTS: tuple[str, ...] = ("foot", "head", "other")
_PASS_HEIGHTS: tuple[str, ...] = ("ground", "low", "high")


def build_event_features(pass_instances_df: pd.DataFrame) -> pd.DataFrame:
    """Compute event-context features from pass event data.

    Does **not** require 360 freeze-frame data.  All features are derived
    solely from the scalar pass event attributes already present in
    ``pass_instances_df``.

    Parameters
    ----------
    pass_instances_df:
        Output of :func:`src.data.join_pass_frames.build_pass_instances`.
        Must contain at least: event_uuid, start_x, start_y, end_x, end_y,
        pass_length, pass_angle (radians), minute, period, possession_id.

    Returns
    -------
    pd.DataFrame
        Indexed by *event_uuid*.  All numeric columns are cast to float32.
    """
    if pass_instances_df is None or pass_instances_df.empty:
        logger.warning("build_event_features: received empty DataFrame")
        return pd.DataFrame()

    df = pass_instances_df.copy()
    logger.info("Building event features for %d passes", len(df))

    out = pd.DataFrame(index=df.index)
    out.index = df["event_uuid"] if "event_uuid" in df.columns else df.index

    # ------------------------------------------------------------------
    # 1. Raw spatial
    # ------------------------------------------------------------------
    out["start_x"] = df["start_x"].astype(float)
    out["start_y"] = df["start_y"].astype(float)
    out["end_x"] = df["end_x"].astype(float)
    out["end_y"] = df["end_y"].astype(float)

    # ------------------------------------------------------------------
    # 2. Pass kinematics
    # ------------------------------------------------------------------
    out["pass_length"] = df["pass_length"].astype(float)

    angle = df["pass_angle"].astype(float)
    out["pass_angle_rad"] = angle
    out["pass_angle_sin"] = np.sin(angle)
    out["pass_angle_cos"] = np.cos(angle)

    # ------------------------------------------------------------------
    # 3. Directionality
    # ------------------------------------------------------------------
    out["is_forward"] = (out["end_x"] > out["start_x"]).astype(float)

    out["x_gain"] = out["end_x"] - out["start_x"]

    # ------------------------------------------------------------------
    # 4. Distance to goal
    # ------------------------------------------------------------------
    sx, sy = out["start_x"], out["start_y"]
    ex, ey = out["end_x"], out["end_y"]

    out["dist_to_goal_start"] = np.sqrt((_GOAL_X - sx) ** 2 + (_GOAL_Y - sy) ** 2)
    out["dist_to_goal_end"] = np.sqrt((_GOAL_X - ex) ** 2 + (_GOAL_Y - ey) ** 2)
    out["goal_dist_gain"] = out["dist_to_goal_start"] - out["dist_to_goal_end"]

    # ------------------------------------------------------------------
    # 5. Under pressure
    # ------------------------------------------------------------------
    under_pressure = df.get("under_pressure", pd.Series(False, index=df.index))
    out["under_pressure"] = under_pressure.fillna(False).astype(float)

    # ------------------------------------------------------------------
    # 6. Temporal
    # ------------------------------------------------------------------
    out["minute"] = df["minute"].astype(float)
    out["period"] = df["period"].astype(float)

    # ------------------------------------------------------------------
    # 7. Body part (one-hot: foot, head, other)
    # ------------------------------------------------------------------
    body_part_raw = df.get(
        "pass_body_part", pd.Series("", index=df.index)
    ).fillna("").str.lower()

    out["body_part_foot"] = body_part_raw.str.contains("foot").astype(float)
    out["body_part_head"] = body_part_raw.str.contains("head").astype(float)
    out["body_part_other"] = (
        (~body_part_raw.str.contains("foot") & ~body_part_raw.str.contains("head"))
        .astype(float)
    )

    # ------------------------------------------------------------------
    # 8. Pass height (one-hot: ground, low, high)
    # ------------------------------------------------------------------
    height_raw = df.get(
        "pass_height", pd.Series("", index=df.index)
    ).fillna("").str.lower()

    out["pass_height_ground"] = height_raw.str.contains("ground").astype(float)
    out["pass_height_low"] = height_raw.str.contains("low").astype(float)
    out["pass_height_high"] = height_raw.str.contains("high").astype(float)

    # ------------------------------------------------------------------
    # 9. Play pattern (one-hot) - if column present
    # ------------------------------------------------------------------
    if "play_pattern_name" in df.columns:
        play_patterns = df["play_pattern_name"].fillna("Unknown")
        pattern_dummies = pd.get_dummies(
            play_patterns, prefix="play_pattern", dtype=float
        )
        # Normalise column names
        pattern_dummies.columns = [
            c.lower().replace(" ", "_") for c in pattern_dummies.columns
        ]
        pattern_dummies.index = out.index
        out = pd.concat([out, pattern_dummies], axis=1)

    # ------------------------------------------------------------------
    # 10. Boolean pass flags
    # ------------------------------------------------------------------
    out["is_switch"] = _to_float_bool(df, "pass_switch")
    out["is_cross"] = _to_float_bool(df, "pass_cross")
    out["is_through_ball"] = _to_float_bool(df, "pass_through_ball")

    # ------------------------------------------------------------------
    # 11. Zone start (1-6)
    # ------------------------------------------------------------------
    out["zone_start"] = _compute_zone(out["start_x"], out["start_y"])

    # ------------------------------------------------------------------
    # 12. Possession length (index within possession)
    # ------------------------------------------------------------------
    if "possession_id" in df.columns:
        out["possession_length"] = (
            df.groupby("possession_id").cumcount().values.astype(float)
        )
    else:
        out["possession_length"] = 0.0

    # ------------------------------------------------------------------
    # Cast all numeric columns to float32
    # ------------------------------------------------------------------
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    logger.info("Event features built: %d columns", out.shape[1])
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float_bool(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a float (0/1) Series from a bool column, defaulting to 0."""
    if col in df.columns:
        return df[col].fillna(False).astype(float)
    return pd.Series(0.0, index=df.index)


def _compute_zone(start_x: pd.Series, start_y: pd.Series) -> pd.Series:
    """Map a start position to a pitch zone (1-6).

    Zones
    -----
    1 – Own third (x < 40)
    2 – Middle third, wide (40 ≤ x < 80, y outside central band)
    3 – Middle third, central (40 ≤ x < 80, y inside central band)
    4 – Final third, wide (80 ≤ x < 102, y outside central band)
    5 – Final third, central (80 ≤ x < 102, y inside central band)
    6 – Box area (x ≥ 102)
    """
    zone = pd.Series(1, index=start_x.index, dtype="int8")

    in_mid = (start_x >= _THIRD_X1) & (start_x < _THIRD_X2)
    in_final = (start_x >= _THIRD_X2) & (start_x < _BOX_X)
    in_box = start_x >= _BOX_X

    central = (start_y >= _CENTRAL_Y_LO) & (start_y <= _CENTRAL_Y_HI)

    zone[in_mid & ~central] = 2
    zone[in_mid & central] = 3
    zone[in_final & ~central] = 4
    zone[in_final & central] = 5
    zone[in_box] = 6

    return zone.astype(float)
