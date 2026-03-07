"""360 freeze-frame geometry feature engineering."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Corridor half-width in metres (StatsBomb units)
_CORRIDOR_WIDTH: float = 5.0

# Radius used for overload calculation
_OVERLOAD_RADIUS: float = 15.0

# Goal centre
_GOAL_X: float = 120.0
_GOAL_Y: float = 40.0


class _LineBand(NamedTuple):
    x_lo: float
    x_hi: float


def build_geometry_features(
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute spatial geometry features from 360 freeze-frame data.

    For passes that have 360 data (``has_360 == True``) the full set of
    geometry features is populated.  For passes without 360 data every
    geometry column is ``NaN``.

    Parameters
    ----------
    pass_instances_df:
        One row per pass; must contain event_uuid, start_x, start_y,
        end_x, end_y, has_360.
    frames_df:
        Per-player-per-event table produced by
        :func:`src.data.parse_360.parse_360_frames`.
        Expected columns: event_uuid, teammate, keeper, actor, x, y.

    Returns
    -------
    pd.DataFrame
        Indexed by *event_uuid*.  Columns are geometry feature names.
        Returns NaN rows for events without 360 data.
    """
    if pass_instances_df is None or pass_instances_df.empty:
        logger.warning("build_geometry_features: empty pass_instances_df")
        return pd.DataFrame()

    logger.info(
        "Building geometry features for %d passes (%d with 360)",
        len(pass_instances_df),
        pass_instances_df.get("has_360", pd.Series(False)).sum(),
    )

    _geometry_cols = [
        "n_defenders_in_corridor",
        "n_defenders_goal_side",
        "nearest_defender_dist_passer",
        "nearest_defender_dist_receiver",
        "n_teammates_visible",
        "n_opponents_visible",
        "team_width",
        "team_depth",
        "opp_width",
        "opp_depth",
        "overload_target_zone",
        "receiver_between_lines",
        "pass_corridor_clear",
        "defensive_compactness",
    ]

    rows: list[dict] = []

    # Pre-index frames for fast lookup
    frames_by_uuid: dict[str, pd.DataFrame] = {}
    if frames_df is not None and not frames_df.empty:
        frames_by_uuid = {
            uuid: grp.reset_index(drop=True)
            for uuid, grp in frames_df.groupby("event_uuid")
        }

    for _, row in pass_instances_df.iterrows():
        uuid = row.get("event_uuid", None)
        has_360 = bool(row.get("has_360", False))

        if not has_360 or uuid not in frames_by_uuid:
            rows.append({"event_uuid": uuid, **{c: np.nan for c in _geometry_cols}})
            continue

        frame = frames_by_uuid[uuid]
        feat = _compute_frame_features(row, frame)
        feat["event_uuid"] = uuid
        rows.append(feat)

    result = pd.DataFrame(rows).set_index("event_uuid")

    # Cast numeric columns to float32
    for col in _geometry_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

    logger.info("Geometry features built: %d rows, %d cols", len(result), result.shape[1])
    return result


# ---------------------------------------------------------------------------
# Single-event geometry computation
# ---------------------------------------------------------------------------

def _compute_frame_features(
    pass_row: "pd.Series",
    frame: pd.DataFrame,
) -> dict:
    """Compute geometry features for a single pass event."""
    sx = float(pass_row["start_x"])
    sy = float(pass_row["start_y"])
    ex = float(pass_row["end_x"])
    ey = float(pass_row["end_y"])

    # Split players
    opponents = frame[~frame["teammate"].fillna(False).astype(bool)].copy()
    # Exclude actor from teammate set for geometry
    teammates = frame[
        frame["teammate"].fillna(False).astype(bool)
        & ~frame["actor"].fillna(False).astype(bool)
    ].copy()

    opp_x = opponents["x"].astype(float)
    opp_y = opponents["y"].astype(float)
    tm_x = teammates["x"].astype(float)
    tm_y = teammates["y"].astype(float)

    # ---- defenders in pass corridor ----
    n_def_corridor = _count_players_in_corridor(opponents, sx, sy, ex, ey, _CORRIDOR_WIDTH)

    # ---- defenders goal-side of receiver ----
    n_def_goal_side = int((opp_x > ex).sum()) if len(opponents) > 0 else 0

    # ---- nearest defender distances ----
    nd_passer = _nearest_dist(opp_x, opp_y, sx, sy)
    nd_receiver = _nearest_dist(opp_x, opp_y, ex, ey)

    # ---- visible player counts ----
    n_tm = int(len(teammates))
    n_opp = int(len(opponents))

    # ---- team spatial spread ----
    team_width = float(tm_y.max() - tm_y.min()) if len(tm_y) >= 2 else 0.0
    team_depth = float(tm_x.max() - tm_x.min()) if len(tm_x) >= 2 else 0.0
    opp_width = float(opp_y.max() - opp_y.min()) if len(opp_y) >= 2 else 0.0
    opp_depth = float(opp_x.max() - opp_x.min()) if len(opp_x) >= 2 else 0.0

    # ---- overload near receiver ----
    tm_near = int(
        (np.sqrt((tm_x - ex) ** 2 + (tm_y - ey) ** 2) <= _OVERLOAD_RADIUS).sum()
    ) if len(teammates) > 0 else 0
    opp_near = int(
        (np.sqrt((opp_x - ex) ** 2 + (opp_y - ey) ** 2) <= _OVERLOAD_RADIUS).sum()
    ) if len(opponents) > 0 else 0
    overload = tm_near - opp_near

    # ---- receiver between lines ----
    lines = _detect_defensive_lines(opponents)
    receiver_btw = _is_between_lines(ex, lines)

    # ---- corridor clear ----
    corridor_clear = 1 if n_def_corridor == 0 else 0

    # ---- defensive compactness ----
    compactness = opp_width * opp_depth

    return {
        "n_defenders_in_corridor": n_def_corridor,
        "n_defenders_goal_side": n_def_goal_side,
        "nearest_defender_dist_passer": nd_passer,
        "nearest_defender_dist_receiver": nd_receiver,
        "n_teammates_visible": n_tm,
        "n_opponents_visible": n_opp,
        "team_width": team_width,
        "team_depth": team_depth,
        "opp_width": opp_width,
        "opp_depth": opp_depth,
        "overload_target_zone": overload,
        "receiver_between_lines": receiver_btw,
        "pass_corridor_clear": corridor_clear,
        "defensive_compactness": compactness,
    }


# ---------------------------------------------------------------------------
# Helper functions (public for testing)
# ---------------------------------------------------------------------------

def _dist_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Compute distance from point (px, py) to line segment (ax,ay)-(bx,by).

    Parameters
    ----------
    px, py:  Query point coordinates.
    ax, ay:  Segment start coordinates.
    bx, by:  Segment end coordinates.

    Returns
    -------
    float
        Euclidean distance from the point to the closest point on the segment.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return float(np.sqrt((px - ax) ** 2 + (py - ay) ** 2))

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))


def _count_players_in_corridor(
    opponents_df: pd.DataFrame,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float = 5.0,
) -> int:
    """Count opponents within *width* metres of the pass line segment.

    Parameters
    ----------
    opponents_df:
        DataFrame of opponent players with 'x' and 'y' columns.
    start_x, start_y:
        Passer position.
    end_x, end_y:
        Intended receiver position.
    width:
        Half-width of the corridor in StatsBomb pitch units (metres).

    Returns
    -------
    int
        Number of opponents within the corridor.
    """
    if opponents_df is None or opponents_df.empty:
        return 0

    count = 0
    for _, p in opponents_df.iterrows():
        d = _dist_point_to_segment(
            float(p["x"]), float(p["y"]),
            start_x, start_y,
            end_x, end_y,
        )
        if d <= width:
            count += 1
    return count


def _detect_defensive_lines(
    opponents_df: pd.DataFrame,
    n_lines: int = 3,
) -> list[_LineBand]:
    """Detect defensive line x-bands using k-means-style clustering on x.

    Parameters
    ----------
    opponents_df:
        DataFrame of opponent players with 'x' column.
    n_lines:
        Number of defensive lines to detect.

    Returns
    -------
    list of _LineBand
        Sorted list of (x_lo, x_hi) bands representing each defensive line.
        Returns empty list when fewer than n_lines opponents are visible.
    """
    if opponents_df is None or opponents_df.empty:
        return []

    xs = opponents_df["x"].dropna().astype(float).values
    if len(xs) < n_lines:
        return []

    xs_sorted = np.sort(xs)

    # Use equal-count bins as a simple proxy for line detection
    indices = np.array_split(np.arange(len(xs_sorted)), n_lines)
    bands: list[_LineBand] = []
    for idx_group in indices:
        if len(idx_group) == 0:
            continue
        group_xs = xs_sorted[idx_group]
        margin = 2.0
        bands.append(_LineBand(
            x_lo=float(group_xs.min()) - margin,
            x_hi=float(group_xs.max()) + margin,
        ))
    return bands


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _nearest_dist(
    opp_x: "pd.Series",
    opp_y: "pd.Series",
    px: float,
    py: float,
) -> float:
    """Return the minimum distance from (px,py) to any opponent, or NaN."""
    if len(opp_x) == 0:
        return np.nan
    dists = np.sqrt((opp_x.values - px) ** 2 + (opp_y.values - py) ** 2)
    return float(dists.min())


def _is_between_lines(ex: float, lines: list[_LineBand]) -> int:
    """Return 1 if ex falls between the 2nd and 3rd defensive line bands."""
    if len(lines) < 3:
        return 0
    # lines are sorted by x; between 2nd and 3rd means in the gap
    second = lines[1]
    third = lines[2]
    if second.x_hi < ex < third.x_lo:
        return 1
    return 0
