"""
src/labels/downstream_outcomes.py
==================================
Continuous threat-gain label based on an empirical zone-value map.

Overview
--------
``threat_gain`` is a transparent, data-driven proxy for the change in
*possession value* attributable to a single pass.  It is conceptually
similar to Expected Threat (xT) but is computed entirely from the StatsBomb
open-data corpus without relying on an external lookup table.

Zone-value map construction
----------------------------
1. Divide the pitch into a ``zone_grid_x × zone_grid_y`` grid
   (default 12 × 8 = 96 zones).
2. For every event in ``events_df``, compute the ball's *arrival position*:

   * **Pass** events: ``(pass_end_x, pass_end_y)``
   * **All other events**: ``(location_x, location_y)``

3. Assign each event to a zone based on arrival position.
4. For each possession, compute a scalar *possession value*::

       v_poss = shots + 0.3 × box_entries + 0.1 × final_third_entries

   where each term is a binary indicator (1 if the possession contained at
   least one such event, else 0).

5. For each zone Z, collect all possessions that visited Z (at least one
   event arrived in Z) and compute::

       zone_value(Z) = mean(v_poss) over those possessions

6. Normalise all zone values to [0, 1] by dividing by the global maximum.

Threat-gain computation
-----------------------
For each pass::

    threat_gain = zone_value(end_zone) - zone_value(start_zone)

where ``end_zone`` is determined by ``(end_x, end_y)`` and ``start_zone``
by ``(start_x, start_y)``.

Documented limitations
----------------------
* This is an *empirical* estimate derived from a finite dataset; zone values
  will be noisy for sparsely visited zones.
* The model uses *possession-level* outcomes, not continuous possession
  trajectories (i.e., it does not model sequences of actions).
* Low-data zones default to 0 zone value, biasing ``threat_gain`` toward 0
  for unusual pass origins/destinations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default pitch dimensions and zone thresholds
_PITCH_X: float = 120.0
_PITCH_Y: float = 80.0
_FINAL_THIRD_X: float = 80.0
_BOX_X: float = 102.0
_BOX_Y_MIN: float = 18.0
_BOX_Y_MAX: float = 62.0

# Outcome weights used in the possession-value formula
_SHOT_WEIGHT: float = 1.0
_BOX_WEIGHT: float = 0.3
_FT_WEIGHT: float = 0.1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_threat_gain(
    pass_instances_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute ``threat_gain`` as a zone-value delta for each pass.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass-instances table.  Required columns: ``event_uuid``,
        ``start_x``, ``start_y``, ``end_x``, ``end_y``.
    events_df:
        Full parsed events table (output of
        :func:`src.data.parse_events.parse_events`).
        Required columns: ``match_id``, ``possession_id``, ``type_name``,
        ``location_x``, ``location_y``.  Pass rows should also have
        ``pass_end_x``, ``pass_end_y``.
    config:
        Dictionary from the ``threat_gain`` section of
        ``configs/labels.yaml``.  Recognised keys:

        * ``zone_grid_x`` (int, default 12) – number of x-axis bins.
        * ``zone_grid_y`` (int, default  8) – number of y-axis bins.

    Returns
    -------
    pd.DataFrame
        Copy of ``pass_instances_df`` with a new (or overwritten) float
        column ``threat_gain``.  Values range from roughly -1 to +1;
        passes that move the ball into higher-value zones have positive
        threat_gain and vice versa.  Passes with missing coordinates receive
        ``NaN``.
    """
    grid_x: int = int(config.get("zone_grid_x", 12))
    grid_y: int = int(config.get("zone_grid_y", 8))

    logger.debug("compute_threat_gain: grid=%d×%d", grid_x, grid_y)

    result = pass_instances_df.copy()

    if events_df is None or events_df.empty:
        logger.warning(
            "compute_threat_gain: events_df is empty; " "all threat_gain values will be NaN."
        )
        result["threat_gain"] = np.nan
        return result

    # ------------------------------------------------------------------
    # 1. Build empirical zone-value map
    # ------------------------------------------------------------------
    zone_value_map = _build_zone_value_map(events_df, grid_x, grid_y)

    # ------------------------------------------------------------------
    # 2. Assign zones to pass start and end positions
    # ------------------------------------------------------------------
    start_zone_x = _to_zone_index(result["start_x"], grid_x, _PITCH_X)
    start_zone_y = _to_zone_index(result["start_y"], grid_y, _PITCH_Y)
    end_zone_x = _to_zone_index(result["end_x"], grid_x, _PITCH_X)
    end_zone_y = _to_zone_index(result["end_y"], grid_y, _PITCH_Y)

    # ------------------------------------------------------------------
    # 3. Look up zone values and compute delta
    # ------------------------------------------------------------------
    def _lookup(zone_col_x: pd.Series, zone_col_y: pd.Series) -> pd.Series:
        """Vectorised zone-value lookup."""
        keys = list(zip(zone_col_x.astype("Int64"), zone_col_y.astype("Int64")))
        return pd.Series(
            [
                zone_value_map.get(k, 0.0) if pd.notna(k[0]) and pd.notna(k[1]) else np.nan
                for k in keys
            ],
            index=zone_col_x.index,
        )

    start_value = _lookup(start_zone_x, start_zone_y)
    end_value = _lookup(end_zone_x, end_zone_y)

    result["threat_gain"] = end_value - start_value

    n_nan = int(result["threat_gain"].isna().sum())
    if n_nan:
        logger.debug("threat_gain: %d NaN values (missing pass coordinates)", n_nan)

    logger.info(
        "threat_gain computed: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
        result["threat_gain"].mean(),
        result["threat_gain"].std(),
        result["threat_gain"].min(),
        result["threat_gain"].max(),
    )

    return result


# ---------------------------------------------------------------------------
# Zone-value map builder
# ---------------------------------------------------------------------------


def _build_zone_value_map(
    events_df: pd.DataFrame,
    grid_x: int,
    grid_y: int,
) -> dict[tuple[int, int], float]:
    """Build a normalised empirical zone-value map.

    Parameters
    ----------
    events_df:
        Full parsed events table.
    grid_x:
        Number of pitch columns in the zone grid.
    grid_y:
        Number of pitch rows in the zone grid.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from ``(zone_x_idx, zone_y_idx)`` → normalised value in
        [0, 1].  Zones absent from the data are not present in the dict
        (callers should default to 0.0).
    """
    events = events_df.copy()

    # ------------------------------------------------------------------
    # Compute ball arrival position for each event
    # ------------------------------------------------------------------
    is_pass = events.get("type_name", pd.Series(dtype=object)) == "Pass"
    pass_end_x = events.get("pass_end_x", pd.Series(np.nan, index=events.index))
    pass_end_y = events.get("pass_end_y", pd.Series(np.nan, index=events.index))
    loc_x = events.get("location_x", pd.Series(np.nan, index=events.index))
    loc_y = events.get("location_y", pd.Series(np.nan, index=events.index))

    events["_ball_x"] = np.where(is_pass & pass_end_x.notna(), pass_end_x, loc_x)
    events["_ball_y"] = np.where(is_pass & pass_end_y.notna(), pass_end_y, loc_y)

    # ------------------------------------------------------------------
    # Assign zones
    # ------------------------------------------------------------------
    events["_zone_x"] = _to_zone_index(events["_ball_x"], grid_x, _PITCH_X)
    events["_zone_y"] = _to_zone_index(events["_ball_y"], grid_y, _PITCH_Y)

    # Drop rows with missing coordinates (cannot assign zone)
    events = events.dropna(subset=["_zone_x", "_zone_y"]).copy()
    events["_zone_x"] = events["_zone_x"].astype(int)
    events["_zone_y"] = events["_zone_y"].astype(int)
    events["_zone"] = list(zip(events["_zone_x"], events["_zone_y"]))

    # ------------------------------------------------------------------
    # Per-event outcome indicators
    # ------------------------------------------------------------------
    events["_is_shot"] = events.get("type_name", pd.Series(dtype=object)) == "Shot"
    events["_is_box"] = (
        (events["_ball_x"].fillna(-1) >= _BOX_X)
        & (events["_ball_y"].fillna(-1) >= _BOX_Y_MIN)
        & (events["_ball_y"].fillna(81) <= _BOX_Y_MAX)
    )
    events["_is_ft"] = events["_ball_x"].fillna(-1) >= _FINAL_THIRD_X

    # ------------------------------------------------------------------
    # Possession-level outcomes (binary: any shot / box / ft in possession)
    # ------------------------------------------------------------------
    _poss_key = ["match_id", "possession_id"]
    # Ensure groupby key columns exist
    for col in _poss_key:
        if col not in events.columns:
            logger.warning("events_df missing '%s'; using 0 as placeholder.", col)
            events[col] = 0

    poss_outcomes = (
        events.groupby(_poss_key, sort=False)
        .agg(
            _poss_shot=("_is_shot", "any"),
            _poss_box=("_is_box", "any"),
            _poss_ft=("_is_ft", "any"),
        )
        .reset_index()
    )

    poss_outcomes["_poss_value"] = (
        _SHOT_WEIGHT * poss_outcomes["_poss_shot"].astype(float)
        + _BOX_WEIGHT * poss_outcomes["_poss_box"].astype(float)
        + _FT_WEIGHT * poss_outcomes["_poss_ft"].astype(float)
    )

    # ------------------------------------------------------------------
    # Unique (match, possession, zone) visits – one row per visit
    # ------------------------------------------------------------------
    zone_visits = (
        events[_poss_key + ["_zone"]]
        .drop_duplicates()
        .merge(poss_outcomes[_poss_key + ["_poss_value"]], on=_poss_key, how="left")
    )

    # ------------------------------------------------------------------
    # Zone value = mean possession value over all possessions that visited
    # ------------------------------------------------------------------
    zone_stats = (
        zone_visits.groupby("_zone", sort=False)["_poss_value"]
        .mean()
        .reset_index()
        .rename(columns={"_poss_value": "_zone_value"})
    )

    # ------------------------------------------------------------------
    # Normalise to [0, 1]
    # ------------------------------------------------------------------
    max_val = zone_stats["_zone_value"].max()
    if max_val > 0:
        zone_stats["_zone_value_norm"] = zone_stats["_zone_value"] / max_val
    else:
        zone_stats["_zone_value_norm"] = 0.0

    zone_map: dict[tuple[int, int], float] = dict(
        zip(zone_stats["_zone"], zone_stats["_zone_value_norm"])
    )

    logger.debug(
        "Zone-value map built: %d zones, max_raw=%.4f",
        len(zone_map),
        float(max_val) if max_val > 0 else 0.0,
    )
    return zone_map


# ---------------------------------------------------------------------------
# Coordinate → zone-index helper
# ---------------------------------------------------------------------------


def _to_zone_index(
    coord: pd.Series,
    n_bins: int,
    pitch_dim: float,
) -> pd.Series:
    """Map a coordinate Series to integer zone indices in ``[0, n_bins-1]``.

    Parameters
    ----------
    coord:
        Series of float pitch coordinates (x or y).
    n_bins:
        Number of equal-width bins along the axis.
    pitch_dim:
        Total pitch length along the axis (120 for x, 80 for y).

    Returns
    -------
    pd.Series
        Integer zone indices.  NaN coordinates remain NaN (using nullable
        Int64 dtype).
    """
    # Clip coordinates to valid pitch range before binning
    clipped = coord.clip(lower=0.0, upper=pitch_dim)
    zone = (clipped / pitch_dim * n_bins).apply(
        lambda v: min(int(v), n_bins - 1) if pd.notna(v) else np.nan
    )
    return zone
