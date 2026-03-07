"""
src/labels/dangerous_progression.py
=====================================
Downstream binary outcome labels for pass events.

For each pass in the canonical ``pass_instances`` table, this module looks
ahead **k** events within the *same possession* and computes four binary
labels:

``final_third_entry_k``
    The ball reaches the attacking final third (x ≥ 80) within k events.

``box_entry_k``
    The ball enters the penalty box (x ≥ 102, 18 ≤ y ≤ 62) within k events.

``shot_within_k``
    A Shot event occurs within k events in the same possession.

``dangerous_progression_k``
    Union label: True if *any* of the three conditions above is satisfied.

Key design constraints
----------------------
* **No information leakage** – only events that occur *after* the pass
  (higher ``index`` value within the same possession) are examined.
* **Possession boundary** – the look-ahead is capped at the last event of the
  current possession; events belonging to future possessions are never used.
* **k defaults to 5** – configurable via ``configs/labels.yaml``.
* **No NaN** – every pass receives a True/False label regardless of 360
  availability; there is no geometry dependency.

Ball-position convention
------------------------
To determine where the ball ends up after each future event:

* **Pass** events  → ball lands at ``(pass_end_x, pass_end_y)``.
* **All other events** → ball is at ``(location_x, location_y)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pitch-zone thresholds (also set in configs/labels.yaml)
# ---------------------------------------------------------------------------
_FINAL_THIRD_X: float = 80.0
_BOX_X: float = 102.0
_BOX_Y_MIN: float = 18.0
_BOX_Y_MAX: float = 62.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_downstream_labels(
    events_df: pd.DataFrame,
    pass_instances_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute binary downstream outcome labels for each pass.

    For each pass in ``pass_instances_df``, inspects the next *k* events
    in the same match possession and sets binary outcome flags.

    Parameters
    ----------
    events_df:
        Full parsed events table, output of
        :func:`src.data.parse_events.parse_events`.  Must contain:
        ``event_uuid``, ``match_id``, ``possession_id``, ``index``,
        ``type_name``, ``location_x``, ``location_y``.
        Pass rows should additionally have ``pass_end_x``, ``pass_end_y``.
    pass_instances_df:
        Canonical pass-instances table.  Must contain: ``event_uuid``,
        ``match_id``, ``possession_id``.
    config:
        Dictionary from ``configs/labels.yaml``.  The ``dangerous_progression``
        section's ``k`` value is used; individual sub-section ``k`` values
        (``final_third_entry.k``, ``box_entry.k``, ``shot_within.k``) are
        respected when present and override the top-level ``k``.

    Returns
    -------
    pd.DataFrame
        Copy of ``pass_instances_df`` with four new (or overwritten) columns:

        * ``final_third_entry_k``   – bool
        * ``box_entry_k``           – bool
        * ``shot_within_k``         – bool
        * ``dangerous_progression_k`` – bool (OR of the three above)

        All columns are ``bool`` dtype; passes with no future events in
        their possession (e.g., the last pass) receive ``False``.
    """
    if events_df is None or events_df.empty:
        logger.warning(
            "compute_downstream_labels: events_df is empty; "
            "all downstream labels will be False."
        )
        return _fill_false_labels(pass_instances_df.copy())

    # ------------------------------------------------------------------
    # Resolve k values (support per-label overrides in config)
    # ------------------------------------------------------------------
    default_k: int = int(config.get("k", 5))
    k_ft: int = int(config.get("final_third_entry", {}).get("k", default_k))
    k_box: int = int(config.get("box_entry", {}).get("k", default_k))
    k_shot: int = int(config.get("shot_within", {}).get("k", default_k))
    k_max: int = max(k_ft, k_box, k_shot)

    # Pitch threshold overrides
    final_third_x: float = float(
        config.get("final_third_entry", {}).get("final_third_x", _FINAL_THIRD_X)
    )
    box_x: float = float(config.get("box_entry", {}).get("box_x", _BOX_X))
    box_y_min: float = float(config.get("box_entry", {}).get("box_y_min", _BOX_Y_MIN))
    box_y_max: float = float(config.get("box_entry", {}).get("box_y_max", _BOX_Y_MAX))

    logger.debug(
        "downstream labels: k_ft=%d, k_box=%d, k_shot=%d, k_max=%d",
        k_ft, k_box, k_shot, k_max,
    )

    # ------------------------------------------------------------------
    # 1. Sort events and assign within-possession rank
    # ------------------------------------------------------------------
    events = events_df.copy()
    _required = {"event_uuid", "match_id", "possession_id", "index", "type_name"}
    _missing = _required - set(events.columns)
    if _missing:
        logger.error("events_df missing required columns: %s", _missing)
        return _fill_false_labels(pass_instances_df.copy())

    events = events.sort_values(
        ["match_id", "possession_id", "index"], kind="mergesort"
    )
    events["_pos_rank"] = (
        events.groupby(["match_id", "possession_id"], sort=False).cumcount()
    )

    # ------------------------------------------------------------------
    # 2. Compute ball destination for each event
    # ------------------------------------------------------------------
    is_pass = events["type_name"] == "Pass"

    pass_end_x = events.get("pass_end_x", pd.Series(np.nan, index=events.index))
    pass_end_y = events.get("pass_end_y", pd.Series(np.nan, index=events.index))
    loc_x = events.get("location_x", pd.Series(np.nan, index=events.index))
    loc_y = events.get("location_y", pd.Series(np.nan, index=events.index))

    # For pass events, ball destination is pass_end; otherwise it is location
    events["_ball_x"] = np.where(
        is_pass & pass_end_x.notna(), pass_end_x, loc_x
    )
    events["_ball_y"] = np.where(
        is_pass & pass_end_y.notna(), pass_end_y, loc_y
    )

    # ------------------------------------------------------------------
    # 3. Pre-compute per-event outcome flags
    # ------------------------------------------------------------------
    events["_is_shot"] = events["type_name"] == "Shot"
    events["_is_ft"] = events["_ball_x"].fillna(-1) >= final_third_x
    events["_is_box"] = (
        (events["_ball_x"].fillna(-1) >= box_x)
        & (events["_ball_y"].fillna(-1) >= box_y_min)
        & (events["_ball_y"].fillna(81) <= box_y_max)
    )

    # ------------------------------------------------------------------
    # 4. Get within-possession rank for each pass in pass_instances_df
    # ------------------------------------------------------------------
    pass_rank_lookup = events[
        ["event_uuid", "match_id", "possession_id", "_pos_rank"]
    ].rename(columns={"_pos_rank": "_pass_rank"})

    result = pass_instances_df.copy()

    # Ensure consistent types for merge keys
    for col in ("match_id", "possession_id"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        if col in pass_rank_lookup.columns:
            pass_rank_lookup[col] = pd.to_numeric(
                pass_rank_lookup[col], errors="coerce"
            )

    result = result.merge(
        pass_rank_lookup,
        on=["event_uuid", "match_id", "possession_id"],
        how="left",
    )

    n_unmatched = result["_pass_rank"].isna().sum()
    if n_unmatched:
        logger.warning(
            "%d passes could not be matched to events_df (possession rank "
            "will be NaN → labels default to False).",
            n_unmatched,
        )

    # ------------------------------------------------------------------
    # 5. Cross-join passes × future events within same possession
    # ------------------------------------------------------------------
    future_cols = [
        "match_id",
        "possession_id",
        "_pos_rank",
        "_is_shot",
        "_is_ft",
        "_is_box",
    ]
    future_events = events[future_cols].copy()

    passes_for_join = result[
        ["event_uuid", "match_id", "possession_id", "_pass_rank"]
    ].copy()

    cross = passes_for_join.merge(
        future_events,
        on=["match_id", "possession_id"],
        how="left",
    )

    # Keep only events strictly after the pass and within k_max steps
    valid_future = (
        cross["_pos_rank"] > cross["_pass_rank"]
    ) & (
        cross["_pos_rank"] <= cross["_pass_rank"] + k_max
    )
    cross = cross.loc[valid_future].copy()

    # Per-label k filtering and aggregation
    def _aggregate_label(
        cross_df: pd.DataFrame,
        flag_col: str,
        k_val: int,
    ) -> pd.Series:
        """Aggregate a boolean flag over the next k_val events per pass."""
        within_k = cross_df[cross_df["_pos_rank"] <= cross_df["_pass_rank"] + k_val]
        return (
            within_k.groupby("event_uuid")[flag_col]
            .any()
            .rename(flag_col)
        )

    shot_agg = _aggregate_label(cross, "_is_shot", k_shot)
    ft_agg = _aggregate_label(cross, "_is_ft", k_ft)
    box_agg = _aggregate_label(cross, "_is_box", k_box)

    agg = (
        pd.DataFrame({"shot_within_k": shot_agg, "final_third_entry_k": ft_agg, "box_entry_k": box_agg})
        .reset_index()
        .rename(columns={"index": "event_uuid"})
    )

    # Drop any pre-existing label columns to avoid pandas _x/_y suffix conflicts
    # (pass_instances from build_pass_instances already has NaN-initialised labels).
    _output_cols = ["shot_within_k", "final_third_entry_k", "box_entry_k", "dangerous_progression_k"]
    result = result.drop(columns=[c for c in _output_cols if c in result.columns], errors="ignore")

    result = result.merge(agg, on="event_uuid", how="left")

    # ------------------------------------------------------------------
    # 6. Fill NaN → False (passes at end of possession / no future events)
    # ------------------------------------------------------------------
    for col in ("shot_within_k", "final_third_entry_k", "box_entry_k"):
        result[col] = result[col].fillna(False).astype(bool)

    result["dangerous_progression_k"] = (
        result["shot_within_k"]
        | result["final_third_entry_k"]
        | result["box_entry_k"]
    )

    # Drop temporary helper column
    result = result.drop(columns=["_pass_rank"], errors="ignore")

    # ------------------------------------------------------------------
    # Logging summary
    # ------------------------------------------------------------------
    n_total = len(result)
    for col in ("final_third_entry_k", "box_entry_k", "shot_within_k", "dangerous_progression_k"):
        n_pos = int(result[col].sum())
        logger.info(
            "%s: %d / %d (%.1f%%)", col, n_pos, n_total, 100.0 * n_pos / n_total if n_total else 0.0
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fill_false_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Set all downstream label columns to False in-place and return df."""
    for col in (
        "final_third_entry_k",
        "box_entry_k",
        "shot_within_k",
        "dangerous_progression_k",
    ):
        df[col] = False
    return df
