"""
src/data/parse_events.py
========================
Parse and normalise a raw StatsBomb events DataFrame.

StatsBomb events are delivered with many sparse columns; this module
extracts the canonical fields needed by the Frame2Threat pipeline and
returns a clean, typed DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# StatsBomb pitch dimensions
PITCH_X_MAX: float = 120.0
PITCH_Y_MAX: float = 80.0

# Pass types that indicate a set piece (excluded from open-play filter)
_SET_PIECE_PASS_TYPES: frozenset[str] = frozenset(
    {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
)

# Play patterns that are not open play
_NON_OPEN_PLAY_PATTERNS: frozenset[str] = frozenset(
    {
        "From Corner",
        "From Free Kick",
        "From Goal Kick",
        "From Keeper",
        "From Throw In",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_coord(series: pd.Series, idx: int) -> pd.Series:
    """Extract a single coordinate (0=x, 1=y) from a list/array-valued Series."""
    def _get(v: object) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        try:
            return float(v[idx])  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            return np.nan

    return series.map(_get)


def _safe_str(series: pd.Series) -> pd.Series:
    """Convert a series to str, mapping NaN/None to pd.NA."""
    return series.where(series.notna(), other=pd.NA).astype(object).where(
        series.notna(), other=pd.NA
    )


def _bool_col(series: pd.Series) -> pd.Series:
    """Coerce a column to nullable boolean."""

    def _to_bool(v: object) -> object:
        if v is True or v is np.bool_(True):
            return True
        if v is False or v is np.bool_(False):
            return False
        if v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        try:
            int_v = int(v)  # type: ignore[arg-type]
            return bool(int_v)
        except (TypeError, ValueError):
            return pd.NA

    return series.map(_to_bool).astype("boolean")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_events(raw_events_df: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalise a raw StatsBomb events DataFrame.

    Extracts the canonical columns required by the Frame2Threat pipeline
    from the flattened events DataFrame produced by ``statsbombpy.sb.events()``.
    Pass-specific columns are populated for pass events and left as NaN
    for all other event types.

    Parameters
    ----------
    raw_events_df:
        DataFrame as returned by ``sb.events(match_id=…, flatten_attrs=True)``.

    Returns
    -------
    pd.DataFrame
        Clean, typed DataFrame.  Pass-event columns are always present but
        only non-null for rows where ``type_name == "Pass"``.
    """
    if raw_events_df.empty:
        logger.warning("parse_events received an empty DataFrame")
        return pd.DataFrame()

    df = raw_events_df.copy()
    logger.debug("Parsing %d raw events", len(df))

    out: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core identity columns
    # ------------------------------------------------------------------
    out["event_uuid"] = df.get("id", pd.Series(dtype=object))
    out["match_id"] = df.get("match_id", pd.Series(dtype="Int64")).astype("Int64")
    out["index"] = df.get("index", pd.Series(dtype="Int64")).astype("Int64")
    out["period"] = df.get("period", pd.Series(dtype="Int64")).astype("Int64")
    out["timestamp"] = df.get("timestamp", pd.Series(dtype=object))
    out["minute"] = df.get("minute", pd.Series(dtype="Int64")).astype("Int64")
    out["second"] = df.get("second", pd.Series(dtype="Int64")).astype("Int64")

    # ------------------------------------------------------------------
    # Categorical / name columns (statsbombpy flattens dicts to strings)
    # ------------------------------------------------------------------
    # In the flat statsbombpy format these are already plain strings
    out["type_name"] = df.get("type", pd.Series(dtype=object)).astype(object)
    out["team_name"] = df.get("team", pd.Series(dtype=object)).astype(object)
    out["player_name"] = df.get("player", pd.Series(dtype=object)).astype(object)
    out["play_pattern_name"] = df.get("play_pattern", pd.Series(dtype=object)).astype(object)
    out["possession_id"] = df.get("possession", pd.Series(dtype="Int64")).astype("Int64")

    # ------------------------------------------------------------------
    # Location (list column → two float columns)
    # ------------------------------------------------------------------
    location = df.get("location", pd.Series([None] * len(df)))
    out["location_x"] = _extract_coord(location, 0)
    out["location_y"] = _extract_coord(location, 1)

    # ------------------------------------------------------------------
    # Pass columns (present for all rows; NaN for non-pass events)
    # ------------------------------------------------------------------
    pass_end_loc = df.get("pass_end_location", pd.Series([None] * len(df)))
    out["pass_end_x"] = _extract_coord(pass_end_loc, 0)
    out["pass_end_y"] = _extract_coord(pass_end_loc, 1)

    out["pass_recipient_name"] = df.get("pass_recipient", pd.Series(dtype=object)).astype(object)
    out["pass_length"] = pd.to_numeric(df.get("pass_length"), errors="coerce")
    out["pass_angle"] = pd.to_numeric(df.get("pass_angle"), errors="coerce")

    # Categorical pass attributes (already flat strings in statsbombpy output)
    for src_col, dst_col in [
        ("pass_body_part", "pass_body_part"),
        ("pass_height", "pass_height"),
        ("pass_type", "pass_type"),
        ("pass_technique", "pass_technique"),
        ("pass_outcome", "pass_outcome_name"),
    ]:
        out[dst_col] = df.get(src_col, pd.Series(dtype=object)).astype(object)

    # Boolean pass flags
    for flag_col in (
        "under_pressure",
        "pass_switch",
        "pass_cross",
        "pass_through_ball",
        "pass_goal_kick",
        "pass_corner",
        "pass_free_kick",
    ):
        # pass_goal_kick / pass_corner / pass_free_kick may not be present in
        # the flattened DataFrame; derive from pass_type when absent
        col = df.get(flag_col, pd.Series([pd.NA] * len(df)))
        out[flag_col] = _bool_col(col)

    # Derive missing flag columns from pass_type if not already populated
    if out["pass_goal_kick"].isna().all():
        out["pass_goal_kick"] = _bool_col(
            out["pass_type"].apply(lambda v: v == "Goal Kick" if pd.notna(v) else pd.NA)
        )
    if out["pass_corner"].isna().all():
        out["pass_corner"] = _bool_col(
            out["pass_type"].apply(lambda v: v == "Corner" if pd.notna(v) else pd.NA)
        )
    if out["pass_free_kick"].isna().all():
        out["pass_free_kick"] = _bool_col(
            out["pass_type"].apply(
                lambda v: v in {"Free Kick", "Kick Off"} if pd.notna(v) else pd.NA
            )
        )

    result = pd.DataFrame(out)

    # ------------------------------------------------------------------
    # Mask pass-only columns for non-pass rows
    # ------------------------------------------------------------------
    _PASS_ONLY_COLS = [
        "pass_recipient_name",
        "pass_length",
        "pass_angle",
        "pass_end_x",
        "pass_end_y",
        "pass_body_part",
        "pass_height",
        "pass_type",
        "pass_technique",
        "pass_outcome_name",
        "pass_switch",
        "pass_cross",
        "pass_through_ball",
        "pass_goal_kick",
        "pass_corner",
        "pass_free_kick",
    ]
    is_not_pass = result["type_name"] != "Pass"
    for col in _PASS_ONLY_COLS:
        if col in result.columns:
            if result[col].dtype == "boolean":
                result.loc[is_not_pass, col] = pd.NA
            else:
                result.loc[is_not_pass, col] = np.nan

    logger.debug("Parsed events: %d rows, %d columns", len(result), len(result.columns))
    return result
