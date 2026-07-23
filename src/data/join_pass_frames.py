"""
src/data/join_pass_frames.py
============================
Join parsed pass events with 360 freeze-frame summaries to produce the
canonical ``pass_instances`` table used by the Frame2Threat ML pipeline.

The resulting table contains every open-play pass event enriched with 360
summary statistics and NaN-initialised label columns ready for annotation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pass types that indicate a deliberate set piece delivery (excluded from
# open-play filter).  "Kick Off" and "Throw-in" are also excluded as they are
# restarts rather than open-play passes.
_EXCLUDED_PASS_TYPES: frozenset[str] = frozenset(
    {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
)

# Play patterns that are inherently not open play
_EXCLUDED_PLAY_PATTERNS: frozenset[str] = frozenset(
    {
        "From Corner",
        "From Free Kick",
        "From Goal Kick",
        "From Throw In",
    }
)

# Label columns initialised to NaN – will be populated by labelling pipeline
_LABEL_COLUMNS: list[str] = [
    "line_break",
    "strict_line_break",
    "loose_line_break",
    "dangerous_progression_k",
    "final_third_entry_k",
    "box_entry_k",
    "shot_within_k",
    "threat_gain",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pass_instances(
    events_df: pd.DataFrame,
    frames_summary_df: pd.DataFrame,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    """Build the canonical pass_instances table.

    Joins open-play pass events with their 360 freeze-frame summary
    statistics and appends NaN-initialised label columns.

    Parameters
    ----------
    events_df:
        Output of :func:`src.data.parse_events.parse_events`.  Expected to
        contain parsed events for one or more matches.
    frames_summary_df:
        Output of :func:`src.data.parse_360.get_frame_summary`.  One row
        per event UUID summarising 360 visible player counts.
    competition_id:
        Optional competition identifier to embed in the output table.
    season_id:
        Optional season identifier to embed in the output table.

    Returns
    -------
    pd.DataFrame
        One row per open-play pass.  See module docstring for full schema.
        Returns an empty DataFrame (with correct columns) if no passes
        remain after filtering.
    """
    if events_df is None or events_df.empty:
        logger.warning("build_pass_instances: events_df is empty")
        return _empty_pass_instances()

    # ------------------------------------------------------------------
    # 1. Filter to pass events only
    # ------------------------------------------------------------------
    passes = events_df[events_df["type_name"] == "Pass"].copy()
    if passes.empty:
        logger.warning("No pass events found in events_df")
        return _empty_pass_instances()

    logger.debug("Total pass events before open-play filter: %d", len(passes))

    # ------------------------------------------------------------------
    # 2. Open-play filter
    # ------------------------------------------------------------------
    passes = _filter_open_play(passes)
    logger.debug("Open-play passes after filter: %d", len(passes))

    if passes.empty:
        logger.warning("No open-play passes remain after filtering")
        return _empty_pass_instances()

    # ------------------------------------------------------------------
    # 3. Embed competition / season identifiers
    # ------------------------------------------------------------------
    passes["competition_id"] = competition_id if competition_id is not None else pd.NA
    passes["season_id"] = season_id if season_id is not None else pd.NA

    # ------------------------------------------------------------------
    # 4. Rename location columns to start_x / start_y
    # ------------------------------------------------------------------
    passes = passes.rename(
        columns={
            "location_x": "start_x",
            "location_y": "start_y",
            "pass_end_x": "end_x",
            "pass_end_y": "end_y",
        }
    )

    # ------------------------------------------------------------------
    # 5. Join 360 summary
    # ------------------------------------------------------------------
    has_frames = frames_summary_df is not None and not frames_summary_df.empty

    if has_frames:
        passes = passes.merge(
            frames_summary_df[
                [
                    "event_uuid",
                    "n_visible_players",
                    "n_visible_teammates",
                    "n_visible_opponents",
                ]
            ],
            on="event_uuid",
            how="left",
        )
        passes["has_360"] = passes["n_visible_players"].notna()
    else:
        passes["has_360"] = False
        passes["n_visible_players"] = pd.NA
        passes["n_visible_teammates"] = pd.NA
        passes["n_visible_opponents"] = pd.NA

    # ------------------------------------------------------------------
    # 6. Select and order final columns
    # ------------------------------------------------------------------
    result = _select_output_columns(passes, competition_id, season_id)

    # ------------------------------------------------------------------
    # 7. Append NaN label columns
    # ------------------------------------------------------------------
    for label_col in _LABEL_COLUMNS:
        result[label_col] = np.nan

    logger.info("Built pass_instances table: %d rows, %d columns", len(result), len(result.columns))
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_open_play(passes: pd.DataFrame) -> pd.DataFrame:
    """Remove set-piece and non-open-play passes."""
    mask = pd.Series(True, index=passes.index)

    # Exclude by pass_type
    if "pass_type" in passes.columns:
        mask &= ~passes["pass_type"].isin(_EXCLUDED_PASS_TYPES).fillna(False)

    # Exclude by play_pattern_name
    if "play_pattern_name" in passes.columns:
        mask &= ~passes["play_pattern_name"].isin(_EXCLUDED_PLAY_PATTERNS).fillna(False)

    # Exclude explicit boolean flag columns when True
    for flag_col in ("pass_goal_kick", "pass_corner", "pass_free_kick"):
        if flag_col in passes.columns:
            mask &= ~passes[flag_col].fillna(False).astype(bool)

    return passes.loc[mask].copy()


def _select_output_columns(
    passes: pd.DataFrame,
    competition_id: int | None,
    season_id: int | None,
) -> pd.DataFrame:
    """Select, rename, and type-coerce the canonical output columns."""

    def _get(col: str, default=np.nan) -> pd.Series:
        if col in passes.columns:
            return passes[col]
        return pd.Series(default, index=passes.index)

    out = pd.DataFrame(index=passes.index)

    # Identity
    out["match_id"] = _get("match_id")
    out["competition_id"] = competition_id if competition_id is not None else _get("competition_id")
    out["season_id"] = season_id if season_id is not None else _get("season_id")
    out["event_uuid"] = _get("event_uuid")
    out["possession_id"] = _get("possession_id")

    # Teams / players
    out["team_name"] = _get("team_name")
    out["player_name"] = _get("player_name")
    out["pass_recipient_name"] = _get("pass_recipient_name")

    # Temporal
    out["minute"] = _get("minute")
    out["second"] = _get("second")
    out["period"] = _get("period")

    # Spatial
    out["start_x"] = _get("start_x")
    out["start_y"] = _get("start_y")
    out["end_x"] = _get("end_x")
    out["end_y"] = _get("end_y")

    # Pass attributes
    out["pass_length"] = _get("pass_length")
    out["pass_angle"] = _get("pass_angle")
    out["pass_body_part"] = _get("pass_body_part")
    out["pass_height"] = _get("pass_height")
    out["pass_type"] = _get("pass_type")
    out["pass_outcome_name"] = _get("pass_outcome_name")
    out["under_pressure"] = _get("under_pressure")
    out["pass_switch"] = _get("pass_switch")
    out["pass_cross"] = _get("pass_cross")

    # 360 summary
    out["has_360"] = _get("has_360", default=False)
    out["n_visible_players"] = _get("n_visible_players")
    out["n_visible_teammates"] = _get("n_visible_teammates")
    out["n_visible_opponents"] = _get("n_visible_opponents")

    # Type coercions
    for int_col in (
        "match_id",
        "competition_id",
        "season_id",
        "possession_id",
        "minute",
        "second",
        "period",
        "n_visible_players",
        "n_visible_teammates",
        "n_visible_opponents",
    ):
        out[int_col] = pd.to_numeric(out[int_col], errors="coerce").astype("Int64")

    for float_col in ("start_x", "start_y", "end_x", "end_y", "pass_length", "pass_angle"):
        out[float_col] = pd.to_numeric(out[float_col], errors="coerce").astype(float)

    for bool_col in ("under_pressure", "pass_switch", "pass_cross", "has_360"):
        col_data = out[bool_col]
        out[bool_col] = col_data.map(
            lambda v: (
                False
                if (v is pd.NA or v is None or (isinstance(v, float) and pd.isna(v)))
                else bool(v)
            )
        ).astype(bool)

    return out


def _empty_pass_instances() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical pass_instances schema."""
    cols = [
        "match_id",
        "competition_id",
        "season_id",
        "event_uuid",
        "possession_id",
        "team_name",
        "player_name",
        "pass_recipient_name",
        "minute",
        "second",
        "period",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_length",
        "pass_angle",
        "pass_body_part",
        "pass_height",
        "pass_type",
        "pass_outcome_name",
        "under_pressure",
        "pass_switch",
        "pass_cross",
        "has_360",
        "n_visible_players",
        "n_visible_teammates",
        "n_visible_opponents",
        *_LABEL_COLUMNS,
    ]
    return pd.DataFrame(columns=cols)
