"""
src/data/parse_360.py
=====================
Normalise StatsBomb 360 freeze-frame data.

StatsBomb 360 data captures the positions of all visible outfield and
goalkeeper players at the moment each event occurs.  This module provides
two functions:

* :func:`parse_360_frames` – normalises raw frame rows into a flat
  per-visible-player table.
* :func:`get_frame_summary` – aggregates that table to one row per event
  with counts of visible players.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_360_frames(raw_frames_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 360 freeze-frame data into a flat per-player-per-event table.

    Accepts the DataFrame produced by ``src.data.ingest.get_360_frames()``
    (which handles both the high-level ``sb.frames()`` output and the manual
    JSON normalisation fallback).

    Input schema expected (any extra columns are ignored):

    * ``id``            – event UUID (may also be ``event_uuid``)
    * ``match_id``      – integer match identifier
    * ``teammate``      – bool; True if the player is on the same team as
                          the actor
    * ``actor``         – bool; True if this entry is the event actor
    * ``keeper``        – bool; True if the player is a goalkeeper
    * ``x``             – pitch x-coordinate (0–120)
    * ``y``             – pitch y-coordinate (0–80)
    * ``player_id``     – optional integer player id
    * ``player_name``   – optional player name string

    Parameters
    ----------
    raw_frames_df:
        DataFrame as returned by ``ingest.get_360_frames()``.

    Returns
    -------
    pd.DataFrame
        One row per visible player per event.  Columns: event_uuid,
        match_id, player_id, player_name, teammate, actor, keeper, x, y.
    """
    if raw_frames_df is None or raw_frames_df.empty:
        logger.warning("parse_360_frames received an empty/None DataFrame")
        return pd.DataFrame(
            columns=[
                "event_uuid",
                "match_id",
                "player_id",
                "player_name",
                "teammate",
                "actor",
                "keeper",
                "x",
                "y",
            ]
        )

    df = raw_frames_df.copy()
    logger.debug("Parsing 360 frames: %d rows", len(df))

    # ------------------------------------------------------------------
    # Normalise event_uuid column name
    # ------------------------------------------------------------------
    # sb.frames() renames event_uuid → id; the raw fallback also uses id
    if "id" in df.columns and "event_uuid" not in df.columns:
        df = df.rename(columns={"id": "event_uuid"})
    elif "event_uuid" not in df.columns:
        logger.error("Neither 'id' nor 'event_uuid' column found in frames DataFrame")
        raise ValueError("Cannot find event UUID column in 360 frames DataFrame")

    # ------------------------------------------------------------------
    # Handle freeze_frame column – present when the DataFrame has NOT yet
    # been exploded (some versions of the pipeline may pass the raw shape)
    # ------------------------------------------------------------------
    if "freeze_frame" in df.columns:
        df = _explode_freeze_frames(df)

    # ------------------------------------------------------------------
    # Ensure required columns exist
    # ------------------------------------------------------------------
    for col in ("teammate", "actor", "keeper"):
        if col not in df.columns:
            df[col] = pd.NA

    if "x" not in df.columns and "location" in df.columns:

        def _loc_coord(v: object, i: int) -> float:
            if v is None or (isinstance(v, float) and v != v):
                return np.nan
            try:
                return float(v[i])  # type: ignore[index]
            except (TypeError, IndexError):
                return np.nan

        df["x"] = df["location"].map(lambda v: _loc_coord(v, 0))
        df["y"] = df["location"].map(lambda v: _loc_coord(v, 1))
        df = df.drop(columns=["location"])

    for coord in ("x", "y"):
        if coord not in df.columns:
            df[coord] = np.nan

    for col in ("player_id", "player_name"):
        if col not in df.columns:
            df[col] = None

    # ------------------------------------------------------------------
    # Select and type-coerce output columns
    # ------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "event_uuid": df["event_uuid"].astype(object),
            "match_id": pd.to_numeric(df.get("match_id"), errors="coerce").astype("Int64"),
            "player_id": pd.to_numeric(df["player_id"], errors="coerce").astype("Int64"),
            "player_name": df["player_name"].astype(object),
            "teammate": _to_nullable_bool(df["teammate"]),
            "actor": _to_nullable_bool(df["actor"]),
            "keeper": _to_nullable_bool(df["keeper"]),
            "x": pd.to_numeric(df["x"], errors="coerce").astype(float),
            "y": pd.to_numeric(df["y"], errors="coerce").astype(float),
        }
    )

    out = out.reset_index(drop=True)
    logger.debug("Parsed 360 frames: %d rows", len(out))
    return out


def get_frame_summary(frames_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-event summary of 360 frame statistics.

    Aggregates the per-player frame table into one row per event with
    counts of visible players, teammates, opponents, and a flag indicating
    whether a goalkeeper is visible.

    Parameters
    ----------
    frames_df:
        Output of :func:`parse_360_frames`.

    Returns
    -------
    pd.DataFrame
        One row per event.  Columns: event_uuid, n_visible_players,
        n_visible_teammates, n_visible_opponents, has_keeper.
    """
    if frames_df is None or frames_df.empty:
        logger.warning("get_frame_summary received an empty/None DataFrame")
        return pd.DataFrame(
            columns=[
                "event_uuid",
                "n_visible_players",
                "n_visible_teammates",
                "n_visible_opponents",
                "has_keeper",
            ]
        )

    logger.debug("Computing frame summary for %d player-frame rows", len(frames_df))

    # Work on boolean copies – fill NA as False for aggregation purposes
    df = frames_df.copy()
    df["_teammate"] = df["teammate"].fillna(False).astype(bool)
    df["_keeper"] = df["keeper"].fillna(False).astype(bool)

    summary = (
        df.groupby("event_uuid", sort=False)
        .agg(
            n_visible_players=("event_uuid", "count"),
            n_visible_teammates=("_teammate", "sum"),
            has_keeper=("_keeper", "any"),
        )
        .reset_index()
    )

    summary["n_visible_opponents"] = (
        summary["n_visible_players"] - summary["n_visible_teammates"]
    ).clip(lower=0)

    # Cast to appropriate types
    for int_col in ("n_visible_players", "n_visible_teammates", "n_visible_opponents"):
        summary[int_col] = summary[int_col].astype("Int64")

    summary["has_keeper"] = summary["has_keeper"].astype(bool)

    logger.debug("Frame summary: %d events", len(summary))
    return summary[
        [
            "event_uuid",
            "n_visible_players",
            "n_visible_teammates",
            "n_visible_opponents",
            "has_keeper",
        ]
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _explode_freeze_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Explode a freeze_frame list column into individual player rows."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        event_uuid = row.get("event_uuid", row.get("id", None))
        match_id = row.get("match_id", None)
        ff_list = row.get("freeze_frame", [])
        if not isinstance(ff_list, list):
            continue
        for player in ff_list:
            if not isinstance(player, dict):
                continue
            loc = player.get("location", [None, None])
            player_info = player.get("player", {})
            rows.append(
                {
                    "event_uuid": event_uuid,
                    "match_id": match_id,
                    "teammate": player.get("teammate"),
                    "actor": player.get("actor"),
                    "keeper": player.get("keeper"),
                    "x": loc[0] if loc and len(loc) > 0 else None,
                    "y": loc[1] if loc and len(loc) > 1 else None,
                    "player_id": player_info.get("id") if isinstance(player_info, dict) else None,
                    "player_name": (
                        player_info.get("name") if isinstance(player_info, dict) else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable boolean dtype."""
    return series.map(
        lambda v: True if v is True or v == 1 else (False if v is False or v == 0 else pd.NA)
    ).astype("boolean")
