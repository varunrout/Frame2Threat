"""
src/data/parse_lineups.py
=========================
Normalise StatsBomb lineup data into a flat per-player DataFrame.

``statsbombpy.sb.lineups()`` returns a dict mapping team names to DataFrames
with nested ``positions`` and ``country`` columns.  This module flattens
those structures into a canonical schema.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def parse_lineups(raw_lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw lineups DataFrame into a flat per-player table.

    Accepts the output of either:

    * ``statsbombpy.sb.lineups(match_id=…)`` – which returns a ``dict``
      ``{team_name: DataFrame}`` – after being pre-combined via
      ``src.data.ingest.get_lineups()``, or
    * The already-combined DataFrame produced by that helper.

    The function handles three input shapes:

    1. The combined DataFrame written by ``ingest.get_lineups()`` which
       already has ``match_id`` and ``team_name`` columns.
    2. A dict ``{team_name: DataFrame}`` returned directly by statsbombpy
       (for convenience when called outside the normal pipeline).
    3. A single-team DataFrame (team name extracted from ``team_name`` col if
       present, otherwise labelled ``"unknown"``).

    Parameters
    ----------
    raw_lineups_df:
        DataFrame (or dict) as described above.

    Returns
    -------
    pd.DataFrame
        One row per player with columns:
        match_id, team_id, team_name, player_id, player_name,
        jersey_number, position_name, country_name.
    """
    if isinstance(raw_lineups_df, dict):
        # Direct dict from sb.lineups – combine and recurse
        frames = []
        for team_name, team_df in raw_lineups_df.items():
            team_df = team_df.copy()
            team_df["team_name"] = team_name
            frames.append(team_df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return parse_lineups(combined)

    if raw_lineups_df.empty:
        logger.warning("parse_lineups received an empty DataFrame")
        return pd.DataFrame()

    df = raw_lineups_df.copy()
    logger.debug("Parsing lineups: %d rows", len(df))

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        match_id = row.get("match_id", pd.NA)
        team_name = row.get("team_name", "unknown")
        team_id = row.get("team_id", pd.NA)
        player_id = row.get("player_id", pd.NA)
        player_name = row.get("player_name", pd.NA)
        jersey_number = row.get("jersey_number", pd.NA)

        # Country may be a dict or a string (depends on statsbombpy version /
        # whether ingest already flattened it)
        country_raw = row.get("country_name", row.get("country", pd.NA))
        if isinstance(country_raw, dict):
            country_name = country_raw.get("name")
        elif isinstance(country_raw, (list,)):
            country_name = None
        elif country_raw is None:
            country_name = None
        else:
            try:
                country_name = None if pd.isna(country_raw) else str(country_raw)
            except (TypeError, ValueError):
                country_name = str(country_raw)

        # Positions: a list/array of dicts [{"position_id": 1, "position": "GK"}, ...]
        positions_raw = row.get("positions", [])
        # Handle both Python lists and numpy object arrays
        if hasattr(positions_raw, "__iter__") and not isinstance(positions_raw, str):
            positions_iter = list(positions_raw)
        else:
            positions_iter = []

        if positions_iter:
            for pos_entry in positions_iter:
                if isinstance(pos_entry, dict):
                    position_name = pos_entry.get("position", pd.NA)
                else:
                    position_name = str(pos_entry)
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player_name,
                        "jersey_number": jersey_number,
                        "position_name": position_name,
                        "country_name": country_name,
                    }
                )
        else:
            # No positional data available – still emit a row
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player_name,
                    "jersey_number": jersey_number,
                    "position_name": pd.NA,
                    "country_name": country_name,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "team_id",
                "team_name",
                "player_id",
                "player_name",
                "jersey_number",
                "position_name",
                "country_name",
            ]
        )

    result = pd.DataFrame(rows)

    # Type coercions
    for int_col in ("match_id", "team_id", "player_id", "jersey_number"):
        if int_col in result.columns:
            result[int_col] = pd.to_numeric(result[int_col], errors="coerce").astype("Int64")

    for str_col in ("team_name", "player_name", "position_name", "country_name"):
        if str_col in result.columns:
            result[str_col] = result[str_col].astype(object)

    # De-duplicate: keep one row per (match_id, player_id, position_name)
    result = result.drop_duplicates(
        subset=["match_id", "player_id", "position_name"], keep="first"
    ).reset_index(drop=True)

    logger.debug("Parsed lineups: %d rows, %d columns", len(result), len(result.columns))
    return result
