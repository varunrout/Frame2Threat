"""
src/data/ingest.py
==================
Download and cache StatsBomb open data using statsbombpy.

All functions check for cached parquet files first; if absent they fetch via
the statsbombpy API, persist to data/raw/, and return the DataFrame.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "data.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _raw_dir() -> Path:
    cfg = _load_config()
    p = _REPO_ROOT / cfg["statsbomb"]["raw_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# statsbombpy import (suppress credentials warning for open data)
# ---------------------------------------------------------------------------


def _sb():
    """Lazy import of statsbombpy.sb, suppressing open-data auth warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsbombpy import sb
    return sb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_competitions() -> pd.DataFrame:
    """Return a DataFrame of all StatsBomb open competitions.

    Checks ``data/raw/competitions.parquet`` first; fetches and caches
    via statsbombpy if not found.

    Returns
    -------
    pd.DataFrame
        Columns: competition_id, season_id, country_name, competition_name,
        competition_gender, season_name, match_available_360, …
    """
    cache = _raw_dir() / "competitions.parquet"
    if cache.exists():
        logger.info("Loading competitions from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Fetching competitions from StatsBomb API")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _sb().competitions()
        df.to_parquet(cache, index=False)
        logger.info("Saved %d competition/season rows to %s", len(df), cache)
        return df
    except Exception as exc:
        logger.error("Failed to fetch competitions: %s", exc)
        raise


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return a DataFrame of matches for a given competition and season.

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier.
    season_id:
        StatsBomb season identifier.

    Returns
    -------
    pd.DataFrame
        One row per match; includes match_id, teams, scores, 360 status, …
    """
    cache = _raw_dir() / f"matches_c{competition_id}_s{season_id}.parquet"
    if cache.exists():
        logger.info("Loading matches from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Fetching matches for competition_id=%d season_id=%d", competition_id, season_id)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _sb().matches(competition_id=competition_id, season_id=season_id)
        df.to_parquet(cache, index=False)
        logger.info("Saved %d matches to %s", len(df), cache)
        return df
    except Exception as exc:
        logger.error(
            "Failed to fetch matches (comp=%d season=%d): %s",
            competition_id,
            season_id,
            exc,
        )
        raise


def get_events(match_id: int) -> pd.DataFrame:
    """Return a flattened events DataFrame for a single match.

    Parameters
    ----------
    match_id:
        StatsBomb match identifier.

    Returns
    -------
    pd.DataFrame
        One row per event; all nested attributes are flattened by statsbombpy.
    """
    cache = _raw_dir() / f"events_{match_id}.parquet"
    if cache.exists():
        logger.debug("Loading events from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Fetching events for match_id=%d", match_id)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _sb().events(match_id=match_id, flatten_attrs=True)
        df.to_parquet(cache, index=False)
        logger.debug("Saved %d events to %s", len(df), cache)
        return df
    except Exception as exc:
        logger.error("Failed to fetch events (match_id=%d): %s", match_id, exc)
        raise


def get_360_frames(match_id: int) -> Optional[pd.DataFrame]:
    """Return a normalised 360 freeze-frame DataFrame for a single match.

    Each row represents one visible player in a single event's freeze frame.
    statsbombpy's ``sb.frames()`` occasionally raises ``InvalidIndexError``
    on newer pandas versions, so this function falls back to loading the raw
    JSON via ``statsbombpy.public`` and normalises it manually.

    Parameters
    ----------
    match_id:
        StatsBomb match identifier.

    Returns
    -------
    pd.DataFrame or None
        Columns: id (event_uuid), match_id, visible_area, teammate, actor,
        keeper, x, y.  Returns ``None`` if no 360 data is available.
    """
    cache = _raw_dir() / f"frames_{match_id}.parquet"
    if cache.exists():
        logger.debug("Loading 360 frames from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Fetching 360 frames for match_id=%d", match_id)

    # First attempt: statsbombpy high-level API
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _sb().frames(match_id=match_id)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _normalise_frames_df(df, match_id)
            df.to_parquet(cache, index=False)
            logger.debug("Saved %d frame rows to %s", len(df), cache)
            return df
    except Exception as exc:
        logger.warning(
            "sb.frames() failed for match_id=%d (%s); falling back to raw JSON",
            match_id,
            exc,
        )

    # Fallback: manual JSON normalisation via statsbombpy.public
    try:
        df = _fetch_frames_raw(match_id)
        if df is None or df.empty:
            logger.info("No 360 data available for match_id=%d", match_id)
            return None
        df.to_parquet(cache, index=False)
        logger.debug("Saved %d frame rows to %s (raw fallback)", len(df), cache)
        return df
    except Exception as exc:
        logger.warning("Raw 360 fetch failed for match_id=%d: %s. Returning None.", match_id, exc)
        return None


def _fetch_frames_raw(match_id: int) -> Optional[pd.DataFrame]:
    """Load 360 JSON directly via statsbombpy.public and normalise manually."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsbombpy import public

    raw: list[dict] = public.frames(match_id=match_id)
    if not raw:
        return None

    rows = []
    for frame in raw:
        event_uuid = frame.get("event_uuid", "")
        visible_area = frame.get("visible_area")
        for player in frame.get("freeze_frame", []):
            loc = player.get("location", [None, None])
            rows.append(
                {
                    "id": event_uuid,
                    "match_id": match_id,
                    "visible_area": str(visible_area) if visible_area else None,
                    "teammate": player.get("teammate"),
                    "actor": player.get("actor"),
                    "keeper": player.get("keeper"),
                    "x": loc[0] if loc and len(loc) > 0 else None,
                    "y": loc[1] if loc and len(loc) > 1 else None,
                    "player_id": (
                        player.get("player", {}).get("id")
                        if isinstance(player.get("player"), dict)
                        else None
                    ),
                    "player_name": (
                        player.get("player", {}).get("name")
                        if isinstance(player.get("player"), dict)
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _normalise_frames_df(df: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Normalise a DataFrame returned by sb.frames() into a consistent schema."""
    # sb.frames renames event_uuid → id; location is split into x/y columns
    # when json_normalize runs on the freeze_frame dicts
    out = df.copy()

    # Ensure match_id column
    if "match_id" not in out.columns:
        out["match_id"] = match_id

    # Rename id → id (already done), handle location list if present
    if "location" in out.columns and "x" not in out.columns:
        locs = out["location"].apply(lambda v: v if isinstance(v, list) else [None, None])
        out["x"] = locs.apply(lambda v: v[0] if v else None)
        out["y"] = locs.apply(lambda v: v[1] if v else None)
        out = out.drop(columns=["location"])

    # player dict column (present in some versions)
    if "player" in out.columns and "player_id" not in out.columns:
        out["player_id"] = out["player"].apply(
            lambda v: v.get("id") if isinstance(v, dict) else None
        )
        out["player_name"] = out["player"].apply(
            lambda v: v.get("name") if isinstance(v, dict) else None
        )
        out = out.drop(columns=["player"])

    for col in ("player_id", "player_name"):
        if col not in out.columns:
            out[col] = None

    return out.reset_index(drop=True)


def get_lineups(match_id: int) -> pd.DataFrame:
    """Return a combined lineups DataFrame for both teams in a match.

    Parameters
    ----------
    match_id:
        StatsBomb match identifier.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, team_name, player_id, player_name, jersey_number,
        player_nickname, country_name.
    """
    cache = _raw_dir() / f"lineups_{match_id}.parquet"
    if cache.exists():
        logger.debug("Loading lineups from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Fetching lineups for match_id=%d", match_id)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = _sb().lineups(match_id=match_id)

        # statsbombpy returns a dict {team_name: DataFrame}
        frames_list = []
        if isinstance(raw, dict):
            for team_name, team_df in raw.items():
                team_df = team_df.copy()
                team_df["team_name"] = team_name
                team_df["match_id"] = match_id
                frames_list.append(team_df)
            df = pd.concat(frames_list, ignore_index=True)
        else:
            df = raw.copy()
            df["match_id"] = match_id

        # Flatten country column if it's a dict
        if "country" in df.columns:
            df["country_name"] = df["country"].apply(
                lambda v: v.get("name") if isinstance(v, dict) else str(v) if pd.notna(v) else None
            )
            df = df.drop(columns=["country"])
        else:
            df["country_name"] = None

        df.to_parquet(cache, index=False)
        logger.debug("Saved %d lineup rows to %s", len(df), cache)
        return df
    except Exception as exc:
        logger.error("Failed to fetch lineups (match_id=%d): %s", match_id, exc)
        raise
