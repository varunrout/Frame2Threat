"""
src/data/inventory.py
=====================
Build and cache an inventory of available StatsBomb open data.

The inventory DataFrame records, for every match in the configured
competitions, whether events, lineups, and 360 frames are available.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "data.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _sb():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsbombpy import sb
    return sb


def build_inventory(competitions_config: list[dict] | None = None) -> pd.DataFrame:
    """Scan available StatsBomb open data and return an inventory DataFrame.

    For every match in the configured (or provided) competitions, the
    inventory records: competition, season, match_id, has_events,
    has_lineups, has_360.

    The result is cached at ``data/interim/inventory.parquet``.

    Parameters
    ----------
    competitions_config:
        Optional list of dicts with ``competition_id`` keys.  Defaults to
        the ``statsbomb.competitions`` list in ``configs/data.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: competition_id, competition_name, season_id, season_name,
        match_id, has_events, has_lineups, has_360.
    """
    cfg = _load_config()
    interim_dir = _REPO_ROOT / cfg["statsbomb"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    cache = interim_dir / "inventory.parquet"

    if competitions_config is None:
        competitions_config = cfg["statsbomb"]["competitions"]

    logger.info("Building inventory for %d competition entries", len(competitions_config))

    sb = _sb()

    # Get all available competitions once to resolve metadata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_comps = sb.competitions()

    rows: list[dict] = []
    competition_ids = [c["competition_id"] for c in competitions_config]

    for comp_id in competition_ids:
        comp_rows = all_comps[all_comps["competition_id"] == comp_id]
        if comp_rows.empty:
            logger.warning("competition_id=%d not found in StatsBomb data", comp_id)
            continue

        for _, comp_row in comp_rows.iterrows():
            season_id = int(comp_row["season_id"])
            competition_name = comp_row.get("competition_name", "")
            season_name = comp_row.get("season_name", "")

            logger.info(
                "Fetching matches for competition=%s (%d) season=%s (%d)",
                competition_name,
                comp_id,
                season_name,
                season_id,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
            except Exception as exc:
                logger.warning(
                    "Could not fetch matches for comp=%d season=%d: %s",
                    comp_id,
                    season_id,
                    exc,
                )
                continue

            for _, match_row in matches_df.iterrows():
                match_id = int(match_row["match_id"])
                match_status = match_row.get("match_status", "")
                status_360 = match_row.get("match_status_360", None)

                has_events = match_status == "available"
                has_lineups = has_events  # lineups available whenever events are
                has_360 = str(status_360).strip().lower() == "available"

                rows.append(
                    {
                        "competition_id": comp_id,
                        "competition_name": competition_name,
                        "season_id": season_id,
                        "season_name": season_name,
                        "match_id": match_id,
                        "has_events": has_events,
                        "has_lineups": has_lineups,
                        "has_360": has_360,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Inventory is empty – check competition configuration.")
        return df

    df = df.sort_values(["competition_id", "season_id", "match_id"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    logger.info("Inventory saved to %s (%d matches)", cache, len(df))
    return df


def get_360_match_ids(inventory_df: pd.DataFrame | None = None) -> List[int]:
    """Return a list of match IDs that have 360 data available.

    Parameters
    ----------
    inventory_df:
        Pre-built inventory DataFrame.  If ``None`` the cached
        ``data/interim/inventory.parquet`` is loaded; if that is also
        absent, :func:`build_inventory` is called first.

    Returns
    -------
    list[int]
        Sorted list of match_ids with ``has_360 == True``.
    """
    if inventory_df is not None:
        df = inventory_df
    else:
        cfg = _load_config()
        interim_dir = _REPO_ROOT / cfg["statsbomb"]["interim_dir"]
        cache = interim_dir / "inventory.parquet"
        if cache.exists():
            logger.info("Loading inventory from cache: %s", cache)
            df = pd.read_parquet(cache)
        else:
            logger.info("No cached inventory found; building now…")
            df = build_inventory()

    if df.empty:
        return []

    match_ids = df.loc[df["has_360"], "match_id"].dropna().astype(int).sort_values().tolist()
    logger.info("Found %d matches with 360 data", len(match_ids))
    return match_ids
