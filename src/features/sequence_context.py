"""Recent possession sequence context features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default sequence window length
_DEFAULT_SEQ_LEN: int = 5


def build_sequence_features(
    pass_instances_df: pd.DataFrame,
    events_df: pd.DataFrame,
    seq_len: int = _DEFAULT_SEQ_LEN,
) -> pd.DataFrame:
    """Compute recent-possession sequence context features for each pass.

    For each pass in *pass_instances_df*, this function looks at the
    ``seq_len`` events that occurred **before** it in the same possession
    and summarises them as scalar features.

    Parameters
    ----------
    pass_instances_df:
        One row per pass (must contain event_uuid, possession_id, minute,
        second, period).
    events_df:
        Full event table (all event types) for the same matches.  Must
        contain possession_id, index (or ordering), minute, second, period,
        type_name, under_pressure.
    seq_len:
        Number of prior events to consider.

    Returns
    -------
    pd.DataFrame
        Indexed by *event_uuid*.  Columns:
        prev_action_type_carry, prev_action_type_duel,
        prev_action_type_pressure, prev_action_type_pass,
        progression_trend, tempo, under_pressure_count.
        All numeric columns are float32.
    """
    if pass_instances_df is None or pass_instances_df.empty:
        logger.warning("build_sequence_features: empty pass_instances_df")
        return pd.DataFrame()

    if events_df is None or events_df.empty:
        logger.warning("build_sequence_features: empty events_df – returning zeros")
        return _zero_sequence_features(pass_instances_df)

    logger.info(
        "Building sequence features for %d passes (seq_len=%d)",
        len(pass_instances_df),
        seq_len,
    )

    # Build a sortable time key within each possession
    events = events_df.copy()
    events = _ensure_sort_key(events)

    # Sort events by possession then time
    events = events.sort_values(["possession_id", "_sort_key"], kind="stable")

    # Pre-index events by possession_id for fast lookup
    poss_groups: dict[int, pd.DataFrame] = {
        pid: grp.reset_index(drop=True)
        for pid, grp in events.groupby("possession_id")
    }

    # Also build event-uuid → (possession_id, position_in_possession)
    pass_poss_ids = pass_instances_df["possession_id"].values
    pass_uuids = pass_instances_df["event_uuid"].values
    pass_minutes = pass_instances_df["minute"].astype(float).values
    pass_seconds = (
        pass_instances_df["second"].astype(float).values
        if "second" in pass_instances_df.columns
        else np.zeros(len(pass_instances_df))
    )
    pass_periods = pass_instances_df["period"].astype(float).values

    rows: list[dict] = []
    for i in range(len(pass_instances_df)):
        uuid = pass_uuids[i]
        poss_id = pass_poss_ids[i]
        minute = pass_minutes[i]
        second = pass_seconds[i]
        period = pass_periods[i]
        pass_time = _time_to_seconds(period, minute, second)

        if poss_id not in poss_groups:
            rows.append(_zero_row(uuid))
            continue

        poss_events = poss_groups[poss_id]
        prior = poss_events[poss_events["_sort_key"] < pass_time].tail(seq_len)

        row = _compute_seq_features(uuid, prior, pass_time, seq_len)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("event_uuid")

    _num_cols = [
        "prev_action_type_carry",
        "prev_action_type_duel",
        "prev_action_type_pressure",
        "prev_action_type_pass",
        "progression_trend",
        "tempo",
        "under_pressure_count",
    ]
    for col in _num_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

    logger.info("Sequence features built: %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_seq_features(
    uuid: str,
    prior: pd.DataFrame,
    pass_time_s: float,
    seq_len: int,
) -> dict:
    """Compute sequence features from the prior event window."""
    n = len(prior)

    if n == 0:
        return _zero_row(uuid)

    # Action type counts
    types = prior["type_name"].fillna("Unknown").str.lower() if "type_name" in prior.columns else pd.Series(dtype=str)
    carry_count = int(types.str.contains("carry").sum())
    duel_count = int(types.str.contains("duel").sum())
    pressure_count = int(types.str.contains("pressure").sum())
    pass_count = int(types.str.contains("pass").sum())

    # Progression trend: mean x_gain over last 3 events
    progression_trend = 0.0
    if "end_x" in prior.columns and "start_x" in prior.columns:
        last3 = prior.tail(3)
        x_gains = (last3["end_x"].astype(float) - last3["start_x"].astype(float)).dropna()
        if len(x_gains) > 0:
            progression_trend = float(x_gains.mean())
    elif "location_x" in prior.columns and "end_x" in prior.columns:
        last3 = prior.tail(3)
        x_gains = (last3["end_x"].astype(float) - last3["location_x"].astype(float)).dropna()
        if len(x_gains) > 0:
            progression_trend = float(x_gains.mean())

    # Tempo: events per minute in the window
    tempo = 0.0
    if n >= 2 and "_sort_key" in prior.columns:
        earliest_time = float(prior["_sort_key"].iloc[0])
        duration_s = max(pass_time_s - earliest_time, 1.0)
        tempo = float(n / (duration_s / 60.0))

    # Under pressure count
    under_pressure_count = 0
    if "under_pressure" in prior.columns:
        under_pressure_count = int(prior["under_pressure"].fillna(False).astype(bool).sum())

    return {
        "event_uuid": uuid,
        "prev_action_type_carry": carry_count,
        "prev_action_type_duel": duel_count,
        "prev_action_type_pressure": pressure_count,
        "prev_action_type_pass": pass_count,
        "progression_trend": progression_trend,
        "tempo": tempo,
        "under_pressure_count": under_pressure_count,
    }


def _zero_row(uuid: str) -> dict:
    return {
        "event_uuid": uuid,
        "prev_action_type_carry": 0,
        "prev_action_type_duel": 0,
        "prev_action_type_pressure": 0,
        "prev_action_type_pass": 0,
        "progression_trend": 0.0,
        "tempo": 0.0,
        "under_pressure_count": 0,
    }


def _zero_sequence_features(pass_instances_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of zeros when events_df is unavailable."""
    uuids = pass_instances_df["event_uuid"].values
    rows = [_zero_row(u) for u in uuids]
    result = pd.DataFrame(rows).set_index("event_uuid")
    for col in result.columns:
        result[col] = result[col].astype("float32")
    return result


def _ensure_sort_key(events: pd.DataFrame) -> pd.DataFrame:
    """Add a ``_sort_key`` column (seconds from kick-off) for ordering."""
    if "_sort_key" in events.columns:
        return events

    period = events.get("period", pd.Series(1, index=events.index)).fillna(1).astype(float)
    minute = events.get("minute", pd.Series(0, index=events.index)).fillna(0).astype(float)
    second = events.get("second", pd.Series(0, index=events.index)).fillna(0).astype(float)

    events = events.copy()
    events["_sort_key"] = _time_to_seconds_series(period, minute, second)
    return events


def _time_to_seconds(period: float, minute: float, second: float) -> float:
    """Convert period + minute + second to a total-seconds sort key."""
    period_offset = (period - 1) * 60 * 60  # generous offset per period
    return period_offset + minute * 60 + second


def _time_to_seconds_series(
    period: pd.Series, minute: pd.Series, second: pd.Series
) -> pd.Series:
    period_offset = (period - 1) * 60 * 60
    return period_offset + minute * 60 + second
