"""
parse_possessions.py
====================
Build the canonical possession_sequences table from events_parsed.parquet.

Each row = one possession.
Columns
-------
Identity  : match_id, possession_id, team_name, period
Spatial   : start_x, start_y, max_x_reached, territory_gained,
            end_x, end_y
Temporal  : n_events, n_passes, n_carries, n_pressures_faced,
            duration_seconds
Meta      : origin_type, mean_pass_length, has_pressure
Sequence  : event_sequence   (list[dict], embedded as object column)
            player_sequence  (list[str],  player name for each event step)

Labels are NOT computed here — they live in
``src/labels/possession_labels.attach_possession_labels()``.
``build_possession_sequences`` calls it automatically before returning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.labels.possession_labels import attach_possession_labels

# ---------------------------------------------------------------------------
# StatsBomb pitch constants
# ---------------------------------------------------------------------------
PITCH_LENGTH = 120.0  # x-axis
PITCH_WIDTH = 80.0  # y-axis

FINAL_THIRD_X = 80.0  # x >= 80  → final third
BOX_X = 102.0  # x >= 102 → penalty area (18-yard box)
BOX_Y_LO = 18.0
BOX_Y_HI = 62.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_possession_sequences(
    events_df: pd.DataFrame,
    *,
    min_events: int = 2,
) -> pd.DataFrame:
    """
    Aggregate a flat events DataFrame into one row per possession.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of ``parse_events.parse_events()`` — all event types.
    min_events : int
        Drop possessions with fewer than this many events (default 2).
        Filters out trivial kick-off / GK restart fragments.

    Returns
    -------
    pd.DataFrame  with shape (n_possessions, ~20 columns).
    """
    required = {
        "match_id",
        "possession_id",
        "index",
        "period",
        "minute",
        "second",
        "type_name",
        "team_name",
        "play_pattern_name",
        "location_x",
        "location_y",
        "under_pressure",
    }
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing columns: {missing}")

    # Sort globally; group operations rely on this ordering
    df = events_df.sort_values(["match_id", "possession_id", "index"]).copy()

    # Boolean coerce
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(bool)

    rows: list[dict] = []
    for (match_id, poss_id), grp in df.groupby(["match_id", "possession_id"], sort=False):
        if len(grp) < min_events:
            continue

        row = _aggregate_possession(match_id, poss_id, grp)
        rows.append(row)

    out = pd.DataFrame(rows)
    _set_dtypes(out)
    out = attach_possession_labels(out, events_df=events_df)
    return out.reset_index(drop=True)


def save_possession_sequences(
    poss_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Persist possession_sequences table to parquet.

    The ``event_sequence`` column contains Python lists, which parquet
    cannot store natively — we serialise it to JSON strings first.
    Call ``load_possession_sequences()`` to round-trip correctly.
    """
    if output_path is None:
        output_path = Path("data/processed/possession_sequences.parquet")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = poss_df.copy()
    if "event_sequence" in to_save.columns:
        to_save["event_sequence"] = to_save["event_sequence"].apply(
            lambda seq: json.dumps(seq) if isinstance(seq, list) else seq
        )

    if "player_sequence" in to_save.columns:
        to_save["player_sequence"] = to_save["player_sequence"].apply(
            lambda seq: json.dumps(seq) if isinstance(seq, list) else seq
        )
    to_save.to_parquet(output_path, index=False)
    return output_path


def load_possession_sequences(
    path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load possession_sequences.parquet and restore event_sequence lists."""
    if path is None:
        path = Path("data/processed/possession_sequences.parquet")
    df = pd.read_parquet(path)
    if "event_sequence" in df.columns:
        df["event_sequence"] = df["event_sequence"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else s
        )
    if "player_sequence" in df.columns:
        df["player_sequence"] = df["player_sequence"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else s
        )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_possession(
    match_id: int,
    poss_id: int,
    grp: pd.DataFrame,
) -> dict:
    """Return a single possession-row dict from a sorted group."""
    first = grp.iloc[0]
    last = grp.iloc[-1]

    # --- identity ---
    team_name = first["team_name"]
    period = int(first["period"])  # type: ignore[arg-type]
    origin = first["play_pattern_name"] if pd.notna(first["play_pattern_name"]) else "Unknown"

    # --- spatial ---
    start_x = float(first["location_x"]) if pd.notna(first["location_x"]) else np.nan
    start_y = float(first["location_y"]) if pd.notna(first["location_y"]) else np.nan
    end_x = float(last["location_x"]) if pd.notna(last["location_x"]) else np.nan
    end_y = float(last["location_y"]) if pd.notna(last["location_y"]) else np.nan

    lx = grp["location_x"].dropna()
    max_x_reached = float(lx.max()) if len(lx) else np.nan
    territory_gained = (
        float(max_x_reached - start_x)
        if (not np.isnan(max_x_reached) and not np.isnan(start_x))
        else np.nan
    )

    # --- temporal / counting ---
    n_events = len(grp)
    types = grp["type_name"].str.lower()
    n_passes = int((types == "pass").sum())
    n_carries = int((types == "carry").sum())
    n_pressures_faced = int(grp["under_pressure"].sum())

    # Duration: convert minute+second to total seconds, diff first→last
    t_start = int(first["minute"]) * 60 + int(first["second"])
    t_end = int(last["minute"]) * 60 + int(last["second"])
    duration_seconds = max(0, t_end - t_start)

    # --- meta ---
    pl = grp["pass_length"].dropna() if "pass_length" in grp.columns else pd.Series(dtype=float)
    mean_pass_length = float(pl.mean()) if len(pl) > 0 else np.nan
    has_pressure = bool(n_pressures_faced > 0)

    # --- sequence ---
    event_sequence = _build_event_sequence(grp)

    return {
        # identity
        "match_id": int(match_id),
        "possession_id": int(poss_id),
        "team_name": team_name,
        "period": period,
        "origin_type": str(origin),
        # spatial
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "max_x_reached": max_x_reached,
        "territory_gained": territory_gained,
        # temporal
        "n_events": n_events,
        "n_passes": n_passes,
        "n_carries": n_carries,
        "n_pressures_faced": n_pressures_faced,
        "duration_seconds": duration_seconds,
        # meta
        "mean_pass_length": mean_pass_length,
        "has_pressure": has_pressure,
        # sequence (serialised later) — labels added by attach_possession_labels
        "event_sequence": event_sequence,
        "player_sequence": _build_player_sequence(grp),
    }


def _build_player_sequence(grp: pd.DataFrame) -> list[str]:
    """
    Return a list of player names aligned 1-to-1 with the event sequence.
    Uses 'Unknown' when the event has no associated player actor.
    """
    has_col = "player_name" in grp.columns
    names: list[str] = []
    for _, row in grp.iterrows():
        if has_col:
            val = row["player_name"]
            names.append(str(val) if (pd.notna(val) and str(val).strip() != "") else "Unknown")
        else:
            names.append("Unknown")
    return names


def attach_player_sequences(
    poss_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Post-hoc: add or refresh the ``player_sequence`` column on an existing
    possession DataFrame without rebuilding the full pipeline.

    Useful when ``possession_sequences.parquet`` was generated before
    ``player_sequence`` was introduced.

    Parameters
    ----------
    poss_df   : possession_sequences DataFrame
    events_df : events_parsed DataFrame (must contain ``player_name``)

    Returns
    -------
    Copy of poss_df with a new / updated ``player_sequence`` column.
    """
    df = events_df.sort_values(["match_id", "possession_id", "index"]).copy()
    player_map: dict[tuple, list[str]] = {}
    for (match_id, poss_id), grp in df.groupby(["match_id", "possession_id"], sort=False):
        player_map[(int(match_id), int(poss_id))] = _build_player_sequence(grp)

    out = poss_df.copy()
    out["player_sequence"] = out.apply(
        lambda r: player_map.get((int(r["match_id"]), int(r["possession_id"])), []),
        axis=1,
    )
    return out


# Event types to include in the sequence representation
_SEQUENCE_TYPES = frozenset(
    {
        "pass",
        "carry",
        "ball receipt*",
        "ball receipt",
        "dribble",
        "shot",
        "pressure",
        "duel",
        "clearance",
        "interception",
        "block",
        "foul committed",
        "foul won",
        "goalkeeper",
        "goal keeper",
    }
)

# Numeric features per event step
_SEQ_FIELDS = [
    "type_id",  # int-encoded event type (see TYPE_VOCAB below)
    "loc_x_norm",  # location_x / 120
    "loc_y_norm",  # location_y / 80
    "end_x_norm",  # pass_end_x / 120  (0 if not a pass / carry)
    "end_y_norm",  # pass_end_y / 80
    "under_pressure",  # 0/1
    "pass_length_norm",  # pass_length / 60 (max ~60 m)
    "minute_norm",  # minute / 90
]

# Compact vocabulary for event types in sequences
TYPE_VOCAB: dict[str, int] = {
    "pass": 1,
    "carry": 2,
    "ball receipt*": 3,
    "ball receipt": 3,
    "dribble": 4,
    "shot": 5,
    "pressure": 6,
    "duel": 7,
    "clearance": 8,
    "interception": 9,
    "block": 10,
    "foul committed": 11,
    "foul won": 12,
    "goalkeeper": 13,
    "goal keeper": 13,
}


def _build_event_sequence(grp: pd.DataFrame) -> list[dict]:
    """
    Return a list of compact per-event dicts for GRU/sequence model input.
    Each dict has float-valued features from ``_SEQ_FIELDS``.
    """
    records = []
    for _, row in grp.iterrows():
        t = str(row["type_name"]).lower() if pd.notna(row["type_name"]) else ""
        type_id = TYPE_VOCAB.get(t, 0)

        lx = float(row["location_x"]) / PITCH_LENGTH if pd.notna(row["location_x"]) else 0.0
        ly = float(row["location_y"]) / PITCH_WIDTH if pd.notna(row["location_y"]) else 0.0

        has_end_x = "pass_end_x" in row.index and pd.notna(row["pass_end_x"])
        has_end_y = "pass_end_y" in row.index and pd.notna(row["pass_end_y"])
        ex = float(row["pass_end_x"]) / PITCH_LENGTH if has_end_x else lx
        ey = float(row["pass_end_y"]) / PITCH_WIDTH if has_end_y else ly

        up = 1.0 if (pd.notna(row["under_pressure"]) and bool(row["under_pressure"])) else 0.0

        has_pl = "pass_length" in row.index and pd.notna(row["pass_length"])
        pl_norm = float(row["pass_length"]) / 60.0 if has_pl else 0.0

        minute = float(row["minute"]) / 90.0 if pd.notna(row["minute"]) else 0.0

        records.append(
            {
                "type_id": type_id,
                "loc_x_norm": round(lx, 4),
                "loc_y_norm": round(ly, 4),
                "end_x_norm": round(ex, 4),
                "end_y_norm": round(ey, 4),
                "under_pressure": up,
                "pass_length_norm": round(pl_norm, 4),
                "minute_norm": round(minute, 4),
            }
        )
    return records


def _set_dtypes(df: pd.DataFrame) -> None:
    """In-place dtype clean-up after DataFrame construction."""
    int_cols = [
        "match_id",
        "possession_id",
        "period",
        "n_events",
        "n_passes",
        "n_carries",
        "n_pressures_faced",
        "duration_seconds",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("int32")

    bool_cols = [
        "has_pressure",  # labels handled by possession_labels.py
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    float_cols = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "max_x_reached",
        "territory_gained",
        "mean_pass_length",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Loading events_parsed.parquet …")
    events = pd.read_parquet("data/processed/events_parsed.parquet")
    print(f"  {len(events):,} events loaded")

    t0 = time.time()
    print("Building possession sequences …")
    poss = build_possession_sequences(events)
    elapsed = time.time() - t0

    print(f"  {len(poss):,} possessions built in {elapsed:.1f}s")
    label_cols = [
        "poss_has_shot",
        "poss_entered_box",
        "poss_entered_final_third",
        "poss_dangerous",
        "poss_broke_pressure",
        "poss_bypassed_lines",
        "poss_recycled",
    ]
    print(f"Label rates:")
    for col in label_cols:
        if col not in poss.columns:
            continue
        rate = poss[col].mean()
        print(f"  {col:32s}: {rate:.1%}  ({int(poss[col].sum()):,} possessions)")
    print()
    print("Phase distribution:")
    print(poss["poss_phase"].value_counts().to_string())
    print()
    print("Outcome tier distribution:")
    print(poss["poss_outcome_tier"].value_counts().sort_index().to_string())

    print("\nSequence length stats (events per possession):")
    lens = poss["n_events"]
    print(
        f"  mean={lens.mean():.1f}  median={lens.median():.0f}  "
        f"p75={lens.quantile(0.75):.0f}  p95={lens.quantile(0.95):.0f}  max={lens.max()}"
    )

    out = save_possession_sequences(poss)
    print(f"\nSaved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    print("\nSample row (no event_sequence):")
    print(poss.drop(columns=["event_sequence"]).head(3).to_string())
