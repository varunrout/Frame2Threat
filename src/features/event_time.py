"""
event_time.py
=============
Event-time (in-play) formulation of possession danger forecasting.

At every event index ``t`` of a possession that has **not yet produced a shot or a box
entry**, a snapshot row is created.  The label ``danger_after_t`` is 1 if a shot or box
entry by the possessing team happens in events ``t+1 .. n-1`` of the same possession.

Everything in :func:`snapshot_features` and :func:`sequence_inputs` is computed from
events ``0..t`` only: no feature uses the possession length ``n``, a fraction of ``n``, or
anything that happens after ``t``.  (``tests/test_event_time.py`` appends arbitrary future
events and asserts that the row for ``t`` is unchanged.)

This replaces the fraction-based v3 formulation, whose prefix length ``ceil(n*f)`` encodes
the possession's total length and whose prefixes frequently already contain the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.event_types import N_EVENT_TYPES, EventType, decode, encode
from src.data.possession_frame import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    is_danger_event,
    possessing_team_at_start,
    to_possession_frame,
)

ORIGINS = [
    "Regular Play",
    "From Throw In",
    "From Free Kick",
    "From Goal Kick",
    "From Corner",
    "From Kick Off",
    "From Keeper",
    "From Counter",
    "Other",
]
_PERIOD_OFFSET_S = {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0, 5: 7200.0}

TYPE_COLS = [f"cur_type_{decode(i).replace(' ', '_')}" for i in range(N_EVENT_TYPES)]
ORIGIN_COLS = [f"origin_{o.lower().replace(' ', '_')}" for o in ORIGINS]
FEATURE_NAMES: list[str] = (
    [
        "cur_x",
        "cur_y",
        "max_x_so_far",
        "events_so_far",
        "n_passes_so_far",
        "n_carries_so_far",
        "n_pressures_so_far",
        "elapsed_s",
        "cur_is_own",
    ]
    + TYPE_COLS
    + ["start_x", "start_y", "period", "game_minute"]
    + ORIGIN_COLS
)
N_SEQ_NUMERIC = 13 + len(ORIGINS)  # columns produced by sequence_inputs()

_ARRAY_FIELDS = [
    "type_id",
    "own",
    "x",
    "y",
    "end_x",
    "end_y",
    "under_pressure",
    "pass_length",
    "abs_s",
]


@dataclass
class PossessionArrays:
    """Per-event arrays for one possession, in StatsBomb event order."""

    match_id: int
    possession_id: int
    team: str
    period: int
    minute: float
    origin: str
    type_id: np.ndarray  # (n,) int
    own: np.ndarray  # (n,) bool, possessing team's event
    x: np.ndarray  # (n,) raw location in the acting team's frame (nan if missing)
    y: np.ndarray
    end_x: np.ndarray  # pass end (nan if not a pass)
    end_y: np.ndarray
    under_pressure: np.ndarray  # (n,) float
    pass_length: np.ndarray  # (n,) metres, nan if not a pass
    abs_s: np.ndarray  # (n,) seconds: period offset + within-period timestamp

    @property
    def n(self) -> int:
        return len(self.type_id)

    def truncated(self, k: int) -> PossessionArrays:
        """First ``k`` events only (used to test causality)."""
        kw = {f: getattr(self, f)[:k] for f in _ARRAY_FIELDS}
        return PossessionArrays(**{**self.__dict__, **kw})


# ---------------------------------------------------------------------------
# Building arrays from events_parsed
# ---------------------------------------------------------------------------


def _timestamp_seconds(ts: pd.Series) -> np.ndarray:
    return pd.to_timedelta(ts.astype(str)).dt.total_seconds().to_numpy(dtype=float)


def build_possession_arrays(
    events_df: pd.DataFrame,
    keys: pd.DataFrame | None = None,
) -> list[PossessionArrays]:
    """One :class:`PossessionArrays` per possession of ``events_df``.

    ``keys`` (columns match_id, possession_id) restricts to those possessions, so the
    result lines up with ``possession_sequences.parquet`` (which drops < 2-event stubs).
    """
    df = events_df.sort_values(["match_id", "possession_id", "index"]).reset_index(drop=True)
    if keys is not None:
        k = keys[["match_id", "possession_id"]].drop_duplicates()
        df = df.merge(k, on=["match_id", "possession_id"], how="inner")
    df["_type_id"] = df["type_name"].map(encode).astype(int)
    df["_abs_s"] = _timestamp_seconds(df["timestamp"]) + df["period"].map(_PERIOD_OFFSET_S).fillna(
        0.0
    )
    df["_up"] = df["under_pressure"].fillna(False).astype(bool).astype(float)
    out: list[PossessionArrays] = []
    for (mid, pid), grp in df.groupby(["match_id", "possession_id"], sort=False):
        team = possessing_team_at_start(grp)
        first = grp.iloc[0]
        origin = first["play_pattern_name"]
        out.append(
            PossessionArrays(
                match_id=int(mid),
                possession_id=int(pid),
                team=str(team),
                period=int(first["period"]),
                minute=float(first["minute"]),
                origin=str(origin) if pd.notna(origin) else "Other",
                type_id=grp["_type_id"].to_numpy(dtype=int),
                own=(grp["team_name"] == team).to_numpy(),
                x=grp["location_x"].to_numpy(dtype=float),
                y=grp["location_y"].to_numpy(dtype=float),
                end_x=grp["pass_end_x"].to_numpy(dtype=float),
                end_y=grp["pass_end_y"].to_numpy(dtype=float),
                under_pressure=grp["_up"].to_numpy(dtype=float),
                pass_length=grp["pass_length"].to_numpy(dtype=float),
                abs_s=grp["_abs_s"].to_numpy(dtype=float),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Labels and eligibility (these look at the future by definition)
# ---------------------------------------------------------------------------


def danger_flags(p: PossessionArrays) -> np.ndarray:
    """Per-event bool: shot or box entry by the possessing team.

    Event location, plus the end location of possessing-team passes (see
    ``possession_frame.is_danger_event``): a pass ending in the box is the entry.
    """
    x = np.nan_to_num(p.x, nan=0.0)
    y = np.nan_to_num(p.y, nan=0.0)
    return is_danger_event(p.type_id, x, y, p.own, p.end_x, p.end_y)


def snapshot_labels(p: PossessionArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(eligible, label, terminal)`` arrays of length n.

    * eligible[t]: t < n-1 and no danger event at or before t.
    * label[t]:    danger event exists in t+1..n-1 (only meaningful where eligible).
    * terminal[t]: t == n-1 and no danger at or before t (sensitivity analysis only).
    """
    n = p.n
    d = danger_flags(p)
    first = int(np.argmax(d)) if d.any() else n
    t = np.arange(n)
    eligible = (t < n - 1) & (t < first)
    label = np.full(n, int(first < n), dtype=int)
    terminal = (t == n - 1) & (first >= n)
    return eligible, label, terminal


# ---------------------------------------------------------------------------
# Features computed from events 0..t only
# ---------------------------------------------------------------------------


def _ffill(a: np.ndarray, init: float) -> np.ndarray:
    """Forward-fill NaNs; leading NaNs take ``init``."""
    out = a.copy()
    if np.isnan(out[0]):
        out[0] = init
    idx = np.where(~np.isnan(out), np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    return out[idx]


def _start_position(p: PossessionArrays) -> tuple[float, float]:
    """Frame-corrected position of the first event (fallback: pitch centre)."""
    fx, fy = to_possession_frame(p.x[:1], p.y[:1], p.own[:1])
    if np.isnan(fx[0]) or np.isnan(fy[0]):
        return PITCH_LENGTH / 2, PITCH_WIDTH / 2
    return float(fx[0]), float(fy[0])


def snapshot_features(p: PossessionArrays) -> np.ndarray:
    """(n, len(FEATURE_NAMES)) float32; row t depends only on events 0..t."""
    n = p.n
    t = np.arange(n)
    sx, sy = _start_position(p)
    cur_x = _ffill(np.where(p.own, p.x, np.nan), sx)
    cur_y = _ffill(np.where(p.own, p.y, np.nan), sy)
    reach = np.fmax(np.where(p.own, p.x, np.nan), np.where(p.own, p.end_x, np.nan))
    reach = np.fmax.accumulate(np.where(np.isnan(reach), -np.inf, reach))
    max_x = np.where(np.isinf(reach), sx, np.maximum(reach, sx))
    elapsed = np.clip(p.abs_s - p.abs_s[0], 0.0, None)

    def cum(mask: np.ndarray) -> np.ndarray:
        return np.cumsum(mask.astype(float))

    cols = [
        cur_x,
        cur_y,
        max_x,
        (t + 1).astype(float),
        cum(p.type_id == EventType.PASS),
        cum(p.type_id == EventType.CARRY),
        cum(p.type_id == EventType.PRESSURE),
        elapsed,
        p.own.astype(float),
    ]
    cols += [(p.type_id == i).astype(float) for i in range(N_EVENT_TYPES)]
    cols += [np.full(n, sx), np.full(n, sy), np.full(n, float(p.period)), np.full(n, p.minute)]
    origin = p.origin if p.origin in ORIGINS else "Other"
    cols += [np.full(n, float(origin == o)) for o in ORIGINS]
    return np.column_stack(cols).astype(np.float32)


def sequence_inputs(p: PossessionArrays) -> tuple[np.ndarray, np.ndarray]:
    """Inputs for the causal GRU: ``(type_ids (n,), numeric (n, N_SEQ_NUMERIC))``.

    Coordinates are expressed in the possessing team's frame (opponent events flipped);
    the numeric block also carries elapsed and inter-event seconds and the possession's
    start context.  Step t uses only events 0..t.
    """
    n = p.n
    cx, cy = PITCH_LENGTH / 2, PITCH_WIDTH / 2
    fx, fy = to_possession_frame(np.nan_to_num(p.x, nan=cx), np.nan_to_num(p.y, nan=cy), p.own)
    ex_raw = np.where(np.isnan(p.end_x), p.x, p.end_x)
    ey_raw = np.where(np.isnan(p.end_y), p.y, p.end_y)
    ex, ey = to_possession_frame(
        np.nan_to_num(ex_raw, nan=cx), np.nan_to_num(ey_raw, nan=cy), p.own
    )
    elapsed = np.clip(p.abs_s - p.abs_s[0], 0.0, None)
    dt = np.clip(np.diff(p.abs_s, prepend=p.abs_s[0]), 0.0, None)
    sx, sy = _start_position(p)
    origin = p.origin if p.origin in ORIGINS else "Other"
    num = np.column_stack(
        [
            fx / PITCH_LENGTH,
            fy / PITCH_WIDTH,
            ex / PITCH_LENGTH,
            ey / PITCH_WIDTH,
            p.own.astype(float),
            p.under_pressure,
            np.nan_to_num(p.pass_length, nan=0.0) / 60.0,
            np.clip(elapsed / 30.0, 0.0, 4.0),
            np.clip(dt / 10.0, 0.0, 2.0),
            np.full(n, sx / PITCH_LENGTH),
            np.full(n, sy / PITCH_WIDTH),
            np.full(n, float(p.period == 2)),
            np.full(n, p.minute / 90.0),
        ]
        + [np.full(n, float(origin == o)) for o in ORIGINS]
    )
    return p.type_id.astype(np.int64), num.astype(np.float32)


# ---------------------------------------------------------------------------
# Snapshot table
# ---------------------------------------------------------------------------


def build_snapshot_table(arrs: list[PossessionArrays]) -> pd.DataFrame:
    """All snapshot rows (eligible, plus terminal ones flagged) with features and label."""
    frames = []
    for i, p in enumerate(arrs):
        elig, lab, term = snapshot_labels(p)
        keep = elig | term
        if not keep.any():
            continue
        df = pd.DataFrame(snapshot_features(p)[keep], columns=FEATURE_NAMES)
        df.insert(0, "t", np.nonzero(keep)[0])
        df.insert(0, "possession_id", p.possession_id)
        df.insert(0, "match_id", p.match_id)
        df.insert(0, "arr_idx", i)
        df["y"] = lab[keep]
        df["terminal"] = term[keep]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
