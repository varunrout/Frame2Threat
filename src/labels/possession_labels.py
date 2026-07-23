"""
possession_labels.py
====================
All possession-level labels for Frame2Threat v2.

Deliberately separated from src/data/parse_possessions.py so that:
  - Labels can be recomputed without re-parsing the full sequence table.
  - Each label is independently unit-testable.
  - New labels can be added without touching the parsing pipeline.

Design contract
---------------
The public API is a single function:

    attach_possession_labels(poss_df, events_df=None) -> pd.DataFrame

It receives the raw structural possession table (output of
``build_possession_sequences`` *before* labels are attached) and returns
a copy with all label columns appended.

Input columns expected in ``poss_df``
--------------------------------------
Scalar  : match_id, possession_id, team_name, period, origin_type,
          start_x, start_y, end_x, end_y, max_x_reached,
          territory_gained, n_events, n_passes, n_carries,
          n_pressures_faced, duration_seconds, mean_pass_length,
          has_pressure
Sequence: event_sequence  (list[dict] with keys: type_id, loc_x_norm,
          loc_y_norm, end_x_norm, end_y_norm, under_pressure,
          pass_length_norm, minute_norm)

Output label columns
---------------------
Group A — Core outcome (moved from parse_possessions.py)
    poss_has_shot              bool
    poss_entered_final_third   bool
    poss_entered_box           bool
    poss_dangerous             bool    = poss_has_shot | poss_entered_box

Group B — Richer outcome
    poss_xg_generated          float   sum of shot xG  (NaN if events_df absent)
    poss_has_goal              bool    (False if events_df absent)
    poss_outcome_tier          int8    0=nothing, 1=FT, 2=box, 3=shot, 4=goal

Group C — Tempo / structural
    poss_tempo                 float   events per second of possession
    poss_verticality           float   territory_gained / (n_events × mean_pass_length + ε)
    poss_recycled              bool    x fell ≥15 m then recovered ≥15 m
    poss_phase                 str     counter | build_up | progression | final_third

Group D — Defensive disruption
    poss_broke_pressure        bool    survived ≥1 pressure event + ≥3 more events after
    poss_bypassed_lines        bool    max_x ≥80 m reached within first 4 events from start_x ≤50
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pitch constants (must match parse_possessions.py)
# ---------------------------------------------------------------------------
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

FINAL_THIRD_X = 80.0  # x >= 80  → final third
BOX_X = 102.0  # x >= 102 → penalty area
BOX_Y_LO = 18.0
BOX_Y_HI = 62.0

# Normalised thresholds (used when reading from event_sequence)
_FT_X_NORM = FINAL_THIRD_X / PITCH_LENGTH  # 0.6667
_BOX_X_NORM = BOX_X / PITCH_LENGTH  # 0.850
_BOX_YLO_NORM = BOX_Y_LO / PITCH_WIDTH  # 0.225
_BOX_YHI_NORM = BOX_Y_HI / PITCH_WIDTH  # 0.775

# TYPE_VOCAB IDs (matching parse_possessions.py)
_SHOT_TYPE_ID = 5
_PRESSURE_TYPE_ID = 6

# Tempo / phase thresholds
_COUNTER_TEMPO_THRESHOLD = 2.0  # events per second
_COUNTER_START_X_THRESHOLD = 50.0  # possession starting in own half
_BUILD_UP_START_X_MAX = 40.0  # deep own half
_FINAL_THIRD_START_X = 80.0  # already in FT

# Recycling / bypass thresholds
_RECYCLE_DROP_M = 15.0  # x must fall by this many metres
_BYPASS_MAX_STEP = 4  # must reach FT within this many events
_BYPASS_START_X = 50.0  # possession must start in own half

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_possession_labels(
    poss_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach all possession-level label columns to ``poss_df``.

    Parameters
    ----------
    poss_df : pd.DataFrame
        Raw structural table from ``build_possession_sequences``.
        Must contain ``event_sequence`` column as list[dict].
    events_df : pd.DataFrame, optional
        Full flat events table. Required for ``poss_xg_generated``,
        ``poss_has_goal``, and ``poss_outcome_tier`` tier-4.

    Returns
    -------
    pd.DataFrame  (copy of ``poss_df`` with label columns appended)
    """
    df = poss_df.copy()

    # Ensure event_sequence is decoded list[dict]
    if "event_sequence" in df.columns:
        df["event_sequence"] = df["event_sequence"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else (s or [])
        )

    # Pre-compute per-row sequence summaries once (avoids repeated iteration)
    seq_data = (
        df["event_sequence"].apply(_parse_sequence)
        if "event_sequence" in df.columns
        else pd.Series([_empty_seq_summary()] * len(df), index=df.index)
    )

    # ── Group A — Core outcome ───────────────────────────────────────────────
    df["poss_has_shot"] = seq_data.apply(lambda s: s["has_shot"])
    df["poss_entered_final_third"] = df["max_x_reached"].fillna(0) >= FINAL_THIRD_X
    df["poss_entered_box"] = seq_data.apply(lambda s: s["entered_box"])
    df["poss_dangerous"] = df["poss_has_shot"] | df["poss_entered_box"]

    # ── Group B — Richer outcome ─────────────────────────────────────────────
    if events_df is not None:
        xg_map, goal_map = _build_xg_goal_maps(events_df)
        key = list(zip(df["match_id"], df["possession_id"]))
        df["poss_xg_generated"] = [float(xg_map.get(k, 0.0)) for k in key]
        df["poss_has_goal"] = [bool(goal_map.get(k, False)) for k in key]
    else:
        df["poss_xg_generated"] = np.nan
        df["poss_has_goal"] = False

    df["poss_outcome_tier"] = _outcome_tier(df).astype("int8")

    # ── Group C — Tempo / structural ─────────────────────────────────────────
    df["poss_tempo"] = (df["n_events"] / df["duration_seconds"].clip(lower=1)).astype("float32")

    df["poss_verticality"] = (
        df["territory_gained"].fillna(0)
        / (df["n_events"] * df["mean_pass_length"].fillna(0) + EPSILON)
    ).astype("float32")

    df["poss_recycled"] = seq_data.apply(lambda s: s["recycled"])

    df["poss_phase"] = _assign_phase(df)

    # ── Group D — Defensive disruption ──────────────────────────────────────
    df["poss_broke_pressure"] = seq_data.apply(lambda s: s["broke_pressure"])
    df["poss_bypassed_lines"] = seq_data.apply(lambda s: s["bypassed_lines"]) & (
        df["start_x"].fillna(PITCH_LENGTH) <= _BYPASS_START_X
    )

    _set_label_dtypes(df)
    return df


# ---------------------------------------------------------------------------
# Sequence parsing — done once per row
# ---------------------------------------------------------------------------


def _parse_sequence(seq: list[dict]) -> dict:
    """
    Extract all sequence-derived label features from one possession's
    event_sequence in a single pass.
    """
    has_shot = False
    entered_box = False
    broke_pressure = False
    recycled = False
    bypassed_lines = False

    n = len(seq)
    if n == 0:
        return _empty_seq_summary()

    xs = [s.get("loc_x_norm", 0.0) * PITCH_LENGTH for s in seq]  # denorm to metres

    # --- shot / box entry ---
    for s in seq:
        if s.get("type_id") == _SHOT_TYPE_ID:
            has_shot = True
        lx = s.get("loc_x_norm", 0.0)
        ly = s.get("loc_y_norm", 0.0)
        if lx >= _BOX_X_NORM and _BOX_YLO_NORM <= ly <= _BOX_YHI_NORM:
            entered_box = True

    # --- broke_pressure: had under_pressure event, then ≥3 events after ---
    last_pressure_idx = -1
    for i, s in enumerate(seq):
        if s.get("under_pressure", 0.0) == 1.0:
            last_pressure_idx = i
    if last_pressure_idx >= 0 and (n - 1 - last_pressure_idx) >= 3:
        broke_pressure = True

    # --- recycled: x dropped ≥15m then recovered ≥15m ---
    peak_x = xs[0]
    trough_x = xs[0]
    for x in xs[1:]:
        if x > peak_x:
            if (peak_x - trough_x) >= _RECYCLE_DROP_M and (x - trough_x) >= _RECYCLE_DROP_M:
                recycled = True
                break
            peak_x = x
            trough_x = x
        elif x < trough_x:
            trough_x = x

    # --- bypassed_lines: reached FT (x≥80) within first BYPASS_MAX_STEP events ---
    for i, s in enumerate(seq[:_BYPASS_MAX_STEP]):
        lx_m = s.get("loc_x_norm", 0.0) * PITCH_LENGTH
        if lx_m >= FINAL_THIRD_X:
            bypassed_lines = True
            break

    return {
        "has_shot": has_shot,
        "entered_box": entered_box,
        "broke_pressure": broke_pressure,
        "recycled": recycled,
        "bypassed_lines": bypassed_lines,
    }


def _empty_seq_summary() -> dict:
    return {
        "has_shot": False,
        "entered_box": False,
        "broke_pressure": False,
        "recycled": False,
        "bypassed_lines": False,
    }


# ---------------------------------------------------------------------------
# xG / goal maps from events_df
# ---------------------------------------------------------------------------


def _build_xg_goal_maps(
    events_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Build (match_id, possession_id) → xG and → has_goal mappings
    from the flat events table.

    Looks for shot_statsbomb_xg (StatsBomb open data column name) and
    shot_outcome_name / type_name.
    """
    shots = events_df[events_df["type_name"] == "Shot"].copy()

    xg_map: dict = {}
    goal_map: dict = {}

    if shots.empty:
        return xg_map, goal_map

    xg_col = None
    for candidate in ("shot_statsbomb_xg", "shot_xg", "xg"):
        if candidate in shots.columns:
            xg_col = candidate
            break

    goal_col = None
    for candidate in ("shot_outcome_name", "shot_outcome"):
        if candidate in shots.columns:
            goal_col = candidate
            break

    for _, row in shots.iterrows():
        k = (row["match_id"], row["possession_id"])
        if xg_col:
            xg_val = float(row[xg_col]) if pd.notna(row[xg_col]) else 0.0
            xg_map[k] = xg_map.get(k, 0.0) + xg_val
        if goal_col:
            outcome = str(row[goal_col]).lower() if pd.notna(row[goal_col]) else ""
            if "goal" in outcome:
                goal_map[k] = True

    return xg_map, goal_map


# ---------------------------------------------------------------------------
# Derived label helpers
# ---------------------------------------------------------------------------


def _outcome_tier(df: pd.DataFrame) -> pd.Series:
    """
    Ordinal outcome tier:
      0 = no meaningful progression
      1 = entered final third only
      2 = entered penalty box
      3 = had a shot
      4 = scored a goal
    """
    tier = pd.Series(0, index=df.index, dtype="int8")
    tier = tier.where(~df["poss_entered_final_third"], 1)
    tier = tier.where(~df["poss_entered_box"], 2)
    tier = tier.where(~df["poss_has_shot"], 3)
    if "poss_has_goal" in df.columns:
        tier = tier.where(~df["poss_has_goal"], 4)
    return tier


def _assign_phase(df: pd.DataFrame) -> pd.Series:
    """
    Possession phase classification:
      counter      — fast transition: tempo ≥ threshold OR possession won in opp half
      final_third  — starts already in the final third (x ≥ 80)
      build_up     — starts deep in own half (x ≤ 40)
      progression  — everything else: midfield build / structured attack
    """
    start_x = df["start_x"].fillna(PITCH_LENGTH / 2)
    tempo = df["poss_tempo"].fillna(0.0)
    origin = df["origin_type"].fillna("").str.lower()

    is_counter = (tempo >= _COUNTER_TEMPO_THRESHOLD) | origin.str.contains("counter")
    is_final_third = (~is_counter) & (start_x >= _FINAL_THIRD_START_X)
    is_build_up = (~is_counter) & (~is_final_third) & (start_x <= _BUILD_UP_START_X_MAX)

    phase = pd.Series("progression", index=df.index, dtype=object)
    phase = phase.where(~is_build_up, "build_up")
    phase = phase.where(~is_final_third, "final_third")
    phase = phase.where(~is_counter, "counter")
    return phase.astype("category")


# ---------------------------------------------------------------------------
# Dtype enforcement for label columns
# ---------------------------------------------------------------------------

_BOOL_LABELS = [
    "poss_has_shot",
    "poss_entered_final_third",
    "poss_entered_box",
    "poss_dangerous",
    "poss_has_goal",
    "poss_recycled",
    "poss_broke_pressure",
    "poss_bypassed_lines",
]

_FLOAT_LABELS = [
    "poss_xg_generated",
    "poss_tempo",
    "poss_verticality",
]


def _set_label_dtypes(df: pd.DataFrame) -> None:
    for col in _BOOL_LABELS:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    for col in _FLOAT_LABELS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    if "poss_outcome_tier" in df.columns:
        df["poss_outcome_tier"] = df["poss_outcome_tier"].astype("int8")


# ---------------------------------------------------------------------------
# Label metadata — useful for notebooks / app
# ---------------------------------------------------------------------------

LABEL_DEFINITIONS: dict[str, dict] = {
    "poss_has_shot": {
        "type": "bool",
        "group": "A",
        "description": "At least one Shot event occurred in the possession.",
    },
    "poss_entered_final_third": {
        "type": "bool",
        "group": "A",
        "description": "At least one event occurred at x ≥ 80 m (final third).",
    },
    "poss_entered_box": {
        "type": "bool",
        "group": "A",
        "description": "At least one event inside the penalty area (x≥102, y∈[18,62]).",
    },
    "poss_dangerous": {
        "type": "bool",
        "group": "A",
        "description": "poss_has_shot OR poss_entered_box — primary binary target.",
    },
    "poss_xg_generated": {
        "type": "float",
        "group": "B",
        "description": "Sum of StatsBomb xG for all shots in the possession. NaN if no events_df.",
    },
    "poss_has_goal": {
        "type": "bool",
        "group": "B",
        "description": "Possession ended with a goal. Requires events_df.",
    },
    "poss_outcome_tier": {
        "type": "int8",
        "group": "B",
        "description": "Ordinal outcome: 0=nothing, 1=FT, 2=box, 3=shot, 4=goal.",
    },
    "poss_tempo": {
        "type": "float",
        "group": "C",
        "description": "Events per second (n_events / duration_seconds, clipped at 1 s).",
    },
    "poss_verticality": {
        "type": "float",
        "group": "C",
        "description": "territory_gained / (n_events × mean_pass_length). High = direct. Low = circulatory.",
    },
    "poss_recycled": {
        "type": "bool",
        "group": "C",
        "description": "x fell ≥15 m then recovered ≥15 m — possession went backwards then rebuilt.",
    },
    "poss_phase": {
        "type": "category",
        "group": "C",
        "description": "counter | build_up | progression | final_third — tactical phase at possession start.",
    },
    "poss_broke_pressure": {
        "type": "bool",
        "group": "D",
        "description": "Survived at least one pressure event with ≥3 subsequent events (played through the press).",
    },
    "poss_bypassed_lines": {
        "type": "bool",
        "group": "D",
        "description": "Reached x≥80 m within the first 4 events, starting from own half (x≤50 m).",
    },
}
