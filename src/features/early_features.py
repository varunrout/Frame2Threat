"""
early_features.py
=================
Feature engineering for v3 early-prediction experiments.

Provides two key functions:

1. ``build_start_features(poss_df)``
   → Features available **before any events occur** (spatial start position,
   origin type, match context).  Used by EXP-015 (start-only XGBoost).

2. ``build_cumulative_tabular_features(poss_df, frac)``
   → Features recomputed from only the first *frac* of each possession's
   event sequence.  Used by EXP-017 (cumulative-features XGBoost).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features.possession_features import (
    build_tabular_features,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------

#: Features available at the *instant a possession begins* — no events needed.
START_ONLY_COLS: list[str] = [
    # raw spatial
    "start_x",
    "start_y",
    # normalised spatial
    "start_x_norm",
    "start_y_norm",
    # derived spatial
    "dist_to_box_start",
    "started_final_third",
    "started_own_half",
    "started_mid_third",
    "start_zone",
    # match context
    "period",
    "is_second_half",
    # origin type one-hots
    "origin_regular_play",
    "origin_from_counter_attack",
    "origin_from_goal_kick",
    "origin_from_keeper",
    "origin_from_free_kick",
    "origin_from_corner",
    "origin_from_throw_in",
    "origin_from_kick_off",
]

#: Features that summarise the *completed* possession — **not** available
#: mid-possession.  Used for documentation / ablation analysis.
COMPLETION_DEPENDENT_COLS: list[str] = [
    # counting / temporal
    "n_events",
    "n_passes",
    "n_carries",
    "n_pressures_faced",
    "duration_seconds",
    "mean_pass_length",
    # rates
    "pass_rate",
    "carry_rate",
    "pressure_rate",
    # spatial completion
    "max_x_reached",
    "territory_gained",
    "progression_speed",
    # pressure
    "has_pressure",
    # possession-label derived
    "poss_tempo",
    "poss_verticality",
    "poss_recycled",
    "poss_broke_pressure",
    "poss_bypassed_lines",
    # phase one-hots
    "phase_counter",
    "phase_build_up",
    "phase_progression",
    "phase_final_third",
]


# ---------------------------------------------------------------------------
# 1. Start-only features  (EXP-015)
# ---------------------------------------------------------------------------


def build_start_features(poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the features known at possession-start.

    Calls the full ``build_tabular_features`` pipeline and then selects
    the :data:`START_ONLY_COLS` subset.  This guarantees consistency with
    the v2 feature engineering.

    Returns DataFrame of shape (N, len(START_ONLY_COLS)).
    """
    full = build_tabular_features(poss_df)
    # Only keep columns that exist (handles older parquets gracefully)
    cols = [c for c in START_ONLY_COLS if c in full.columns]
    return full[cols].copy()


# ---------------------------------------------------------------------------
# 2. Cumulative features  (EXP-017)
# ---------------------------------------------------------------------------

_EVENT_TYPE_PASS = 0
_EVENT_TYPE_CARRY = 1


def _recompute_aggregates_from_prefix(
    poss_df: pd.DataFrame,
    frac: float,
) -> pd.DataFrame:
    """
    Overwrite the completion-dependent aggregate columns in *poss_df* using
    only the first *frac* of each possession's event sequence.

    Possession-label features (poss_tempo, poss_verticality, …) are zeroed
    because they cannot be reliably derived from a partial sequence.
    """
    df = poss_df.copy()

    new_n_events: list[int] = []
    new_n_passes: list[int] = []
    new_n_carries: list[int] = []
    new_n_pressures: list[int] = []
    new_max_x: list[float] = []
    new_territory: list[float] = []
    new_duration: list[float] = []
    new_mean_pass_len: list[float] = []
    new_has_pressure: list[int] = []

    for _, row in df.iterrows():
        events = row["event_sequence"]
        if isinstance(events, str):
            events = json.loads(events)

        n = len(events) if events else 0
        t = max(1, int(np.ceil(n * frac)))
        prefix = events[:t] if events else []

        # Counts
        np_ = sum(1 for e in prefix if int(e.get("type_id", 9)) == _EVENT_TYPE_PASS)
        nc = sum(1 for e in prefix if int(e.get("type_id", 9)) == _EVENT_TYPE_CARRY)
        nprs = sum(int(e.get("under_pressure", 0)) for e in prefix)

        new_n_events.append(len(prefix))
        new_n_passes.append(np_)
        new_n_carries.append(nc)
        new_n_pressures.append(nprs)

        # Spatial — convert normalised coords back to pitch coords
        xs = [float(e.get("loc_x_norm", 0)) * PITCH_LENGTH for e in prefix] + [
            float(e.get("end_x_norm", 0)) * PITCH_LENGTH for e in prefix
        ]
        mx = max(xs) if xs else float(row.get("start_x", 0))
        new_max_x.append(mx)
        new_territory.append(mx - float(row.get("start_x", 0)))

        # Duration (minute_norm → minutes → seconds, rough estimate)
        if len(prefix) >= 2:
            m0 = float(prefix[0].get("minute_norm", 0))
            m1 = float(prefix[-1].get("minute_norm", 0))
            dur = abs(m1 - m0) * 90.0 * 60.0  # de-normalise: 1.0 ≈ 90 min
        else:
            dur = 0.0
        new_duration.append(dur)

        # Mean pass length
        pass_lens = [
            float(e.get("pass_length_norm", 0))
            for e in prefix
            if int(e.get("type_id", 9)) == _EVENT_TYPE_PASS
        ]
        new_mean_pass_len.append(float(np.mean(pass_lens)) if pass_lens else 0.0)

        # Has pressure
        new_has_pressure.append(int(nprs > 0))

    # Overwrite DataFrame columns
    df["n_events"] = new_n_events
    df["n_passes"] = new_n_passes
    df["n_carries"] = new_n_carries
    df["n_pressures_faced"] = new_n_pressures
    df["max_x_reached"] = new_max_x
    df["territory_gained"] = new_territory
    df["duration_seconds"] = new_duration
    df["mean_pass_length"] = new_mean_pass_len
    df["has_pressure"] = new_has_pressure

    # Zero possession-label features (not derivable from partial sequence)
    for col in [
        "poss_tempo",
        "poss_verticality",
        "poss_recycled",
        "poss_broke_pressure",
        "poss_bypassed_lines",
    ]:
        if col in df.columns:
            df[col] = 0.0
    if "poss_phase" in df.columns:
        df["poss_phase"] = "progression"  # neutral default

    return df


def build_cumulative_tabular_features(
    poss_df: pd.DataFrame,
    frac: float = 1.0,
) -> pd.DataFrame:
    """
    Build tabular features using only the first *frac* of each possession.

    At ``frac = 1.0`` this is equivalent to ``build_tabular_features()``.
    At ``frac = 0.25`` only the first quarter of events contribute to
    counting / spatial / rate features; possession-label features are zeroed.

    Parameters
    ----------
    poss_df : DataFrame from ``load_possession_sequences()``
    frac    : fraction of events to use (0 < frac ≤ 1.0)

    Returns
    -------
    pd.DataFrame with the same columns as ``build_tabular_features()``.
    """
    if frac >= 1.0:
        return build_tabular_features(poss_df)
    modified = _recompute_aggregates_from_prefix(poss_df, frac)
    return build_tabular_features(modified)


def build_prefix_sequence_tensors(
    poss_df: pd.DataFrame,
    frac: float = 1.0,
    max_seq_len: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequence tensors truncated to the first *frac* of each possession.

    Useful for training prefix-specific GRU models.
    """
    from src.features.possession_features import build_sequence_tensors, N_SEQ_FEAT

    if frac >= 1.0:
        return build_sequence_tensors(poss_df, max_seq_len=max_seq_len)

    # Truncate event sequences in a copy
    df = poss_df.copy()
    seqs: list = []
    for raw in df["event_sequence"]:
        events = json.loads(raw) if isinstance(raw, str) else list(raw)
        t = max(1, int(np.ceil(len(events) * frac)))
        seqs.append(events[:t])
    df["event_sequence"] = seqs

    return build_sequence_tensors(df, max_seq_len=max_seq_len)
