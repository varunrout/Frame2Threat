"""
possession_features.py
======================
Feature engineering for the possession-level model.

Two outputs
-----------
1. ``build_tabular_features(poss_df)``   → pd.DataFrame  (N × ~39 cols)
   Suitable for XGBoost / LightGBM.

2. ``build_sequence_tensors(poss_df)``   → np.ndarray  (N, MAX_SEQ_LEN, N_FEAT)
   Each possession's event sequence, padded / truncated, for GRU / LSTM.

Both functions accept the DataFrame produced by
``parse_possessions.build_possession_sequences()``
(or loaded via ``parse_possessions.load_possession_sequences()``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 40  # truncate long possessions, pad short ones
N_SEQ_FEAT = 8  # matches parse_possessions._SEQ_FIELDS

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
BOX_X = 102.0
BOX_Y_LO = 18.0
BOX_Y_HI = 62.0
FINAL_THIRD = 80.0
MID_THIRD = 40.0

ORIGIN_VOCAB: dict[str, int] = {
    "regular play": 0,
    "from counter attack": 1,
    "from goal kick": 2,
    "from keeper": 3,
    "from free kick": 4,
    "from corner": 5,
    "from throw in": 6,
    "from kick off": 7,
}

# poss_phase categories (matches possession_labels.py)
PHASE_CATS: list[str] = ["counter", "build_up", "progression", "final_third"]


# ---------------------------------------------------------------------------
# 1. Tabular features
# ---------------------------------------------------------------------------


def build_tabular_features(poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ~30 tabular features from the possession_sequences table.

    Returns a DataFrame with the same index as ``poss_df``.
    No label columns are included.
    """
    df = poss_df.copy()
    feats: dict[str, pd.Series] = {}

    # --- raw spatial pass-throughs ---
    # max_x_reached and territory_gained summarise the completed possession.
    # They are valid for retrospective analysis, but not leakage-free early prediction.
    feats["start_x"] = df["start_x"].astype(float)
    feats["start_y"] = df["start_y"].astype(float)
    feats["max_x_reached"] = df["max_x_reached"].astype(float)
    feats["territory_gained"] = df["territory_gained"].astype(float)

    # --- spatial derived ---
    feats["start_x_norm"] = feats["start_x"] / PITCH_LENGTH
    feats["start_y_norm"] = feats["start_y"] / PITCH_WIDTH
    # distance from possession start to the near post of attacking box
    box_centre_y = (BOX_Y_LO + BOX_Y_HI) / 2.0  # 40.0
    feats["dist_to_box_start"] = np.sqrt(
        (BOX_X - feats["start_x"]).clip(0) ** 2 + (feats["start_y"] - box_centre_y) ** 2
    )
    feats["started_final_third"] = (feats["start_x"] >= FINAL_THIRD).astype(int)
    feats["started_own_half"] = (feats["start_x"] < MID_THIRD).astype(int)
    feats["started_mid_third"] = (
        (feats["start_x"] >= MID_THIRD) & (feats["start_x"] < FINAL_THIRD)
    ).astype(int)

    # start_y zone: 0=left flank, 1=central, 2=right flank (using attacking side)
    feats["start_zone"] = (
        pd.cut(df["start_y"], bins=[0, 26.7, 53.3, 80.0], labels=[0, 1, 2], right=True)
        .astype("Int8")
        .fillna(1)
        .astype(int)
    )

    # --- temporal / counting ---
    feats["n_events"] = df["n_events"].astype(float)
    feats["n_passes"] = df["n_passes"].astype(float)
    feats["n_carries"] = df["n_carries"].astype(float)
    feats["n_pressures_faced"] = df["n_pressures_faced"].astype(float)
    feats["duration_seconds"] = df["duration_seconds"].astype(float)
    feats["mean_pass_length"] = df["mean_pass_length"].fillna(0).astype(float)

    # rates
    n_ev = df["n_events"].astype(float).clip(lower=1)
    feats["pass_rate"] = df["n_passes"].astype(float) / n_ev
    feats["carry_rate"] = df["n_carries"].astype(float) / n_ev
    feats["pressure_rate"] = df["n_pressures_faced"].astype(float) / n_ev

    # progression speed (x per second)
    dur = df["duration_seconds"].astype(float).clip(lower=1)
    feats["progression_speed"] = df["territory_gained"].astype(float).fillna(0) / dur

    # --- period ---
    feats["period"] = df["period"].astype(int)
    feats["is_second_half"] = (df["period"] >= 2).astype(int)

    # --- pressure flag ---
    feats["has_pressure"] = df["has_pressure"].astype(int)

    # --- origin type (one-hot, 8 classes) ---
    origin_norm = df["origin_type"].str.lower().str.strip().map(ORIGIN_VOCAB).fillna(0).astype(int)
    for name, idx in ORIGIN_VOCAB.items():
        safe = name.replace(" ", "_")
        feats[f"origin_{safe}"] = (origin_norm == idx).astype(int)

    # --- new possession-label derived features ---
    # Only added if each column is present; fallback to neutral default otherwise
    # so the function stays compatible with old parquets and tests.
    # Note: outcome-encoding labels (poss_has_shot, poss_entered_box, poss_dangerous,
    # poss_outcome_tier, poss_xg_generated, poss_has_goal) are intentionally excluded
    # as they directly encode the prediction target.
    for col, default in [
        ("poss_tempo", 0.0),
        ("poss_verticality", 0.0),
        ("poss_recycled", 0.0),
        ("poss_broke_pressure", 0.0),
        ("poss_bypassed_lines", 0.0),
    ]:
        if col in df.columns:
            vals = df[col].fillna(default).astype(float)
            if col == "poss_verticality":
                cap = vals.quantile(0.99) if vals.max() > 0 else 1.0
                vals = vals.clip(upper=float(cap))
        else:
            vals = pd.Series(float(default), index=df.index, dtype=float)
        feats[col] = vals

    # poss_phase → 4 one-hot columns
    if "poss_phase" in df.columns:
        phase = df["poss_phase"].astype(str)
    else:
        phase = pd.Series("progression", index=df.index)
    for cat in PHASE_CATS:
        feats[f"phase_{cat}"] = (phase == cat).astype(int)

    result = pd.DataFrame(feats, index=df.index)
    return result


# ---------------------------------------------------------------------------
# 2. Sequence tensors for GRU
# ---------------------------------------------------------------------------


def build_sequence_tensors(
    poss_df: pd.DataFrame,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build padded sequence tensors from the ``event_sequence`` column.

    Parameters
    ----------
    poss_df     : DataFrame from ``load_possession_sequences()``
    max_seq_len : truncate / pad to this many steps (default 40)

    Returns
    -------
    X : np.ndarray  shape (N, max_seq_len, N_SEQ_FEAT)  float32
    lengths : np.ndarray  shape (N,)  int32 — true sequence length before padding
    """
    N = len(poss_df)
    X = np.zeros((N, max_seq_len, N_SEQ_FEAT), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int32)

    for i, seq_raw in enumerate(poss_df["event_sequence"]):
        # Handle both JSON string (from parquet) and Python list (in-memory)
        if isinstance(seq_raw, str):
            seq: list[dict] = json.loads(seq_raw)
        else:
            seq = seq_raw  # type: ignore[assignment]

        T = min(len(seq), max_seq_len)
        lengths[i] = T

        for t, ev in enumerate(seq[:T]):
            X[i, t, 0] = ev.get("type_id", 0)
            X[i, t, 1] = ev.get("loc_x_norm", 0)
            X[i, t, 2] = ev.get("loc_y_norm", 0)
            X[i, t, 3] = ev.get("end_x_norm", 0)
            X[i, t, 4] = ev.get("end_y_norm", 0)
            X[i, t, 5] = ev.get("under_pressure", 0)
            X[i, t, 6] = ev.get("pass_length_norm", 0)
            X[i, t, 7] = ev.get("minute_norm", 0)

    return X, lengths


# ---------------------------------------------------------------------------
# 3. Combined pipeline — returns everything needed for notebooks
# ---------------------------------------------------------------------------


def prepare_features_and_labels(
    poss_df: pd.DataFrame,
    label_col: str = "poss_dangerous",
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict:
    """
    One-shot feature preparation used by notebooks 07 and 08.

    Returns
    -------
    dict with keys:
      "X_tab"    : pd.DataFrame  tabular features
      "X_seq"    : np.ndarray    (N, max_seq_len, N_SEQ_FEAT) sequence tensors
      "lengths"  : np.ndarray    (N,) true sequence lengths
      "y"        : pd.Series     binary label
      "meta"     : pd.DataFrame  identity cols (match_id, possession_id, team_name)
    """
    X_tab = build_tabular_features(poss_df)
    X_seq, lengths = build_sequence_tensors(poss_df, max_seq_len=max_seq_len)
    y = poss_df[label_col].astype(int)
    meta = poss_df[["match_id", "possession_id", "team_name"]].copy()

    return {
        "X_tab": X_tab,
        "X_seq": X_seq,
        "lengths": lengths,
        "y": y,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Utility: get feature names for XGBoost / SHAP
# ---------------------------------------------------------------------------


def get_tabular_feature_names() -> list[str]:
    """Return the ordered list of tabular feature column names."""
    dummy = pd.DataFrame(
        {
            "start_x": [60.0],
            "start_y": [40.0],
            "max_x_reached": [90.0],
            "territory_gained": [30.0],
            "end_x": [90.0],
            "end_y": [40.0],
            "n_events": [15],
            "n_passes": [6],
            "n_carries": [5],
            "n_pressures_faced": [2],
            "duration_seconds": [20],
            "mean_pass_length": [15.0],
            "has_pressure": [True],
            "period": [1],
            "origin_type": ["Regular Play"],
            "poss_has_shot": [False],
            "poss_entered_box": [False],
            "poss_entered_final_third": [True],
            "poss_dangerous": [False],
            "event_sequence": ["[]"],
            # new label-derived features
            "poss_tempo": [1.0],
            "poss_verticality": [0.5],
            "poss_recycled": [False],
            "poss_broke_pressure": [False],
            "poss_bypassed_lines": [False],
            "poss_phase": ["counter"],
        }
    )
    return build_tabular_features(dummy).columns.tolist()


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.parse_possessions import load_possession_sequences

    print("Loading possession_sequences.parquet …")
    poss = load_possession_sequences()
    print(f"  {len(poss):,} possessions")

    print("\nBuilding tabular features …")
    X_tab = build_tabular_features(poss)
    print(f"  Shape  : {X_tab.shape}")
    print(f"  Columns: {X_tab.columns.tolist()}")
    print(f"  NaNs   : {X_tab.isna().sum().sum()}")

    print("\nBuilding sequence tensors …")
    X_seq, lengths = build_sequence_tensors(poss)
    print(f"  Tensor shape  : {X_seq.shape}")
    print(f"  Max length    : {lengths.max()}")
    print(f"  Mean length   : {lengths.mean():.1f}")
    print(f"  Padded (< 40) : {(lengths < MAX_SEQ_LEN).sum():,}")

    print("\nSample tabular row:")
    print(X_tab.iloc[0].to_string())
