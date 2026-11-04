"""
possession_data.py
==================
Pure (Streamlit-free) helpers for inspecting scored possessions: artefact checks, held-out match
selection, the per-step danger frame, key events and the retrospective attribution lookups.

* The danger trajectory comes from the **event-time** models (prospective: step ``t`` uses
  events ``0..t`` only).
* The attribution helpers read **retrospective** cross-fitted leave-one-out drops
  (``reports/attribution``) and must be labelled as such wherever they are shown.

The interactive page lives in ``enhanced_possession_page.py``; these helpers are shared by the
analysis scripts and unit-tested in ``tests/test_possession_data.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.event_time import PossessionArrays

REQUIRED = {
    "possession table": "data/processed/possession_sequences.parquet",
    "event-time arrays": "data/processed/event_time_arrays.pkl",
    "split manifest": "data/processed/split_manifest.csv",
    "event-time metadata": "models/eventtime_meta.json",
    "event-time XGBoost": "models/eventtime_xgboost.joblib",
    "event-time GRU": "models/eventtime_gru.pt",
}


def missing_artefacts(root: Path) -> list[str]:
    """Names (with paths) of the artefacts the page needs that are not on disk."""
    return [f"{k} ({v})" for k, v in REQUIRED.items() if not (root / v).exists()]


def held_out_match_ids(manifest: pd.DataFrame, split: str = "test") -> set[int]:
    return {int(m) for m in manifest.loc[manifest["split"] == split, "match_id"]}


def index_arrays(arrs: list[PossessionArrays]) -> dict[tuple[int, int], PossessionArrays]:
    return {(a.match_id, a.possession_id): a for a in arrs}


def load_arrays(path: Path) -> dict[tuple[int, int], PossessionArrays]:
    return index_arrays(pickle.loads(path.read_bytes()))


def steps_frame(res: dict, arr: PossessionArrays) -> pd.DataFrame:
    """One row per event: prediction, change in prediction, alert flags, real coordinates."""
    df = pd.DataFrame(res["steps"]).rename(columns={"p": "p_danger"})
    df["event_idx"] = df["t"]
    df["elapsed_s"] = np.clip(arr.abs_s - arr.abs_s[0], 0.0, None)
    df["x"], df["y"] = arr.x, arr.y
    df["delta_p"] = df["p_danger"].diff()
    return df[
        [
            "event_idx",
            "event",
            "own",
            "elapsed_s",
            "x",
            "y",
            "p_danger",
            "delta_p",
            "alert",
            "tipping",
            "realised",
        ]
    ]


def key_events(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The ``k`` events after which the prediction moved most (largest ``|delta_p|``)."""
    scored = df.dropna(subset=["delta_p"])
    return scored.reindex(scored["delta_p"].abs().sort_values(ascending=False).index).head(k)


def possession_attribution(att: pd.DataFrame, match_id: int, possession_id: int) -> pd.DataFrame:
    """Cross-fitted retrospective LOO drops for one possession (seed 42 run)."""
    sel = att[(att["match_id"] == match_id) & (att["possession_id"] == possession_id)]
    if "seed" in sel.columns:
        sel = sel[sel["seed"] == sel["seed"].min()]
    return sel.sort_values("event_idx")


def leaderboard(players: pd.DataFrame, min_events: int) -> pd.DataFrame:
    """Players with at least ``min_events`` possessing-team events, best first."""
    out = players[players["events"] >= min_events].copy()
    out["interval_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out.sort_values("mean_loo", ascending=False)
