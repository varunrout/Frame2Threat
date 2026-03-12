"""
possession_attribution.py
=========================
Per-event attribution within a possession for the GRU and XGBoost models.

Key functions
-------------
load_gru_model(path)                              → (PossessionGRU, config_dict)
cumulative_danger_scores(event_seq, model)        → np.ndarray (T,)
leave_one_out_attribution(event_seq, model)       → np.ndarray (T,)
attribute_possession(poss_row, gru_model)         → dict
player_attribution_summary(poss_df, gru_model)   → pd.DataFrame
unlock_event_index(attributions)                  → int

Usage example
-------------
    from src.evaluation.possession_attribution import (
        load_gru_model, attribute_possession, player_attribution_summary
    )
    gru, cfg = load_gru_model("models/gru_poss_dangerous.pt")
    poss_df  = load_possession_sequences("data/processed/possession_sequences.parquet")

    # attribute one possession
    row    = poss_df.iloc[42]
    report = attribute_possession(row, gru)

    # aggregate by player across all possessions
    summary = player_attribution_summary(poss_df, gru)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (must match parse_possessions / possession_features)
# ---------------------------------------------------------------------------
PITCH_LENGTH = 120.0
PITCH_WIDTH  =  80.0
MAX_SEQ_LEN  =  40
N_SEQ_FEAT   =   8   # [type_id, loc_x_norm, loc_y_norm, end_x_norm, end_y_norm,
                     #  under_pressure, pass_length_norm, minute_norm]

EVENT_TYPE_LABELS: dict[int, str] = {
    0: "Pass",
    1: "Carry",
    2: "Shot",
    3: "Dribble",
    4: "Pressure",
    5: "Ball Receipt",
    6: "Duel",
    7: "Clearance",
    8: "Interception",
    9: "Other",
}


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_gru_model(path: str | Path = "models/gru_poss_dangerous.pt"):
    """
    Load a saved PossessionGRU checkpoint.

    Returns
    -------
    model  : PossessionGRU  (eval mode, CPU)
    config : dict           (architecture hyperparams)
    """
    from src.models.gru_possession import PossessionGRU

    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model  = PossessionGRU(
        input_size   = config.get("input_size",   8),
        hidden_size  = config.get("hidden_size",  64),
        num_layers   = config.get("num_layers",   1),
        dropout      = config.get("dropout",      0.0),
        bidirectional= config.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# 2. Sequence helpers
# ---------------------------------------------------------------------------

def _parse_event_seq(poss_row: pd.Series) -> list[dict]:
    """Return the event_sequence list from a possession row."""
    raw = poss_row.get("event_sequence", "[]")
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)


def _events_to_tensor(events: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of event dicts to (x_tensor, length_tensor) ready for GRU.

    x_tensor : (1, T_trunc, 8)   float32
    length   : (1,)              int64
    """
    T     = min(len(events), MAX_SEQ_LEN)
    arr   = np.zeros((1, T, N_SEQ_FEAT), dtype=np.float32)
    for t, ev in enumerate(events[:T]):
        arr[0, t, 0] = float(ev.get("type_id",            0))
        arr[0, t, 1] = float(ev.get("loc_x_norm",         0))
        arr[0, t, 2] = float(ev.get("loc_y_norm",         0))
        arr[0, t, 3] = float(ev.get("end_x_norm",         0))
        arr[0, t, 4] = float(ev.get("end_y_norm",         0))
        arr[0, t, 5] = float(ev.get("under_pressure",     0))
        arr[0, t, 6] = float(ev.get("pass_length_norm",   0))
        arr[0, t, 7] = float(ev.get("minute_norm",        0))
    x_t  = torch.from_numpy(arr)
    len_t = torch.tensor([T], dtype=torch.long)
    return x_t, len_t


@torch.no_grad()
def _score_sequence(
    model: "PossessionGRU",
    x: torch.Tensor,
    length: torch.Tensor,
) -> float:
    """Return sigmoid probability for a single prepared tensor."""
    logit = model(x, length)
    return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# 3. Core attribution methods
# ---------------------------------------------------------------------------

@torch.no_grad()
def cumulative_danger_scores(
    events: list[dict],
    model: "PossessionGRU",
) -> np.ndarray:
    """
    Danger probability after each prefix [0..t+1] of the possession.

    Returns array of shape (T,) where T = min(len(events), MAX_SEQ_LEN).
    scores[t] = P(dangerous | first t+1 events).

    Implementation: batches all T prefix sequences in one forward pass using
    packed sequences with varying lengths — ~T× speedup over individual calls.
    """
    T        = min(len(events), MAX_SEQ_LEN)
    x_one, _ = _events_to_tensor(events)  # (1, T, 8)
    x_one    = x_one.squeeze(0)           # (T, 8)

    # Repeat the full sequence T times; mask timesteps > t for prefix t
    batch   = x_one.unsqueeze(0).expand(T, -1, -1).clone()  # (T, T, 8)
    for t in range(T):
        batch[t, t + 1:, :] = 0.0   # zero-pad the tail

    lengths = torch.arange(1, T + 1, dtype=torch.long)  # [1,2,...,T]
    logits  = model(batch, lengths)                       # (T,)
    probs   = torch.sigmoid(logits).numpy()
    return probs.astype(np.float32)


@torch.no_grad()
def leave_one_out_attribution(
    events: list[dict],
    model: "PossessionGRU",
) -> np.ndarray:
    """
    Attribution by leave-one-out: how much does the final danger score drop
    when event t is zeroed out (replaced with all-zeros feature vector)?

    attribution[t] = score_full − score_with_event_t_zeroed

    Positive values → event elevated danger.
    Negative values → event suppressed danger (e.g. the ball going backwards).

    Returns array of shape (T,).

    Implementation: batches all T masked sequences in a single forward pass
    (batch size = T+1) for ~40x speedup over individual calls.
    """
    T         = min(len(events), MAX_SEQ_LEN)
    x_one, _  = _events_to_tensor(events)   # (1, T, 8)
    x_one     = x_one.squeeze(0)             # (T, 8)

    # Build batch of T+1 sequences: [full, mask_t=0, mask_t=1, ...]
    # All have the same length T.
    batch = x_one.unsqueeze(0).expand(T + 1, -1, -1).clone()  # (T+1, T, 8)
    for t in range(T):
        batch[t + 1, t, :] = 0.0   # row t+1 has event t zeroed

    lengths = torch.full((T + 1,), T, dtype=torch.long)
    logits  = model(batch, lengths)               # (T+1,)
    probs   = torch.sigmoid(logits).numpy()

    score_full = probs[0]
    attrs      = score_full - probs[1:]           # (T,)
    return attrs.astype(np.float32)


def unlock_event_index(attributions: np.ndarray) -> int:
    """
    Index of the single event with the highest positive attribution.
    Returns -1 if all attributions are non-positive.
    """
    if attributions.max() <= 0:
        return -1
    return int(np.argmax(attributions))


# ---------------------------------------------------------------------------
# 4. Full possession report
# ---------------------------------------------------------------------------

def attribute_possession(
    poss_row: pd.Series,
    gru_model: "PossessionGRU",
) -> dict[str, Any]:
    """
    Full attribution report for a single possession.

    Returns
    -------
    dict with keys:
        "possession_id"   : str
        "match_id"        : int | str
        "team"            : str
        "n_events"        : int
        "poss_dangerous"  : bool
        "final_score"     : float       — GRU P(dangerous) on full sequence
        "cumulative"      : list[float] — P(dangerous) after each event
        "loo_attr"        : list[float] — leave-one-out attributions
        "unlock_index"    : int         — argmax of loo_attr (or -1)
        "events"          : list[dict]  — original event dicts with added fields:
                              cum_score, loo_attr, type_label
    """
    events = _parse_event_seq(poss_row)
    if not events:
        return {"error": "empty event sequence"}

    T          = min(len(events), MAX_SEQ_LEN)
    events_tr  = events[:T]

    cum    = cumulative_danger_scores(events_tr, gru_model)
    loo    = leave_one_out_attribution(events_tr, gru_model)
    unlock = unlock_event_index(loo)

    enriched = []
    for t, ev in enumerate(events_tr):
        enriched.append({
            **ev,
            "step"        : t,
            "type_label"  : EVENT_TYPE_LABELS.get(int(ev.get("type_id", 9)), "Other"),
            "cum_score"   : float(cum[t]),
            "loo_attr"    : float(loo[t]),
            "is_unlock"   : (t == unlock),
        })

    return {
        "possession_id" : str(poss_row.get("possession_id", "")),
        "match_id"      : poss_row.get("match_id"),
        "team"          : str(poss_row.get("team_name", "")),
        "n_events"      : T,
        "poss_dangerous": bool(poss_row.get("poss_dangerous", False)),
        "final_score"   : float(cum[-1]) if len(cum) else 0.0,
        "cumulative"    : cum.tolist(),
        "loo_attr"      : loo.tolist(),
        "unlock_index"  : unlock,
        "events"        : enriched,
    }


# ---------------------------------------------------------------------------
# 5. Player-level aggregation
# ---------------------------------------------------------------------------

def player_attribution_summary(
    poss_df: pd.DataFrame,
    gru_model: "PossessionGRU",
    *,
    min_touches: int = 10,
    dangerous_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate per-player attribution scores across all possessions.

    Requires the possession DataFrame to have a ``player_sequence`` column
    (list of player names aligned with event_sequence).  If missing, falls
    back to ``team_name`` as the player identifier.

    Parameters
    ----------
    poss_df        : possession_sequences DataFrame
    gru_model      : loaded PossessionGRU (eval mode)
    min_touches    : minimum events attributed to a player to appear in output
    dangerous_only : if True, only consider possessions labelled dangerous
    verbose        : print progress every 500 rows

    Returns
    -------
    pd.DataFrame with columns:
        player, team, n_touches, n_unlocks, mean_loo_attr,
        p90_loo_attr, mean_score_at_touch, n_possessions
    """
    rows: list[dict] = []

    if dangerous_only:
        poss_df = poss_df[poss_df["poss_dangerous"] == 1]

    has_players = "player_sequence" in poss_df.columns

    for i, (_, row) in enumerate(poss_df.iterrows()):
        if verbose and i % 500 == 0:
            print(f"  Attributing possession {i}/{len(poss_df)} …", end="\r")

        events = _parse_event_seq(row)
        if not events:
            continue

        T      = min(len(events), MAX_SEQ_LEN)
        events = events[:T]

        try:
            loo = leave_one_out_attribution(events, gru_model)
            cum = cumulative_danger_scores(events, gru_model)
        except Exception:
            continue

        # resolve player name per step
        if has_players:
            raw_players = row["player_sequence"]
            if isinstance(raw_players, str):
                raw_players = json.loads(raw_players)
            players = (list(raw_players) + ["Unknown"] * T)[:T]
        else:
            players = [str(row.get("team_name", "Unknown"))] * T

        team = str(row.get("team_name", ""))
        unlock_idx = unlock_event_index(loo)

        for t in range(T):
            rows.append({
                "player"           : players[t],
                "team"             : team,
                "loo_attr"         : float(loo[t]),
                "cum_score"        : float(cum[t]),
                "is_unlock"        : (t == unlock_idx and unlock_idx >= 0),
                "possession_id"    : str(row.get("possession_id", "")),
                "match_id"         : row.get("match_id"),
            })

    if verbose:
        print()

    if not rows:
        return pd.DataFrame(columns=[
            "player", "team", "n_touches", "n_unlocks",
            "mean_loo_attr", "p90_loo_attr", "mean_score_at_touch", "n_possessions"
        ])

    touches = pd.DataFrame(rows)

    summary = (
        touches.groupby(["player", "team"])
        .agg(
            n_touches          = ("loo_attr",      "count"),
            n_unlocks          = ("is_unlock",     "sum"),
            mean_loo_attr      = ("loo_attr",      "mean"),
            p90_loo_attr       = ("loo_attr",      lambda s: np.percentile(s, 90)),
            mean_score_at_touch= ("cum_score",     "mean"),
            n_possessions      = ("possession_id", "nunique"),
        )
        .reset_index()
    )

    summary = summary[summary["n_touches"] >= min_touches].copy()
    summary["unlock_rate"] = (
        summary["n_unlocks"] / summary["n_touches"]
    ).round(4)
    summary = summary.sort_values("mean_loo_attr", ascending=False)
    return summary


# ---------------------------------------------------------------------------
# 6. Batch scoring utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_all_possessions(
    poss_df: pd.DataFrame,
    gru_model: "PossessionGRU",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Score every possession in ``poss_df`` using the GRU model.

    Returns np.ndarray of shape (N,) with P(dangerous) for each possession.
    Much faster than calling attribute_possession row-by-row when you only
    need the final score.
    """
    from src.features.possession_features import build_sequence_tensors
    from src.models.gru_possession import make_dataloader

    X, L = build_sequence_tensors(poss_df)
    # dummy labels
    y = np.zeros(len(poss_df), dtype=np.float32)
    loader = make_dataloader(X, L, y, batch_size=batch_size, shuffle=False)

    all_proba: list[np.ndarray] = []
    gru_model.eval()
    for X_b, L_b, _ in loader:
        logits = gru_model(X_b, L_b)
        all_proba.append(torch.sigmoid(logits).numpy())

    return np.concatenate(all_proba)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    warnings.filterwarnings("ignore")

    from src.data.parse_possessions import load_possession_sequences

    print("Loading GRU model …")
    gru, cfg = load_gru_model("models/gru_poss_dangerous.pt")
    print(f"  Config: {cfg}")

    print("Loading possessions …")
    poss = load_possession_sequences("data/processed/possession_sequences.parquet")
    dangerous = poss[poss["poss_dangerous"] == 1]
    print(f"  {len(poss):,} total | {len(dangerous):,} dangerous")

    # Attribute a sample possession
    row    = dangerous.iloc[0]
    report = attribute_possession(row, gru)
    print(f"\nPossession: {report['possession_id']}  |  team: {report['team']}")
    print(f"  n_events     : {report['n_events']}")
    print(f"  poss_dangerous: {report['poss_dangerous']}")
    print(f"  final_score  : {report['final_score']:.3f}")
    unlock = report["unlock_index"]
    if unlock >= 0:
        ev = report["events"][unlock]
        print(f"  unlock event : step {unlock} — {ev['type_label']} "
              f"(attr={ev['loo_attr']:+.3f})")

    # Mini player summary on 200 possessions
    print("\nBuilding player attribution summary (sample 200) …")
    sample = poss.sample(200, random_state=42)
    summary = player_attribution_summary(sample, gru, min_touches=3, verbose=True)
    print(f"\nTop 10 players by mean LOO attribution:")
    print(summary.head(10).to_string(index=False))
