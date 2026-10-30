"""
live_scoring.py
===============
Event-by-event scoring with the event-time (in-play) models, used by ``frame2threat
score-live``.

Input events use the ``possession_sequences.parquet`` ``event_sequence`` schema
(``type_id``, ``loc_x_norm``, ``loc_y_norm``, ``end_x_norm``, ``end_y_norm``,
``under_pressure``, ``pass_length_norm``) plus two optional keys:

* ``own_team``        1.0 / 0.0 (default 1.0): event performed by the possessing team;
* ``elapsed_seconds`` seconds since the possession's first event.  If absent, events are
  assumed to be spaced by the training-set median inter-event time (a warning is returned).

Step ``t`` is scored from events ``0..t`` only.  Scoring stops once the possessing team has
produced a shot or box entry: from then on the outcome is realised and there is nothing
left to forecast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.event_types import EventType
from src.data.event_types import label as type_label
from src.data.possession_frame import PITCH_LENGTH, PITCH_WIDTH
from src.features.event_time import (
    FEATURE_NAMES,
    PossessionArrays,
    danger_flags,
    sequence_inputs,
    snapshot_features,
)

DEFAULT_ORIGIN = "Regular Play"


def events_to_arrays(
    events: list[dict],
    *,
    origin: str = DEFAULT_ORIGIN,
    period: int = 1,
    minute: float = 45.0,
    median_dt: float = 2.0,
) -> tuple[PossessionArrays, list[str]]:
    """Convert schema dicts to :class:`PossessionArrays`; returns (arrays, warnings)."""
    warns: list[str] = []
    n = len(events)
    type_id = np.array([int(e.get("type_id", 0)) for e in events], dtype=int)
    x = np.array([float(e.get("loc_x_norm", 0.0)) * PITCH_LENGTH for e in events])
    y = np.array([float(e.get("loc_y_norm", 0.0)) * PITCH_WIDTH for e in events])
    is_pass = type_id == int(EventType.PASS)
    end_x = np.where(
        is_pass, [float(e.get("end_x_norm", 0.0)) * PITCH_LENGTH for e in events], np.nan
    )
    end_y = np.where(
        is_pass, [float(e.get("end_y_norm", 0.0)) * PITCH_WIDTH for e in events], np.nan
    )
    own = np.array([float(e.get("own_team", 1.0)) >= 0.5 for e in events])
    if all("elapsed_seconds" in e for e in events):
        abs_s = np.array([float(e["elapsed_seconds"]) for e in events])
    else:
        warns.append(
            f"no 'elapsed_seconds' in the input: assuming {median_dt:.1f}s between events "
            "(training median); time-dependent features are approximate."
        )
        abs_s = np.arange(n, dtype=float) * median_dt
    arr = PossessionArrays(
        match_id=0,
        possession_id=0,
        team="possessing",
        period=period,
        minute=minute,
        origin=origin,
        type_id=type_id,
        own=own,
        x=x,
        y=y,
        end_x=end_x,
        end_y=end_y,
        under_pressure=np.array([float(e.get("under_pressure", 0.0)) for e in events]),
        pass_length=np.where(
            is_pass, [float(e.get("pass_length_norm", 0.0)) * 60.0 for e in events], np.nan
        ),
        abs_s=abs_s,
    )
    return arr, warns


def load_meta(root: Path) -> dict:
    return json.loads((root / "models" / "eventtime_meta.json").read_text(encoding="utf-8"))


def score_stream(
    events: list[dict],
    root: Path,
    *,
    model: str = "gru",
    threshold: float | None = None,
    origin: str = DEFAULT_ORIGIN,
    period: int = 1,
    minute: float = 45.0,
) -> dict:
    """Score each event of a possession; returns steps, threshold and warnings."""
    meta = load_meta(root)
    arr, warns = events_to_arrays(
        events,
        origin=origin,
        period=period,
        minute=minute,
        median_dt=float(meta.get("median_inter_event_seconds", 2.0)),
    )
    return score_arrays(arr, root, model=model, threshold=threshold, warnings=warns)


def score_arrays(
    arr: PossessionArrays,
    root: Path,
    *,
    model: str = "gru",
    threshold: float | None = None,
    warnings: list[str] | None = None,
) -> dict:
    """Score a prepared :class:`PossessionArrays` event by event (step ``t`` uses events ``0..t``)."""
    meta = load_meta(root)
    thr = float(threshold if threshold is not None else meta[f"{model}_threshold"])
    warns = list(warnings or [])
    if model == "gru":
        import torch

        from src.models.event_time import EventTimeGRU

        ck = torch.load(root / "models" / "eventtime_gru.pt", map_location="cpu")
        net = EventTimeGRU(ck["n_numeric"], embed_dim=ck["embed_dim"], hidden_size=ck["hidden"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        tid, num = sequence_inputs(arr)
        with torch.no_grad():
            probs = torch.sigmoid(net(torch.from_numpy(tid)[None], torch.from_numpy(num)[None]))[
                0
            ].numpy()
    else:
        import joblib
        import pandas as pd

        art = joblib.load(root / "models" / "eventtime_xgboost.joblib")
        feats = pd.DataFrame(snapshot_features(arr), columns=FEATURE_NAMES)
        probs = art["model"].predict_proba(feats[art["features"]])[:, 1]

    danger = danger_flags(arr)
    realised_at = int(np.argmax(danger)) if danger.any() else None
    steps = []
    alerted = False
    for t in range(arr.n):
        if realised_at is not None and t >= realised_at:
            steps.append(
                {
                    "t": t,
                    "event": type_label(int(arr.type_id[t])),
                    "own": bool(arr.own[t]),
                    "p": None,
                    "alert": False,
                    "tipping": False,
                    "realised": t == realised_at,
                }
            )
            continue
        p = float(probs[t])
        tip = p >= thr and not alerted
        alerted = alerted or p >= thr
        steps.append(
            {
                "t": t,
                "event": type_label(int(arr.type_id[t])),
                "own": bool(arr.own[t]),
                "p": p,
                "alert": p >= thr,
                "tipping": tip,
                "realised": False,
            }
        )
    return {
        "steps": steps,
        "threshold": thr,
        "model": model,
        "warnings": warns,
        "realised_at": realised_at,
    }
