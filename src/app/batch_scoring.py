"""
batch_scoring.py
================
Batch scoring of every event of every possession with the event-time (in-play) models, used by
``frame2threat score-batch``.  The input is a flat events table (``events_parsed.parquet``
schema), because the event-time features need per-event timestamps and teams.

For each event ``t`` of a possession that has not yet produced a shot or box entry the XGBoost and GRU
probabilities of ``P(shot or box entry by the possessing team in a later event | events 0..t)`` are
returned with the alert flag (validation-chosen thresholds in ``models/eventtime_meta.json``) and,
where defined, the label.  Step ``t`` uses events ``0..t`` only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.event_types import label as type_label
from src.features.event_time import (
    FEATURE_NAMES,
    build_possession_arrays,
    danger_flags,
    sequence_inputs,
    snapshot_features,
    snapshot_labels,
)

COLUMNS = [
    "match_id",
    "possession_id",
    "event_idx",
    "event_type",
    "own_team",
    "elapsed_s",
    "eligible",
    "p_xgb",
    "p_gru",
    "alert_xgb",
    "alert_gru",
    "label_danger_after",
    "outcome_realised",
]


def load_models(root: Path):
    """Return ``(xgb_artefact, gru_net, meta)`` from ``models/eventtime_*``."""
    import joblib
    import torch

    from src.models.event_time import EventTimeGRU

    meta = json.loads((root / "models" / "eventtime_meta.json").read_text(encoding="utf-8"))
    art = joblib.load(root / "models" / "eventtime_xgboost.joblib")
    ck = torch.load(root / "models" / "eventtime_gru.pt", map_location="cpu")
    net = EventTimeGRU(ck["n_numeric"], embed_dim=ck["embed_dim"], hidden_size=ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return art, net, meta


def score_events(
    events_df: pd.DataFrame,
    root: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
    keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score every event of (a sample of) the possessions in ``events_df``."""
    from src.models.event_time import SeqSample, predict_steps

    art, net, meta = load_models(root)
    if keys is None:
        sizes = events_df.groupby(["match_id", "possession_id"]).size().reset_index(name="n")
        keys = sizes[sizes["n"] >= 2][["match_id", "possession_id"]]
    if limit is not None and limit < len(keys):
        keys = keys.sample(limit, random_state=seed)
    arrs = build_possession_arrays(events_df, keys)

    feats = np.vstack([snapshot_features(p) for p in arrs])
    p_xgb_all = art["model"].predict_proba(
        pd.DataFrame(feats, columns=FEATURE_NAMES)[art["features"]]
    )[:, 1]
    samples = []
    for p in arrs:
        elig, lab, _ = snapshot_labels(p)
        tid, num = sequence_inputs(p)
        samples.append(SeqSample(tid, num, elig, int(lab[0])))
    p_gru = predict_steps(net, samples)

    frames = []
    off = 0
    thr_x, thr_g = float(meta["xgb_threshold"]), float(meta["gru_threshold"])
    for i, p in enumerate(arrs):
        n = p.n
        elig, lab, _ = snapshot_labels(p)
        danger = danger_flags(p)
        first = int(np.argmax(danger)) if danger.any() else n
        t = np.arange(n)
        live = t < first  # outcome not yet realised: the models are defined here
        px = np.where(live, p_xgb_all[off : off + n], np.nan)
        pg = np.where(live, p_gru[i], np.nan)
        off += n
        frames.append(
            pd.DataFrame(
                {
                    "match_id": p.match_id,
                    "possession_id": p.possession_id,
                    "event_idx": t,
                    "event_type": [type_label(int(x)) for x in p.type_id],
                    "own_team": p.own,
                    "elapsed_s": np.clip(p.abs_s - p.abs_s[0], 0.0, None),
                    "eligible": elig,
                    "p_xgb": px,
                    "p_gru": pg,
                    "alert_xgb": px >= thr_x,
                    "alert_gru": pg >= thr_g,
                    "label_danger_after": pd.Series(lab, dtype="Int64").where(elig),
                    "outcome_realised": t >= first,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[COLUMNS]
