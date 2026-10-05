"""Regenerate the committed evidence pack (``reports/repro/scored_possessions_sample.csv``).

25 possessions are drawn with a fixed seed from the **test-split** matches, scored with the
event-time (in-play) XGBoost and GRU, plus the retrospective models (retrained on the corrected
labels; ``models/v2_corrected_*``) clearly labelled as retrospective.

Usage:  python scripts/make_evidence_pack.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.batch_scoring import score_events  # noqa: E402
from src.data.parse_possessions import load_possession_sequences  # noqa: E402
from src.evaluation.possession_attribution import load_gru_model  # noqa: E402
from src.features.possession_features import (
    build_sequence_tensors,
    build_tabular_features,
)  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(".")
SEED = 42
N = 25
OUT = Path("reports/repro")


def main() -> None:
    import torch

    from src.models.gru_possession import evaluate, make_dataloader

    poss = load_possession_sequences("data/processed/possession_sequences.parquet")
    test_ids = set(pd.read_parquet("data/processed/test.parquet")["match_id"].unique())
    pool = poss[poss["match_id"].isin(test_ids) & (poss["n_events"] >= 4)]
    sample = (
        pool.sample(N, random_state=SEED)
        .sort_values(["match_id", "possession_id"])
        .reset_index(drop=True)
    )
    events = pd.read_parquet("data/processed/events_parsed.parquet")
    keys = sample[["match_id", "possession_id"]]
    ev = score_events(events, ROOT, keys=keys)

    rows = []
    for r in sample.itertuples(index=False):
        g = ev[(ev["match_id"] == r.match_id) & (ev["possession_id"] == r.possession_id)]
        live = g[~g["outcome_realised"]]
        first_danger = (
            g.loc[g["outcome_realised"], "event_idx"].min()
            if g["outcome_realised"].any()
            else np.nan
        )

        def first_alert(col):
            hit = live.loc[live[col], "event_idx"]
            return hit.min() if len(hit) else np.nan

        fx, fg = first_alert("alert_xgb"), first_alert("alert_gru")
        rows.append(
            {
                "match_id": r.match_id,
                "possession_id": r.possession_id,
                "team_name": r.team_name,
                "period": r.period,
                "origin_type": r.origin_type,
                "n_events": r.n_events,
                "poss_dangerous": bool(r.poss_dangerous),
                "first_danger_event_idx": first_danger,
                "eventtime_xgb_max_p": (
                    round(float(live["p_xgb"].max()), 4) if len(live) else np.nan
                ),
                "eventtime_gru_max_p": (
                    round(float(live["p_gru"].max()), 4) if len(live) else np.nan
                ),
                "eventtime_xgb_first_alert_idx": fx,
                "eventtime_gru_first_alert_idx": fg,
                "eventtime_xgb_lead_events": (
                    first_danger - fx if not (np.isnan(first_danger) or np.isnan(fx)) else np.nan
                ),
                "eventtime_gru_lead_events": (
                    first_danger - fg if not (np.isnan(first_danger) or np.isnan(fg)) else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)

    # retrospective models (corrected labels): these see the whole possession
    xgb = joblib.load("models/v2_corrected_xgboost_poss_dangerous.joblib")
    out["RETROSPECTIVE_xgb_prob"] = np.round(
        xgb.predict_proba(build_tabular_features(sample))[:, 1], 4
    )
    gru, _ = load_gru_model("models/v2_corrected_gru_poss_dangerous.pt")
    X, L = build_sequence_tensors(sample)
    loader = make_dataloader(
        X, L, sample["poss_dangerous"].astype(int).to_numpy(), batch_size=64, shuffle=False
    )
    out["RETROSPECTIVE_gru_prob"] = np.round(evaluate(gru, loader, torch.device("cpu"))[1], 4)
    out.to_csv(OUT / "scored_possessions_sample.csv", index=False)

    meta = json.loads(Path("models/eventtime_meta.json").read_text())
    manifest = {
        "generated_by": "scripts/make_evidence_pack.py",
        "seed": SEED,
        "sampling": f"{N} possessions with >= 4 events drawn at random (seed {SEED}) from the TEST-split matches of data/processed/possession_sequences.parquet",
        "generated_from": {
            "data": [
                "data/processed/possession_sequences.parquet",
                "data/processed/events_parsed.parquet",
            ],
            "models": [
                "models/eventtime_xgboost.joblib",
                "models/eventtime_gru.pt",
                "models/eventtime_meta.json",
                "models/v2_corrected_xgboost_poss_dangerous.joblib",
                "models/v2_corrected_gru_poss_dangerous.pt",
            ],
        },
        "n_rows": int(len(out)),
        "columns": list(out.columns),
        "thresholds": {
            "xgb": meta["xgb_threshold"],
            "gru": meta["gru_threshold"],
            "rule": meta["threshold_rule"],
        },
        "notes": [
            "eventtime_* columns come from the in-play event-time models: step t uses events 0..t only; "
            "scores are defined while no shot / box entry (incl. a possessing-team pass ending in the box) has occurred.",
            "first_danger_event_idx is the first shot / box-entry event (NaN if none); lead_events = first_danger_event_idx - first alert.",
            "RETROSPECTIVE_* columns are full-possession models retrained on the frame-corrected labels; they see the "
            "whole possession and are NOT early predictions.",
            "The fraction-based v3 columns (xgb_start_prob, xgb_50pct_prob, xgb_75pct_prob) were withdrawn: the prefix length encodes the possession length.",
        ],
    }
    (OUT / "scored_possessions_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(out.head(8).to_string())
    print(f"{len(out)} rows written")


if __name__ == "__main__":
    main()
