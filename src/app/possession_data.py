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


FROZEN_TEST_RESULTS = "reports/repro/frozen_metrics_test.json"
HOLDOUT_RESULTS = "reports/repro/sealed_holdout.json"


def _auc(block: dict, model: str) -> dict:
    return block[model]["roc_auc"]


def limitation_lines(root: Path) -> list[str]:
    """Known-limitations bullets for the dashboard, with every number read from committed artefacts.

    Sources: ``reports/repro/frozen_metrics_test.json`` (original test split) and
    ``reports/repro/sealed_holdout.json`` (Euro 2024), both written by ``scripts/sealed_holdout.py`` with
    the frozen false-alert and ECE definitions. A missing file gives an explicit "not available" line.
    """
    import json

    lines: list[str] = []
    test_path = root / FROZEN_TEST_RESULTS
    if test_path.exists():
        rec = json.loads(test_path.read_text(encoding="utf-8"))
        r = rec["results"]
        ba, bo, al, ece = (
            r["bootstrap_all"],
            r["bootstrap_one_per_possession"],
            r["alerts"],
            r["ece"],
        )
        diff = _auc(ba, "GRU - XGB")
        lines += [
            f"**Discrimination is modest:** event-time ROC AUC {_auc(ba, 'XGB')['estimate']:.3f} "
            f"(XGBoost) / {_auc(ba, 'GRU')['estimate']:.3f} (GRU) on all test snapshots, "
            f"{_auc(bo, 'XGB')['estimate']:.3f} / {_auc(bo, 'GRU')['estimate']:.3f} with one snapshot "
            "per possession",
            f"**Ball position dominates:** logistic regression on ball (x, y) alone reaches "
            f"{_auc(ba, 'LR_xy')['estimate']:.3f} / {_auc(bo, 'LR_xy')['estimate']:.3f}",
            "**XGBoost and GRU are statistically indistinguishable:** GRU minus XGBoost AUC "
            f"{diff['estimate']:+.3f}, 95% interval [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]",
            f"**False alerts:** {al['XGB']['false_alert_rate']:.1%} (XGBoost) / "
            f"{al['GRU']['false_alert_rate']:.1%} (GRU) of safe test possessions raise at least one "
            "alert at the validation-chosen threshold",
            f"**Calibration is imperfect:** ECE {ece['XGB']:.3f} / {ece['GRU']:.3f} (snapshot-weighted, "
            "ten equal-width bins, absolute probability)",
            f"**Small test cohort:** {r['matches']} matches (World Cup 2022 and Barcelona La Liga 2020/21)",
        ]
    else:
        lines.append(f"In-cohort test metrics not available ({FROZEN_TEST_RESULTS} missing)")
    hold_path = root / HOLDOUT_RESULTS
    if hold_path.exists():
        rec = json.loads(hold_path.read_text(encoding="utf-8"))
        r = rec["results"]
        ba = r["bootstrap_all"]
        x, g = _auc(ba, "XGB"), _auc(ba, "GRU")
        lines.append(
            f"**One external test only:** on {rec['data']['competition']} ({r['matches']} unseen "
            f"matches, scored once) ROC AUC {x['estimate']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}] "
            f"(XGBoost) / {g['estimate']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}] (GRU), ECE "
            f"{r['ece']['XGB']:.3f} / {r['ece']['GRU']:.3f}; other leagues, seasons and women's "
            "football are untested"
        )
    else:
        lines.append("**No external validation:** results may not transfer to other competitions")
    lines += [
        "**Partial 360 coverage:** freeze frames exist only for the World Cup 2022 matches; not "
        "continuous tracking",
        "**No causal claims:** attribution is retrospective correlation, not causal contribution",
    ]
    return lines
