"""
early_evaluation.py
===================
Evaluation utilities for v3 early-prediction experiments.

Key functions
-------------
prefix_gru_auc_curve(poss_df, gru_model, fracs)
    → AUC / AP at each prefix fraction (EXP-016)

tipping_point_analysis(poss_df, gru_model, threshold)
    → Where in the possession does the GRU "decide"? (EXP-019)
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

if TYPE_CHECKING:
    from src.models.gru_possession import PossessionGRU


# ---------------------------------------------------------------------------
# 1. Prefix GRU evaluation  (EXP-016)
# ---------------------------------------------------------------------------


def _get_prefix_proba(
    events: list[dict],
    model: "PossessionGRU",
    frac: float,
) -> float:
    """
    Danger probability from the GRU after seeing only the first *frac* of events.
    """
    from src.evaluation.possession_attribution import _events_to_tensor, _score_sequence

    n = len(events)
    t = max(1, int(np.ceil(n * frac)))
    prefix = events[:t]
    x, length = _events_to_tensor(prefix)
    return _score_sequence(model, x, length)


@torch.no_grad()
def prefix_gru_auc_curve(
    poss_df: pd.DataFrame,
    gru_model: "PossessionGRU",
    label_col: str = "poss_dangerous",
    fracs: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
    min_events: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute ROC-AUC and PR-AUC at each prefix fraction.

    Only possessions with ≥ *min_events* events are included (so a 25 %
    prefix has at least 1 event).

    Parameters
    ----------
    poss_df    : test-split possession DataFrame
    gru_model  : loaded PossessionGRU (eval mode)
    label_col  : binary label column
    fracs      : prefix fractions to evaluate
    min_events : minimum sequence length filter
    verbose    : print progress

    Returns
    -------
    pd.DataFrame with columns: frac, n_poss, roc_auc, pr_auc
    """
    from src.evaluation.possession_attribution import cumulative_danger_scores

    gru_model.eval()

    # Pre-parse all event sequences
    rows_with_events: list[tuple[Any, list[dict]]] = []
    y_all: list[int] = []
    for idx, row in poss_df.iterrows():
        raw = row.get("event_sequence", "[]")
        events = json.loads(raw) if isinstance(raw, str) else list(raw)
        if len(events) < min_events:
            continue
        rows_with_events.append((idx, events))
        y_all.append(int(row[label_col]))

    y_arr = np.array(y_all, dtype=int)
    n_total = len(rows_with_events)
    if verbose:
        print(f"  Evaluating {n_total:,} possessions (≥{min_events} events) …")

    results: list[dict] = []

    for frac in fracs:
        probas: list[float] = []
        for i, (idx, events) in enumerate(rows_with_events):
            if verbose and i % 500 == 0 and i > 0:
                print(f"    frac={frac:.0%}  {i}/{n_total}", end="\r")

            p = _get_prefix_proba(events, gru_model, frac)
            probas.append(p)

        y_pred = np.array(probas)
        roc = roc_auc_score(y_arr, y_pred)
        pr = average_precision_score(y_arr, y_pred)

        results.append(
            {
                "frac": frac,
                "pct_label": f"{frac:.0%}",
                "n_poss": n_total,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
            }
        )
        if verbose:
            print(f"    frac={frac:.0%}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Tipping-point analysis  (EXP-019)
# ---------------------------------------------------------------------------


def tipping_point_analysis(
    poss_df: pd.DataFrame,
    gru_model: "PossessionGRU",
    label_col: str = "poss_dangerous",
    threshold: float = 0.50,
    min_events: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each possession, find the *first event* where P(dangerous) ≥ threshold.

    This reveals how early the GRU recognises danger and which event types
    tend to be the "tipping point".

    Returns a DataFrame with one row per possession:
        possession_id, match_id, team_name,
        n_events, poss_dangerous,
        tipping_step, tipping_frac, tipping_event_type,
        final_score, max_delta_step, max_delta_type, max_delta_value
    """
    from src.evaluation.possession_attribution import (
        cumulative_danger_scores,
        _parse_event_seq,
        EVENT_TYPE_LABELS,
    )

    gru_model.eval()
    records: list[dict] = []

    for i, (_, row) in enumerate(poss_df.iterrows()):
        if verbose and i % 500 == 0 and i > 0:
            print(f"  Tipping-point analysis: {i}/{len(poss_df)}", end="\r")

        events = _parse_event_seq(row)
        if len(events) < min_events:
            continue

        cum = cumulative_danger_scores(events, gru_model)
        T = len(cum)

        # Find first crossing
        tipping_step = -1
        tipping_type = "none"
        for t in range(T):
            if cum[t] >= threshold:
                tipping_step = t
                tipping_type = EVENT_TYPE_LABELS.get(int(events[t].get("type_id", 9)), "Other")
                break

        # Largest single-step delta
        if T >= 2:
            deltas = np.diff(cum)
            max_delta_idx = int(np.argmax(deltas))
            max_delta_val = float(deltas[max_delta_idx])
            max_delta_type = EVENT_TYPE_LABELS.get(
                int(events[max_delta_idx + 1].get("type_id", 9)), "Other"
            )
        else:
            max_delta_idx = 0
            max_delta_val = 0.0
            max_delta_type = "N/A"

        records.append(
            {
                "possession_id": str(row.get("possession_id", "")),
                "match_id": row.get("match_id"),
                "team_name": str(row.get("team_name", "")),
                "n_events": T,
                "poss_dangerous": int(row.get(label_col, 0)),
                "tipping_step": tipping_step,
                "tipping_frac": round(tipping_step / T, 3) if tipping_step >= 0 else np.nan,
                "tipping_event_type": tipping_type,
                "final_score": float(cum[-1]),
                "max_delta_step": max_delta_idx + 1,
                "max_delta_type": max_delta_type,
                "max_delta_value": round(max_delta_val, 4),
            }
        )

    if verbose:
        print()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Summary helpers for notebook plots
# ---------------------------------------------------------------------------


def tipping_summary_by_origin(tip_df: pd.DataFrame, poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tipping-point data with origin_type and summarise.

    Returns a DataFrame grouped by origin_type with:
        n, pct_tipped, median_tipping_frac, mean_final_score
    """
    left = tip_df.copy()
    left["match_id"] = left["match_id"].astype(str)
    left["possession_id"] = left["possession_id"].astype(str)
    right = poss_df[["match_id", "possession_id", "origin_type"]].drop_duplicates().copy()
    right["match_id"] = right["match_id"].astype(str)
    right["possession_id"] = right["possession_id"].astype(str)

    merged = left.merge(
        right,
        on=["match_id", "possession_id"],
        how="left",
    )
    # Only possessions that crossed the threshold
    tipped = merged[merged["tipping_step"] >= 0]
    all_groups = merged.groupby("origin_type").agg(n=("possession_id", "count"))
    tip_groups = tipped.groupby("origin_type").agg(
        n_tipped=("possession_id", "count"),
        median_tipping_frac=("tipping_frac", "median"),
        mean_final_score=("final_score", "mean"),
    )
    summary = all_groups.join(tip_groups, how="left").fillna(0)
    summary["pct_tipped"] = (summary["n_tipped"] / summary["n"] * 100).round(1)
    return summary.sort_values("pct_tipped", ascending=False)


def tipping_event_type_distribution(tip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of event types at the tipping point (only possessions
    that crossed the threshold).
    """
    tipped = tip_df[tip_df["tipping_step"] >= 0].copy()
    if tipped.empty:
        return pd.DataFrame(columns=["event_type", "count", "pct"])
    counts = tipped["tipping_event_type"].value_counts().reset_index()
    counts.columns = ["event_type", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def max_delta_event_type_distribution(tip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of event types at the max single-step danger increase.
    """
    counts = tip_df["max_delta_type"].value_counts().reset_index()
    counts.columns = ["event_type", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts
