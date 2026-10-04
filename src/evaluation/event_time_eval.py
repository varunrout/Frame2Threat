"""
event_time_eval.py
==================
Evaluation utilities for the event-time (in-play) danger models: pooled metrics,
reliability table, bucketed metrics, one-snapshot-per-possession sampling, and
match-level bootstrap confidence intervals (including paired differences).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EVENT_BUCKETS = [(1, 3), (4, 6), (7, 10), (11, 10**9)]
SECOND_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 10**9)]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC AUC, average precision, Brier score and log loss (NaN AUC/AP if one class)."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    one_class = len(np.unique(y)) < 2
    return {
        "roc_auc": np.nan if one_class else float(roc_auc_score(y, p)),
        "avg_precision": np.nan if one_class else float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: count, mean predicted, observed positive rate."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append(
            {
                "bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}",
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                "observed": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bucket_metrics(
    rows: pd.DataFrame,
    preds: Mapping[str, np.ndarray],
    column: str,
    buckets: Sequence[tuple[float, float]],
) -> pd.DataFrame:
    """Metrics per model within each ``[lo, hi]`` (events) / ``[lo, hi)`` (seconds) bucket."""
    out = []
    for lo, hi in buckets:
        m = (
            (rows[column] >= lo) & (rows[column] <= hi)
            if column == "events_so_far"
            else (rows[column] >= lo) & (rows[column] < hi)
        ).to_numpy()
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        for name, p in preds.items():
            if m.sum() == 0:
                continue
            out.append(
                {
                    "bucket": label,
                    "model": name,
                    "n": int(m.sum()),
                    "prevalence": float(rows["y"].to_numpy()[m].mean()),
                    **metrics(rows["y"].to_numpy()[m], np.asarray(p)[m]),
                }
            )
    return pd.DataFrame(out)


def one_snapshot_per_possession(rows: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Positional indices of one uniformly sampled eligible snapshot per possession."""
    rng = np.random.default_rng(seed)
    keys = rows["match_id"].astype(np.int64) * 10_000_000 + rows["possession_id"].astype(np.int64)
    order = rng.permutation(len(rows))
    shuffled = keys.to_numpy()[order]
    _, first = np.unique(shuffled, return_index=True)
    return np.sort(order[first])


def bootstrap_by_match(
    rows: pd.DataFrame,
    preds: Mapping[str, np.ndarray],
    *,
    deltas: Sequence[tuple[str, str]] = (),
    stats: Sequence[str] = ("roc_auc", "avg_precision", "brier"),
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """95% percentile intervals, resampling whole matches with replacement.

    ``deltas`` are ``(model_a, model_b)`` pairs reported as a - b on the same resamples.
    """
    y = rows["y"].to_numpy()
    mids = rows["match_id"].to_numpy()
    uniq = np.unique(mids)
    idx_by_match = {m: np.nonzero(mids == m)[0] for m in uniq}
    rng = np.random.default_rng(seed)
    point = {n: metrics(y, p) for n, p in preds.items()}
    draws: dict[str, list[float]] = {}

    def _add(key: str, v: float) -> None:
        draws.setdefault(key, []).append(v)

    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_match[m] for m in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        res = {n: metrics(y[ix], np.asarray(p)[ix]) for n, p in preds.items()}
        for n in preds:
            for s in stats:
                _add(f"{n}|{s}", res[n][s])
        for a, b in deltas:
            for s in stats:
                _add(f"{a} - {b}|{s}", res[a][s] - res[b][s])

    out = []
    for key, vals in draws.items():
        name, stat = key.split("|")
        if " - " in name:
            a, b = name.split(" - ")
            est = point[a][stat] - point[b][stat]
        else:
            est = point[name][stat]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.append({"model": name, "stat": stat, "estimate": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def fmt_ci(df: pd.DataFrame, model: str, stat: str, nd: int = 3) -> str:
    r = df[(df.model == model) & (df.stat == stat)].iloc[0]
    return f"{r.estimate:.{nd}f} [{r.ci_low:.{nd}f}, {r.ci_high:.{nd}f}]"


def threshold_matching_prevalence(y: np.ndarray, p: Sequence[float]) -> float:
    """Score threshold whose predicted-positive rate equals the observed prevalence.

    Chosen on **validation** only: with the positive rate fixed at the base rate, precision
    and recall coincide, so the alert level is neither trigger-happy nor conservative.
    """
    prev = float(np.mean(y))
    return float(np.quantile(np.asarray(p), 1.0 - prev))
