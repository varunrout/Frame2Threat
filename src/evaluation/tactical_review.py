"""Tactical quality review – high/low score examples and segment breakdowns."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pitch zone names (matches zone IDs in event_features.py)
_ZONE_LABELS: dict[int, str] = {
    1: "Own Third",
    2: "Mid Third – Wide",
    3: "Mid Third – Central",
    4: "Final Third – Wide",
    5: "Final Third – Central",
    6: "Box Area",
}


def get_top_scoring_passes(
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    n: int = 20,
) -> pd.DataFrame:
    """Return the top-n passes by predicted score.

    Parameters
    ----------
    pass_instances_df:
        One row per pass event; must contain ``event_uuid``.
    scores:
        Predicted danger/progression scores aligned with ``pass_instances_df``.
    n:
        Number of top passes to return.

    Returns
    -------
    pd.DataFrame
        Subset of ``pass_instances_df`` for the top-n passes, with an
        additional ``predicted_score`` column, sorted descending.
    """
    df = pass_instances_df.copy()
    df["predicted_score"] = np.asarray(scores, dtype=float)
    top = df.nlargest(n, "predicted_score").reset_index(drop=True)
    logger.debug("get_top_scoring_passes: returning %d rows (n=%d)", len(top), n)
    return top


def get_false_positives(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold: float = 0.5,
    n: int = 20,
) -> pd.DataFrame:
    """Return false-positive passes (predicted dangerous, actually safe).

    Parameters
    ----------
    pass_instances_df:
        One row per pass event.
    y_true:
        Ground-truth binary labels aligned with ``pass_instances_df``.
    y_pred:
        Predicted probabilities aligned with ``pass_instances_df``.
    threshold:
        Probability cutoff for positive prediction.
    n:
        Maximum number of rows to return.

    Returns
    -------
    pd.DataFrame
        False-positive rows with ``y_true`` and ``predicted_score`` columns,
        sorted by predicted score descending.
    """
    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["predicted_score"] = np.asarray(y_pred, dtype=float)
    df["y_pred_bin"] = (df["predicted_score"] >= threshold).astype(int)

    fp = df[(df["y_pred_bin"] == 1) & (df["y_true"] == 0)]
    fp = fp.nlargest(n, "predicted_score").reset_index(drop=True)
    logger.debug("get_false_positives: %d FPs (n=%d)", len(fp), n)
    return fp


def get_false_negatives(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold: float = 0.5,
    n: int = 20,
) -> pd.DataFrame:
    """Return false-negative passes (actually dangerous, predicted safe).

    Parameters
    ----------
    pass_instances_df:
        One row per pass event.
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted probabilities.
    threshold:
        Probability cutoff.
    n:
        Maximum number of rows to return.

    Returns
    -------
    pd.DataFrame
        False-negative rows with ``y_true`` and ``predicted_score`` columns,
        sorted by predicted score ascending (worst misses first).
    """
    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["predicted_score"] = np.asarray(y_pred, dtype=float)
    df["y_pred_bin"] = (df["predicted_score"] >= threshold).astype(int)

    fn = df[(df["y_pred_bin"] == 0) & (df["y_true"] == 1)]
    fn = fn.nsmallest(n, "predicted_score").reset_index(drop=True)
    logger.debug("get_false_negatives: %d FNs (n=%d)", len(fn), n)
    return fn


def breakdown_by_zone(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute per-zone classification metrics.

    Uses the ``zone_start`` column (int 1-6) if present; otherwise derives
    zones from ``start_x`` and ``start_y``.

    Parameters
    ----------
    pass_instances_df:
        Pass events DataFrame.
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities.

    Returns
    -------
    pd.DataFrame
        One row per zone with columns: zone, zone_label, n_passes,
        prevalence, roc_auc, pr_auc, brier_score, mean_predicted_score.
    """
    from src.evaluation.metrics import classification_metrics

    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_prob"] = np.asarray(y_prob, dtype=float)

    if "zone_start" not in df.columns:
        df["zone_start"] = _derive_zone(df)

    rows: list[dict[str, Any]] = []
    for zone_id, group in df.groupby("zone_start"):
        n = len(group)
        yt = group["y_true"].values
        yp = group["y_prob"].values

        row: dict[str, Any] = {
            "zone": int(zone_id),
            "zone_label": _ZONE_LABELS.get(int(zone_id), f"Zone {zone_id}"),
            "n_passes": n,
            "prevalence": float(yt.mean()) if n > 0 else float("nan"),
            "mean_predicted_score": float(yp.mean()) if n > 0 else float("nan"),
        }
        if n >= 5 and yt.sum() > 0:
            m = classification_metrics(yt, yp)
            row.update(
                {
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "brier_score": m["brier_score"],
                }
            )
        else:
            row.update(
                {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier_score": float("nan")}
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def breakdown_by_score_state(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute per-score-state (winning / drawing / losing) metrics.

    Uses ``score_state`` column when present (expected values: ``'winning'``,
    ``'drawing'``, ``'losing'``).  If absent, the entire dataset is treated as
    a single ``'unknown'`` group.

    Returns
    -------
    pd.DataFrame
        One row per score state.
    """
    from src.evaluation.metrics import classification_metrics

    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_prob"] = np.asarray(y_prob, dtype=float)

    state_col = "score_state" if "score_state" in df.columns else None
    if state_col is None:
        df["_score_state"] = "unknown"
        state_col = "_score_state"

    rows = _group_metrics(df, state_col, classification_metrics)
    return pd.DataFrame(rows).reset_index(drop=True)


def breakdown_by_minute(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    bins: list[int] | None = None,
) -> pd.DataFrame:
    """Compute metrics broken down by game-minute interval.

    Parameters
    ----------
    pass_instances_df:
        Must contain a ``minute`` column.
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities.
    bins:
        Left edges of minute intervals; defaults to [0, 15, 30, 45, 60, 75, 90].

    Returns
    -------
    pd.DataFrame
        One row per minute interval.
    """
    from src.evaluation.metrics import classification_metrics

    if bins is None:
        bins = [0, 15, 30, 45, 60, 75, 90]

    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_prob"] = np.asarray(y_prob, dtype=float)

    minute_col = "minute" if "minute" in df.columns else None
    if minute_col is None:
        logger.warning("breakdown_by_minute: 'minute' column not found, returning empty")
        return pd.DataFrame()

    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    df["_minute_bin"] = pd.cut(
        df["minute"],
        bins=bins + [999],
        labels=labels,
        right=False,
        include_lowest=True,
    )

    rows = _group_metrics(df, "_minute_bin", classification_metrics)
    result = pd.DataFrame(rows).rename(columns={"_minute_bin": "minute_interval"})
    return result.reset_index(drop=True)


def breakdown_by_competition(
    pass_instances_df: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute metrics broken down by competition.

    Uses ``competition_id`` or ``competition_name`` column when present.

    Returns
    -------
    pd.DataFrame
        One row per competition.
    """
    from src.evaluation.metrics import classification_metrics

    df = pass_instances_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_prob"] = np.asarray(y_prob, dtype=float)

    comp_col = (
        "competition_name"
        if "competition_name" in df.columns
        else "competition_id" if "competition_id" in df.columns else None
    )
    if comp_col is None:
        logger.warning("breakdown_by_competition: no competition column found")
        return pd.DataFrame()

    rows = _group_metrics(df, comp_col, classification_metrics)
    return pd.DataFrame(rows).reset_index(drop=True)


def player_progression_profile(
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    min_passes: int = 20,
) -> pd.DataFrame:
    """Compute player-level progression statistics.

    Parameters
    ----------
    pass_instances_df:
        Must contain ``player_id`` and/or ``player_name`` columns.
    scores:
        Predicted danger/progression scores aligned with ``pass_instances_df``.
    min_passes:
        Minimum passes for a player to appear in the output.

    Returns
    -------
    pd.DataFrame
        One row per player with columns: player_id, player_name (when
        available), n_passes, mean_score, median_score, top10_pct_score,
        score_std, progressive_pass_rate (score ≥ 0.5).
    """
    df = pass_instances_df.copy()
    df["_score"] = np.asarray(scores, dtype=float)

    id_col = "player_id" if "player_id" in df.columns else None
    name_col = "player_name" if "player_name" in df.columns else None
    group_col = id_col or name_col

    if group_col is None:
        logger.warning("player_progression_profile: no player column found")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for player_key, group in df.groupby(group_col):
        if len(group) < min_passes:
            continue
        s = group["_score"].values
        row: dict[str, Any] = {
            "player_id": group[id_col].iloc[0] if id_col else None,
            "player_name": group[name_col].iloc[0] if name_col else None,
            "n_passes": len(group),
            "mean_score": float(np.mean(s)),
            "median_score": float(np.median(s)),
            "top10_pct_score": float(np.percentile(s, 90)),
            "score_std": float(np.std(s)),
            "progressive_pass_rate": float((s >= 0.5).mean()),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_score", ascending=False).reset_index(drop=True)
    return result


def team_progression_profile(
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    min_passes: int = 50,
) -> pd.DataFrame:
    """Compute team-level progression statistics.

    Parameters
    ----------
    pass_instances_df:
        Must contain ``team_id`` and/or ``team_name`` columns.
    scores:
        Predicted danger/progression scores.
    min_passes:
        Minimum passes for a team to appear in the output.

    Returns
    -------
    pd.DataFrame
        One row per team with columns analogous to
        :func:`player_progression_profile` plus ``passes_per_game`` when
        ``match_id`` is available.
    """
    df = pass_instances_df.copy()
    df["_score"] = np.asarray(scores, dtype=float)

    id_col = "team_id" if "team_id" in df.columns else None
    name_col = "team_name" if "team_name" in df.columns else None
    group_col = id_col or name_col

    if group_col is None:
        logger.warning("team_progression_profile: no team column found")
        return pd.DataFrame()

    has_match = "match_id" in df.columns

    rows: list[dict[str, Any]] = []
    for team_key, group in df.groupby(group_col):
        if len(group) < min_passes:
            continue
        s = group["_score"].values
        row: dict[str, Any] = {
            "team_id": group[id_col].iloc[0] if id_col else None,
            "team_name": group[name_col].iloc[0] if name_col else None,
            "n_passes": len(group),
            "mean_score": float(np.mean(s)),
            "median_score": float(np.median(s)),
            "top10_pct_score": float(np.percentile(s, 90)),
            "score_std": float(np.std(s)),
            "progressive_pass_rate": float((s >= 0.5).mean()),
        }
        if has_match:
            n_games = group["match_id"].nunique()
            row["n_games"] = n_games
            row["passes_per_game"] = len(group) / max(n_games, 1)
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_score", ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_zone(df: pd.DataFrame) -> pd.Series:
    """Derive zone_start (1–6) from start_x / start_y coordinates."""
    sx = df.get("start_x", pd.Series(60.0, index=df.index))
    sy = df.get("start_y", pd.Series(40.0, index=df.index))

    zone = pd.Series(1, index=df.index, dtype=int)
    zone[(sx >= 40) & (sx < 80) & ~((sy >= 26) & (sy <= 54))] = 2
    zone[(sx >= 40) & (sx < 80) & (sy >= 26) & (sy <= 54)] = 3
    zone[(sx >= 80) & (sx < 102) & ~((sy >= 26) & (sy <= 54))] = 4
    zone[(sx >= 80) & (sx < 102) & (sy >= 26) & (sy <= 54)] = 5
    zone[sx >= 102] = 6
    return zone


def _group_metrics(
    df: pd.DataFrame,
    group_col: str,
    metrics_fn: Any,
) -> list[dict[str, Any]]:
    """Helper: compute metrics for each group defined by *group_col*."""
    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(group_col, observed=True):
        n = len(group)
        yt = group["y_true"].values
        yp = group["y_prob"].values
        row: dict[str, Any] = {
            group_col: group_key,
            "n_passes": n,
            "prevalence": float(yt.mean()) if n > 0 else float("nan"),
            "mean_predicted_score": float(yp.mean()) if n > 0 else float("nan"),
        }
        if n >= 5 and yt.sum() > 0:
            m = metrics_fn(yt, yp)
            row.update(
                {
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "brier_score": m["brier_score"],
                }
            )
        else:
            row.update(
                {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier_score": float("nan")}
            )
        rows.append(row)
    return rows
