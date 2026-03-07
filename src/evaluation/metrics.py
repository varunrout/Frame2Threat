"""Predictive evaluation metrics for classification, ranking, and regression tasks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_prob:
        Predicted positive-class probabilities in [0, 1].
    threshold:
        Decision threshold for converting probabilities to class predictions.

    Returns
    -------
    dict
        Keys: roc_auc, pr_auc, log_loss, brier_score, f1, precision,
        recall, accuracy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_total = len(y_true)
    logger.debug(
        "classification_metrics: n=%d, n_pos=%d (%.1f%%)",
        n_total,
        n_pos,
        100 * n_pos / max(n_total, 1),
    )

    if n_pos == 0 or n_pos == n_total:
        logger.warning("classification_metrics: only one class present, some metrics will be NaN")

    results: dict[str, float] = {}

    try:
        results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        results["roc_auc"] = float("nan")

    try:
        results["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        results["pr_auc"] = float("nan")

    try:
        results["log_loss"] = float(log_loss(y_true, y_prob))
    except ValueError:
        results["log_loss"] = float("nan")

    results["brier_score"] = float(brier_score_loss(y_true, y_prob))
    results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    results["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    results["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    results["accuracy"] = float(accuracy_score(y_true, y_pred))

    return results


def calibration_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration quality metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted positive-class probabilities.
    n_bins:
        Number of equal-width probability bins.

    Returns
    -------
    dict
        Keys: ece, mce, fraction_of_positives (list), mean_predicted_value (list).
        ``ece`` is the Expected Calibration Error; ``mce`` is the Maximum
        Calibration Error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # 0-indexed bins

    fraction_of_positives: list[float] = []
    mean_predicted_value: list[float] = []
    bin_counts: list[int] = []

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            fraction_of_positives.append(float(y_true[mask].mean()))
            mean_predicted_value.append(float(y_prob[mask].mean()))
        else:
            # Use bin midpoint for empty bins
            mid = (bin_edges[b] + bin_edges[b + 1]) / 2.0
            fraction_of_positives.append(float("nan"))
            mean_predicted_value.append(float(mid))

    counts_arr = np.array(bin_counts, dtype=float)
    fop_arr = np.array(fraction_of_positives, dtype=float)
    mpv_arr = np.array(mean_predicted_value, dtype=float)

    # Only include non-empty bins in ECE/MCE
    non_empty = counts_arr > 0
    if non_empty.sum() == 0:
        ece = float("nan")
        mce = float("nan")
    else:
        weights = counts_arr[non_empty] / counts_arr[non_empty].sum()
        abs_diff = np.abs(fop_arr[non_empty] - mpv_arr[non_empty])
        ece = float(np.sum(weights * abs_diff))
        mce = float(abs_diff.max())

    return {
        "ece": ece,
        "mce": mce,
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "bin_counts": bin_counts,
    }


def ranking_metrics(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray | pd.Series,
    k: int = 5,
) -> dict[str, float]:
    """Compute ranking quality metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary relevance labels (1 = relevant).
    y_scores:
        Predicted scores; higher means more relevant.
    k:
        Cutoff for NDCG and hit-rate metrics.

    Returns
    -------
    dict
        Keys: ndcg_at_k, mrr, top_k_hit_rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)

    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_relevance = y_true[sorted_indices]

    # NDCG@k
    def _dcg(relevances: np.ndarray, n: int) -> float:
        top = relevances[:n]
        gains = top / np.log2(np.arange(2, len(top) + 2))
        return float(gains.sum())

    ideal_relevances = np.sort(y_true)[::-1]
    dcg_k = _dcg(sorted_relevance, k)
    idcg_k = _dcg(ideal_relevances, k)
    ndcg_at_k = dcg_k / idcg_k if idcg_k > 0 else 0.0

    # Mean Reciprocal Rank
    first_relevant = np.where(sorted_relevance > 0)[0]
    mrr = 1.0 / (first_relevant[0] + 1) if len(first_relevant) > 0 else 0.0

    # Top-k Hit Rate
    top_k_hit_rate = float(sorted_relevance[:k].sum() > 0)

    return {
        "ndcg_at_k": float(ndcg_at_k),
        "mrr": float(mrr),
        "top_k_hit_rate": float(top_k_hit_rate),
    }


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth continuous values.
    y_pred:
        Predicted continuous values.

    Returns
    -------
    dict
        Keys: rmse, mae, r2, spearman_r.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) < 3:
        spearman_r = float("nan")
    else:
        corr, _ = spearmanr(y_true, y_pred)
        spearman_r = float(corr) if not np.isnan(corr) else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman_r": spearman_r}


def summarise_all_tasks(
    results_dict: dict[str, dict[str, np.ndarray | pd.Series]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute a tidy summary DataFrame across multiple tasks.

    Parameters
    ----------
    results_dict:
        Mapping of ``{task_name: {"y_true": ..., "y_prob": ...}}``.
        Each value must contain ``y_true`` and ``y_prob`` arrays.
    threshold:
        Probability threshold for classification metrics.

    Returns
    -------
    pd.DataFrame
        One row per task, columns are metric names.
        Also includes calibration ECE and MCE columns.
    """
    rows: list[dict[str, Any]] = []

    for task_name, data in results_dict.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]

        row: dict[str, Any] = {"task": task_name}
        row.update(classification_metrics(y_true, y_prob, threshold=threshold))

        cal = calibration_metrics(y_true, y_prob)
        row["ece"] = cal["ece"]
        row["mce"] = cal["mce"]

        ranking = ranking_metrics(y_true, y_prob, k=5)
        row.update(ranking)

        rows.append(row)
        logger.info(
            "Task '%s': roc_auc=%.4f, pr_auc=%.4f, brier=%.4f, ece=%.4f",
            task_name,
            row.get("roc_auc", float("nan")),
            row.get("pr_auc", float("nan")),
            row.get("brier_score", float("nan")),
            row.get("ece", float("nan")),
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("task")
    return df
