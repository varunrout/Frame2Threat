"""Calibration analysis and plotting for binary classifiers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available; plotting functions will raise ImportError")


def plot_calibration_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    task_name: str = "",
    ax: "Axes | None" = None,
    n_bins: int = 10,
) -> "Axes":
    """Plot a reliability diagram (calibration curve) for a binary classifier.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted positive-class probabilities.
    task_name:
        Label used in the plot title and legend.
    ax:
        Optional existing matplotlib Axes to draw on.  A new figure is
        created when ``None``.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the reliability diagram.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_calibration_curve")

    data = compute_reliability_diagram_data(y_true, y_prob, n_bins=n_bins)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    mpv = np.array(data["mean_predicted_value"], dtype=float)
    fop = np.array(data["fraction_of_positives"], dtype=float)
    counts = np.array(data["bin_counts"], dtype=float)

    # Only plot non-empty bins
    non_empty = ~np.isnan(fop)
    mpv_plot = mpv[non_empty]
    fop_plot = fop[non_empty]
    counts_plot = counts[non_empty]

    # Bubble size proportional to count
    bubble_sizes = 50 + 200 * counts_plot / max(counts_plot.max(), 1)

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    ax.scatter(
        mpv_plot,
        fop_plot,
        s=bubble_sizes,
        alpha=0.7,
        color="steelblue",
        edgecolors="navy",
        zorder=3,
        label=task_name or "Model",
    )
    ax.plot(mpv_plot, fop_plot, "-o", color="steelblue", alpha=0.5, ms=0)

    ece = data.get("ece", float("nan"))
    mce = data.get("mce", float("nan"))

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    title = f"Calibration curve – {task_name}" if task_name else "Calibration curve"
    ax.set_title(f"{title}\nECE={ece:.4f}, MCE={mce:.4f}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def compute_reliability_diagram_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute bin-level data needed for a reliability diagram.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of equal-width probability bins.

    Returns
    -------
    dict
        Keys:

        * ``bin_edges`` – array of length ``n_bins + 1``.
        * ``bin_counts`` – list[int], samples per bin.
        * ``fraction_of_positives`` – list[float | nan], observed rate per bin.
        * ``mean_predicted_value`` – list[float], mean predicted prob per bin.
        * ``ece`` – float, Expected Calibration Error.
        * ``mce`` – float, Maximum Calibration Error.
    """
    # Import here to avoid circular dependency
    from src.evaluation.metrics import calibration_metrics

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    cal = calibration_metrics(y_true, y_prob, n_bins=n_bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_counts": cal["bin_counts"],
        "fraction_of_positives": cal["fraction_of_positives"],
        "mean_predicted_value": cal["mean_predicted_value"],
        "ece": cal["ece"],
        "mce": cal["mce"],
    }


def calibrate_model(
    model: Any,
    X_cal: np.ndarray | pd.DataFrame,
    y_cal: np.ndarray | pd.Series,
    method: str = "isotonic",
) -> Any:
    """Wrap a fitted estimator with post-hoc probability calibration.

    Uses :class:`sklearn.calibration.CalibratedClassifierCV` in
    ``cv='prefit'`` mode, meaning ``model`` must already be fitted and
    ``X_cal`` / ``y_cal`` are used as the calibration hold-out set.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible estimator that exposes
        ``predict_proba``.
    X_cal:
        Calibration feature matrix (held-out set, not used during training).
    y_cal:
        Calibration labels.
    method:
        Calibration method – ``'isotonic'`` or ``'sigmoid'`` (Platt scaling).

    Returns
    -------
    CalibratedClassifierCV
        A fitted calibrated wrapper around ``model``.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")

    logger.info(
        "Calibrating model with method='%s' on %d samples", method, len(y_cal)
    )
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    X_arr = X_cal.values if isinstance(X_cal, pd.DataFrame) else X_cal
    calibrated.fit(X_arr, np.asarray(y_cal))
    logger.info("Calibration complete")
    return calibrated
