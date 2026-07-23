"""SHAP-based and geometric explanations for model predictions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False
    logger.warning("shap not available; SHAP explanation functions will raise ImportError")

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

# Human-readable descriptions for known feature names
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "n_defenders_in_corridor": "Number of defenders in pass corridor",
    "n_defenders_goal_side": "Defenders between receiver and goal",
    "nearest_defender_dist_passer": "Distance from nearest defender to passer (m)",
    "nearest_defender_dist_receiver": "Distance from nearest defender to receiver (m)",
    "n_teammates_visible": "Visible teammates in freeze frame",
    "n_opponents_visible": "Visible opponents in freeze frame",
    "receiver_between_lines": "Receiver is between defensive lines",
    "pass_corridor_clear": "Pass corridor is unobstructed",
    "overload_target_zone": "Numerical overload at target zone",
    "defensive_compactness": "Opponent defensive compactness score",
    "x_gain": "Horizontal distance gained by pass (m)",
    "goal_dist_gain": "Reduction in distance-to-goal",
    "dist_to_goal_end": "Receiver distance to opponent goal (m)",
    "is_forward": "Pass moves forward",
    "is_through_ball": "Pass is a through ball",
    "is_cross": "Pass is a cross",
    "is_switch": "Pass is a wide switch",
    "under_pressure": "Passer was under pressure",
    "pass_length": "Pass length (m)",
    "pass_angle_rad": "Pass direction angle (radians)",
    "minute": "Minute in game",
    "zone_start": "Pitch zone at pass start (1=own third, 6=box)",
    "team_width": "Attacking team width",
    "opp_width": "Defending team width",
}


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Compute SHAP values for a fitted model.

    Automatically selects the appropriate SHAP explainer:

    * :class:`shap.TreeExplainer` for XGBoost / LightGBM / sklearn tree models.
    * :class:`shap.LinearExplainer` for linear models (logistic regression).
    * :class:`shap.KernelExplainer` as a generic fallback (slow).

    Parameters
    ----------
    model:
        A fitted sklearn-compatible estimator.
    X:
        Feature matrix to explain (can be training data, validation, or
        individual instances).
    feature_names:
        List of column names aligned with ``X``.

    Returns
    -------
    dict
        Keys:

        * ``shap_values`` – np.ndarray of shape (n_samples, n_features).
        * ``expected_value`` – float baseline expected value.
        * ``feature_names`` – list of feature name strings.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is required for compute_shap_values")

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    explainer = _build_explainer(model, X_arr)

    logger.info("Computing SHAP values for %d samples", len(X_arr))
    raw = explainer(X_arr)

    if hasattr(raw, "values"):
        shap_vals = raw.values
        expected_value = float(raw.base_values[0] if raw.base_values.ndim > 0 else raw.base_values)
    else:
        shap_vals = np.asarray(raw)
        exp = explainer.expected_value
        expected_value = float(exp[1] if hasattr(exp, "__len__") and len(exp) > 1 else exp)

    # For binary classifiers that return (n, 2) SHAP values, take class-1 slice
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    elif shap_vals.ndim == 2 and shap_vals.shape[1] == 2 * len(feature_names):
        # Some explainers concatenate both class SHAP values
        shap_vals = shap_vals[:, len(feature_names) :]

    logger.info("SHAP computation complete, shape=%s", shap_vals.shape)
    return {
        "shap_values": shap_vals,
        "expected_value": expected_value,
        "feature_names": feature_names,
    }


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str = "",
    ax: Any | None = None,
    max_features: int = 20,
) -> Any:
    """Plot a bar chart of mean absolute SHAP values.

    Parameters
    ----------
    shap_values:
        SHAP values array of shape (n_samples, n_features).
    feature_names:
        Feature name strings (length must match ``shap_values.shape[1]``).
    title:
        Plot title.
    ax:
        Optional existing matplotlib Axes.
    max_features:
        Maximum number of features to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_shap_summary")

    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_features]
    ordered_names = [feature_names[i] for i in order]
    ordered_values = mean_abs[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(ordered_names) * 0.35)))
    else:
        fig = ax.get_figure()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ordered_names)))  # type: ignore[attr-defined]
    bars = ax.barh(
        range(len(ordered_names)),
        ordered_values[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title or "Feature Importance (SHAP)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def explain_single_event(
    event_uuid: str,
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    shap_values: np.ndarray | None = None,
    task_name: str = "dangerous_progression",
) -> dict[str, Any]:
    """Generate a structured explanation for a single pass event.

    Computes or reuses pre-computed SHAP values for the event row and
    combines them with geometric context to produce an analyst-friendly
    explanation dictionary.

    Parameters
    ----------
    event_uuid:
        UUID of the pass to explain.
    pass_instances_df:
        Pass events DataFrame; must contain ``event_uuid`` (or ``id``) and
        all columns listed in ``feature_names``.
    frames_df:
        360 freeze-frame DataFrame.
    model:
        Fitted classifier with ``predict_proba`` method.
    feature_names:
        Feature columns to include in explanation.
    shap_values:
        Optional pre-computed SHAP matrix (n_samples × n_features) aligned
        with ``pass_instances_df``.  When ``None``, SHAP values are computed
        on-the-fly for the single event.
    task_name:
        Task name used in the narrative text.

    Returns
    -------
    dict
        Keys: event_uuid, predicted_score, top_reasons, narrative,
        geometry_context.
    """
    uuid_col = "event_uuid" if "event_uuid" in pass_instances_df.columns else "id"
    event_rows = pass_instances_df[pass_instances_df[uuid_col] == event_uuid]
    if event_rows.empty:
        raise ValueError(f"event_uuid '{event_uuid}' not found in pass_instances_df")

    event_idx = event_rows.index[0]
    row_pos = pass_instances_df.index.get_loc(event_idx)
    event_row = event_rows.iloc[0]

    # Extract feature vector for this event
    valid_features = [f for f in feature_names if f in pass_instances_df.columns]
    X_event = event_row[valid_features].fillna(0.0).values.reshape(1, -1)

    # Predicted score
    try:
        prob = model.predict_proba(X_event)[0, 1]
    except Exception as exc:
        logger.error("predict_proba failed for event %s: %s", event_uuid, exc)
        prob = float("nan")
    predicted_score = float(prob)

    # SHAP contributions
    if shap_values is not None:
        shap_event = shap_values[row_pos]
    else:
        if not _SHAP_AVAILABLE:
            logger.warning("shap not available; SHAP contributions will be zeros")
            shap_event = np.zeros(len(valid_features))
        else:
            sv_dict = compute_shap_values(model, X_event, valid_features)
            shap_event = sv_dict["shap_values"][0]

    # Build top reasons (top 5 by absolute SHAP value)
    abs_contribs = np.abs(shap_event[: len(valid_features)])
    top_indices = np.argsort(abs_contribs)[::-1][:5]
    top_reasons = []
    for i in top_indices:
        feat = valid_features[i]
        val = float(X_event[0, i])
        contrib = float(shap_event[i]) if i < len(shap_event) else 0.0
        desc = _FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title())
        top_reasons.append(
            {
                "feature": feat,
                "value": val,
                "contribution": contrib,
                "description": desc,
            }
        )

    # Geometry context from freeze frame
    geometry_context = _extract_geometry_context(event_uuid, frames_df, event_row)

    # Generate narrative
    narrative = generate_explanation_narrative(top_reasons, predicted_score, task_name)

    return {
        "event_uuid": event_uuid,
        "predicted_score": predicted_score,
        "top_reasons": top_reasons,
        "narrative": narrative,
        "geometry_context": geometry_context,
    }


def generate_explanation_narrative(
    top_reasons: list[dict[str, Any]],
    predicted_score: float,
    task_name: str = "dangerous_progression",
) -> str:
    """Generate an analyst-friendly text explanation for a pass score.

    Parameters
    ----------
    top_reasons:
        List of reason dicts with keys: feature, value, contribution, description.
    predicted_score:
        Predicted probability / score in [0, 1].
    task_name:
        Descriptive task name used in the narrative header.

    Returns
    -------
    str
        Multi-sentence narrative string explaining the model's prediction.

    Examples
    --------
    "This pass scores 78% dangerous progression probability because:
    (1) The receiver is between defensive lines (contribution: +0.23).
    (2) Only 1 defender in the pass corridor (contribution: +0.18).
    ..."
    """
    pct = f"{predicted_score * 100:.1f}%"
    task_label = task_name.replace("_", " ")

    # Positive vs negative contributors
    positive = [r for r in top_reasons if r["contribution"] > 0]
    negative = [r for r in top_reasons if r["contribution"] < 0]

    parts = [f"This pass scores {pct} {task_label} probability because:"]

    for idx, reason in enumerate(top_reasons[:5], start=1):
        direction = "↑ increases" if reason["contribution"] > 0 else "↓ decreases"
        val_str = _format_feature_value(reason["feature"], reason["value"])
        parts.append(
            f"  ({idx}) {reason['description']}: {val_str} "
            f"[{direction} score by {abs(reason['contribution']):.3f}]"
        )

    if positive:
        p_descs = [r["description"] for r in positive[:2]]
        parts.append(f"\nKey factors boosting the score: {', '.join(p_descs)}.")

    if negative:
        n_descs = [r["description"] for r in negative[:2]]
        parts.append(f"Key factors reducing the score: {', '.join(n_descs)}.")

    # Qualitative rating
    if predicted_score >= 0.75:
        parts.append(
            "\nOverall: Highly dangerous progression pass with multiple favourable "
            "geometric conditions."
        )
    elif predicted_score >= 0.5:
        parts.append("\nOverall: Moderately dangerous progression pass.")
    elif predicted_score >= 0.25:
        parts.append("\nOverall: Below-average progression value with limiting factors.")
    else:
        parts.append("\nOverall: Low danger – limited spatial advantage for the attacking team.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_explainer(model: Any, X_background: np.ndarray) -> Any:
    """Select and build the most appropriate SHAP explainer."""
    model_class = type(model).__name__.lower()

    if any(name in model_class for name in ("xgb", "lgbm", "lgb", "forest", "tree", "gradient")):
        try:
            explainer = shap.TreeExplainer(model)
            logger.debug("Using TreeExplainer for %s", model_class)
            return explainer
        except Exception as exc:
            logger.warning("TreeExplainer failed (%s), trying LinearExplainer", exc)

    if any(name in model_class for name in ("logistic", "linear", "ridge", "lasso")):
        try:
            background = shap.maskers.Independent(
                X_background, max_samples=min(100, len(X_background))
            )
            explainer = shap.LinearExplainer(model, background)
            logger.debug("Using LinearExplainer for %s", model_class)
            return explainer
        except Exception as exc:
            logger.warning("LinearExplainer failed (%s), falling back to KernelExplainer", exc)

    # Generic fallback – slow but universal
    background_sample = shap.sample(X_background, min(50, len(X_background)))
    logger.debug("Using KernelExplainer (slow) for %s", model_class)
    return shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background_sample)


def _extract_geometry_context(
    event_uuid: str,
    frames_df: pd.DataFrame,
    event_row: pd.Series,
) -> dict[str, Any]:
    """Build a geometry context dict from the freeze frame and event row."""
    uuid_col = "event_uuid" if "event_uuid" in frames_df.columns else "id"
    frame_rows = frames_df[frames_df[uuid_col] == event_uuid]

    n_teammates = (
        int((frame_rows["teammate"] == True).sum()) if not frame_rows.empty else 0
    )  # noqa: E712
    n_opponents = (
        int((frame_rows["teammate"] == False).sum()) if not frame_rows.empty else 0
    )  # noqa: E712

    context: dict[str, Any] = {
        "n_players_visible": len(frame_rows),
        "n_teammates_visible": n_teammates,
        "n_opponents_visible": n_opponents,
        "passer_x": float(event_row.get("start_x", float("nan"))),
        "passer_y": float(event_row.get("start_y", float("nan"))),
        "receiver_x": float(event_row.get("end_x", float("nan"))),
        "receiver_y": float(event_row.get("end_y", float("nan"))),
        "pass_length": float(event_row.get("pass_length", float("nan"))),
    }

    # Add pre-computed geometry features if present
    for col in [
        "n_defenders_in_corridor",
        "receiver_between_lines",
        "pass_corridor_clear",
        "overload_target_zone",
        "defensive_compactness",
    ]:
        if col in event_row.index:
            context[col] = float(event_row[col]) if not pd.isna(event_row[col]) else None

    return context


def _format_feature_value(feature_name: str, value: float) -> str:
    """Return a human-readable representation of a feature value."""
    bool_features = {
        "is_forward",
        "is_cross",
        "is_through_ball",
        "is_switch",
        "under_pressure",
        "receiver_between_lines",
        "pass_corridor_clear",
    }
    if feature_name in bool_features:
        return "Yes" if value >= 0.5 else "No"

    count_features = {
        "n_defenders_in_corridor",
        "n_defenders_goal_side",
        "n_teammates_visible",
        "n_opponents_visible",
    }
    if feature_name in count_features:
        return str(int(round(value)))

    return f"{value:.2f}"
