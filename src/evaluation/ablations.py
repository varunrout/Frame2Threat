"""Ablation study runner comparing model configurations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Feature group keys expected in *feature_groups* dict
_FEATURE_SET_NAMES = ("event_only", "event+360", "graph_only", "hybrid")


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: dict[str, np.ndarray | pd.Series],
    X_val: pd.DataFrame,
    y_val: dict[str, np.ndarray | pd.Series],
    feature_groups: dict[str, list[str]],
    tasks: list[str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train models under different feature-set ablations and compare results.

    Each ablation trains a separate model using only the columns specified in
    the corresponding ``feature_groups`` entry.  The following canonical sets
    are evaluated when present in ``feature_groups``:

    * ``event_only``  – event-context features (no 360 freeze-frame data)
    * ``event+360``   – event features ∪ geometry/360 features
    * ``graph_only``  – GNN-derived graph features only
    * ``hybrid``      – all available features

    Parameters
    ----------
    X_train:
        Full training feature matrix (all columns).
    y_train:
        Mapping of ``{task_name: label_array}`` for training.
    X_val:
        Full validation feature matrix.
    y_val:
        Mapping of ``{task_name: label_array}`` for validation.
    feature_groups:
        Mapping of ``{ablation_name: [col_name, ...]}``.  Only columns that
        actually exist in ``X_train`` are used (extras are silently dropped).
    tasks:
        Task names to evaluate (must be keys in ``y_train``).
    config:
        Optional model hyper-parameter dict forwarded to the classifier.

    Returns
    -------
    pd.DataFrame
        Columns: model_name, task, roc_auc, pr_auc, brier_score.
        One row per (ablation, task) combination.
    """
    from src.evaluation.metrics import classification_metrics

    config = config or {}
    rows: list[dict[str, Any]] = []

    for ablation_name, feature_cols in feature_groups.items():
        # Restrict to columns that exist in X_train
        valid_cols = [c for c in feature_cols if c in X_train.columns]
        if not valid_cols:
            logger.warning(
                "Ablation '%s': no valid columns found, skipping", ablation_name
            )
            continue

        logger.info(
            "Ablation '%s': training with %d features", ablation_name, len(valid_cols)
        )
        X_tr = X_train[valid_cols].fillna(0.0)
        X_vl = X_val[valid_cols].fillna(0.0)

        for task in tasks:
            if task not in y_train:
                logger.warning("Task '%s' not in y_train, skipping", task)
                continue

            y_tr = np.asarray(y_train[task], dtype=float)
            y_vl_task = np.asarray(y_val.get(task, np.zeros(len(X_vl))), dtype=float)

            model = _build_ablation_model(config)
            try:
                model.fit(X_tr.values, y_tr)
                y_prob = model.predict_proba(X_vl.values)[:, 1]
            except Exception as exc:
                logger.error(
                    "Ablation '%s', task '%s' failed: %s", ablation_name, task, exc
                )
                rows.append(
                    {
                        "model_name": ablation_name,
                        "task": task,
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "brier_score": float("nan"),
                    }
                )
                continue

            metrics = classification_metrics(y_vl_task, y_prob)
            rows.append(
                {
                    "model_name": ablation_name,
                    "task": task,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "brier_score": metrics["brier_score"],
                }
            )
            logger.info(
                "  [%s / %s] roc_auc=%.4f, pr_auc=%.4f, brier=%.4f",
                ablation_name,
                task,
                metrics["roc_auc"],
                metrics["pr_auc"],
                metrics["brier_score"],
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model_name", "task", "roc_auc", "pr_auc", "brier_score"]
    )


def compare_multitask_vs_singletask(
    X_train: pd.DataFrame,
    y_dict_train: dict[str, np.ndarray | pd.Series],
    X_val: pd.DataFrame,
    y_dict_val: dict[str, np.ndarray | pd.Series],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare multitask training against individually trained single-task models.

    Trains one multitask model (shared feature matrix, one classifier per
    task) and one independent single-task model per task, then evaluates both
    on the validation set.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_dict_train:
        Mapping of ``{task_name: label_array}`` for training.
    X_val:
        Validation feature matrix.
    y_dict_val:
        Mapping of ``{task_name: label_array}`` for validation.
    config:
        Optional model hyper-parameter dict.

    Returns
    -------
    pd.DataFrame
        Columns: approach, task, roc_auc, pr_auc, brier_score.
        Rows for each (approach, task) combination where approach ∈
        {``'multitask'``, ``'singletask'``}.
    """
    from src.evaluation.metrics import classification_metrics

    config = config or {}
    X_tr = X_train.fillna(0.0)
    X_vl = X_val.fillna(0.0)
    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Multitask: one classifier per task, but treated as a joint model
    # ------------------------------------------------------------------
    multitask_models: dict[str, Any] = {}
    for task, y_tr in y_dict_train.items():
        model = _build_ablation_model(config)
        try:
            model.fit(X_tr.values, np.asarray(y_tr, dtype=float))
            multitask_models[task] = model
        except Exception as exc:
            logger.error("Multitask training failed for task '%s': %s", task, exc)

    for task, model in multitask_models.items():
        y_vl = np.asarray(y_dict_val.get(task, np.zeros(len(X_vl))), dtype=float)
        try:
            y_prob = model.predict_proba(X_vl.values)[:, 1]
            metrics = classification_metrics(y_vl, y_prob)
        except Exception as exc:
            logger.error("Multitask prediction failed for task '%s': %s", task, exc)
            metrics = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier_score": float("nan")}

        rows.append(
            {
                "approach": "multitask",
                "task": task,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
            }
        )

    # ------------------------------------------------------------------
    # 2. Singletask: separate model per task, identical architecture
    # ------------------------------------------------------------------
    for task, y_tr in y_dict_train.items():
        model = _build_ablation_model(config)
        y_vl = np.asarray(y_dict_val.get(task, np.zeros(len(X_vl))), dtype=float)
        try:
            model.fit(X_tr.values, np.asarray(y_tr, dtype=float))
            y_prob = model.predict_proba(X_vl.values)[:, 1]
            metrics = classification_metrics(y_vl, y_prob)
        except Exception as exc:
            logger.error("Singletask training/eval failed for task '%s': %s", task, exc)
            metrics = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier_score": float("nan")}

        rows.append(
            {
                "approach": "singletask",
                "task": task,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
            }
        )
        logger.info(
            "[singletask / %s] roc_auc=%.4f, pr_auc=%.4f",
            task,
            metrics["roc_auc"],
            metrics["pr_auc"],
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["approach", "task", "roc_auc", "pr_auc", "brier_score"]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_ablation_model(config: dict[str, Any]) -> Any:
    """Construct a fast classifier for ablation experiments.

    Prefers XGBoost when available; falls back to logistic regression.
    """
    model_type = config.get("model_type", "xgboost")

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=int(config.get("n_estimators", 200)),
                max_depth=int(config.get("max_depth", 5)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                subsample=float(config.get("subsample", 0.8)),
                colsample_bytree=float(config.get("colsample_bytree", 0.8)),
                eval_metric="logloss",
                random_state=int(config.get("random_state", 42)),
                n_jobs=int(config.get("n_jobs", -1)),
                verbosity=0,
            )
        except ImportError:
            logger.warning("XGBoost not available; falling back to LogisticRegression")

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=int(config.get("n_estimators", 200)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                random_state=int(config.get("random_state", 42)),
                n_jobs=int(config.get("n_jobs", -1)),
                class_weight="balanced",
                verbose=-1,
            )
        except ImportError:
            logger.warning("LightGBM not available; falling back to LogisticRegression")

    # Fallback
    return LogisticRegression(
        C=float(config.get("C", 1.0)),
        max_iter=int(config.get("max_iter", 1000)),
        class_weight="balanced",
        solver="lbfgs",
        random_state=int(config.get("random_state", 42)),
    )
