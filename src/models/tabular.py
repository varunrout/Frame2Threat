"""Logistic regression and gradient boosting tabular models."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Supported model types
_SUPPORTED_MODELS = frozenset({"logistic", "xgboost", "lightgbm"})


class TabularClassifier:
    """Unified interface for logistic regression, XGBoost, and LightGBM.

    Handles class imbalance automatically via ``scale_pos_weight`` (tree
    models) or ``class_weight='balanced'`` (logistic regression).

    Parameters
    ----------
    model_type:
        One of ``'logistic'``, ``'xgboost'``, ``'lightgbm'``.
    task:
        Label/task name (used for logging and file names).
    config:
        Optional configuration dict with model-specific hyper-parameters.
        Keys recognised: n_estimators, max_depth, learning_rate, n_jobs,
        random_state, max_iter, C, reg_alpha, reg_lambda, num_leaves.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        task: str = "line_break",
        config: dict | None = None,
    ) -> None:
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {_SUPPORTED_MODELS}, got '{model_type}'"
            )
        self.model_type = model_type
        self.task = task
        self.config: dict = config or {}
        self._model: Any = None
        self._feature_names: list[str] | None = None
        self._build_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        cfg = self.config
        rs = int(cfg.get("random_state", 42))

        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                C=float(cfg.get("C", 1.0)),
                max_iter=int(cfg.get("max_iter", 1000)),
                class_weight="balanced",
                solver="lbfgs",
                random_state=rs,
            )

        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=int(cfg.get("n_estimators", 300)),
                max_depth=int(cfg.get("max_depth", 6)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                subsample=float(cfg.get("subsample", 0.8)),
                colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
                reg_alpha=float(cfg.get("reg_alpha", 0.1)),
                reg_lambda=float(cfg.get("reg_lambda", 1.0)),
                eval_metric="logloss",
                random_state=rs,
                n_jobs=int(cfg.get("n_jobs", -1)),
            )

        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            self._model = LGBMClassifier(
                n_estimators=int(cfg.get("n_estimators", 300)),
                max_depth=int(cfg.get("max_depth", -1)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                num_leaves=int(cfg.get("num_leaves", 31)),
                reg_alpha=float(cfg.get("reg_alpha", 0.1)),
                reg_lambda=float(cfg.get("reg_lambda", 1.0)),
                class_weight="balanced",
                random_state=rs,
                n_jobs=int(cfg.get("n_jobs", -1)),
                verbose=-1,
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "TabularClassifier":
        """Fit the model.

        Parameters
        ----------
        X_train, y_train:
            Training features and labels.
        X_val, y_val:
            Optional validation set.  Used for early stopping in XGBoost /
            LightGBM when both are provided.

        Returns
        -------
        self
        """
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)

        y_arr = np.asarray(y_train)
        pos = y_arr.sum()
        neg = len(y_arr) - pos
        spw = neg / max(pos, 1)
        logger.info(
            "[%s/%s] Fitting on %d samples (pos=%d, neg=%d, spw=%.2f)",
            self.task, self.model_type, len(y_arr), pos, neg, spw,
        )

        # Set scale_pos_weight for tree models
        if self.model_type == "xgboost":
            self._model.set_params(scale_pos_weight=spw)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            if self.model_type == "xgboost":
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False
            elif self.model_type == "lightgbm":
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = _lgbm_early_stop_callback(50)

        # Keep DataFrame for XGBoost/LightGBM (preserves feature names);
        # convert to numpy only for logistic regression.
        if self.model_type == "logistic":
            X_fit = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        else:
            X_fit = X_train
        self._model.fit(X_fit, y_arr, **fit_kwargs)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability estimates for each class.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict(X_arr)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted :class:`pd.Series`.

        For logistic regression the absolute coefficient values are used.
        For tree models the built-in ``feature_importances_`` are used.

        Returns
        -------
        pd.Series
            Index: feature names (or int indices if names are unavailable).
            Values: importance scores, sorted descending.
        """
        names = (
            self._feature_names
            if self._feature_names is not None
            else list(range(self._get_n_features()))
        )

        if self.model_type == "logistic":
            coefs = self._model.coef_
            if coefs.ndim == 2:
                coefs = coefs[0]
            importances = np.abs(coefs)
        else:
            importances = self._model.feature_importances_

        series = pd.Series(importances, index=names)
        return series.sort_values(ascending=False)

    def _get_n_features(self) -> int:
        try:
            return self._model.n_features_in_
        except AttributeError:
            return len(self._feature_names or [])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Serialise model and metadata to *path* using joblib."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "task": self.task,
            "config": self.config,
            "model": self._model,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, path)
        logger.info("Saved TabularClassifier to %s", path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "TabularClassifier":
        """Load a :class:`TabularClassifier` from *path*."""
        path = pathlib.Path(path)
        payload = joblib.load(path)
        instance = cls(
            model_type=payload["model_type"],
            task=payload["task"],
            config=payload["config"],
        )
        instance._model = payload["model"]
        instance._feature_names = payload.get("feature_names")
        logger.info("Loaded TabularClassifier from %s", path)
        return instance

    def __repr__(self) -> str:
        return f"TabularClassifier(model_type={self.model_type!r}, task={self.task!r})"


class MultitaskTabular:
    """Wrapper that trains one :class:`TabularClassifier` per task.

    Parameters
    ----------
    model_type:
        Shared model type for all task classifiers.
    config:
        Shared configuration dict forwarded to each :class:`TabularClassifier`.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        config: dict | None = None,
    ) -> None:
        self.model_type = model_type
        self.config: dict = config or {}
        self._classifiers: dict[str, TabularClassifier] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_dict_train: dict[str, pd.Series | np.ndarray],
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_dict_val: dict[str, pd.Series | np.ndarray] | None = None,
    ) -> "MultitaskTabular":
        """Fit one classifier per task.

        Parameters
        ----------
        X_train:
            Shared feature matrix.
        y_dict_train:
            Mapping of task name → label array.
        X_val, y_dict_val:
            Optional validation data per task.

        Returns
        -------
        self
        """
        for task_name, y_train in y_dict_train.items():
            logger.info("MultitaskTabular: fitting task '%s'", task_name)
            clf = TabularClassifier(
                model_type=self.model_type,
                task=task_name,
                config=self.config,
            )
            y_val = None
            if y_dict_val is not None:
                y_val = y_dict_val.get(task_name)
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            self._classifiers[task_name] = clf
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba_all(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return probability estimates for all tasks.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of task name → probability array of shape (n_samples, 2).
        """
        return {
            task: clf.predict_proba(X)
            for task, clf in self._classifiers.items()
        }

    def predict_all(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return binary predictions for all tasks."""
        return {
            task: clf.predict(X)
            for task, clf in self._classifiers.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str | pathlib.Path) -> None:
        """Save each task classifier to *dir_path/<task>.joblib*."""
        dir_path = pathlib.Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for task, clf in self._classifiers.items():
            clf.save(dir_path / f"{task}.joblib")
        logger.info("Saved MultitaskTabular to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str | pathlib.Path) -> "MultitaskTabular":
        """Load all task classifiers from *dir_path*."""
        dir_path = pathlib.Path(dir_path)
        instance = cls()
        for p in sorted(dir_path.glob("*.joblib")):
            task = p.stem
            instance._classifiers[task] = TabularClassifier.load(p)
            logger.info("Loaded classifier for task '%s'", task)
        return instance

    @property
    def tasks(self) -> list[str]:
        """List of fitted task names."""
        return list(self._classifiers.keys())

    def __repr__(self) -> str:
        return (
            f"MultitaskTabular(model_type={self.model_type!r}, "
            f"tasks={self.tasks!r})"
        )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _lgbm_early_stop_callback(stopping_rounds: int):  # type: ignore[return]
    """Return a LightGBM early-stopping callback if available."""
    try:
        from lightgbm import early_stopping
        return [early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    except ImportError:
        return []
