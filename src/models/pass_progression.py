"""
pass_progression.py
===================
Final pass-level models on ``progression_beyond_pass_k``: XGBoost, early stopping on validation,
**no** ``scale_pos_weight``, optional validation-only calibrator.

Artefacts (``models/pass_progression_event_only.joblib`` and ``..._event_360.joblib``) are a
:class:`PassProgressionModel`; the JSON metadata next to each holds the label definition, the ordered
feature list, match counts, the git commit, test metrics with match-level bootstrap intervals, the
reliability table and calibration intercept / slope.

This module has no Streamlit or torch dependency, so the app, scripts and tests share it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

LABEL = "progression_beyond_pass_k"
LABEL_DEFINITION = (
    "Within the next 5 events of the same possession, ignoring the receipt of this pass and the "
    "opponent's events: a shot, a new box entry, or a gain of at least 10 m beyond the pass end into "
    "a more advanced zone (reports/label_methodology.md, Label G)."
)
LABEL_PLAIN_ENGLISH = (
    "Does the passing team make *further* dangerous progress in the next five events: a shot, a "
    "first entry into the penalty area, or a gain of at least ten metres beyond where the pass "
    "ended into a more advanced zone?  The pass's own receipt and the opponent's events are ignored."
)
EVENT_ONLY = "pass_progression_event_only"
EVENT_360 = "pass_progression_event_360"
DEFAULT_MODEL = EVENT_ONLY
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ArtefactError(RuntimeError):
    """A model artefact or a required feature is missing; the message names it."""


@dataclass
class PassProgressionModel:
    """XGBoost classifier (+ optional calibrator fitted on validation only) with its feature order."""

    model: Any
    features: list[str]
    calibrator: Any = None
    label: str = LABEL
    kind: str = "event_only"

    def predict_proba_frame(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities in [0, 1]; raises :class:`ArtefactError` naming any missing feature."""
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ArtefactError(f"missing feature(s) {missing} required by the {self.kind} model")
        p = self.model.predict_proba(X[self.features].astype(float))[:, 1]
        if self.calibrator is not None:
            p = (
                self.calibrator.predict(p)
                if hasattr(self.calibrator, "predict")
                else self.calibrator(p)
            )
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    # sklearn-style alias for code that expects a two-column predict_proba
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba_frame(X)
        return np.column_stack([1.0 - p, p])


def model_path(name: str = DEFAULT_MODEL, models_dir: Path | None = None) -> Path:
    return (models_dir or MODELS_DIR) / f"{name}.joblib"


def load_model(name: str = DEFAULT_MODEL, models_dir: Path | None = None) -> PassProgressionModel:
    path = model_path(name, models_dir)
    if not path.exists():
        raise ArtefactError(
            f"model artefact not found: {path} (train it with scripts/train_pass_models.py)"
        )
    return joblib.load(path)


def load_metadata(name: str = DEFAULT_MODEL, models_dir: Path | None = None) -> dict:
    path = model_path(name, models_dir).with_suffix(".json")
    if not path.exists():
        raise ArtefactError(f"model metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_feature_frame(
    pass_df: pd.DataFrame, geometry: pd.DataFrame | None = None, with_360: bool = False
) -> pd.DataFrame:
    """Engineered features in the order the models were trained on.

    ``geometry`` is indexed by ``event_uuid`` (14 geometry columns); passes without a row get zeros,
    as in training.  Raises :class:`ArtefactError` when the 360 model is requested without geometry.
    """
    from src.features.event_features import build_event_features

    X = build_event_features(pass_df)
    if not with_360:
        return X
    if geometry is None:
        raise ArtefactError(
            "geometry features are required by the event+360 model but were not supplied"
        )
    g = geometry.reindex(pass_df["event_uuid"].values).reset_index(drop=True)
    g.index = X.index
    return pd.concat([X, g.fillna(0.0)], axis=1)


def player_profile_table(
    pass_df: pd.DataFrame, scores: np.ndarray, actual: np.ndarray, min_passes: int = 50
) -> pd.DataFrame:
    """Per player: passes, mean predicted progression probability, actual rate, actual minus predicted.

    Players with fewer than ``min_passes`` passes are dropped (the app lets the user lower the bar).
    """
    df = pd.DataFrame(
        {
            "player": pass_df["player_name"].to_numpy(),
            "team": pass_df["team_name"].to_numpy(),
            "pred": np.asarray(scores, float),
            "actual": np.asarray(actual, float),
        }
    ).dropna(subset=["player", "actual"])
    g = df.groupby("player").agg(
        team=("team", lambda s: s.mode().iloc[0]),
        passes=("pred", "size"),
        mean_predicted=("pred", "mean"),
        actual_rate=("actual", "mean"),
    )
    g["actual_minus_predicted"] = g["actual_rate"] - g["mean_predicted"]
    return (
        g[g["passes"] >= min_passes]
        .sort_values("actual_minus_predicted", ascending=False)
        .reset_index()
    )


def geometry_for(
    pass_rows: pd.DataFrame, frames_df: pd.DataFrame, cache: pd.DataFrame | None = None
) -> pd.DataFrame:
    """14 geometry features indexed by ``event_uuid``: from ``cache`` where available, else computed."""
    from src.features.geometry_features import build_geometry_features

    uuids = pass_rows["event_uuid"].tolist()
    if cache is not None and all(u in cache.index for u in uuids):
        return cache.loc[uuids]
    return build_geometry_features(pass_rows, frames_df)


def load_actual_labels(pass_df: pd.DataFrame, processed_dir: Path) -> np.ndarray:
    """Actual ``progression_beyond_pass_k`` per pass (aligned by ``event_uuid``).

    Reads ``pass_labels_beyond.parquet`` when present; otherwise computes it from ``events_parsed.parquet``.
    Raises :class:`ArtefactError` if neither source exists.
    """
    cached = processed_dir / "pass_labels_beyond.parquet"
    if cached.exists():
        lab = pd.read_parquet(cached).set_index("event_uuid")[LABEL]
    else:
        events = processed_dir / "events_parsed.parquet"
        if not events.exists():
            raise ArtefactError(f"actual labels unavailable: neither {cached} nor {events} exists")
        from src.labels.progression_beyond_pass import compute_progression_beyond_pass

        lab = compute_progression_beyond_pass(pd.read_parquet(events), pass_df, {}).set_index(
            "event_uuid"
        )[LABEL]
    out = lab.reindex(pass_df["event_uuid"].to_numpy())
    if out.isna().any():
        raise ArtefactError(f"actual labels missing for {int(out.isna().sum())} passes")
    return out.astype(float).to_numpy()


def score_real_data(pass_df: pd.DataFrame, models_dir: Path | None = None) -> np.ndarray:
    """Event-only progression probability for every pass; raises (never falls back to placeholders)."""
    model = load_model(EVENT_ONLY, models_dir)
    return model.predict_proba_frame(build_feature_frame(pass_df))
