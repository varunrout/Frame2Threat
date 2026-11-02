"""Load portable event-time XGBoost models, with legacy joblib compatibility."""

from pathlib import Path

from src.app.validation import InputError
from src.features.event_time import FEATURE_NAMES


def load_event_time_xgb(root: Path, meta: dict) -> dict:
    native = root / "models/eventtime_xgboost.ubj"
    if native.exists():
        from xgboost import XGBClassifier

        if meta.get("features") != FEATURE_NAMES:
            raise InputError(
                "native XGBoost feature schema differs from inference; retrain the model"
            )
        model = XGBClassifier()
        model.load_model(native)
        art = {"model": model, "features": meta["features"]}
    else:
        import joblib

        art = joblib.load(root / "models/eventtime_xgboost.joblib")
    if list(art["features"]) != list(FEATURE_NAMES):
        raise InputError("XGBoost feature schema differs from inference; retrain the model")
    return art
