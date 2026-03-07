"""Tests for model implementations."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import RuleBasedDangerousProgression, RuleBasedLineBreak
from src.models.tabular import MultitaskTabular, TabularClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic tabular feature matrix and binary labels."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "start_x": rng.uniform(20, 90, n),
            "start_y": rng.uniform(10, 70, n),
            "end_x": rng.uniform(30, 110, n),
            "end_y": rng.uniform(10, 70, n),
            "pass_length": rng.uniform(5, 35, n),
            "x_gain": rng.uniform(-10, 30, n),
            "goal_dist_gain": rng.uniform(-5, 15, n),
        }
    )
    y = (X["x_gain"] > 10).astype(int).values
    return X, y


def _make_pass_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic pass_instances-like DataFrame for rule-based models."""
    rng = np.random.default_rng(seed)
    start_x = rng.uniform(20, 90, n)
    x_gain = rng.uniform(-5, 30, n)
    return pd.DataFrame(
        {
            "event_uuid": [f"evt_{i:04d}" for i in range(n)],
            "match_id": [1001] * n,
            "start_x": start_x,
            "start_y": rng.uniform(10, 70, n),
            "end_x": start_x + x_gain,
            "end_y": rng.uniform(10, 70, n),
            "pass_length": rng.uniform(5, 35, n),
            "x_gain": x_gain,
            "has_360": [True] * n,
        }
    )


# ---------------------------------------------------------------------------
# Rule-based baselines
# ---------------------------------------------------------------------------


class TestRuleBasedLineBreak:
    def test_returns_series(self):
        df = _make_pass_df()
        model = RuleBasedLineBreak()
        result = model.predict(df)
        assert hasattr(result, "__len__")
        assert len(result) == len(df)

    def test_binary_output(self):
        df = _make_pass_df()
        model = RuleBasedLineBreak()
        result = model.predict(df)
        unique = set(np.asarray(result).ravel())
        assert unique.issubset({0, 1, True, False})

    def test_high_end_x_more_often_positive(self):
        """Passes ending deep in opponent territory should fire more often."""
        df_deep = _make_pass_df(200)
        df_deep["end_x"] = 90.0
        df_deep["pass_length"] = 20.0
        model = RuleBasedLineBreak()
        preds = np.asarray(model.predict(df_deep)).astype(int)
        assert preds.mean() > 0.3, "Expected more positives for deep passes"


class TestRuleBasedDangerousProgression:
    def test_returns_correct_length(self):
        df = _make_pass_df()
        model = RuleBasedDangerousProgression()
        result = model.predict(df)
        assert len(result) == len(df)

    def test_binary_output(self):
        df = _make_pass_df()
        model = RuleBasedDangerousProgression()
        result = model.predict(df)
        unique = set(np.asarray(result).ravel())
        assert unique.issubset({0, 1, True, False})


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------


class TestTabularClassifierLogistic:
    def test_fit_predict(self):
        X, y = _make_xy()
        clf = TabularClassifier(model_type="logistic", task="test_task")
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(X),) or probs.shape == (len(X), 2)
        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_in_0_1(self):
        X, y = _make_xy()
        clf = TabularClassifier(model_type="logistic")
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        p = probs[:, 1] if probs.ndim == 2 else probs
        assert (p >= 0).all() and (p <= 1).all()

    def test_save_and_load(self):
        X, y = _make_xy()
        clf = TabularClassifier(model_type="logistic")
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            clf.save(path)
            loaded = TabularClassifier.load(path)
            p_orig = clf.predict_proba(X)
            p_load = loaded.predict_proba(X)
            np.testing.assert_allclose(p_orig, p_load, rtol=1e-5)


class TestTabularClassifierXGBoost:
    def test_fit_predict(self):
        pytest.importorskip("xgboost")
        X, y = _make_xy(n=150)
        clf = TabularClassifier(model_type="xgboost")
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert len(probs) == len(X)

    def test_feature_importance_returned(self):
        pytest.importorskip("xgboost")
        X, y = _make_xy(n=150)
        clf = TabularClassifier(model_type="xgboost")
        clf.fit(X, y)
        imp = clf.get_feature_importance()
        assert imp is not None
        assert len(imp) == X.shape[1]


class TestMultitaskTabular:
    def test_all_tasks_trained(self):
        X, _ = _make_xy(n=200)
        rng = np.random.default_rng(1)
        y_dict = {
            "line_break": rng.integers(0, 2, len(X)),
            "shot_within_k": rng.integers(0, 2, len(X)),
        }
        mt = MultitaskTabular(model_type="logistic")
        mt.fit(X, y_dict)
        preds = mt.predict_proba_all(X)
        assert "line_break" in preds
        assert "shot_within_k" in preds
        assert len(preds["line_break"]) == len(X)

    def test_predict_all_tasks_have_probs_in_range(self):
        X, _ = _make_xy(n=100)
        rng = np.random.default_rng(2)
        y_dict = {
            "task_a": rng.integers(0, 2, len(X)),
            "task_b": rng.integers(0, 2, len(X)),
        }
        mt = MultitaskTabular(model_type="logistic")
        mt.fit(X, y_dict)
        preds = mt.predict_proba_all(X)
        for task, probs in preds.items():
            p = probs[:, 1] if np.asarray(probs).ndim == 2 else np.asarray(probs)
            assert (p >= 0).all() and (p <= 1).all(), f"{task} probs out of range"
