"""Final pass models (progression_beyond_pass_k): artefact contract and the app's real-data rules."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features.event_features import build_event_features
from src.models import pass_progression as pp

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app" / "app.py"


def _train_tiny(
    df: pd.DataFrame, tmp_path: Path, name: str = pp.EVENT_ONLY
) -> pp.PassProgressionModel:
    X = pp.build_feature_frame(df)
    y = (np.arange(len(X)) % 3 == 0).astype(int)
    model = XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)  # no scale_pos_weight
    art = pp.PassProgressionModel(model, list(X.columns), None, pp.LABEL, kind="event_only")
    joblib.dump(art, pp.model_path(name, tmp_path))
    return art


def test_saved_feature_list_matches_build_event_features_order_exactly(
    sample_pass_instances_df, tmp_path
):
    art = _train_tiny(sample_pass_instances_df, tmp_path)
    loaded = pp.load_model(pp.EVENT_ONLY, tmp_path)
    assert loaded.features == list(build_event_features(sample_pass_instances_df).columns)
    assert loaded.label == "progression_beyond_pass_k"
    assert art.model.get_params().get("scale_pos_weight") in (None, 1, 1.0)


def test_predictions_are_probabilities_and_missing_features_are_named(
    sample_pass_instances_df, tmp_path
):
    _train_tiny(sample_pass_instances_df, tmp_path)
    m = pp.load_model(pp.EVENT_ONLY, tmp_path)
    p = m.predict_proba_frame(pp.build_feature_frame(sample_pass_instances_df))
    assert p.shape == (len(sample_pass_instances_df),) and ((p >= 0) & (p <= 1)).all()
    two = m.predict_proba(pp.build_feature_frame(sample_pass_instances_df))
    np.testing.assert_allclose(two.sum(axis=1), 1.0)
    broken = pp.build_feature_frame(sample_pass_instances_df).drop(columns=["end_x"])
    with pytest.raises(pp.ArtefactError, match="end_x"):
        m.predict_proba_frame(broken)
    with pytest.raises(pp.ArtefactError, match="geometry"):
        pp.build_feature_frame(sample_pass_instances_df, None, with_360=True)


def test_missing_artefact_error_names_the_file(tmp_path):
    with pytest.raises(pp.ArtefactError, match="pass_progression_event_only.joblib"):
        pp.load_model(pp.EVENT_ONLY, tmp_path)
    with pytest.raises(pp.ArtefactError, match="pass_progression_event_only.json"):
        pp.load_metadata(pp.EVENT_ONLY, tmp_path)
    with pytest.raises(pp.ArtefactError):
        pp.score_real_data(pd.DataFrame({"event_uuid": ["a"]}), tmp_path)


def test_committed_artefacts_match_the_feature_builder_and_metadata():
    only, full = pp.model_path(pp.EVENT_ONLY), pp.model_path(pp.EVENT_360)
    if not (only.exists() and full.exists()):
        pytest.skip("model artefacts not present (models/ is not committed)")
    m_only, m_360 = pp.load_model(pp.EVENT_ONLY), pp.load_model(pp.EVENT_360)
    assert (
        len(m_only.features) == 27
        and m_only.features == m_360.features[:27]
        and len(m_360.features) == 41
    )
    meta = json.loads(only.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["label"] == pp.LABEL and meta["features"] == m_only.features
    assert {"roc_auc", "avg_precision", "brier", "log_loss"} <= set(
        meta["test_all_passes"]["metrics_with_95ci_match_bootstrap"]
    )
    assert (
        len(meta["test_all_passes"]["reliability_table"]) == 10
        and "slope" in meta["test_all_passes"]["calibration"]
    )
    assert "scale_pos_weight" not in json.dumps(meta["hyperparameters"])


def test_app_loader_default_is_the_new_event_only_model():
    assert pp.DEFAULT_MODEL == pp.EVENT_ONLY == "pass_progression_event_only"
    assert pp.model_path().name == "pass_progression_event_only.joblib"
    src = APP.read_text(encoding="utf-8")
    assert "pp.DEFAULT_MODEL" in src and "xgboost_dp_event" not in src
    assert (
        "pass_progression_event_only" not in src or "pp.DEFAULT_MODEL" in src
    )  # no hard-coded old path


def _parents(tree):
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _guarded_by_demo_mode(node, parents) -> bool:
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If):
            t = parent.test
            in_body = any(child is b or child in list(ast.walk(b)) for b in parent.body)
            if isinstance(t, ast.Name) and t.id == "demo_mode" and in_body:
                return True
            if (
                isinstance(t, ast.UnaryOp)
                and isinstance(t.op, ast.Not)
                and isinstance(t.operand, ast.Name)
                and t.operand.id == "demo_mode"
                and not in_body
            ):
                return True
        child = parent
    return False


def test_no_random_score_path_when_real_data_is_loaded():
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    parents = _parents(tree)
    demo_fns = {"_make_demo_scores", "_make_ranked_options", "_show_demo_explanation"}
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in demo_fns
    ]
    assert calls, "expected the demo helpers to be used in demo mode"
    for c in calls:
        assert _guarded_by_demo_mode(
            c, parents
        ), f"{c.func.id} (line {c.lineno}) is reachable with real data"
    assert "random_placeholder" not in src  # the old fallback label is gone
    # a scoring failure with real data stops the page with a message instead of falling back
    assert "Cannot score passes with the trained model" in src and "st.stop()" in src


def test_player_profile_table_filters_and_computes_actual_minus_predicted():
    n_a, n_b = 60, 10
    df = pd.DataFrame({"player_name": ["A"] * n_a + ["B"] * n_b, "team_name": ["T"] * (n_a + n_b)})
    pred = np.r_[np.full(n_a, 0.2), np.full(n_b, 0.5)]
    actual = np.r_[np.r_[np.ones(15), np.zeros(45)], np.ones(n_b)]
    table = pp.player_profile_table(df, pred, actual)  # default minimum 50 passes
    assert table["player"].tolist() == ["A"] and table.loc[0, "passes"] == 60
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.2)
    assert table.loc[0, "actual_rate"] == pytest.approx(0.25)
    assert table.loc[0, "actual_minus_predicted"] == pytest.approx(0.05)
    assert pp.player_profile_table(df, pred, actual, min_passes=5)["player"].tolist() == ["B", "A"]


def test_actual_labels_are_aligned_by_event_uuid_and_missing_sources_raise(tmp_path):
    df = pd.DataFrame({"event_uuid": ["a", "b", "c"]})
    pd.DataFrame({"event_uuid": ["c", "a", "b"], pp.LABEL: [1, 0, 1]}).to_parquet(
        tmp_path / "pass_labels_beyond.parquet"
    )
    assert pp.load_actual_labels(df, tmp_path).tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(pp.ArtefactError):
        pp.load_actual_labels(df, tmp_path / "nowhere")
