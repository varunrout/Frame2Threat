"""Regression tests for genuine event-time input fidelity and invalid data."""

import numpy as np
import pytest

from src.app.live_scoring import arrays_to_payload, events_to_arrays, score_stream
from src.app.validation import InputError, validate_event_list, validate_events_table
from src.features.event_time import sequence_inputs, snapshot_features
from tests.test_batch_scoring import tiny_root, _events  # noqa: F401
from tests.test_event_time import make_arrays, P, PR, C


@pytest.mark.parametrize(
    "field,value",
    [
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", -1),
        ("type_id", 1.5),
        ("type_id", float("inf")),
        ("type_id", True),
        ("own_team", 0.5),
        ("under_pressure", float("nan")),
    ],
)
def test_invalid_event_values(field, value):
    event = {"type_id": P, "loc_x_norm": 0.5, "loc_y_norm": 0.5, field: value}
    with pytest.raises(InputError):
        validate_event_list([event])


def test_input_roundtrip_preserves_training_features():
    arr = make_arrays(
        [P, PR, C],
        [35.123456, 95, np.nan],
        [23, 10, np.nan],
        own=[True, False, True],
        seconds=[90.123, 90.6, 92.789],
    )
    arr.minute = 68
    payload = arrays_to_payload(arr)
    restored, warnings = events_to_arrays(
        payload["event_sequence"],
        origin=payload["origin"],
        minute=payload["minute"],
        period=payload["period"],
    )
    assert not warnings
    np.testing.assert_allclose(snapshot_features(arr), snapshot_features(restored))
    np.testing.assert_allclose(sequence_inputs(arr)[1], sequence_inputs(restored)[1])


@pytest.mark.parametrize("model", ["xgb", "gru"])
def test_predictions_are_prefix_invariant(tiny_root, model):  # noqa: F811
    arr = make_arrays([P, PR, C], [30, 80, 60], [40, 40, 40], own=[True, False, True])
    payload = arrays_to_payload(arr)
    kwargs = {"origin": arr.origin, "period": arr.period, "minute": arr.minute, "model": model}
    full = score_stream(payload["event_sequence"], tiny_root, **kwargs)
    for n in range(1, arr.n + 1):
        prefix = score_stream(payload["event_sequence"][:n], tiny_root, **kwargs)
        np.testing.assert_allclose(
            [s["p"] for s in prefix["steps"]], [s["p"] for s in full["steps"][:n]], atol=1e-7
        )


@pytest.mark.parametrize(
    "column", ["timestamp", "pass_end_x", "under_pressure", "play_pattern_name"]
)
def test_batch_missing_feature_columns(column):
    with pytest.raises(InputError, match=column):
        validate_events_table(_events().drop(columns=column))


def test_incompatible_model_schema_is_rejected(tiny_root):  # noqa: F811
    import joblib

    path = tiny_root / "models/eventtime_xgboost.joblib"
    art = joblib.load(path)
    art["features"] = list(reversed(art["features"]))
    joblib.dump(art, path)
    events = [{"type_id": P, "loc_x_norm": 0.5, "loc_y_norm": 0.5}]
    with pytest.raises(InputError, match="schema"):
        score_stream(events, tiny_root, model="xgb")


def test_training_rejects_overlapping_match_manifest(tmp_path, monkeypatch):
    import pandas as pd
    from scripts import run_event_time

    monkeypatch.setattr(run_event_time, "PROC", tmp_path)
    for name, mid in (("train", 1), ("val", 2), ("test", 1)):
        pd.DataFrame({"match_id": [mid]}).to_parquet(tmp_path / f"{name}.parquet")
    pd.DataFrame({"match_id": [1, 2, 1], "split": ["train", "val", "test"]}).to_csv(
        tmp_path / "split_manifest.csv", index=False
    )
    with pytest.raises(ValueError, match="exactly once"):
        run_event_time.load_split_ids()


@pytest.mark.parametrize(
    "column,value", [("location_x", np.inf), ("minute", np.nan), ("timestamp", "bad")]
)
def test_batch_rejects_malformed_values(column, value):
    events = _events()
    events.loc[0, column] = value
    with pytest.raises(InputError, match=column):
        validate_events_table(events)


def test_batch_rejects_duplicate_observations():
    import pandas as pd

    events = _events()
    with pytest.raises(InputError, match="duplicate"):
        validate_events_table(pd.concat([events, events.iloc[:1]], ignore_index=True))


def test_native_model_roundtrip_without_joblib(tiny_root):  # noqa: F811
    import json
    import joblib
    from src.features.event_time import FEATURE_NAMES
    from src.app.validation import require_artefacts

    events = [{"type_id": P, "loc_x_norm": 0.5, "loc_y_norm": 0.5}]
    before = score_stream(events, tiny_root, model="xgb")
    path = tiny_root / "models/eventtime_xgboost.joblib"
    joblib.load(path)["model"].save_model(tiny_root / "models/eventtime_xgboost.ubj")
    meta_path = tiny_root / "models/eventtime_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["features"] = FEATURE_NAMES
    meta_path.write_text(json.dumps(meta))
    path.unlink()
    require_artefacts(tiny_root, "xgb")
    after = score_stream(events, tiny_root, model="xgb")
    assert after["steps"] == before["steps"]


@pytest.mark.parametrize("provider_owner", [True, False])
def test_raw_future_events_cannot_reassign_possession_owner(provider_owner):
    import pandas as pd
    from src.features.event_time import build_possession_arrays

    prefix = _events().query("possession_id == 2").copy()
    if provider_owner:
        prefix["possession_team_name"] = "A"
    future = pd.concat([prefix.iloc[:1]] * 8, ignore_index=True)
    future["team_name"] = "B"
    future["index"] = np.arange(100, 108)
    whole = pd.concat([prefix, future], ignore_index=True)
    early = build_possession_arrays(prefix)[0]
    full = build_possession_arrays(whole)[0]
    assert full.team == early.team == "A"
    np.testing.assert_array_equal(full.own[: early.n], early.own)
    np.testing.assert_allclose(snapshot_features(full)[: early.n], snapshot_features(early))
    np.testing.assert_allclose(sequence_inputs(full)[1][: early.n], sequence_inputs(early)[1])
