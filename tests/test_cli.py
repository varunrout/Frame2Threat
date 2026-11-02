"""CLI smoke tests: validation, clear failures, exports, and the held-out sample loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.app import cli
from src.app.validation import (
    InputError,
    require_artefacts,
    validate_event_list,
    validate_events_table,
)
from tests.test_batch_scoring import _events, tiny_root  # noqa: F401  (fixture re-use)

ROOT = Path(__file__).resolve().parents[1]


def _ev(type_id=1, **kw):
    base = {"type_id": type_id, "loc_x_norm": 0.5, "loc_y_norm": 0.5}
    base.update(kw)
    return base


def _run(monkeypatch, root, argv):
    monkeypatch.setattr(cli, "_project_root", lambda: root)
    monkeypatch.setattr("sys.argv", ["frame2threat", *argv])
    cli.main()


# --- validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "events, message",
    [
        ([], "non-empty"),
        ("nope", "non-empty"),
        ([1], "expected an object"),
        ([{"type_id": 1}], "missing required key"),
        ([_ev(type_id=99)], "unknown type_id"),
        ([_ev(type_id="x")], "integer"),
        ([_ev(loc_x_norm=1.7)], "outside the normalised range"),
        ([_ev(elapsed_seconds=1.0), _ev()], "for every event or for none"),
        ([_ev(elapsed_seconds=5.0), _ev(elapsed_seconds=1.0)], "non-decreasing"),
    ],
)
def test_validate_event_list_rejects_bad_input(events, message):
    with pytest.raises(InputError, match=message):
        validate_event_list(events)


def test_validate_event_list_accepts_good_input():
    good = [_ev(elapsed_seconds=0.0), _ev(type_id=2, elapsed_seconds=1.5)]
    assert validate_event_list(good) is good


def test_validate_events_table_reports_missing_columns():
    with pytest.raises(InputError, match="missing column"):
        validate_events_table(pd.DataFrame({"match_id": [1]}))
    validate_events_table(_events())


def test_require_artefacts_names_the_missing_files(tmp_path):
    (tmp_path / "models").mkdir()
    with pytest.raises(InputError, match="eventtime_meta.json.*run_event_time"):
        require_artefacts(tmp_path, "both")


# --- score-live ------------------------------------------------------------------------


def test_score_live_prints_mode_and_exports(tiny_root, tmp_path, monkeypatch, capsys):  # noqa: F811
    events = [
        _ev(1, loc_x_norm=0.3, end_x_norm=0.4, end_y_norm=0.5, pass_length_norm=0.1),
        _ev(2, loc_x_norm=0.5),
        _ev(1, loc_x_norm=0.6, end_x_norm=0.7, end_y_norm=0.5, pass_length_norm=0.1),
        _ev(5, loc_x_norm=0.9),
        _ev(1, loc_x_norm=0.4),
    ]
    src = tmp_path / "poss.json"
    src.write_text(json.dumps({"event_sequence": events, "period": 1}), encoding="utf-8")
    for fmt in ("csv", "parquet"):
        out = tmp_path / f"scores.{fmt}"
        _run(monkeypatch, tiny_root, ["score-live", str(src), "--model", "xgb", "-o", str(out)])
        text = capsys.readouterr().out
        assert "event-time (prospective)" in text and "OUTCOME REALISED" in text
        table = pd.read_csv(out) if fmt == "csv" else pd.read_parquet(out)
        assert len(table) == 5
        assert table["p_danger"].iloc[:3].between(0, 1).all()
        assert table["p_danger"].iloc[3:].isna().all()  # the shot realises the outcome
        assert bool(table["realised"].iloc[3])


@pytest.mark.parametrize(
    "setup, message",
    [
        ("missing", "input JSON not found"),
        ("badjson", "not valid JSON"),
        ("badschema", "missing required key"),
    ],
)
def test_score_live_fails_clearly(
    tiny_root, tmp_path, monkeypatch, capsys, setup, message  # noqa: F811
):
    path = tmp_path / "in.json"
    if setup == "badjson":
        path.write_text("{oops", encoding="utf-8")
    elif setup == "badschema":
        path.write_text(json.dumps([{"type_id": 1}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tiny_root, ["score-live", str(path)])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_score_live_without_model_artefacts(tmp_path, monkeypatch, capsys):
    (tmp_path / "models").mkdir()
    path = tmp_path / "in.json"
    path.write_text(json.dumps([_ev()]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, ["score-live", str(path)])
    assert exc.value.code == 2 and "missing model artefact" in capsys.readouterr().err


# --- score-batch -----------------------------------------------------------------------


def test_score_batch_writes_parquet_and_rejects_bad_extension(
    tiny_root, tmp_path, monkeypatch, capsys  # noqa: F811
):
    src = tmp_path / "events.parquet"
    _events().to_parquet(src)
    out = tmp_path / "scores.parquet"
    _run(monkeypatch, tiny_root, ["score-batch", str(src), "-o", str(out)])
    assert "event-time (prospective)" in capsys.readouterr().out
    table = pd.read_parquet(out)
    assert {"p_xgb", "p_gru", "outcome_realised"} <= set(table.columns) and len(table) == 9
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tiny_root, ["score-batch", str(src), "-o", str(tmp_path / "x.txt")])
    assert exc.value.code == 2


def test_score_batch_rejects_wrong_schema(tiny_root, tmp_path, monkeypatch, capsys):  # noqa: F811
    src = tmp_path / "events.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(src)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tiny_root, ["score-batch", str(src)])
    assert exc.value.code == 2 and "missing column" in capsys.readouterr().err


# --- sample-possession -----------------------------------------------------------------


def test_sample_possession_uses_only_test_matches(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    seq = json.dumps([_ev()] * 10)
    pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "possession_id": [1, 2, 1, 2],
            "team_name": "T",
            "origin_type": "Regular Play",
            "period": 1,
            "n_events": 10,
            "poss_dangerous": [0, 1, 0, 1],
            "event_sequence": seq,
        }
    ).to_parquet(proc / "possession_sequences.parquet")
    events = _events().assign(match_id=2, minute=68)
    events.to_parquet(proc / "events_parsed.parquet", index=False)
    pd.DataFrame({"match_id": [1, 2], "split": ["train", "test"]}).to_csv(
        proc / "split_manifest.csv", index=False
    )
    out = tmp_path / "sample.json"
    _run(monkeypatch, tmp_path, ["sample-possession", "-o", str(out), "--dangerous", "1"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["match_id"] == 2 and payload["retrospective_label_poss_dangerous"] == 1
    assert payload["minute"] == 68
    validate_event_list(payload["event_sequence"])


# --- real artefacts (skipped when the gitignored data / models are absent) ---------------

_HAVE_REAL = all(
    (ROOT / p).exists()
    for p in (
        "models/eventtime_gru.pt",
        "models/eventtime_xgboost.joblib",
        "models/eventtime_meta.json",
        "data/processed/possession_sequences.parquet",
        "data/processed/split_manifest.csv",
    )
)


@pytest.mark.skipif(not _HAVE_REAL, reason="trained models / processed data not present")
@pytest.mark.parametrize("model", ["gru", "xgb"])
def test_score_live_on_real_heldout_possession(tmp_path, monkeypatch, capsys, model):
    sample = tmp_path / "s.json"
    _run(monkeypatch, ROOT, ["sample-possession", "-o", str(sample), "--seed", "7"])
    manifest = pd.read_csv(ROOT / "data" / "processed" / "split_manifest.csv")
    payload = json.loads(sample.read_text(encoding="utf-8"))
    assert manifest.set_index("match_id").loc[payload["match_id"], "split"] == "test"
    out = tmp_path / f"o_{model}.csv"
    _run(monkeypatch, ROOT, ["score-live", str(sample), "--model", model, "-o", str(out)])
    table = pd.read_csv(out)
    assert len(table) == len(payload["event_sequence"])
    scored = table["p_danger"].dropna()
    assert len(scored) > 0 and scored.between(0, 1).all()
