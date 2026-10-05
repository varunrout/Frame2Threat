"""score-batch uses the event-time models; train-early is deprecated."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.app import cli
from src.features.event_time import FEATURE_NAMES, N_SEQ_NUMERIC


def _events() -> pd.DataFrame:
    rows, idx = [], 0
    specs = [
        ("A", ["Pass", "Carry", "Pass", "Shot", "Pass"], [30, 45, 60, 100, 50]),
        ("A", ["Pass", "Pressure", "Carry", "Pass"], [30, 70, 55, 80]),
    ]
    for pid, (team, names, xs) in enumerate(specs, start=1):
        for k, (n, x) in enumerate(zip(names, xs)):
            idx += 1
            rows.append(
                {
                    "match_id": 1,
                    "possession_id": pid,
                    "index": idx,
                    "period": 1,
                    "minute": 1,
                    "second": k,
                    "timestamp": f"00:00:{k:02d}.000",
                    "type_name": n,
                    "team_name": "B" if n == "Pressure" else team,
                    "play_pattern_name": "Regular Play",
                    "location_x": float(x),
                    "location_y": 40.0,
                    "under_pressure": False,
                    "pass_end_x": float(x + 5) if n == "Pass" else np.nan,
                    "pass_end_y": 40.0 if n == "Pass" else np.nan,
                    "pass_length": 10.0 if n == "Pass" else np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_root(tmp_path):
    torch = pytest.importorskip("torch")
    from xgboost import XGBClassifier

    from src.models.event_time import EventTimeGRU

    (tmp_path / "models").mkdir()
    rng = np.random.default_rng(0)
    xgb = XGBClassifier(n_estimators=5, max_depth=2).fit(
        pd.DataFrame(rng.normal(size=(60, len(FEATURE_NAMES))), columns=FEATURE_NAMES),
        rng.integers(0, 2, 60),
    )
    joblib.dump(
        {"model": xgb, "features": FEATURE_NAMES}, tmp_path / "models" / "eventtime_xgboost.joblib"
    )
    torch.manual_seed(0)
    net = EventTimeGRU(N_SEQ_NUMERIC, embed_dim=8, hidden_size=64)
    torch.save(
        {"state_dict": net.state_dict(), "n_numeric": N_SEQ_NUMERIC, "hidden": 64, "embed_dim": 8},
        tmp_path / "models" / "eventtime_gru.pt",
    )
    (tmp_path / "models" / "eventtime_meta.json").write_text(
        json.dumps({"xgb_threshold": 0.4, "gru_threshold": 0.4, "median_inter_event_seconds": 1.0})
    )
    return tmp_path


def test_score_events_columns_probabilities_and_realised_events(tiny_root):
    from src.app.batch_scoring import COLUMNS, score_events

    out = score_events(_events(), tiny_root)
    assert list(out.columns) == COLUMNS
    assert not any(
        c.startswith("gru_prob") or c == "fraction" for c in out.columns
    )  # no fraction output
    live = out[~out["outcome_realised"]]
    assert live[["p_xgb", "p_gru"]].apply(lambda s: s.between(0, 1)).all().all()
    # possession 1: the shot (event 3) realises the outcome; nothing is scored at or after it
    p1 = out[out["possession_id"] == 1].sort_values("event_idx")
    assert p1["outcome_realised"].tolist() == [False, False, False, True, True]
    assert p1.loc[p1["outcome_realised"], ["p_xgb", "p_gru"]].isna().all().all()
    assert p1["label_danger_after"].iloc[:3].tolist() == [1, 1, 1]
    assert p1["label_danger_after"].iloc[3:].isna().all()
    # possession 2 never reaches danger: last event is not eligible, so no label
    p2 = out[out["possession_id"] == 2].sort_values("event_idx")
    assert p2["label_danger_after"].iloc[:3].tolist() == [0, 0, 0] and pd.isna(
        p2["label_danger_after"].iloc[3]
    )
    assert p2["alert_xgb"].dtype == bool


def test_cli_has_no_fracs_and_train_early_is_deprecated(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["score-batch", "events.parquet", "--limit", "5"])
    assert args.limit == 5 and not hasattr(args, "fracs")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_train_early(parser.parse_args(["train-early"]))
    assert exc.value.code == 2
    assert "DEPRECATED" in capsys.readouterr().err
