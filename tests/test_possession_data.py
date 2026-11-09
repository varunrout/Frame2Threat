"""Pure helpers of the Possession Inspector page."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app import possession_data as pg
from src.features.event_time import PossessionArrays


def _arr(n=4):
    z = np.zeros(n)
    return PossessionArrays(
        match_id=7,
        possession_id=3,
        team="A",
        period=1,
        minute=10.0,
        origin="Regular Play",
        type_id=np.ones(n, int),
        own=np.ones(n, bool),
        x=np.arange(n) * 10.0,
        y=z + 40,
        end_x=z,
        end_y=z,
        under_pressure=z,
        pass_length=z,
        abs_s=np.array([5.0, 6.0, 8.0, 9.0])[:n],
    )


def _res():
    ps = [0.2, 0.5, 0.45, None]
    return {
        "steps": [
            {
                "t": t,
                "event": "Pass",
                "own": True,
                "p": p,
                "alert": bool(p and p > 0.4),
                "tipping": t == 1,
                "realised": p is None,
            }
            for t, p in enumerate(ps)
        ]
    }


def test_steps_frame_uses_real_time_and_delta():
    df = pg.steps_frame(_res(), _arr())
    assert df["elapsed_s"].tolist() == [0.0, 1.0, 3.0, 4.0]
    assert df["delta_p"].iloc[1] == 0.5 - 0.2 and np.isnan(df["delta_p"].iloc[3])
    assert df["x"].tolist() == [0.0, 10.0, 20.0, 30.0]


def test_key_events_orders_by_absolute_change():
    df = pg.steps_frame(_res(), _arr())
    assert pg.key_events(df, 1)["event_idx"].tolist() == [1]


def test_held_out_ids_and_artefact_report(tmp_path):
    m = pd.DataFrame({"match_id": [1, 2, 3], "split": ["train", "test", "test"]})
    assert pg.held_out_match_ids(m) == {2, 3}
    assert any("event-time GRU" in x for x in pg.missing_artefacts(tmp_path))


def test_leaderboard_applies_minimum_events_and_flags_intervals():
    players = pd.DataFrame(
        {
            "player": ["a", "b", "c"],
            "events": [150, 99, 400],
            "mean_loo": [0.05, 0.09, -0.01],
            "ci_low": [0.01, 0.0, -0.02],
            "ci_high": [0.09, 0.2, 0.0],
        }
    )
    out = pg.leaderboard(players, 100)
    assert out["player"].tolist() == ["a", "c"]  # b excluded, sorted by mean_loo
    assert out["interval_excludes_zero"].tolist() == [True, False]


def test_possession_attribution_selects_one_seed():
    att = pd.DataFrame(
        {
            "match_id": 1,
            "possession_id": 2,
            "event_idx": [1, 0, 1, 0],
            "seed": [42, 42, 43, 43],
            "loo_drop": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = pg.possession_attribution(att, 1, 2)
    assert out["seed"].unique().tolist() == [42] and out["event_idx"].tolist() == [0, 1]


def test_practical_usefulness_labels_come_from_later_danger_not_realised_steps():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from practical_usefulness import scored_steps

    steps = [{"t": 0, "p": 0.2}, {"t": 1, "p": 0.4}, {"t": 2, "p": None, "realised": True}]
    dangerous = scored_steps({"steps": steps, "realised_at": 2})
    assert [y for _, y in dangerous] == [1, 1]  # the old code labelled these 0
    safe = scored_steps({"steps": [{"t": 0, "p": 0.1}, {"t": 1, "p": 0.1}], "realised_at": None})
    assert [y for _, y in safe] == [0]  # last event of a safe possession is not eligible


def test_limitation_lines_read_numbers_from_artefacts(tmp_path):
    import json
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "reports/repro").mkdir(parents=True)
    for name in ("frozen_metrics_test.json", "sealed_holdout.json"):
        shutil.copy(root / "reports/repro" / name, tmp_path / "reports/repro" / name)
    rec = json.loads((tmp_path / "reports/repro/frozen_metrics_test.json").read_text())
    rec["results"]["alerts"]["XGB"]["false_alert_rate"] = 0.123
    (tmp_path / "reports/repro/frozen_metrics_test.json").write_text(json.dumps(rec))
    text = "\n".join(pg.limitation_lines(tmp_path))
    assert "12.3% (XGBoost)" in text  # follows the artefact, not typed-in text
    assert "UEFA Euro 2024" in text


def test_limitation_lines_without_artefacts(tmp_path):
    text = "\n".join(pg.limitation_lines(tmp_path))
    assert "not available" in text and "No external validation" in text
