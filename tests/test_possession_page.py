"""Pure helpers of the Possession Inspector page."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app import possession_page as pg
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
