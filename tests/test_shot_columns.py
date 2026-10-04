"""Shot xG and outcome are parsed and reach the possession labels (poss_xg_generated,
poss_has_goal, outcome tier 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.parse_events import parse_events
from src.data.parse_possessions import build_possession_sequences


def _raw_events() -> pd.DataFrame:
    rows = []
    spec = [
        ("Pass", [60.0, 40.0], np.nan, None),
        ("Carry", [75.0, 40.0], np.nan, None),
        ("Shot", [108.0, 38.0], 0.42, "Goal"),
        ("Pass", [60.0, 40.0], np.nan, None),
        ("Carry", [70.0, 40.0], np.nan, None),
        ("Shot", [105.0, 20.0], 0.08, "Saved"),
    ]
    for i, (t, loc, xg, outcome) in enumerate(spec):
        poss = 1 if i < 3 else 2
        rows.append(
            {
                "id": f"e{i}",
                "match_id": 1,
                "index": i + 1,
                "period": 1,
                "timestamp": f"00:00:{i:02d}.000",
                "minute": 0,
                "second": i,
                "type": t,
                "team": "A",
                "player": "p",
                "play_pattern": "Regular Play",
                "possession": poss,
                "location": loc,
                "shot_statsbomb_xg": xg,
                "shot_outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def test_parse_events_keeps_shot_xg_and_outcome_only_for_shots():
    out = parse_events(_raw_events())
    shots = out[out.type_name == "Shot"]
    assert shots.shot_statsbomb_xg.tolist() == [0.42, 0.08]
    assert shots.shot_outcome_name.tolist() == ["Goal", "Saved"]
    assert out.loc[out.type_name != "Shot", "shot_statsbomb_xg"].isna().all()
    assert out.loc[out.type_name != "Shot", "shot_outcome_name"].isna().all()


def test_goal_scoring_shot_gives_goal_tier_4_and_positive_xg():
    events = parse_events(_raw_events())
    poss = build_possession_sequences(events, min_events=2).set_index("possession_id")
    goal, saved = poss.loc[1], poss.loc[2]
    assert bool(goal["poss_has_goal"]) and int(goal["poss_outcome_tier"]) == 4
    assert goal["poss_xg_generated"] > 0.4
    assert not bool(saved["poss_has_goal"]) and int(saved["poss_outcome_tier"]) == 3
    assert abs(saved["poss_xg_generated"] - 0.08) < 1e-6
