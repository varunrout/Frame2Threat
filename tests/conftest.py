"""Shared pytest fixtures for Frame2Threat tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """Return the absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> pathlib.Path:
    """Return the absolute path to the configs/ directory."""
    return CONFIGS_DIR


@pytest.fixture
def sample_events_df():
    """Synthetic events DataFrame matching StatsBomb schema."""
    n = 50
    rng = np.random.default_rng(42)

    data = {
        'id': [f'evt_{i:03d}' for i in range(n)],
        'match_id': [1001] * 30 + [1002] * 20,
        'index': list(range(n)),
        'period': [1] * 25 + [2] * 25,
        'timestamp': [f'00:{i:02d}:00.000' for i in range(n)],
        'minute': list(range(n)),
        'second': [0] * n,
        'type': [{'id': 30, 'name': 'Pass'}] * 40 + [{'id': 16, 'name': 'Shot'}] * 5 + [{'id': 23, 'name': 'Under Pressure'}] * 5,
        'team': [{'id': 1, 'name': 'Team A'}] * n,
        'player': [{'id': 101, 'name': 'Player A'}] * n,
        'possession': list(range(1, n + 1)),
        'play_pattern': [{'id': 1, 'name': 'Regular Play'}] * n,
        'location': [[rng.uniform(30, 90), rng.uniform(20, 60)] for _ in range(n)],
    }

    passes = []
    for i in range(n):
        if i < 40:
            end_x = float(data['location'][i][0]) + rng.uniform(5, 25)
            end_y = float(data['location'][i][1]) + rng.uniform(-10, 10)
            passes.append({
                'recipient': {'id': 102, 'name': 'Player B'},
                'length': rng.uniform(5, 30),
                'angle': rng.uniform(-np.pi, np.pi),
                'end_location': [end_x, end_y],
                'body_part': {'id': 70, 'name': 'Right Foot'},
                'height': {'id': 1, 'name': 'Ground Pass'},
                'type': {'id': 65, 'name': 'Kick Off'},
                'outcome': None,
                'switch': False,
                'cross': False,
                'through_ball': False,
                'goal_kick': False,
                'corner': False,
                'free_kick': False,
            })
        else:
            passes.append(None)

    data['pass'] = passes
    data['under_pressure'] = [True if i % 5 == 0 else None for i in range(n)]
    data['shot'] = [
        {'outcome': {'id': 72, 'name': 'Off T'}, 'statsbomb_xg': 0.1}
        if 40 <= i < 45 else None
        for i in range(n)
    ]

    return pd.DataFrame(data)


@pytest.fixture
def sample_pass_instances_df():
    """Canonical pass_instances DataFrame."""
    rng = np.random.default_rng(42)
    n = 100

    start_x = rng.uniform(20, 90, n)
    start_y = rng.uniform(10, 70, n)
    end_x = start_x + rng.uniform(5, 30, n)
    end_y = start_y + rng.uniform(-15, 15, n)

    return pd.DataFrame({
        'match_id': [1001] * 50 + [1002] * 50,
        'competition_id': [43] * n,
        'season_id': [3] * n,
        'event_uuid': [f'evt_{i:04d}' for i in range(n)],
        'possession_id': list(range(1, n + 1)),
        'team_name': ['Team A'] * n,
        'player_name': ['Player A'] * 50 + ['Player B'] * 50,
        'pass_recipient_name': ['Player B'] * n,
        'minute': rng.integers(0, 90, n).tolist(),
        'second': rng.integers(0, 60, n).tolist(),
        'period': [1] * 50 + [2] * 50,
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x.clip(0, 120),
        'end_y': end_y.clip(0, 80),
        'pass_length': rng.uniform(5, 35, n),
        'pass_angle': rng.uniform(-3.14, 3.14, n),
        'pass_body_part': ['Right Foot'] * n,
        'pass_height': ['Ground Pass'] * n,
        'pass_type': ['Open Play'] * n,
        'pass_outcome_name': [None] * n,
        'under_pressure': rng.choice([True, False, None], n).tolist(),
        'pass_switch': [False] * n,
        'pass_cross': [False] * n,
        'pass_through_ball': [False] * n,
        'play_pattern_name': ['Regular Play'] * n,
        'has_360': [True] * 60 + [False] * 40,
        'n_visible_players': rng.integers(4, 20, n).tolist(),
        'n_visible_teammates': rng.integers(2, 10, n).tolist(),
        'n_visible_opponents': rng.integers(2, 10, n).tolist(),
        'strict_line_break': [None] * n,
        'loose_line_break': [None] * n,
        'dangerous_progression_k': [None] * n,
        'final_third_entry_k': [None] * n,
        'box_entry_k': [None] * n,
        'shot_within_k': [None] * n,
        'threat_gain': [None] * n,
    })


@pytest.fixture
def sample_frames_df():
    """Synthetic 360 freeze frames DataFrame."""
    rng = np.random.default_rng(42)
    rows = []

    for i in range(60):
        event_uuid = f'evt_{i:04d}'
        n_players = rng.integers(8, 18)
        for j in range(n_players):
            rows.append({
                'event_uuid': event_uuid,
                'player_id': j + 1,
                'player_name': f'Player_{j}',
                'teammate': j < n_players // 2,
                'actor': j == 0,
                'keeper': j == n_players - 1,
                'x': float(rng.uniform(0, 120)),
                'y': float(rng.uniform(0, 80)),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def label_config():
    """Label configuration matching configs/labels.yaml."""
    return {
        'line_break': {
            'min_forward_gain_m': 5.0,
            'min_y_advance_frac': 0.08,
            'defensive_layer_gap_m': 3.0,
            'open_play_only': True,
        },
        'dangerous_progression': {'k': 5},
        'final_third_entry': {'k': 5, 'final_third_x': 80.0},
        'box_entry': {'k': 5, 'box_x': 102.0, 'box_y_min': 18.0, 'box_y_max': 62.0},
        'shot_within': {'k': 5},
        'threat_gain': {'zone_grid_x': 12, 'zone_grid_y': 8},
    }


@pytest.fixture
def feature_config():
    """Feature configuration."""
    return {
        'graph': {
            'knn_k': 5,
            'node_features': ['x', 'y', 'teammate', 'is_keeper', 'is_actor', 'is_receiver', 'dist_to_goal', 'dist_to_passer'],
        }
    }


# ---------------------------------------------------------------------------
# Legacy fixtures kept for backward compatibility
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_pass_events() -> pd.DataFrame:
    """Minimal pass events DataFrame (legacy fixture)."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        "id": [f"evt-{i:04d}" for i in range(n)],
        "match_id": rng.integers(100, 110, size=n),
        "minute": rng.integers(1, 90, size=n),
        "second": rng.integers(0, 59, size=n),
        "start_x": rng.uniform(0, 120, size=n),
        "start_y": rng.uniform(0, 80, size=n),
        "end_x": rng.uniform(0, 120, size=n),
        "end_y": rng.uniform(0, 80, size=n),
        "pass_length": rng.uniform(1, 40, size=n),
        "pass_angle_rad": rng.uniform(-np.pi, np.pi, size=n),
        "under_pressure": rng.choice([True, False], size=n),
        "body_part": rng.choice(["foot", "head", "no_touch"], size=n),
        "pass_height": rng.choice(["ground", "low", "high"], size=n),
        "play_pattern": rng.choice(["regular", "set_piece", "counter"], size=n),
        "possession_length": rng.integers(1, 15, size=n),
    })


@pytest.fixture()
def synthetic_freeze_frames() -> list[list[dict]]:
    """Minimal freeze-frame list (legacy fixture)."""
    rng = np.random.default_rng(99)
    frames = []
    for _ in range(20):
        n_players = int(rng.integers(5, 11))
        frame = [
            {
                "x": float(rng.uniform(0, 120)),
                "y": float(rng.uniform(0, 80)),
                "teammate": bool(rng.choice([True, False])),
                "keeper": bool(i == 0),
            }
            for i in range(n_players)
        ]
        frames.append(frame)
    return frames
