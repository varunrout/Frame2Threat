"""
Pytest configuration and shared fixtures for the Frame2Threat test suite.

Fixtures defined here are available to all test modules without explicit import.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Minimal synthetic pass event DataFrame (no StatsBomb API call required)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_pass_events() -> pd.DataFrame:
    """
    Return a small DataFrame that mimics the schema produced by
    ``src.data.loader`` after filtering for pass events with 360 data.

    Columns mirror the StatsBomb flat event format used throughout the project.
    """
    rng = np.random.default_rng(42)
    n = 20

    df = pd.DataFrame(
        {
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
        }
    )
    return df


# ---------------------------------------------------------------------------
# Minimal synthetic freeze-frame list
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_freeze_frames() -> list[list[dict]]:
    """
    Return a list of 20 freeze-frame snapshots, each containing between 5 and
    10 player entries with ``x``, ``y``, ``teammate``, and ``keeper`` keys.
    """
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
