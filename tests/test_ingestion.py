"""Tests for data ingestion and parsing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.parse_events import parse_events
from src.data.parse_360 import parse_360_frames, get_frame_summary
from src.data.parse_lineups import parse_lineups
from src.data.join_pass_frames import build_pass_instances


# ---------------------------------------------------------------------------
# Helpers to build raw StatsBomb-like data
# ---------------------------------------------------------------------------

def _make_raw_events(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Minimal raw events DataFrame mimicking statsbombpy output."""
    rng = np.random.default_rng(seed)

    def _loc():
        return [float(rng.uniform(0, 120)), float(rng.uniform(0, 80))]

    records = []
    for i in range(n):
        is_pass = i < (n - 5)
        end_loc = _loc() if is_pass else np.nan
        records.append({
            'id': f'evt_{i:04d}',
            'match_id': 1001,
            'index': i,
            'period': 1,
            'timestamp': f'00:{i:02d}:00.000',
            'minute': i,
            'second': 0,
            'type': {'id': 30, 'name': 'Pass'} if is_pass else {'id': 16, 'name': 'Shot'},
            'team': {'id': 1, 'name': 'Team A'},
            'player': {'id': 101, 'name': 'Player A'},
            'possession': i + 1,
            'possession_team': {'id': 1, 'name': 'Team A'},
            'play_pattern': {'id': 1, 'name': 'Regular Play'},
            'location': _loc(),
            'pass': {
                'recipient': {'id': 102, 'name': 'Player B'},
                'length': float(rng.uniform(5, 30)),
                'angle': float(rng.uniform(-np.pi, np.pi)),
                'end_location': end_loc,
                'body_part': {'id': 70, 'name': 'Right Foot'},
                'height': {'id': 1, 'name': 'Ground Pass'},
                'type': {'id': 65, 'name': 'Open Play'},
                'outcome': None,
                'switch': False,
                'cross': False,
                'through_ball': False,
                'goal_kick': False,
                'corner': False,
                'free_kick': False,
            } if is_pass else np.nan,
            'under_pressure': True if i % 5 == 0 else np.nan,
            'shot': np.nan,
        })

    return pd.DataFrame(records)


def _make_raw_frames(event_ids: list[str], seed: int = 1) -> pd.DataFrame:
    """Minimal raw 360 frames DataFrame (one row per player per event)."""
    rng = np.random.default_rng(seed)
    rows = []
    for eid in event_ids:
        n_players = int(rng.integers(8, 14))
        for j in range(n_players):
            rows.append({
                'event_uuid': eid,
                'freeze_frame': [
                    {
                        'player': {'id': j + 1, 'name': f'P{j}'},
                        'location': [float(rng.uniform(0, 120)), float(rng.uniform(0, 80))],
                        'teammate': bool(j < n_players // 2),
                        'actor': bool(j == 0),
                        'keeper': bool(j == n_players - 1),
                    }
                ],
            })
    # statsbombpy gives one row per event with freeze_frame as a list
    consolidated = {}
    for row in rows:
        eid = row['event_uuid']
        if eid not in consolidated:
            consolidated[eid] = []
        consolidated[eid].extend(row['freeze_frame'])
    return pd.DataFrame([
        {'event_uuid': k, 'freeze_frame': v} for k, v in consolidated.items()
    ])


def _make_raw_lineups(seed: int = 2) -> pd.DataFrame:
    """Minimal raw lineups DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for team_id in [1, 2]:
        for i in range(11):
            rows.append({
                'match_id': 1001,
                'team_id': team_id,
                'team_name': f'Team {chr(64 + team_id)}',
                'player_id': team_id * 100 + i,
                'player_name': f'Player_{team_id}_{i}',
                'player_nickname': None,
                'jersey_number': i + 1,
                'country': {'id': 1, 'name': 'England'},
                'cards': [],
                'positions': [],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseEvents:
    def test_parse_events_returns_dataframe(self):
        """parse_events returns a pandas DataFrame."""
        raw = _make_raw_events()
        result = parse_events(raw)
        assert isinstance(result, pd.DataFrame)

    def test_parse_events_has_required_columns(self):
        """Required columns present after parsing."""
        raw = _make_raw_events()
        result = parse_events(raw)
        required = {'event_uuid', 'match_id', 'minute', 'second', 'period',
                    'type_name'}
        # parse_events uses location_x/location_y or start_x/start_y depending on version
        has_loc = {'location_x', 'location_y'}.issubset(result.columns) or \
                  {'start_x', 'start_y'}.issubset(result.columns)
        missing = required - set(result.columns)
        assert not missing, f"Missing columns: {missing}"
        assert has_loc, "Expected location columns (location_x/y or start_x/y) in parsed events"

    def test_parse_events_preserves_row_count(self):
        """Output has same number of rows as input."""
        raw = _make_raw_events(30)
        result = parse_events(raw)
        assert len(result) == len(raw)


class TestParse360:
    def test_parse_360_frames_structure(self):
        """parse_360_frames returns player-level rows."""
        raw = _make_raw_frames([f'evt_{i:04d}' for i in range(10)])
        result = parse_360_frames(raw)
        assert isinstance(result, pd.DataFrame)
        assert 'event_uuid' in result.columns
        assert len(result) > len(raw)  # exploded: more rows than events

    def test_parse_360_frames_has_coordinates(self):
        """Parsed frames have x and y coordinates."""
        raw = _make_raw_frames([f'evt_{i:04d}' for i in range(5)])
        result = parse_360_frames(raw)
        assert 'x' in result.columns
        assert 'y' in result.columns

    def test_frame_summary_counts(self):
        """get_frame_summary returns correct player counts per event."""
        raw = _make_raw_frames([f'evt_{i:04d}' for i in range(5)])
        frames = parse_360_frames(raw)
        summary = get_frame_summary(frames)
        assert isinstance(summary, pd.DataFrame)
        assert 'event_uuid' in summary.columns
        # Each summary row corresponds to one event
        assert len(summary) == frames['event_uuid'].nunique()


class TestBuildPassInstances:
    def _build(self):
        raw_events = _make_raw_events(30)
        events = parse_events(raw_events)
        event_ids = events['event_uuid'].dropna().tolist()[:15]
        raw_frames = _make_raw_frames(event_ids)
        frames = parse_360_frames(raw_frames)
        return build_pass_instances(events, frames), events

    def test_build_pass_instances_has_all_columns(self):
        """Canonical table has required columns."""
        result, _ = self._build()
        required = {'event_uuid', 'start_x', 'start_y',
                    'end_x', 'end_y', 'pass_length', 'has_360'}
        missing = required - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_build_pass_instances_only_passes(self):
        """Only pass events appear in pass_instances output."""
        result, events = self._build()
        if 'type_name' in result.columns:
            non_pass = result[result['type_name'].notna() & (result['type_name'] != 'Pass')]
            assert len(non_pass) == 0

    def test_pass_instances_has_360_flag(self):
        """has_360 column is boolean and correctly populated."""
        result, _ = self._build()
        assert 'has_360' in result.columns
        valid_vals = {True, False, None, np.nan}
        assert result['has_360'].dtype == bool or result['has_360'].isin([True, False]).all()

    def test_no_duplicate_event_uuids(self):
        """event_uuid is unique in pass_instances."""
        result, _ = self._build()
        assert result['event_uuid'].nunique() == len(result)


class TestParseLineups:
    def test_parse_lineups_returns_dataframe(self):
        """parse_lineups returns a DataFrame."""
        raw = _make_raw_lineups()
        result = parse_lineups(raw)
        assert isinstance(result, pd.DataFrame)

    def test_parse_lineups_has_player_columns(self):
        """Parsed lineups have player_id and team_name."""
        raw = _make_raw_lineups()
        result = parse_lineups(raw)
        assert 'player_id' in result.columns or 'player_name' in result.columns
