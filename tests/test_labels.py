"""Tests for label computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels.line_break import compute_line_break_labels, _count_defenders_between
from src.labels.dangerous_progression import compute_downstream_labels
from src.labels.downstream_outcomes import compute_threat_gain
from src.labels.validation_sampling import label_prevalence_table, label_sanity_checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events_for_possession(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Minimal parsed events DataFrame with possession-level structure."""
    rng = np.random.default_rng(seed)
    rows = []
    poss_id = 0
    for i in range(n):
        if i % 5 == 0:
            poss_id += 1
        type_name = "Pass" if i % 5 != 4 else "Shot"
        start_x = float(rng.uniform(30, 90))
        start_y = float(rng.uniform(20, 60))
        rows.append(
            {
                "event_uuid": f"evt_{i:04d}",
                "match_id": 1001,
                "index": i,
                "possession_id": poss_id,
                "period": 1,
                "minute": i,
                "second": 0,
                "type_name": type_name,
                "location_x": start_x,
                "location_y": start_y,
                "pass_end_x": (
                    start_x + float(rng.uniform(5, 20)) if type_name == "Pass" else np.nan
                ),
                "pass_end_y": (
                    start_y + float(rng.uniform(-5, 5)) if type_name == "Pass" else np.nan
                ),
                "start_x": start_x,
                "start_y": start_y,
                "end_x": start_x + float(rng.uniform(5, 20)) if type_name == "Pass" else np.nan,
                "end_y": start_y + float(rng.uniform(-5, 5)) if type_name == "Pass" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _make_pass_instances(events_df: pd.DataFrame) -> pd.DataFrame:
    """Derive pass_instances from events (only pass rows)."""
    passes = events_df[events_df["type_name"] == "Pass"].copy()
    rng = np.random.default_rng(0)
    n = len(passes)
    passes["team_name"] = "Team A"
    passes["player_name"] = "Player A"
    passes["pass_recipient_name"] = "Player B"
    passes["pass_length"] = rng.uniform(5, 30, n)
    passes["pass_angle"] = rng.uniform(-3.14, 3.14, n)
    passes["pass_body_part"] = "Right Foot"
    passes["pass_height"] = "Ground Pass"
    passes["pass_type"] = "Open Play"
    passes["pass_outcome_name"] = None
    passes["under_pressure"] = False
    passes["pass_switch"] = False
    passes["pass_cross"] = False
    passes["pass_through_ball"] = False
    passes["play_pattern_name"] = "Regular Play"
    passes["has_360"] = [True if i < n // 2 else False for i in range(n)]
    passes["competition_id"] = 43
    passes["season_id"] = 3
    passes["n_visible_players"] = 14
    passes["n_visible_teammates"] = 6
    passes["n_visible_opponents"] = 7
    return passes.reset_index(drop=True)


def _make_frames_df(event_uuids: list[str], seed: int = 7) -> pd.DataFrame:
    """Minimal parsed freeze frames with opponent and teammate rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for eid in event_uuids:
        for j in range(10):
            rows.append(
                {
                    "event_uuid": eid,
                    "player_id": j + 1,
                    "teammate": j < 5,
                    "actor": j == 0,
                    "keeper": j == 9,
                    "x": float(rng.uniform(20, 100)),
                    "y": float(rng.uniform(10, 70)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _count_defenders_between unit tests
# ---------------------------------------------------------------------------


class TestCountDefendersBetween:
    def test_basic_count(self):
        """Counts defenders whose x is strictly between start_x and end_x."""
        defenders = pd.DataFrame({"x": [30.0, 45.0, 60.0, 80.0]})
        assert _count_defenders_between(defenders, 35.0, 65.0) == 2

    def test_empty_defenders(self):
        """Returns 0 for empty defenders DataFrame."""
        defenders = pd.DataFrame({"x": []})
        assert _count_defenders_between(defenders, 10.0, 50.0) == 0

    def test_backward_pass(self):
        """Returns 0 when end_x <= start_x (backward or lateral pass)."""
        defenders = pd.DataFrame({"x": [30.0, 40.0]})
        assert _count_defenders_between(defenders, 50.0, 20.0) == 0

    def test_boundary_exclusive(self):
        """Boundary values are excluded (strictly between)."""
        defenders = pd.DataFrame({"x": [20.0, 50.0, 80.0]})
        assert _count_defenders_between(defenders, 20.0, 80.0) == 1


# ---------------------------------------------------------------------------
# Line break label tests
# ---------------------------------------------------------------------------


class TestLineBreakLabels:
    def _run(self, seed: int = 0):
        events = _make_events_for_possession(30, seed=seed)
        passes = _make_pass_instances(events)
        uuids_with_360 = passes[passes["has_360"]]["event_uuid"].tolist()
        frames = _make_frames_df(uuids_with_360, seed=seed)
        config = {"min_forward_gain_m": 5.0, "open_play_only": False}
        return compute_line_break_labels(passes, frames, config)

    def test_strict_leq_loose(self):
        """strict_line_break is always <= loose_line_break (no case where strict=True, loose=False)."""
        result = self._run()
        strict = result["strict_line_break"]
        loose = result["loose_line_break"]
        known_mask = strict.notna() & loose.notna()
        if known_mask.sum() > 0:
            violations = ((strict == True) & (loose == False))[known_mask]
            assert not violations.any(), "strict_line_break is True but loose_line_break is False"

    def test_line_break_no_360_gets_nan(self):
        """Passes without 360 data receive NaN for both label columns."""
        events = _make_events_for_possession(30)
        passes = _make_pass_instances(events)
        frames = _make_frames_df([], seed=0)  # No frames at all
        config = {"min_forward_gain_m": 5.0, "open_play_only": False}
        result = compute_line_break_labels(passes, frames, config)
        no_360 = result[~result["has_360"]]
        assert no_360["strict_line_break"].isna().all()
        assert no_360["loose_line_break"].isna().all()

    def test_output_is_copy_with_new_columns(self):
        """Result is a DataFrame with strict_line_break and loose_line_break columns."""
        result = self._run()
        assert "strict_line_break" in result.columns
        assert "loose_line_break" in result.columns


# ---------------------------------------------------------------------------
# Downstream label tests
# ---------------------------------------------------------------------------


class TestDownstreamLabels:
    def _run(self):
        events = _make_events_for_possession(40, seed=1)
        passes = _make_pass_instances(events)
        config = {"dangerous_progression": {"k": 5}}
        return compute_downstream_labels(events, passes, config), events, passes

    def test_all_columns_present(self):
        """All 4 downstream label columns are added."""
        result, _, _ = self._run()
        expected = {
            "final_third_entry_k",
            "box_entry_k",
            "shot_within_k",
            "dangerous_progression_k",
        }
        missing = expected - set(result.columns)
        assert not missing, f"Missing label columns: {missing}"

    def test_downstream_labels_no_future_leakage(self):
        """Each label only uses events within the same possession."""
        result, events, _ = self._run()
        # Spot check: for passes with no more events in possession, dp should be False
        last_events = events.groupby("possession_id")["index"].max()
        last_event_uuids = events[events["index"].isin(last_events.values)]["event_uuid"].tolist()
        if last_event_uuids:
            last_passes = result[result["event_uuid"].isin(last_event_uuids)]
            if len(last_passes) > 0:
                # The last event of a possession cannot have any future events
                assert not last_passes["shot_within_k"].any()

    def test_dangerous_progression_is_union(self):
        """dangerous_progression_k = ft_entry OR box_entry OR shot."""
        result, _, _ = self._run()
        manual_union = (
            result["final_third_entry_k"].fillna(False).astype(bool)
            | result["box_entry_k"].fillna(False).astype(bool)
            | result["shot_within_k"].fillna(False).astype(bool)
        )
        computed = result["dangerous_progression_k"].fillna(False).astype(bool)
        assert (manual_union == computed).all()

    def test_no_nan_in_downstream_labels(self):
        """Downstream labels are all bool (no NaN) per design."""
        result, _, _ = self._run()
        for col in [
            "final_third_entry_k",
            "box_entry_k",
            "shot_within_k",
            "dangerous_progression_k",
        ]:
            assert not result[col].isna().any(), f"{col} contains NaN"


# ---------------------------------------------------------------------------
# Threat gain tests
# ---------------------------------------------------------------------------


class TestThreatGain:
    def _run(self):
        events = _make_events_for_possession(40, seed=2)
        passes = _make_pass_instances(events)
        config = {"zone_grid_x": 12, "zone_grid_y": 8}
        return compute_threat_gain(passes, events, config)

    def test_threat_gain_column_added(self):
        """threat_gain column is in output."""
        result = self._run()
        assert "threat_gain" in result.columns

    def test_threat_gain_range(self):
        """threat_gain values are in roughly [-1, 1]."""
        result = self._run()
        vals = result["threat_gain"].dropna()
        assert (vals >= -1.1).all(), "Some threat_gain < -1"
        assert (vals <= 1.1).all(), "Some threat_gain > 1"

    def test_threat_gain_numeric(self):
        """threat_gain is numeric dtype."""
        result = self._run()
        assert pd.api.types.is_float_dtype(result["threat_gain"])


# ---------------------------------------------------------------------------
# Validation / sampling tests
# ---------------------------------------------------------------------------


class TestLabelPrevalence:
    def _make_labelled_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame(
            {
                "event_uuid": [f"e{i}" for i in range(n)],
                "strict_line_break": rng.choice([True, False, None], n).tolist(),
                "loose_line_break": rng.choice([True, False, None], n).tolist(),
                "dangerous_progression_k": rng.choice([True, False], n).tolist(),
                "final_third_entry_k": rng.choice([True, False], n).tolist(),
                "box_entry_k": rng.choice([True, False], n).tolist(),
                "shot_within_k": rng.choice([True, False], n).tolist(),
                "threat_gain": rng.uniform(-0.5, 0.5, n),
                "start_x": rng.uniform(20, 90, n),
                "start_y": rng.uniform(10, 70, n),
            }
        )

    def test_label_prevalence_table_runs(self):
        """label_prevalence_table returns a DataFrame without error."""
        df = self._make_labelled_df()
        result = label_prevalence_table(df)
        assert isinstance(result, pd.DataFrame)

    def test_label_prevalence_table_columns(self):
        """label_prevalence_table has required columns."""
        df = self._make_labelled_df()
        result = label_prevalence_table(df)
        if len(result) > 0:
            expected_cols = {"label", "n_total", "n_positive", "prevalence"}
            assert expected_cols.issubset(set(result.columns))

    def test_sanity_checks_pass_on_clean_data(self):
        """label_sanity_checks passes on well-formed data."""
        df = self._make_labelled_df()
        # Make strict <= loose to avoid any sanity failures
        df["strict_line_break"] = False
        df["loose_line_break"] = True
        df["dangerous_progression_k"] = (
            df["final_third_entry_k"] | df["box_entry_k"] | df["shot_within_k"]
        )
        checks = label_sanity_checks(df)
        assert isinstance(checks, dict)


# ---------------------------------------------------------------------------
# progression_beyond_pass_k  (non-tautological pass label)
# ---------------------------------------------------------------------------

from src.labels.progression_beyond_pass import (  # noqa: E402
    compute_progression_beyond_pass,
    label_window,
    zone_index,
)

_NAN = float("nan")


def _ev(type_, team="A", x=_NAN, y=40.0, end_x=_NAN, end_y=_NAN):
    return {"type": type_, "team": team, "x": x, "y": y, "end_x": end_x, "end_y": end_y}


def test_zone_index_ordering():
    assert [zone_index(10, 40), zone_index(50, 40), zone_index(90, 40), zone_index(110, 40)] == [
        0,
        1,
        2,
        3,
    ]
    assert zone_index(110, 5) == 2  # wide of the box is still final third


def test_receipt_of_the_pass_alone_never_makes_a_positive():
    # pass ends at x=95; the only follow-up is its receipt at the end location
    win = [_ev("ball receipt*", x=95.0), _ev("pressure", team="B", x=20.0)]
    assert label_window("A", (95.0, 40.0), win) == 0


def test_label_depends_on_events_after_the_receipt():
    win = [_ev("ball receipt*", x=60.0), _ev("carry", x=60.0), _ev("shot", x=60.0)]
    assert label_window("A", (60.0, 40.0), win) == 1


def test_opponent_events_are_ignored_even_in_the_box_zone():
    win = [_ev("ball receipt*", x=60.0), _ev("pressure", team="B", x=110.0), _ev("shot", team="B")]
    assert label_window("A", (60.0, 40.0), win) == 0


def test_box_entry_only_counts_if_pass_did_not_already_end_in_the_box():
    win = [_ev("ball receipt*", x=105.0), _ev("pass", x=105.0, end_x=108.0, end_y=40.0)]
    assert label_window("A", (105.0, 40.0), win) == 0  # already in the box at the pass end
    win2 = [_ev("ball receipt*", x=95.0), _ev("pass", x=95.0, end_x=106.0, end_y=40.0)]
    assert label_window("A", (95.0, 40.0), win2) == 1  # enters the box beyond the pass end


def test_progression_needs_ten_metres_and_a_more_advanced_zone():
    base = [_ev("ball receipt*", x=50.0)]
    assert label_window("A", (50.0, 40.0), base + [_ev("pass", x=50, end_x=62.0, end_y=40.0)]) == 0
    assert label_window("A", (50.0, 40.0), base + [_ev("pass", x=50, end_x=79.0, end_y=40.0)]) == 0
    assert label_window("A", (50.0, 40.0), base + [_ev("pass", x=50, end_x=82.0, end_y=40.0)]) == 1
    # +12 m but still in the same zone (final third) -> negative
    base2 = [_ev("ball receipt*", x=82.0)]
    assert label_window("A", (82.0, 40.0), base2 + [_ev("pass", x=82, end_x=94.0, end_y=40.0)]) == 0


def test_a_second_receipt_is_not_dropped_and_missing_receipt_is_fine():
    # receipt is only dropped before any own ball action; a later receipt keeps its position
    win = [_ev("pass", x=50.0, end_x=50.0, end_y=40.0), _ev("ball receipt*", x=105.0, y=40.0)]
    assert label_window("A", (50.0, 40.0), win) == 1
    assert label_window("A", (50.0, 40.0), []) == 0


def _toy_events():
    rows = []
    spec = [
        ("Pass", "A", 50.0, 95.0),  # idx 0: target pass, ends x=95
        ("Ball Receipt*", "A", 95.0, _NAN),
        ("Carry", "A", 95.0, _NAN),
        ("Pass", "A", 95.0, 96.0),
        ("Ball Receipt*", "A", 96.0, _NAN),
        ("Carry", "A", 96.0, _NAN),
        ("Shot", "A", 100.0, _NAN),  # idx 6: beyond k=5 slots after the pass
    ]
    for i, (t, team, x, ex) in enumerate(spec):
        rows.append(
            {
                "event_uuid": f"u{i}",
                "match_id": 1,
                "possession_id": 1,
                "index": i + 1,
                "type_name": t,
                "team_name": team,
                "location_x": x,
                "location_y": 40.0,
                "pass_end_x": ex,
                "pass_end_y": 40.0 if ex == ex else _NAN,
            }
        )
    return pd.DataFrame(rows)


def test_window_is_exactly_k_events_and_ignores_the_past():
    ev = _toy_events()
    passes = pd.DataFrame({"event_uuid": ["u0"], "match_id": [1], "possession_id": [1]})
    out = compute_progression_beyond_pass(ev, passes, {"k": 5})
    assert not out["progression_beyond_pass_k"].iloc[0]  # the shot is the 6th event after
    out6 = compute_progression_beyond_pass(ev, passes, {"k": 6})
    assert out6["progression_beyond_pass_k"].iloc[0]
    # events before the pass and after the window never change the label
    ev2 = pd.concat([ev.assign(index=ev["index"] + 10), ev.iloc[:0]], ignore_index=True)
    shot_before = ev2.iloc[[6]].assign(event_uuid="b0", index=1)
    ev3 = pd.concat([shot_before, ev2], ignore_index=True)
    out3 = compute_progression_beyond_pass(ev3, passes, {"k": 5})
    assert not out3["progression_beyond_pass_k"].iloc[0]


def test_old_label_would_have_been_positive_for_the_same_pass():
    # sanity of the motivation: the old definition fires on the receipt at the pass end
    from src.labels.dangerous_progression import compute_downstream_labels

    ev = _toy_events()
    passes = pd.DataFrame({"event_uuid": ["u0"], "match_id": [1], "possession_id": [1]})
    old = compute_downstream_labels(ev, passes, {"k": 5})
    assert bool(old["dangerous_progression_k"].iloc[0])  # receipt at x=95 is a final-third entry


def test_last_pass_of_possession_is_negative():
    ev = _toy_events()
    passes = pd.DataFrame({"event_uuid": ["u3"], "match_id": [1], "possession_id": [1]})
    out = compute_progression_beyond_pass(ev.iloc[:4], passes, {"k": 5})
    assert not out["progression_beyond_pass_k"].iloc[0]
