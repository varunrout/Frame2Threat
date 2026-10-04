"""Tests for the event-time (in-play) formulation and the possession coordinate frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.event_types import EventType
from src.data.possession_frame import (
    in_box,
    is_danger_event,
    own_team_flags,
    possessing_team,
    to_possession_frame,
)
from src.evaluation.event_time_eval import (
    bootstrap_by_match,
    metrics,
    one_snapshot_per_possession,
    reliability_table,
    threshold_matching_prevalence,
)
from src.features.event_time import (
    FEATURE_NAMES,
    N_SEQ_NUMERIC,
    PossessionArrays,
    build_snapshot_table,
    sequence_inputs,
    snapshot_features,
    snapshot_labels,
)

P, C, R, S, PR, D = (
    int(EventType.PASS),
    int(EventType.CARRY),
    int(EventType.BALL_RECEIPT),
    int(EventType.SHOT),
    int(EventType.PRESSURE),
    int(EventType.DUEL),
)


def make_arrays(types, xs, ys, own=None, seconds=None, end_xs=None, pid=1, mid=1):
    n = len(types)
    own = np.ones(n, dtype=bool) if own is None else np.asarray(own, dtype=bool)
    seconds = np.arange(n, dtype=float) * 2.0 if seconds is None else np.asarray(seconds, float)
    end_x = np.full(n, np.nan) if end_xs is None else np.asarray(end_xs, float)
    return PossessionArrays(
        match_id=mid,
        possession_id=pid,
        team="A",
        period=1,
        minute=10.0,
        origin="Regular Play",
        type_id=np.asarray(types, dtype=int),
        own=own,
        x=np.asarray(xs, float),
        y=np.asarray(ys, float),
        end_x=end_x,
        end_y=np.where(np.isnan(end_x), np.nan, 40.0),
        under_pressure=np.zeros(n),
        pass_length=np.where(np.asarray(types) == P, 12.0, np.nan),
        abs_s=seconds,
    )


# ---------------------------------------------------------------- coordinate frame


def test_flip_and_box():
    fx, fy = to_possession_frame([10.0, 10.0], [20.0, 20.0], [True, False])
    assert fx.tolist() == [10.0, 110.0] and fy.tolist() == [20.0, 60.0]
    assert in_box(105.0, 40.0) and not in_box(101.0, 40.0) and not in_box(105.0, 10.0)


def test_opponent_event_in_its_own_box_zone_is_not_danger():
    # opponent pressure at x=110 (opponent frame) is near the possessing team's OWN goal
    assert not is_danger_event(PR, 110.0, 40.0, False)
    assert is_danger_event(PR, 110.0, 40.0, True)
    assert is_danger_event(S, 50.0, 40.0, True)
    assert not is_danger_event(S, 50.0, 40.0, False)


def test_possessing_team_majority_and_flags():
    g = pd.DataFrame(
        {
            "type_name": ["Pressure", "Pass", "Carry", "Pressure", "Pass"],
            "team_name": ["B", "A", "A", "B", "A"],
        }
    )
    assert possessing_team(g) == "A"
    assert own_team_flags(g).tolist() == [False, True, True, False, True]


# ---------------------------------------------------------------- eligibility / labels


def test_eligibility_stops_at_first_danger_and_label_is_future_looking():
    p = make_arrays([P, C, P, C, S, P], [30, 40, 50, 60, 110, 50], [40] * 6)
    elig, lab, term = snapshot_labels(p)
    assert elig.tolist() == [True, True, True, True, False, False]  # shot at index 4
    assert lab[:4].tolist() == [1, 1, 1, 1]
    assert not term.any()


def test_safe_possession_has_n_minus_1_rows_and_one_terminal_row():
    p = make_arrays([P, C, P, C], [30, 35, 40, 45], [40] * 4)
    elig, lab, term = snapshot_labels(p)
    assert elig.tolist() == [True, True, True, False]
    assert lab.tolist() == [0, 0, 0, 0] and term.tolist() == [False, False, False, True]


def test_box_entry_by_opponent_event_does_not_count():
    own = [True, True, False, True, True]
    p = make_arrays([P, C, PR, P, C], [30, 40, 110, 50, 60], [40] * 5, own=own)
    elig, lab, _ = snapshot_labels(p)
    assert lab.max() == 0 and elig[:4].all()


# ---------------------------------------------------------------- no future information


def _random_possession(rng, n, pid=1):
    types = rng.choice([P, C, R, PR, D], size=n)
    own = rng.random(n) < 0.8
    xs = rng.uniform(0, 120, n)
    ys = rng.uniform(0, 80, n)
    xs[rng.random(n) < 0.1] = np.nan
    end_x = np.where(types == P, rng.uniform(0, 120, n), np.nan)
    secs = np.cumsum(rng.uniform(0.2, 4.0, n))
    return make_arrays(types, xs, ys, own=own, seconds=secs, end_xs=end_x, pid=pid)


@pytest.mark.parametrize("seed", range(5))
def test_features_at_t_unchanged_by_future_events(seed):
    rng = np.random.default_rng(seed)
    short = _random_possession(rng, 9)
    longer = short.truncated(9)
    # build a longer possession whose first 9 events are identical, then arbitrary futures
    extra = _random_possession(rng, 14)
    for f in ("type_id", "own", "x", "y", "end_x", "end_y", "under_pressure", "pass_length"):
        setattr(extra, f, np.concatenate([getattr(short, f), getattr(extra, f)[9:]]))
    extra.abs_s = np.concatenate([short.abs_s, short.abs_s[-1] + np.cumsum(rng.uniform(0.1, 9, 5))])
    a = snapshot_features(short)
    b = snapshot_features(extra)[:9]
    np.testing.assert_array_equal(a, b)
    ta, na = sequence_inputs(short)
    tb, nb = sequence_inputs(extra)
    np.testing.assert_array_equal(ta, tb[:9])
    np.testing.assert_array_equal(na, nb[:9])
    assert longer.n == 9


def test_feature_names_match_matrix_and_no_length_proxy_beyond_so_far():
    p = _random_possession(np.random.default_rng(0), 7)
    assert snapshot_features(p).shape == (7, len(FEATURE_NAMES))
    assert not any(c in FEATURE_NAMES for c in ("n_events", "frac", "duration_seconds", "n"))
    _, num = sequence_inputs(p)
    assert num.shape == (7, N_SEQ_NUMERIC)


def test_elapsed_seconds_come_from_timestamps():
    p = make_arrays([P, C, P], [30, 40, 50], [40] * 3, seconds=[100.0, 103.5, 110.0])
    f = snapshot_features(p)
    col = FEATURE_NAMES.index("elapsed_s")
    assert f[:, col].tolist() == [0.0, 3.5, 10.0]


def test_cur_xy_ignores_opponent_events_and_max_x_is_monotone():
    p = make_arrays([P, PR, C, P], [30, 110, 45, 70], [40, 20, 41, 42], own=[1, 0, 1, 1])
    f = snapshot_features(p)
    cx, mx = FEATURE_NAMES.index("cur_x"), FEATURE_NAMES.index("max_x_so_far")
    assert f[:, cx].tolist() == [30.0, 30.0, 45.0, 70.0]
    assert np.all(np.diff(f[:, mx]) >= 0) and f[1, mx] == 30.0


def test_snapshot_table_rows_match_eligibility():
    arrs = [
        make_arrays([P, C, S, P], [30, 40, 100, 50], [40] * 4, pid=1),
        make_arrays([P, C, P], [30, 35, 40], [40] * 3, pid=2),
    ]
    t = build_snapshot_table(arrs)
    assert len(t[~t.terminal]) == 2 + 2 and t[t.terminal].shape[0] == 1
    assert t[t.possession_id == 1].y.tolist() == [1, 1]


# ---------------------------------------------------------------- GRU causality


def test_gru_output_at_t_depends_only_on_prefix():
    torch = pytest.importorskip("torch")
    from src.models.event_time import EventTimeGRU

    torch.manual_seed(0)
    model = EventTimeGRU(N_SEQ_NUMERIC).eval()
    p = _random_possession(np.random.default_rng(3), 12)
    tid, num = sequence_inputs(p)
    full = model(torch.from_numpy(tid)[None], torch.from_numpy(num)[None])[0]
    for k in (3, 7):
        pre = model(torch.from_numpy(tid[:k])[None], torch.from_numpy(num[:k])[None])[0]
        assert torch.allclose(full[:k], pre, atol=1e-5)
    assert sum(p_.numel() for p_ in model.gru.parameters()) > 0 and model.gru.num_layers == 1
    assert model.gru.hidden_size == 64


# ---------------------------------------------------------------- evaluation helpers


def _rows():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(20), 30),
            "possession_id": np.tile(np.repeat(np.arange(6), 5), 20),
            "y": rng.integers(0, 2, n),
        }
    )


def test_one_snapshot_per_possession_is_unique_and_seeded():
    rows = _rows()
    a = one_snapshot_per_possession(rows, 1)
    b = one_snapshot_per_possession(rows, 1)
    assert (a == b).all() and len(a) == rows.groupby(["match_id", "possession_id"]).ngroups
    assert not (a == one_snapshot_per_possession(rows, 2)).all()


def test_metrics_reliability_threshold_bootstrap():
    rows = _rows()
    rng = np.random.default_rng(1)
    p = np.clip(rows.y.to_numpy() * 0.3 + rng.random(len(rows)) * 0.7, 0, 1)
    m = metrics(rows.y.to_numpy(), p)
    assert 0.5 < m["roc_auc"] <= 1 and 0 < m["brier"] < 1
    assert np.isnan(metrics(np.zeros(5), np.full(5, 0.2))["roc_auc"])
    rel = reliability_table(rows.y.to_numpy(), p)
    assert rel.n.sum() == len(rows)
    thr = threshold_matching_prevalence(rows.y.to_numpy(), p)
    assert abs((p >= thr).mean() - rows.y.mean()) < 0.02
    boot = bootstrap_by_match(
        rows, {"a": p, "b": rng.random(len(rows))}, deltas=[("a", "b")], n_boot=30, seed=0
    )
    row = boot[(boot.model == "a - b") & (boot.stat == "roc_auc")].iloc[0]
    assert row.ci_low <= row.estimate <= row.ci_high or row.ci_low <= row.ci_high


# ---------------------------------------------------------------- parse_possessions frame fix


def test_parse_possessions_excludes_opponent_events_from_labels():
    from src.data.parse_possessions import build_possession_sequences

    rows = []
    spec = [
        ("A", "Pass", 30.0),
        ("A", "Carry", 40.0),
        ("B", "Pressure", 112.0),  # opponent frame: really near A's own goal
        ("A", "Pass", 50.0),
    ]
    for i, (team, name, x) in enumerate(spec):
        rows.append(
            {
                "match_id": 1,
                "possession_id": 1,
                "index": i + 1,
                "period": 1,
                "minute": 1,
                "second": i,
                "type_name": name,
                "team_name": team,
                "play_pattern_name": "Regular Play",
                "location_x": x,
                "location_y": 40.0,
                "under_pressure": False,
                "pass_end_x": np.nan,
                "pass_end_y": np.nan,
                "pass_length": np.nan,
            }
        )
    out = build_possession_sequences(pd.DataFrame(rows), min_events=2)
    r = out.iloc[0]
    assert r["team_name"] == "A"
    assert r["max_x_reached"] == 50.0
    assert not r["poss_entered_box"] and not r["poss_dangerous"]
    flags = [e["own_team"] for e in r["event_sequence"]]
    assert flags == [1.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------- live scoring input


def test_live_events_to_arrays_defaults_and_timestamps():
    from src.app.live_scoring import events_to_arrays

    ev = [
        {
            "type_id": int(EventType.PASS),
            "loc_x_norm": 0.25,
            "loc_y_norm": 0.5,
            "end_x_norm": 0.5,
            "end_y_norm": 0.5,
            "pass_length_norm": 0.2,
        },
        {"type_id": int(EventType.CARRY), "loc_x_norm": 0.5, "loc_y_norm": 0.5, "own_team": 0.0},
    ]
    arr, warns = events_to_arrays(ev, median_dt=2.0)
    assert warns and arr.abs_s.tolist() == [0.0, 2.0] and arr.own.tolist() == [True, False]
    assert arr.x.tolist() == [30.0, 60.0] and arr.end_x[0] == 60.0 and np.isnan(arr.end_x[1])
    for e in ev:
        e["elapsed_seconds"] = 5.0
    arr2, warns2 = events_to_arrays(ev)
    assert not warns2 and arr2.abs_s.tolist() == [5.0, 5.0]


# ---------------------------------------------------------------- same-moment leak (pass ending in the box)


def test_pass_ending_in_box_is_a_danger_event_and_ends_eligibility():
    # the only danger is a pass that ends in the box; no event LOCATION is ever in the box
    p = make_arrays(
        [P, P, P],
        [30.0, 50.0, 60.0],
        [40.0, 40.0, 40.0],
        end_xs=[50.0, 106.0, 70.0],
    )
    elig, lab, term = snapshot_labels(p)
    assert elig.tolist() == [True, False, False]  # nothing at or after the pass is eligible
    assert lab[0] == 1 and not term.any()


def test_pass_ending_in_box_without_receipt_and_with_receipt_behave_the_same():
    no_receipt = make_arrays([P, C, P], [30, 40, 50], [40] * 3, end_xs=[40, np.nan, 105])
    with_receipt = make_arrays(
        [P, C, P, R], [30, 40, 50, 105], [40] * 4, end_xs=[40, np.nan, 105, np.nan]
    )
    assert snapshot_labels(no_receipt)[0].tolist() == [True, True, False]
    assert snapshot_labels(with_receipt)[0].tolist() == [True, True, False, False]


def test_opponent_pass_or_pass_ending_outside_box_is_not_danger():
    opp = make_arrays([P, P, P], [30, 50, 60], [40] * 3, own=[1, 0, 1], end_xs=[40, 106, 70])
    assert snapshot_labels(opp)[1].max() == 0
    outside = make_arrays([P, P, P], [30, 50, 60], [40] * 3, end_xs=[40, 101, 70])
    assert snapshot_labels(outside)[1].max() == 0
    assert is_danger_event(P, 50.0, 40.0, True, 106.0, 40.0)
    assert not is_danger_event(P, 50.0, 40.0, True)  # no end location: location-only rule
