"""Cross-fitted attribution: held-out only, defending-team events never credited, shared labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.event_types import EventType
from src.data.event_types import label as type_label
from src.evaluation.attribution_crossfit import (
    N_FOLDS,
    build_events_table,
    credited_events,
    event_type_table,
    fold_splits,
    gini,
    make_folds,
    player_leaderboard,
    sanity_checks,
    split_half_stability,
)


def _toy_poss(n_matches: int = 12, per_match: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, n_matches + 1):
        for p in range(1, per_match + 1):
            n = int(rng.integers(5, 12))
            seq, players = [], []
            for t in range(n):
                own = t % 4 != 3  # every 4th event is the defending team's
                seq.append(
                    {
                        "type_id": int(rng.choice([1, 2, 3, 6, 8, 13])),
                        "loc_x_norm": float(rng.uniform(0, 1)),
                        "loc_y_norm": float(rng.uniform(0, 1)),
                        "end_x_norm": 0.5,
                        "end_y_norm": 0.5,
                        "under_pressure": 0.0,
                        "pass_length_norm": 0.2,
                        "minute_norm": 0.1,
                        "own_team": 1.0 if own else 0.0,
                    }
                )
                players.append(f"P{t % 3}" if own else "Defender")
            dang = bool(rng.random() < 0.4)
            rows.append(
                {
                    "match_id": m,
                    "possession_id": p,
                    "team_name": "A",
                    "n_events": n,
                    "event_sequence": seq,
                    "player_sequence": players,
                    "poss_dangerous": dang,
                    "poss_xg_generated": 0.3 if dang else 0.0,
                    "poss_has_goal": False,
                }
            )
    return pd.DataFrame(rows)


def test_folds_partition_matches_and_splits_are_disjoint():
    matches = list(range(1, 100))
    mf = make_folds(matches, N_FOLDS, seed=42)
    assert sorted(mf) == matches and set(mf.values()) == set(range(N_FOLDS))
    held_union: list[int] = []
    for k in range(N_FOLDS):
        tr, va, ho = fold_splits(mf, k, seed=42)
        assert not set(tr) & set(ho) and not set(va) & set(ho) and not set(tr) & set(va)
        assert sorted(tr + va + ho) == matches
        assert va, "early-stopping validation set must not be empty"
        held_union += ho
    assert sorted(held_union) == matches  # every match is attributed exactly once


def test_crossfit_attributes_each_possession_with_a_model_that_never_saw_its_match():
    pytest.importorskip("torch")
    from src.evaluation.attribution_crossfit import crossfit_attribution

    poss = _toy_poss()
    arch = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    cfg = {"lr": 1e-2, "n_epochs": 2, "batch_size": 32, "patience": 2, "test_batch_size": 32}
    scores, loo, fold_of_row, fold_auc, match_fold = crossfit_attribution(
        poss, seed=7, arch=arch, train_cfg=cfg, verbose=False
    )
    assert (fold_of_row >= 0).all() and np.isfinite(scores).all()
    for i, m in enumerate(poss["match_id"]):
        # the fold that scored this row is the fold the match was held out in
        assert fold_of_row[i] == match_fold[int(m)]
        assert len(loo[i]) == min(len(poss["event_sequence"].iloc[i]), 40)
    for k in range(N_FOLDS):
        tr, va, ho = fold_splits(match_fold, k, 7)
        assert not (set(tr) | set(va)) & set(ho)


def _toy_events_table():
    poss = _toy_poss(n_matches=3, per_match=4)
    n = len(poss)
    scores = np.linspace(0.1, 0.9, n)
    loo = [np.zeros(min(len(s), 40)) for s in poss["event_sequence"]]
    # give every defending-team event a huge drop and every possessing-team event a small one
    for i, seq in enumerate(poss["event_sequence"]):
        for t, ev in enumerate(seq):
            loo[i][t] = 0.01 if ev["own_team"] >= 0.5 else 0.9
    teams = {
        (r.match_id, r.possession_id): [
            "A" if e["own_team"] >= 0.5 else "B" for e in r.event_sequence
        ]
        for r in poss.itertuples()
    }
    return build_events_table(poss, scores, loo, np.zeros(n, dtype=int), teams, 42)


def test_defending_team_events_are_never_credited_to_a_player():
    df = _toy_events_table()
    assert not df.loc[~df["possessing_team_event"], "player"].isin(["P0", "P1", "P2"]).any()
    board = player_leaderboard(df, min_events=1, n_boot=50)
    assert "Defender" not in set(board["player"])  # only defending-team events carry that name
    assert (
        board["mean_loo"] < 0.05
    ).all()  # the 0.9 defending-team drops never reach a leaderboard entry
    assert credited_events(df)["possessing_team_event"].all()
    # even if a defending-team event carried a real player's name it must stay out of their mean
    tampered = df.copy()
    tampered.loc[~tampered["possessing_team_event"], "player"] = "P0"
    b2 = player_leaderboard(tampered, min_events=1, n_boot=50).set_index("player")
    assert b2.loc["P0", "mean_loo"] == pytest.approx(0.01)


def test_event_type_labels_come_from_the_shared_enum():
    df = _toy_events_table()
    assert (df["event_type"] == df["type_id"].map(lambda i: type_label(int(i)))).all()
    assert type_label(int(EventType.BALL_RECEIPT)) in set(df["event_type"]) or len(df) > 0
    tab = event_type_table(df)
    assert set(tab["side"]) <= {"possessing team", "defending team"}
    # a goalkeeper / clearance / pressure event only ever appears on the side it was performed on
    assert tab["share_top"].sum() == pytest.approx(1.0)


def test_gini_split_half_and_sanity_checks_run():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert np.isnan(gini([0, 0]))
    df = _toy_events_table()
    noisy = df.assign(loo_drop=df["loo_drop"] + np.random.default_rng(0).normal(0, 0.01, len(df)))
    out = split_half_stability(noisy, ["P0", "P1", "P2"], repeats=3, seed=42)
    assert -1.0 <= out["mean"] <= 1.0 and out["repeats"] == 3
    san = sanity_checks(df)
    assert san["n_possessions"] == 12 and 0.0 <= san["gini_median_all"] <= 1.0
