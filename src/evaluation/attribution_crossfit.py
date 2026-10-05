"""
attribution_crossfit.py
=======================
Retrospective event attribution with **cross-fitting** (out-of-fold) and the leaderboards,
reliability checks and sanity checks built on it.

Why cross-fitting
-----------------
The v2 attribution scored possessions with a GRU that had been trained on them (in-sample).
Here the 99 matches are split into ``N_FOLDS`` match-grouped folds; for each fold a GRU is
trained on the other folds (with an inner match-level split for early stopping) and leave-one-out
(LOO) attribution is computed **only** on the held-out fold's possessions.  Every possession is
therefore attributed by a model that never saw it (or any other possession of its match).

What the numbers mean
---------------------
LOO drop = score(full possession) - score(possession with event t zeroed).  It measures which
events the *retrospective* model relies on to call a possession dangerous.  It is not causal
credit: the zeroed-event sequences are unrealistic, and defending-team events (pressure,
clearance, block, goalkeeper ...) act as *evidence* of danger rather than attacking actions.
Only the first ``MAX_SEQ_LEN`` (40) events of a possession are seen by the GRU and attributed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.event_types import label as type_label

N_FOLDS = 5
MAX_SEQ_LEN = 40
UNKNOWN_PLAYER = "Unknown"


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def make_folds(match_ids, n_folds: int = N_FOLDS, seed: int = 42) -> dict[int, int]:
    """Assign every match to one of ``n_folds`` folds (shuffled round-robin, seeded)."""
    ids = np.array(sorted({int(m) for m in match_ids}))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {int(m): i % n_folds for i, m in enumerate(ids)}


def fold_splits(
    match_fold: dict[int, int], k: int, seed: int = 42, val_frac: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train_matches, val_matches, heldout_matches)`` for fold ``k``.

    The held-out fold is attributed; the remaining matches are split by match into train and an
    early-stopping validation set.  The three lists are disjoint.
    """
    held = sorted(m for m, f in match_fold.items() if f == k)
    rest = sorted(m for m, f in match_fold.items() if f != k)
    rng = np.random.default_rng(seed + 1000 * (k + 1))
    rng.shuffle(rest)
    n_val = max(1, int(round(val_frac * len(rest))))
    return sorted(rest[n_val:]), sorted(rest[:n_val]), held


# ---------------------------------------------------------------------------
# Cross-fitted scoring + LOO (needs torch)
# ---------------------------------------------------------------------------


def crossfit_attribution(
    poss: pd.DataFrame,
    *,
    seed: int = 42,
    n_folds: int = N_FOLDS,
    arch: dict,
    train_cfg: dict,
    label: str = "poss_dangerous",
    verbose: bool = True,
):
    """Train fold models and attribute held-out possessions only.

    Returns ``(scores, loo, fold_of_row, fold_auc, match_fold)`` where ``scores`` is the GRU
    probability for each possession (row order of ``poss``), ``loo`` a list of per-event LOO drops
    (length ``min(n_events, 40)``) and ``fold_of_row`` the fold whose model scored the row.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    from src.evaluation.possession_attribution import leave_one_out_attribution
    from src.features.possession_features import build_sequence_tensors
    from src.models.gru_possession import evaluate, make_dataloader, train_gru

    match_fold = make_folds(poss["match_id"].unique(), n_folds, seed)
    n = len(poss)
    scores = np.full(n, np.nan)
    loo: list[np.ndarray | None] = [None] * n
    fold_of_row = np.full(n, -1, dtype=int)
    fold_auc: dict[int, float] = {}
    y_all = poss[label].astype(int).to_numpy()
    for k in range(n_folds):
        tr_m, va_m, ho_m = fold_splits(match_fold, k, seed)
        assert not (set(tr_m) | set(va_m)) & set(ho_m), "held-out matches leaked into training"
        tr = poss[poss["match_id"].isin(tr_m)]
        va = poss[poss["match_id"].isin(va_m)]
        ho_idx = np.flatnonzero(poss["match_id"].isin(ho_m).to_numpy())
        ho = poss.iloc[ho_idx]
        X_tr, L_tr = build_sequence_tensors(tr)
        X_va, L_va = build_sequence_tensors(va)
        torch.manual_seed(seed * 100 + k)
        model, hist = train_gru(
            X_tr,
            L_tr,
            tr[label].astype(int).to_numpy(),
            X_va,
            L_va,
            va[label].astype(int).to_numpy(),
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch["dropout"],
            tab_size=0,
            lr=train_cfg["lr"],
            n_epochs=train_cfg["n_epochs"],
            batch_size=train_cfg["batch_size"],
            patience=train_cfg["patience"],
            verbose=False,
        )
        model.eval()
        X_ho, L_ho = build_sequence_tensors(ho)
        loader = make_dataloader(
            X_ho, L_ho, y_all[ho_idx], batch_size=train_cfg["test_batch_size"], shuffle=False
        )
        auc, proba = evaluate(model, loader, torch.device("cpu"))
        scores[ho_idx] = proba
        fold_of_row[ho_idx] = k
        fold_auc[k] = float(auc)
        for i in ho_idx:
            events = poss["event_sequence"].iloc[i][:MAX_SEQ_LEN]
            loo[i] = leave_one_out_attribution(events, model).astype(float)
        if verbose:
            print(
                f"  fold {k}: train {len(tr_m)} / val {len(va_m)} / held-out {len(ho_m)} matches, "
                f"{len(hist)} epochs, held-out AUC {auc:.4f}",
                flush=True,
            )
        del model
    assert (fold_of_row >= 0).all() and all(x is not None for x in loo)
    pooled = float(roc_auc_score(y_all, scores))
    fold_auc[-1] = pooled
    return scores, loo, fold_of_row, fold_auc, match_fold


# ---------------------------------------------------------------------------
# Per-event table
# ---------------------------------------------------------------------------


def build_events_table(
    poss: pd.DataFrame,
    scores: np.ndarray,
    loo: list[np.ndarray],
    fold_of_row: np.ndarray,
    teams_by_poss: dict[tuple[int, int], list],
    seed: int,
) -> pd.DataFrame:
    """One row per attributed event (first 40 events of every possession)."""
    rows = []
    for i, r in enumerate(poss.itertuples(index=False)):
        seq = r.event_sequence[:MAX_SEQ_LEN]
        players = list(r.player_sequence)[:MAX_SEQ_LEN]
        teams = teams_by_poss.get((int(r.match_id), int(r.possession_id)), [None] * len(seq))
        for t, ev in enumerate(seq):
            rows.append(
                (
                    int(r.match_id),
                    int(r.possession_id),
                    t,
                    int(ev["type_id"]),
                    type_label(int(ev["type_id"])),
                    players[t] if t < len(players) else UNKNOWN_PLAYER,
                    teams[t] if t < len(teams) else None,
                    bool(ev.get("own_team", 1.0) >= 0.5),
                    (
                        float(ev["loc_x_norm"]) * 120.0
                        if ev.get("own_team", 1.0) >= 0.5
                        else 120.0 - float(ev["loc_x_norm"]) * 120.0
                    ),
                    (
                        float(ev["loc_y_norm"]) * 80.0
                        if ev.get("own_team", 1.0) >= 0.5
                        else 80.0 - float(ev["loc_y_norm"]) * 80.0
                    ),
                    float(scores[i]),
                    float(loo[i][t]),
                    int(fold_of_row[i]),
                    seed,
                    bool(r.poss_dangerous),
                    float(r.poss_xg_generated),
                    bool(r.poss_has_goal),
                    str(r.team_name),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "possession_id",
            "event_idx",
            "type_id",
            "event_type",
            "player",
            "event_team",
            "possessing_team_event",
            "x",
            "y",
            "possession_score",
            "loo_drop",
            "fold",
            "seed",
            "poss_dangerous",
            "poss_xg",
            "poss_has_goal",
            "possessing_team",
        ],
    )


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------


def credited_events(df: pd.DataFrame) -> pd.DataFrame:
    """Events that may be credited to a player: possessing-team events with a named player."""
    return df[df["possessing_team_event"] & (df["player"] != UNKNOWN_PLAYER) & df["player"].notna()]


def player_leaderboard(
    df: pd.DataFrame, min_events: int = 100, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Per-player mean LOO attribution over their **possessing-team** events.

    The 95% interval is a player-level cluster bootstrap: the player's possessions are resampled
    with replacement (events within a possession are correlated).
    """
    ev = credited_events(df)
    rng = np.random.default_rng(seed)
    out = []
    for player, g in ev.groupby("player"):
        if len(g) < min_events:
            continue
        per_poss = g.groupby(["match_id", "possession_id"]).agg(
            s=("loo_drop", "sum"), c=("loo_drop", "size"), d=("poss_dangerous", "first")
        )
        s, c = per_poss["s"].to_numpy(), per_poss["c"].to_numpy()
        idx = rng.integers(0, len(per_poss), size=(n_boot, len(per_poss)))
        means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        out.append(
            {
                "player": player,
                "team": g["possessing_team"].mode().iloc[0],
                "events": len(g),
                "possessions": len(per_poss),
                "mean_loo": float(g["loo_drop"].mean()),
                "ci_low": float(np.percentile(means, 2.5)),
                "ci_high": float(np.percentile(means, 97.5)),
                "share_dangerous_possessions": float(per_poss["d"].mean()),
            }
        )
    cols = [
        "player",
        "team",
        "events",
        "possessions",
        "mean_loo",
        "ci_low",
        "ci_high",
        "share_dangerous_possessions",
    ]
    return (
        pd.DataFrame(out, columns=cols)
        .sort_values("mean_loo", ascending=False)
        .reset_index(drop=True)
    )


def team_leaderboard(df: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    """Per possessing team: possessions, danger rate, mean xG per possession, mean own-event LOO."""
    g = poss.groupby("team_name").agg(
        possessions=("possession_id", "size"),
        danger_rate=("poss_dangerous", "mean"),
        mean_xg_per_possession=("poss_xg_generated", "mean"),
        goals=("poss_has_goal", "sum"),
    )
    own = df[df["possessing_team_event"]].groupby("possessing_team")["loo_drop"].mean()
    g["mean_own_event_loo"] = own
    return g.sort_values("mean_own_event_loo", ascending=False).reset_index()


def event_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean attribution, share of top-attributed events, base frequency and rate per type/side."""
    d = df.assign(side=np.where(df["possessing_team_event"], "possessing team", "defending team"))
    top = d.loc[d.groupby(["match_id", "possession_id"])["loo_drop"].idxmax()]
    top = top[top["loo_drop"] > 0]
    n_all, n_top = len(d), len(top)
    base = d.groupby(["event_type", "side"]).size().rename("n_events")
    tp = top.groupby(["event_type", "side"]).size().rename("n_top")
    tab = pd.concat([base, tp], axis=1).fillna(0)
    tab["mean_loo"] = d.groupby(["event_type", "side"])["loo_drop"].mean()
    tab["mean_abs_loo"] = d.groupby(["event_type", "side"])["loo_drop"].apply(
        lambda s: s.abs().mean()
    )
    tab["base_freq"] = tab["n_events"] / n_all
    tab["share_top"] = tab["n_top"] / n_top
    tab["rate"] = tab["share_top"] / tab["base_freq"]
    return tab.reset_index().sort_values("rate", ascending=False)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def split_half_stability(
    df: pd.DataFrame, players: list[str], repeats: int = 50, seed: int = 42
) -> dict:
    """Spearman correlation of player rankings between random halves of their possessions."""
    ev = credited_events(df)
    ev = ev[ev["player"].isin(players)]
    by_player = {
        p: g.groupby(["match_id", "possession_id"])["loo_drop"].agg(["sum", "count"]).to_numpy()
        for p, g in ev.groupby("player")
    }
    rng = np.random.default_rng(seed)
    rhos = []
    names = sorted(by_player)
    for _ in range(repeats):
        a, b = [], []
        for p in names:
            arr = by_player[p]
            perm = rng.permutation(len(arr))
            h = (len(arr) + 1) // 2
            x, y = arr[perm[:h]], arr[perm[h:]]
            a.append(x[:, 0].sum() / x[:, 1].sum())
            b.append(y[:, 0].sum() / y[:, 1].sum())
        rhos.append(spearmanr(a, b)[0])
    return {
        "mean": float(np.mean(rhos)),
        "sd": float(np.std(rhos)),
        "min": float(np.min(rhos)),
        "max": float(np.max(rhos)),
        "n_players": len(names),
        "repeats": repeats,
    }


def cross_run_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame, players: list[str]) -> dict:
    """Agreement between two independent cross-fitted attribution runs (different seeds)."""
    key = ["match_id", "possession_id", "event_idx"]
    m = df_a[key + ["loo_drop"]].merge(df_b[key + ["loo_drop"]], on=key, suffixes=("_a", "_b"))
    ev_corr = float(np.corrcoef(m["loo_drop_a"], m["loo_drop_b"])[0, 1])
    pa = credited_events(df_a).groupby("player")["loo_drop"].mean()
    pb = credited_events(df_b).groupby("player")["loo_drop"].mean()
    both = [p for p in players if p in pa.index and p in pb.index]
    rho = float(spearmanr(pa[both], pb[both])[0])
    poss_a = df_a.groupby(["match_id", "possession_id"])["possession_score"].first()
    poss_b = df_b.groupby(["match_id", "possession_id"])["possession_score"].first()
    return {
        "event_pearson": ev_corr,
        "player_spearman": rho,
        "n_players": len(both),
        "possession_score_spearman": float(spearmanr(poss_a, poss_b.reindex(poss_a.index))[0]),
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def gini(x) -> float:
    """Gini coefficient of non-negative values (NaN if they sum to zero)."""
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    if x.sum() == 0 or len(x) == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def sanity_checks(df: pd.DataFrame) -> dict:
    per = df.groupby(["match_id", "possession_id"]).agg(
        score=("possession_score", "first"),
        xg=("poss_xg", "first"),
        goal=("poss_has_goal", "first"),
        dangerous=("poss_dangerous", "first"),
    )
    ginis = df.groupby(["match_id", "possession_id"])["loo_drop"].apply(gini)
    per["gini"] = ginis
    dang = per[per["dangerous"]]
    goal_idx, nongoal_idx = dang[dang["goal"]].index, dang[~dang["goal"]].index
    keyed = df.set_index(["match_id", "possession_id"])
    own = keyed[keyed["possessing_team_event"]]

    def mean_loo(frame, idx):
        return float(frame.loc[frame.index.isin(idx), "loo_drop"].mean())

    return {
        "score_vs_xg_spearman": float(spearmanr(per["score"], per["xg"])[0]),
        "score_vs_xg_pearson": float(np.corrcoef(per["score"], per["xg"])[0, 1]),
        "n_possessions": int(len(per)),
        "n_goal_possessions": int(len(goal_idx)),
        "n_dangerous_nongoal": int(len(nongoal_idx)),
        "mean_loo_all_events_goal": mean_loo(keyed, goal_idx),
        "mean_loo_all_events_dangerous_nongoal": mean_loo(keyed, nongoal_idx),
        "mean_loo_own_events_goal": mean_loo(own, goal_idx),
        "mean_loo_own_events_dangerous_nongoal": mean_loo(own, nongoal_idx),
        "gini_median_dangerous": float(per.loc[per["dangerous"], "gini"].median()),
        "gini_median_safe": float(per.loc[~per["dangerous"], "gini"].median()),
        "gini_iqr_dangerous": [
            float(per.loc[per["dangerous"], "gini"].quantile(q)) for q in (0.25, 0.75)
        ],
        "gini_iqr_safe": [
            float(per.loc[~per["dangerous"], "gini"].quantile(q)) for q in (0.25, 0.75)
        ],
        "gini_median_all": float(per["gini"].median()),
    }
