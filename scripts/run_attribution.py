"""EXP-031: cross-fitted retrospective attribution on the corrected possession table.

Runs two independent 5-fold match-grouped cross-fitting rounds (seed 42 = primary, seed 43 =
reliability check), writes ``data/processed/attribution_events.parquet`` (primary) and
``attribution_events_seed43.parquet``, the leaderboards / tables / figures under
``reports/attribution/`` and ``reports/figures/attribution/``, and the report.

Usage:  python scripts/run_attribution.py [--skip-fit]   (--skip-fit reuses the parquet files)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parse_possessions import load_possession_sequences  # noqa: E402
from src.evaluation.attribution_crossfit import (  # noqa: E402
    N_FOLDS,
    build_events_table,
    credited_events,
    crossfit_attribution,
    cross_run_agreement,
    event_type_table,
    player_leaderboard,
    sanity_checks,
    split_half_stability,
    team_leaderboard,
)

warnings.filterwarnings("ignore")
PROC = Path("data/processed")
REP = Path("reports/attribution")
FIG = Path("reports/figures/attribution")
MIN_EVENTS = 100


def md(df: pd.DataFrame, fmt: dict | None = None, n: int | None = None) -> str:
    fmt = fmt or {}
    d = df.head(n) if n else df
    cols = list(d.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in d.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if c in fmt:
                cells.append(format(v, fmt[c]))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def word(rho: float) -> str:
    return "low" if rho < 0.3 else "moderate" if rho < 0.6 else "reasonably high"


def run_fit(poss, teams, seed, arch, train_cfg, out_path):
    print(f"Cross-fitting, seed {seed} ...", flush=True)
    scores, loo, fold, fold_auc, match_fold = crossfit_attribution(
        poss, seed=seed, arch=arch, train_cfg=train_cfg
    )
    df = build_events_table(poss, scores, loo, fold, teams, seed)
    df.to_parquet(out_path, index=False)
    return df, fold_auc


def figures(df, board, etab, poss):
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    FIG.mkdir(parents=True, exist_ok=True)
    # --- player leaderboard
    top = board.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(top))
    ax.errorbar(
        top["mean_loo"],
        y,
        xerr=[top["mean_loo"] - top["ci_low"], top["ci_high"] - top["mean_loo"]],
        fmt="o",
        color="#c0392b",
        ecolor="#888",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} ({t})" for p, t in zip(top["player"], top["team"])], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Mean LOO attribution of possessing-team events (95% cluster bootstrap CI)")
    ax.set_title(
        f"Top 20 players by mean attribution (>= {MIN_EVENTS} events), cross-fitted", fontsize=10
    )
    plt.tight_layout()
    fig.savefig(FIG / "player_leaderboard_top20.png", dpi=150)
    plt.close(fig)
    # --- event-type rate chart
    t = etab[etab["n_events"] >= 200].copy()
    order = t.groupby("event_type")["rate"].max().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 6))
    h = 0.38
    for j, (side, col) in enumerate(
        (("possessing team", "#2c7fb8"), ("defending team", "#d95f0e"))
    ):
        s = t[t["side"] == side].set_index("event_type").reindex(order)
        ax.barh(
            np.arange(len(order)) + (j - 0.5) * h,
            s["rate"].fillna(0),
            height=h,
            color=col,
            label=side,
        )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(1, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Rate = share of top-attributed events / base frequency  (1 = in proportion)")
    ax.set_title("Which events does the retrospective model lean on?", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "event_type_rate.png", dpi=150)
    plt.close(fig)
    # --- 4 example possessions (fixed seed)
    rng = np.random.default_rng(42)
    per = df.groupby(["match_id", "possession_id"]).agg(
        n=("event_idx", "size"), d=("poss_dangerous", "first")
    )
    cand = per[(per["n"] >= 12) & (per["n"] <= 25)]
    chosen = []
    for flag in (True, True, False, False):
        pool = cand[(cand["d"] == flag) & ~cand.index.isin(chosen)]
        chosen.append(pool.index[rng.integers(len(pool))])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pitch = Pitch(pitch_type="statsbomb", line_color="#999")
    vmax = max(0.05, df["loo_drop"].abs().quantile(0.99))
    for ax, key in zip(axes.ravel(), chosen):
        g = df[(df["match_id"] == key[0]) & (df["possession_id"] == key[1])].sort_values(
            "event_idx"
        )
        pitch.draw(ax=ax)
        pitch.lines(
            g["x"].values[:-1],
            g["y"].values[:-1],
            g["x"].values[1:],
            g["y"].values[1:],
            ax=ax,
            color="#bbb",
            lw=1,
            zorder=1,
        )
        sc = pitch.scatter(
            g["x"],
            g["y"],
            ax=ax,
            c=g["loo_drop"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            s=60 + 900 * np.abs(g["loo_drop"]) / vmax,
            marker="o",
            edgecolors=np.where(g["possessing_team_event"], "k", "#d95f0e"),
            linewidths=np.where(g["possessing_team_event"], 0.8, 2.0),
            zorder=3,
        )
        for r in g.itertuples():
            ax.text(r.x, r.y, str(r.event_idx), fontsize=5, ha="center", va="center", zorder=5)
        top5 = g.reindex(g["loo_drop"].abs().sort_values(ascending=False).index[:5])
        side = lambda r: "att" if r.possessing_team_event else "def"  # noqa: E731
        listing = (
            "largest drops:"
            + chr(10)
            + chr(10).join(
                f"{r.event_idx}: {r.event_type} ({side(r)}) {r.loo_drop:+.2f}"
                for r in top5.itertuples()
            )
        )
        ax.text(
            0.01,
            0.99,
            listing,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbb", "alpha": 0.9},
        )
        kind = "dangerous" if g["poss_dangerous"].iloc[0] else "safe"
        ax.set_title(
            f"{kind} possession {key[1]} (match {key[0]}), {g['possessing_team'].iloc[0]}, "
            f"score {g['possession_score'].iloc[0]:.2f}",
            fontsize=9,
        )
    fig.colorbar(
        sc, ax=axes, shrink=0.6, label="LOO drop in possession score (red = event raises the score)"
    )
    fig.suptitle(
        "Per-event LOO drops (possessing-team frame; orange outline = defending-team event)",
        fontsize=11,
    )
    fig.savefig(FIG / "example_possessions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fit", action="store_true")
    args = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    poss = load_possession_sequences(PROC / "possession_sequences.parquet")
    arch = yaml.safe_load(open("configs/model_gru.yaml"))["architecture"]
    train_cfg = yaml.safe_load(open("configs/model_gru.yaml"))["training"]
    p42, p43 = PROC / "attribution_events.parquet", PROC / "attribution_events_seed43.parquet"
    fold_auc = {}
    if args.skip_fit and p42.exists() and p43.exists():
        d42, d43 = pd.read_parquet(p42), pd.read_parquet(p43)
        raw_auc = json.loads((REP / "attribution_results.json").read_text())["fold_auc"]
        fold_auc = {k: {int(i): v for i, v in d.items()} for k, d in raw_auc.items()}
    else:
        ev = pd.read_parquet(
            PROC / "events_parsed.parquet",
            columns=["match_id", "possession_id", "index", "team_name"],
        )
        ev = ev.sort_values(["match_id", "possession_id", "index"])
        teams = {
            (int(m), int(p)): g["team_name"].tolist()
            for (m, p), g in ev.groupby(["match_id", "possession_id"])
        }
        assert all(
            len(teams[(int(r.match_id), int(r.possession_id))]) == len(r.event_sequence)
            for r in poss.itertuples(index=False)
        ), "events_parsed and event_sequence are misaligned"
        d42, auc42 = run_fit(poss, teams, 42, arch, train_cfg, p42)
        d43, auc43 = run_fit(poss, teams, 43, arch, train_cfg, p43)
        fold_auc = {"seed42": auc42, "seed43": auc43}

    # ---------------------------------------------------------------- analyses
    board = player_leaderboard(d42, MIN_EVENTS)
    n_players_all = credited_events(d42)["player"].nunique()
    teams_tbl = team_leaderboard(d42, poss)
    etab = event_type_table(d42)
    qual = board["player"].tolist()
    stab = split_half_stability(d42, qual, repeats=50, seed=42)
    agree = cross_run_agreement(d42, d43, qual)
    san = sanity_checks(d42)
    board.to_csv(REP / "player_leaderboard.csv", index=False)
    teams_tbl.to_csv(REP / "team_leaderboard.csv", index=False)
    etab.to_csv(REP / "event_type_table.csv", index=False)
    chosen = figures(d42, board, etab, poss)

    aucs = fold_auc["seed42"]
    results = {
        "fold_auc": fold_auc,
        "split_half": stab,
        "cross_seed": agree,
        "sanity": san,
        "players_qualifying": len(board),
        "players_with_credited_events": int(n_players_all),
        "example_possessions": [[int(a), int(b)] for a, b in chosen],
    }
    (REP / "attribution_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- report
    n_events, n_poss = len(d42), d42.groupby(["match_id", "possession_id"]).ngroups
    own_share = d42["possessing_team_event"].mean()
    trunc = float((poss["n_events"] > 40).mean())
    fold_rows = pd.DataFrame(
        [{"fold": f"fold {k}", "held-out ROC AUC": v} for k, v in aucs.items() if k >= 0]
        + [{"fold": "pooled (all 17,065 possessions)", "held-out ROC AUC": aucs[-1]}]
    )
    qrel = word(stab["mean"])
    brel = word(agree["player_spearman"])
    et = etab[etab["n_events"] >= 200]
    top_rows = md(
        board[
            [
                "player",
                "team",
                "events",
                "possessions",
                "mean_loo",
                "ci_low",
                "ci_high",
                "share_dangerous_possessions",
            ]
        ].head(15),
        {"share_dangerous_possessions": ".3f"},
    )
    bot_rows = md(
        board[
            [
                "player",
                "team",
                "events",
                "possessions",
                "mean_loo",
                "ci_low",
                "ci_high",
                "share_dangerous_possessions",
            ]
        ].tail(10),
        {"share_dangerous_possessions": ".3f"},
    )
    sig_pos = int((board["ci_low"] > 0).sum())
    sig_neg = int((board["ci_high"] < 0).sum())
    lines = f"""# Retrospective attribution report (cross-fitted)

Generated by `scripts/run_attribution.py` (EXP-031).  Everything here is **retrospective**: the model
sees the whole possession (first {40} events) and is asked which events it relies on.  It is not a
forecast and not causal credit.

## 1. Method

- Data: the corrected possession table (frame-fixed labels, xG and goals populated), {n_poss:,} possessions
  from 99 matches (64 World Cup 2022, 35 La Liga 2020/21 Barcelona).
- **Cross-fitting.** The 99 matches are split into {N_FOLDS} match-grouped folds (seed 42).  For each fold a
  v2 GRU (`configs/model_gru.yaml`: 1 layer, hidden 64, lr 1e-3, early stopping on validation AUC, the same
  class-weighted loss as v2) is trained on the other four folds, with an inner match-level split (15% of the
  matches) for early stopping.  LOO attribution is computed **only** on the held-out fold, so every possession
  is attributed by a model that never saw that possession or any other possession of its match
  (`tests/test_attribution.py` checks this).  The v2 numbers (leaderboards, Gini 0.495) were in-sample.
- **LOO drop** = score(full possession) - score(possession with event *t* replaced by an all-zero vector).
  Events beyond the 40th are not seen by the GRU and are not attributed ({trunc:.1%} of possessions are longer).
- Scores are GRU outputs trained with class weighting, so they rank possessions well but are **not calibrated
  probabilities**; attribution values are in score units.
- {n_events:,} attributed events, of which {own_share:.1%} were performed by the possessing team and the rest by
  the defending team (their coordinates are stored in the opposite frame; the v2 GRU input was left unchanged).

## 2. Cross-fitted accuracy (retrospective)

{md(fold_rows)}
Pooled out-of-fold ROC AUC **{aucs[-1]:.3f}** (retrospective, full-possession input: not an early-warning figure).
The second cross-fitting round (seed 43) gave a pooled {fold_auc['seed43'][-1]:.3f}.

## 3. Sanity checks against real outcomes

- Possession score vs possession xG: Spearman **{san['score_vs_xg_spearman']:.3f}**, Pearson {san['score_vs_xg_pearson']:.3f} ({san['n_possessions']:,} possessions).
- Mean LOO of events in goal-scoring possessions ({san['n_goal_possessions']}) vs dangerous non-goal possessions ({san['n_dangerous_nongoal']:,}):
  all events {san['mean_loo_all_events_goal']:+.4f} vs {san['mean_loo_all_events_dangerous_nongoal']:+.4f};
  possessing-team events {san['mean_loo_own_events_goal']:+.4f} vs {san['mean_loo_own_events_dangerous_nongoal']:+.4f}.
- Gini of |LOO| per possession (concentration of attribution; 0 = spread evenly, 1 = one event): median
  **{san['gini_median_dangerous']:.3f}** for dangerous possessions (IQR {san['gini_iqr_dangerous'][0]:.3f}-{san['gini_iqr_dangerous'][1]:.3f}),
  **{san['gini_median_safe']:.3f}** for safe ones (IQR {san['gini_iqr_safe'][0]:.3f}-{san['gini_iqr_safe'][1]:.3f}); overall median {san['gini_median_all']:.3f}.
  (v2's in-sample figure was 0.495; the definition here is on |LOO| over the attributed events.)

## 4. Event-type table (possessing vs defending team)

Rate = share of each possession's single top-attributed event (positive drops only) / base frequency of that
event type and side among all attributed events.  Rows with at least 200 events.

{md(et[['event_type','side','n_events','mean_loo','mean_abs_loo','base_freq','share_top','rate']].head(40), {'n_events': ',d', 'base_freq': '.4f', 'share_top': '.4f', 'mean_loo': '+.4f', 'mean_abs_loo': '.4f', 'rate': '.2f'})}
Goalkeeper, clearance, block, foul-committed and similar events appear almost only on the **defending**
side.  The model uses them as evidence that the opponent was under pressure or that a shot happened (a
goalkeeper action nearly always follows a shot); they are not attacking actions by the players who carry
those events, and are therefore excluded from the player leaderboard.

![Event-type rate](../figures/attribution/event_type_rate.png)

## 5. Player leaderboard

Only possessing-team events count toward a player (defending-team events are never credited).  Minimum
{MIN_EVENTS} such events: **{len(board)} of {n_players_all} players qualify.**  Mean LOO is per event; the 95% interval is a
player-level cluster bootstrap over the player's possessions (1,000 resamples).  {sig_pos} players have an
interval entirely above zero and {sig_neg} entirely below.

Top 15:

{top_rows}
Bottom 10:

{bot_rows}
![Top 20 players](../figures/attribution/player_leaderboard_top20.png)

Full table: `player_leaderboard.csv`.

## 6. Team leaderboard

Possessions, danger rate, mean xG per possession (now available) and mean attribution of the team's own events.
Top and bottom five by mean own-event attribution; full table in `team_leaderboard.csv`.

{md(pd.concat([teams_tbl.head(5), teams_tbl.tail(5)]), {'possessions': 'd', 'danger_rate': '.3f', 'mean_xg_per_possession': '.4f', 'goals': '.0f', 'mean_own_event_loo': '+.4f'})}
## 7. How reliable is the leaderboard?

- **Split-half stability.**  Each qualifying player's possessions were randomly split into two halves
  (seed 42, {stab['repeats']} repeats) and players ranked on each half: Spearman correlation between halves
  **{stab['mean']:.3f} +/- {stab['sd']:.3f}** (range {stab['min']:.3f} to {stab['max']:.3f}; {stab['n_players']} players).
  That is **{qrel}** stability: {"the ranking is mostly noise at this sample size and should not be used to order players" if stab['mean'] < 0.3 else "players keep part of their position from one half of their possessions to the other, but individual positions are not reliable" if stab['mean'] < 0.6 else "the ordering is reasonably stable"}.
- **Cross-run agreement.**  A possession is scored by one fold model only, so agreement between fold models on
  the same possession is not available.  Instead the whole cross-fitting was repeated with a different seed
  (different fold assignment, inner split and initialisation).  Per-event LOO correlation between the two runs:
  Pearson **{agree['event_pearson']:.3f}**; player-ranking Spearman **{agree['player_spearman']:.3f}** ({agree['n_players']} players,
  **{brel}**); possession-score Spearman {agree['possession_score_spearman']:.3f}.

- **Stable is not the same as meaningful.**  The top of the table is dominated by forwards (strikers and wide
  attackers) and the bottom by centre-backs and full-backs, so most of the stable signal is player **role**: attackers
  touch the ball closer to goal in possessions that end dangerously.  Nothing is adjusted for position, team strength
  or opponent, so a high rank says "this player's possessing-team events are the ones the model leans on", not
  "this player is better".  Split-half stability ({stab['mean']:.2f}) corresponds to roughly {2*stab['mean']/(1+stab['mean']):.2f} for a
  full-length sample (Spearman-Brown), which is enough to describe role but not to separate similar players.
- Teams other than the large-sample ones (Barcelona, the World Cup sides) appear only through a few matches
  (about 120 possessions each), so the team table is noisy.

## 8. Example possessions

Two dangerous and two safe possessions, chosen by a fixed seed (42) from 12-25-event possessions; each event
is plotted in the possessing team's frame, numbered by event index; the five largest drops are listed in each panel (att = possessing team, def = defending team).

![Example possessions](../figures/attribution/example_possessions.png)

## 9. What this does and does not measure

- It shows **which events the retrospective model relies on** when it calls a possession dangerous.
- It is **not causal credit**: zeroing an event creates a sequence that never happens, so a drop reflects how the
  model uses that event, not what would have happened without it.
- Defending-team events (pressure, goalkeeper, clearance, block) act as **evidence of danger**; attribution to
  them says nothing about the defenders' quality.
- Player means mix role (a striker touches the ball near the box more) with contribution; there is no adjustment
  for position, team strength or opponent.
- **Saturation:** when a possession's score is already close to 1, removing any single event leaves it close to 1, so every drop is near zero (see the second example above, score 1.00, all drops +0.00).  LOO then says nothing about which events mattered; it is least informative exactly for the most obviously dangerous possessions.
- Scores are uncalibrated and only the first 40 events are used.

## 10. How to use it

- **Tag key moments for video review:** open the possessions where the top-attributed event is a possessing-team
  carry/pass and check whether the moment really mattered.
- **Scouting shortlists to inspect:** players with intervals above zero and enough possessions are a list of names
  to watch, not a ranking.  Given the stability results above, do not rank players on this alone.
- Do **not** use it for recruitment decisions on its own.

## 11. Recommended next step

Replace retrospective LOO with a **forward-looking action value model** (expected threat, xT, or a VAEP-style
model that values each action by the change in scoring and conceding probability over the next few actions).
That credits actions for what they make more likely, is defined for every action including defending ones, and
does not depend on the whole possession being visible.

Files: `player_leaderboard.csv`, `team_leaderboard.csv`, `event_type_table.csv`, `attribution_results.json`;
per-event table `data/processed/attribution_events.parquet` (gitignored).
"""
    (REP / "ATTRIBUTION_REPORT.md").write_text(lines, encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
