"""Self-audit: shot xG / outcome parsing (poss_xg_generated, poss_has_goal, outcome tier).

Compares the rebuilt possession table with the pre-fix one and prints sanity checks.
    python scripts/audit_shot_columns.py --out reports/audit/shot_columns_after.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ev = pd.read_parquet(PROC / "events_parsed.parquet")
    new = pd.read_parquet(PROC / "possession_sequences.parquet")
    old = pd.read_parquet(PROC / "possession_sequences_pre_xg.parquet")
    inv = pd.read_parquet("data/interim/inventory.parquet")[
        ["match_id", "competition_id", "season_id"]
    ]
    shots = ev[ev.type_name == "Shot"].merge(inv, on="match_id", how="left")
    goals = shots[shots.shot_outcome_name == "Goal"]
    n_matches = ev.match_id.nunique()
    shootout = goals[goals.period == 5]
    lines = ["# Shot columns audit (after the fix)", ""]
    lines += [
        "## Before / after in `possession_sequences.parquet`",
        "",
        f"- Before: poss_xg_generated sum = {old.poss_xg_generated.sum():.1f}; "
        f"poss_has_goal true = {int(old.poss_has_goal.sum())}; "
        f"max outcome tier = {int(old.poss_outcome_tier.max())}",
        f"- After: poss_xg_generated sum = {new.poss_xg_generated.sum():.1f}; "
        f"poss_has_goal true = {int(new.poss_has_goal.sum())}; "
        f"max outcome tier = {int(new.poss_outcome_tier.max())}",
        "",
        "Columns that changed (everything else is identical to the pre-fix table):",
        "",
    ]
    changed = []
    for c in old.columns:
        if c in ("event_sequence", "player_sequence"):
            same = (old[c].astype(str) == new[c].astype(str)).all()
        elif old[c].dtype.kind in "fi":
            same = np.allclose(old[c].astype(float), new[c].astype(float), equal_nan=True)
        else:
            same = (old[c].astype(str) == new[c].astype(str)).all()
        if not same:
            changed.append(c)
    lines += [f"- {', '.join(changed)}", ""]
    lines += ["## Sanity checks", ""]
    lines += [
        f"- Shots: {len(shots):,}; goals: {len(goals)} ({len(shootout)} in penalty shootouts); "
        f"matches: {n_matches}",
        f"- Goals per match: **{len(goals) / n_matches:.2f}** including shootouts, "
        f"**{(len(goals) - len(shootout)) / n_matches:.2f}** excluding them",
    ]
    for (cid, sid), g in shots.groupby(["competition_id", "season_id"]):
        gg = g[(g.shot_outcome_name == "Goal") & (g.period != 5)]
        lines.append(
            f"  - competition {int(cid)} season {int(sid)}: {g.match_id.nunique()} matches, "
            f"{len(gg) / g.match_id.nunique():.2f} goals per match (excl. shootouts), "
            f"xG {g.shot_statsbomb_xg.sum() / g.match_id.nunique():.2f} per match"
        )
    lines += [
        f"- Total xG {shots.shot_statsbomb_xg.sum():.1f} vs {len(goals)} goals "
        f"(ratio {shots.shot_statsbomb_xg.sum() / len(goals):.2f}); possession table xG sum "
        f"{new.poss_xg_generated.sum():.1f}",
        f"- Share of possessions with a goal: {new.poss_has_goal.mean():.2%} "
        f"({int(new.poss_has_goal.sum())} of {len(new):,}); share with a shot: "
        f"{new.poss_has_shot.mean():.2%}",
        "- Outcome tier distribution: "
        + ", ".join(
            f"tier {k}: {v}" for k, v in new.poss_outcome_tier.value_counts().sort_index().items()
        ),
        f"- Goals not attached to a possession row (possessions under 2 events dropped): "
        f"{len(goals) - int(goals.merge(new[['match_id','possession_id']], on=['match_id','possession_id']).shape[0])}",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
