"""Inspectable demonstration evidence from real held-out data.

Scores one held-out (test-split) possession that reached danger with the event-time XGBoost and GRU
and saves (a) ``reports/figures/demo/possession_demo.png`` (danger over time + ball path) and
(b) ``reports/repro/cli_demo.txt`` (the exact CLI session).  Needs the gitignored data / models.

Usage:  python scripts/make_demo_evidence.py [--seed 42]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app import possession_page as pg  # noqa: E402
from src.app.live_scoring import score_arrays  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    missing = pg.missing_artefacts(ROOT)
    if missing:
        raise SystemExit("missing: " + "; ".join(missing))
    poss = pd.read_parquet(ROOT / pg.REQUIRED["possession table"])
    manifest = pd.read_csv(ROOT / pg.REQUIRED["split manifest"])
    pool = poss[
        poss["match_id"].isin(pg.held_out_match_ids(manifest))
        & poss["poss_dangerous"].astype(bool)
        & poss["n_events"].between(12, 30)
    ]
    row = pool.sample(1, random_state=args.seed).iloc[0]
    arr = pg.load_arrays(ROOT / pg.REQUIRED["event-time arrays"])[
        (int(row["match_id"]), int(row["possession_id"]))
    ]
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for model, colour in (("xgb", "darkorange"), ("gru", "steelblue")):
        res = score_arrays(arr, ROOT, model=model)
        df = pg.steps_frame(res, arr).dropna(subset=["p_danger"])
        ax.plot(df.event_idx + 1, df.p_danger, marker="o", ms=3, color=colour, label=model.upper())
        ax.axhline(res["threshold"], color=colour, ls="--", lw=0.8)
    full = pg.steps_frame(res, arr)
    real = full[full["realised"]]
    if len(real):
        ax.axvline(real.event_idx.iloc[0] + 1, color="black", label="shot / box entry")
    ax.set_ylim(0, 1)
    ax.set_xlabel("event in possession")
    ax.set_ylabel("P(shot or box entry in a later event)")
    ax.set_title(
        f"Event-time prediction, held-out match {int(row['match_id'])}, possession "
        f"{int(row['possession_id'])} ({row['team_name']})",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    try:
        from mplsoccer import Pitch

        pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
        pitch.draw(ax=axp)
        own = full[full["own"]]
        axp.plot(own.x, own.y, color="white", lw=1.2, marker="o", ms=3)
        axp.set_title("Ball path, possessing-team frame (attacks left to right)", fontsize=9)
    except ImportError:
        axp.axis("off")
    out = ROOT / "reports" / "figures" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "possession_demo.png", dpi=130)
    print("wrote", out / "possession_demo.png")

    sample = ROOT / "reports" / "repro" / "demo_possession.json"
    steps = [
        ["sample-possession", "-o", str(sample), "--dangerous", "1", "--seed", str(args.seed)],
        ["score-live", str(sample), "--model", "gru"],
    ]
    log = []
    for tail in steps:
        c = [sys.executable, "-W", "ignore", "-m", "src.app.cli", *tail]
        shown = "frame2threat " + " ".join(Path(a).name if a.endswith(".json") else a for a in tail)
        p = subprocess.run(c, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        log += [f"$ {shown}", p.stdout.rstrip(), p.stderr.rstrip(), ""]
    (ROOT / "reports" / "repro" / "cli_demo.txt").write_text("\n".join(log), encoding="utf-8")
    print("wrote reports/repro/cli_demo.txt and", sample)


if __name__ == "__main__":
    main()
