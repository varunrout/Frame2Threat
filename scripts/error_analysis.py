"""Where does the event-time model work and where does it fail?

Re-scores the committed event-time XGBoost and GRU on the held-out test snapshots (no retraining) and
reports ROC AUC per segment: competition, possession origin, period and possession start minute.
Each AUC carries a match-level bootstrap interval; segments with few positives, few negatives or few matches are flagged.

This is descriptive. Segments are not corrected for multiple comparisons and the test split has 15
matches, so differences between segments should be read as hypotheses, not findings.

Writes ``reports/event_time/error_analysis.csv`` and ``.md`` and
``reports/figures/error_analysis/auc_by_segment.png``.  Needs the gitignored ``data/processed`` and
``models`` files.

Usage:  python scripts/error_analysis.py [--n-boot 300]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.app.batch_scoring import load_models  # noqa: E402
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    build_snapshot_table,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps  # noqa: E402

PROC = ROOT / "data" / "processed"
MIN_CLASS = 100  # minimum snapshots of the rarer class
MIN_MATCHES = 5


def competition_of(match_id: int) -> str:
    """Competition from the match-id range (La Liga 2020/21 ids are below 3,800,000)."""
    return "La Liga 2020/21 (Barcelona)" if match_id < 3_800_000 else "World Cup 2022"


def minute_band(minute: float) -> str:
    for upper, name in ((30, "0-30"), (60, "30-60"), (90, "60-90")):
        if minute < upper:
            return name
    return "90+"


def score_test_snapshots() -> pd.DataFrame:
    test_ids = set(pd.read_csv(PROC / "split_manifest.csv").query("split == 'test'")["match_id"])
    arrs = [
        a
        for a in pickle.loads((PROC / "event_time_arrays.pkl").read_bytes())
        if a.match_id in test_ids
    ]
    art, net, _ = load_models(ROOT)
    snaps = build_snapshot_table(arrs)
    snaps = snaps[~snaps["terminal"].to_numpy().astype(bool)].reset_index(drop=True)
    samples = []
    for arr in arrs:
        eligible, label, _ = snapshot_labels(arr)
        type_ids, numeric = sequence_inputs(arr)
        samples.append(SeqSample(type_ids, numeric, eligible, int(label[0])))
    steps = predict_steps(net, samples)
    out = pd.DataFrame(
        {
            "match_id": [arrs[i].match_id for i in snaps["arr_idx"]],
            "origin": [arrs[i].origin for i in snaps["arr_idx"]],
            "period": [arrs[i].period for i in snaps["arr_idx"]],
            "minute": [arrs[i].minute for i in snaps["arr_idx"]],
            "possession_id": [arrs[i].possession_id for i in snaps["arr_idx"]],
            "y": snaps["y"].astype(int).to_numpy(),
            "p_xgb": art["model"].predict_proba(snaps[FEATURE_NAMES][art["features"]])[:, 1],
            "p_gru": [steps[i][t] for i, t in zip(snaps["arr_idx"], snaps["t"])],
        }
    )
    out["competition"] = out["match_id"].map(competition_of)
    out["minute_band"] = out["minute"].map(minute_band)
    out["period"] = out["period"].map(lambda p: f"period {int(p)}")
    return out


def auc_with_ci(frame: pd.DataFrame, column: str, n_boot: int, rng: np.random.Generator):
    y = frame["y"].to_numpy()
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y, frame[column])
    groups = [g.index.to_numpy() for _, g in frame.groupby("match_id")]
    scores = frame[column].to_numpy()
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        draws.append(roc_auc_score(y[idx], scores[idx]))
    if len(draws) < 20:
        return point, np.nan, np.nan
    return point, *np.percentile(draws, [2.5, 97.5])


def segment_table(snaps: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for dimension in ("competition", "origin", "period", "minute_band"):
        for value, part in snaps.groupby(dimension):
            part = part.reset_index(drop=True)
            row = {
                "dimension": dimension,
                "segment": value,
                "snapshots": len(part),
                "possessions": part[["match_id", "possession_id"]].drop_duplicates().shape[0],
                "matches": part["match_id"].nunique(),
                "positives": int(part["y"].sum()),
                "prevalence": float(part["y"].mean()),
            }
            for name in ("xgb", "gru"):
                point, low, high = auc_with_ci(part, f"p_{name}", n_boot, rng)
                row[f"auc_{name}"], row[f"auc_{name}_low"], row[f"auc_{name}_high"] = (
                    point,
                    low,
                    high,
                )
            rarer = min(row["positives"], row["snapshots"] - row["positives"])
            row["small_sample"] = rarer < MIN_CLASS or row["matches"] < MIN_MATCHES
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, overall: dict, path: Path, n_boot: int) -> None:
    lines = [
        "# Error analysis: event-time models by segment",
        "",
        "Committed event-time XGBoost and GRU re-scored on the held-out test snapshots (15 matches, no retraining).",
        f"ROC AUC with a {n_boot}-resample match-level bootstrap 95% interval. **Descriptive only**: segments are not",
        "corrected for multiple comparisons and the test split is small. Rows marked `small` have fewer than "
        f"{MIN_CLASS} snapshots in the rarer class or fewer than {MIN_MATCHES} matches; do not read anything into them.",
        "Competition is derived from the match-id range and is confounded with 360 coverage and with club.",
        "",
        f"Overall: XGBoost {overall['xgb']:.4f}, GRU {overall['gru']:.4f} on {overall['n']:,} snapshots.",
    ]
    for dimension, part in table.groupby("dimension", sort=False):
        lines += [
            "",
            f"## By {dimension.replace('_', ' ')}",
            "",
            "| Segment | Snapshots | Matches | Prevalence | XGBoost AUC [95% CI] | GRU AUC [95% CI] | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in part.iterrows():

            def cell(name: str) -> str:
                if np.isnan(r[f"auc_{name}"]):
                    return "n/a"
                ci = (
                    ""
                    if np.isnan(r[f"auc_{name}_low"])
                    else f" [{r[f'auc_{name}_low']:.3f}, {r[f'auc_{name}_high']:.3f}]"
                )
                return f"{r[f'auc_{name}']:.3f}{ci}"

            note = "small" if r["small_sample"] else ""
            lines.append(
                f"| {r['segment']} | {int(r['snapshots']):,} | {int(r['matches'])} | {r['prevalence']:.3f} "
                f"| {cell('xgb')} | {cell('gru')} | {note} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(table: pd.DataFrame, overall: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = table[~table["small_sample"] & table["auc_xgb"].notna()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(shown) + 1.6))
    ypos = np.arange(len(shown))[::-1]
    for offset, name, colour in ((0.15, "xgb", "#d97706"), (-0.15, "gru", "#2563eb")):
        auc = shown[f"auc_{name}"].to_numpy()
        low = np.where(np.isnan(shown[f"auc_{name}_low"]), auc, shown[f"auc_{name}_low"])
        high = np.where(np.isnan(shown[f"auc_{name}_high"]), auc, shown[f"auc_{name}_high"])
        ax.errorbar(
            auc,
            ypos + offset,
            xerr=[auc - low, high - auc],
            fmt="o",
            ms=4,
            lw=1,
            color=colour,
            label=name.upper(),
        )
    ax.axvline(overall["xgb"], color="#d97706", ls=":", lw=1)
    ax.axvline(overall["gru"], color="#2563eb", ls=":", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.dimension}: {r.segment}" for r in shown.itertuples()], fontsize=8)
    ax.set_xlabel("ROC AUC on held-out test snapshots (match-bootstrap 95% CI; dotted = overall)")
    ax.set_title("Event-time model AUC by segment (small segments omitted)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    snaps = score_test_snapshots()
    overall = {
        "n": len(snaps),
        "xgb": float(roc_auc_score(snaps["y"], snaps["p_xgb"])),
        "gru": float(roc_auc_score(snaps["y"], snaps["p_gru"])),
    }
    table = segment_table(snaps, args.n_boot)
    out_dir = ROOT / "reports" / "event_time"
    table.to_csv(out_dir / "error_analysis.csv", index=False)
    write_markdown(table, overall, out_dir / "error_analysis.md", args.n_boot)
    plot(table, overall, ROOT / "reports" / "figures" / "error_analysis" / "auc_by_segment.png")
    print(f"overall: {overall}")
    print(
        table[
            ["dimension", "segment", "snapshots", "matches", "auc_xgb", "auc_gru", "small_sample"]
        ]
    )


if __name__ == "__main__":
    main()
