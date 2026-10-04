"""Before/after tables for the same-moment-leak fix (EXP-026), from the saved result files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OLD = Path("reports/event_time/superseded_pre_same_moment_fix")
NEW = Path("reports/event_time")
MODELS = ["LR_xy", "LR_len", "XGB_start", "XGB", "GRU"]


def tbl(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in df.itertuples(index=False):
        out += "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return out


def ci(boot: list[dict], model: str, stat: str) -> str:
    r = next(b for b in boot if b["model"] == model and b["stat"] == stat)
    return f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def main() -> None:
    o = json.loads((OLD / "event_time_results.json").read_text())
    n = json.loads((NEW / "event_time_results.json").read_text())
    L = [
        "# Same-moment leak fix: before / after (EXP-026)",
        "",
        "*Before* = `superseded_pre_same_moment_fix/` (danger judged on event location only); "
        "*after* = a possessing-team pass ending in the box is itself the danger event.  "
        "Same split, same hyper-parameter search on validation, same seeds.",
        "",
        f"Eligible test rows: {o['info']['rows']['test']:,} -> {n['info']['rows']['test']:,}; test possessions "
        f"with rows: {o['info']['possessions_with_rows']['test']:,} -> {n['info']['possessions_with_rows']['test']:,}; "
        f"row prevalence (train/val/test): "
        + " / ".join(
            f"{o['info']['prevalence_rows'][s]:.3f}->{n['info']['prevalence_rows'][s]:.3f}"
            for s in ("train", "val", "test")
        ),
        "",
    ]
    for key, title in (
        ("pooled_all", "All eligible snapshots"),
        ("pooled_one_per_possession", "One random snapshot per possession (seed 42)"),
    ):
        rows = []
        for m in MODELS:
            a = next(r for r in o[key] if r["model"] == m)
            b = next(r for r in n[key] if r["model"] == m)
            rows.append(
                {
                    "model": m,
                    "AUC before": a["roc_auc"],
                    "AUC after": b["roc_auc"],
                    "AP before": a["avg_precision"],
                    "AP after": b["avg_precision"],
                    "Brier before": a["brier"],
                    "Brier after": b["brier"],
                }
            )
        L += [f"## {title}", "", tbl(pd.DataFrame(rows))]
    L += [
        "## Mean (sd) ROC AUC over 20 one-per-possession draws",
        "",
        "| model | before | after |",
        "|---|---|---|",
    ]
    for m in MODELS:
        a, b = (
            o["mean_auc_20_one_per_possession_draws"][m],
            n["mean_auc_20_one_per_possession_draws"][m],
        )
        L.append(f"| {m} | {a[0]:.3f} ({a[1]:.3f}) | {b[0]:.3f} ({b[1]:.3f}) |")
    for key, title in (
        ("bootstrap_all", "all snapshots"),
        ("bootstrap_one", "one snapshot per possession"),
    ):
        L += [
            "",
            f"## Match-level bootstrap 95% intervals, ROC AUC ({title})",
            "",
            "| model / difference | before | after |",
            "|---|---|---|",
        ]
        names = MODELS + [
            "XGB - LR_xy",
            "XGB - LR_len",
            "GRU - LR_xy",
            "GRU - LR_len",
            "GRU - XGB",
            "XGB - XGB_start",
        ]
        for m in names:
            L.append(f"| {m} | {ci(o[key], m, 'roc_auc')} | {ci(n[key], m, 'roc_auc')} |")
    L += [
        "",
        "## Alerts, lead time and thresholds",
        "",
        "| model | threshold before -> after | dangerous alerted | safe alerted (false alarm) | median lead (events) | median lead (s) |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("XGB", "GRU"):
        a, b = o["tipping"][m], n["tipping"][m]
        L.append(
            f"| {m} | {a['threshold']:.3f} -> {b['threshold']:.3f} | "
            f"{a['dangerous_possessions_tipped']:.1%} -> {b['dangerous_possessions_tipped']:.1%} | "
            f"{a['safe_possessions_tipped (false alarm)']:.1%} -> {b['safe_possessions_tipped (false alarm)']:.1%} | "
            f"{a['median_lead_events']:.0f} -> {b['median_lead_events']:.0f} | "
            f"{a['median_lead_seconds']:.1f} -> {b['median_lead_seconds']:.1f} |"
        )
    for name, title in (
        ("buckets_events.csv", "events-so-far bucket"),
        ("buckets_seconds.csv", "elapsed-seconds bucket"),
    ):
        a = pd.read_csv(OLD / name)
        b = pd.read_csv(NEW / name)
        m = a.merge(b, on=["bucket", "model"], suffixes=(" before", " after"))
        m = m[m.model.isin(["XGB", "GRU", "LR_xy"])][
            ["bucket", "model", "n before", "n after", "roc_auc before", "roc_auc after"]
        ]
        m.columns = ["bucket", "model", "n before", "n after", "AUC before", "AUC after"]
        L += ["", f"## ROC AUC by {title}", "", tbl(m)]
    (NEW / "same_moment_fix_before_after.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
