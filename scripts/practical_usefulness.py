#!/usr/bin/env python
"""
Practical Usefulness Evaluation
================================
Evaluates operational usefulness of event-time models beyond ROC-AUC.
Metrics: Precision/Recall at validation thresholds, false alerts, lead time,
calibration, Brier score, performance by possession phase.

Outputs:
- reports/repro/practical_usefulness.json
- reports/figures/practical_usefulness.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.app import possession_data as pg
from src.app.live_scoring import score_arrays


def load_test_data(root: Path):
    """Load test split data."""
    missing = pg.missing_artefacts(root)
    if missing:
        raise SystemExit("Missing artefacts: " + "; ".join(missing))

    poss = pd.read_parquet(root / pg.REQUIRED["possession table"])
    manifest = pd.read_csv(root / pg.REQUIRED["split manifest"])
    arrays = pg.load_arrays(root / pg.REQUIRED["event-time arrays"])

    ids = pg.held_out_match_ids(manifest, "test")
    pool = poss[poss["match_id"].isin(ids) & (poss["n_events"] >= 4)]

    test_arrays = []
    for _, row in pool.iterrows():
        arr = arrays.get((int(row.match_id), int(row.possession_id)))
        if arr is not None:
            test_arrays.append(arr)

    return test_arrays, pool


def scored_steps(res: dict) -> list[tuple[dict, int]]:
    """(step, label) for every scored step; label = ``danger_after_t`` of the event-time formulation.

    A step before the outcome has label 1 iff the possession later produces a shot or box entry.  The
    last event of a possession that never does is not eligible (nothing follows it) and is dropped.
    Realised steps carry no score, so they must never be used as the label source.
    """
    dangerous = res.get("realised_at") is not None
    last = len(res["steps"]) - 1
    return [
        (s, int(dangerous))
        for s in res["steps"]
        if s["p"] is not None and (dangerous or s["t"] < last)
    ]


def evaluate_at_threshold(test_arrays, root: Path, model: str, threshold: float):
    """Evaluate model at a specific threshold."""
    all_probs = []
    all_labels = []
    all_alerts = []
    all_first_alerts = []
    all_lead_times = []
    false_alerts = 0
    true_alerts = 0

    for arr in test_arrays:
        res = score_arrays(arr, root, model=model, threshold=threshold)
        pairs = scored_steps(res)
        probs = [s["p"] for s, _ in pairs]
        labels = [y for _, y in pairs]

        # Per-possession evaluation
        realised_at = res.get("realised_at")
        first_alert = next((s["t"] for s in res["steps"] if s["tipping"]), None)

        if realised_at is not None:
            # Dangerous possession
            if first_alert is not None:
                true_alerts += 1
                lead_time = realised_at - first_alert
                all_lead_times.append(lead_time)
            else:
                # Missed detection
                pass
        else:
            # Safe possession
            if first_alert is not None:
                false_alerts += 1

        all_probs.extend(probs)
        all_labels.extend(labels)
        all_alerts.extend([s["alert"] for s, _ in pairs])
        if first_alert is not None:
            all_first_alerts.append(first_alert)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_alerts = np.array(all_alerts)

    # Precision/Recall at threshold
    tp = np.sum((all_alerts == 1) & (all_labels == 1))
    fp = np.sum((all_alerts == 1) & (all_labels == 0))
    fn = np.sum((all_alerts == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # False alert rate (per possession)
    n_dangerous = sum(
        1
        for arr in test_arrays
        if score_arrays(arr, root, model=model).get("realised_at") is not None
    )
    n_safe = len(test_arrays) - n_dangerous
    false_alert_rate = false_alerts / n_safe if n_safe > 0 else 0

    # Lead time stats
    lead_time_stats = {}
    if all_lead_times:
        lead_time_stats = {
            "mean": float(np.mean(all_lead_times)),
            "median": float(np.median(all_lead_times)),
            "min": int(np.min(all_lead_times)),
            "max": int(np.max(all_lead_times)),
            "std": float(np.std(all_lead_times)),
        }

    # Calibration
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10, strategy="uniform")
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Brier score
    brier = np.mean((all_probs - all_labels) ** 2)

    # Performance by event count bucket
    bucket_stats = {}
    for bucket_name, (min_e, max_e) in [
        ("1-3", (1, 3)),
        ("4-6", (4, 6)),
        ("7-10", (7, 10)),
        ("11+", (11, 100)),
    ]:
        bucket_probs = []
        bucket_labels = []
        for arr in test_arrays:
            res = score_arrays(arr, root, model=model, threshold=threshold)
            for s, y in scored_steps(res):
                if min_e <= s["t"] + 1 <= max_e:
                    bucket_probs.append(s["p"])
                    bucket_labels.append(y)

        if bucket_probs:
            bucket_probs = np.array(bucket_probs)
            bucket_labels = np.array(bucket_labels)
            from sklearn.metrics import roc_auc_score, average_precision_score

            bucket_stats[bucket_name] = {
                "n": len(bucket_probs),
                "prevalence": float(np.mean(bucket_labels)),
                "roc_auc": (
                    float(roc_auc_score(bucket_labels, bucket_probs))
                    if len(np.unique(bucket_labels)) > 1
                    else 0.5
                ),
                "avg_precision": (
                    float(average_precision_score(bucket_labels, bucket_probs))
                    if len(np.unique(bucket_labels)) > 1
                    else np.mean(bucket_labels)
                ),
                "brier": float(np.mean((bucket_probs - bucket_labels) ** 2)),
            }

    return {
        "model": model,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alert_rate": false_alert_rate,
        "false_alerts": false_alerts,
        "true_alerts": true_alerts,
        "lead_time": lead_time_stats,
        "ece": float(ece),
        "brier": float(brier),
        "bucket_stats": bucket_stats,
        "n_possessions": len(test_arrays),
        "n_dangerous": n_dangerous,
        "n_safe": n_safe,
    }


def plot_practical_usefulness(xgb_results, gru_results, output_path: Path):
    """Create practical usefulness visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Precision-Recall at threshold
    ax = axes[0, 0]
    models = ["XGBoost", "GRU"]
    precisions = [xgb_results["precision"], gru_results["precision"]]
    recalls = [xgb_results["recall"], gru_results["recall"]]
    f1s = [xgb_results["f1"], gru_results["f1"]]

    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, precisions, width, label="Precision", color="#E67E22", alpha=0.8)
    ax.bar(x, recalls, width, label="Recall", color="#3498DB", alpha=0.8)
    ax.bar(x + width, f1s, width, label="F1", color="#27AE60", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 at Validation Threshold")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # 2. False Alert Rate
    ax = axes[0, 1]
    far = [xgb_results["false_alert_rate"], gru_results["false_alert_rate"]]
    ax.bar(models, far, color=["#E67E22", "#3498DB"], alpha=0.8)
    ax.set_ylabel("False Alert Rate (per safe possession)")
    ax.set_title("False Alert Rate")
    ax.grid(axis="y", alpha=0.3)

    # 3. Lead Time Distribution
    ax = axes[0, 2]
    xgb_lead = xgb_results["lead_time"]
    gru_lead = gru_results["lead_time"]
    if xgb_lead and gru_lead:
        ax.bar(
            ["XGBoost", "GRU"],
            [xgb_lead["median"], gru_lead["median"]],
            yerr=[
                [xgb_lead["median"] - xgb_lead["min"], gru_lead["median"] - gru_lead["min"]],
                [xgb_lead["max"] - xgb_lead["median"], gru_lead["max"] - gru_lead["median"]],
            ],
            color=["#E67E22", "#3498DB"],
            alpha=0.8,
            capsize=5,
        )
        ax.set_ylabel("Lead Time (events)")
        ax.set_title("Median Lead Time Before Outcome")
        ax.grid(axis="y", alpha=0.3)

    # 4. Calibration
    ax = axes[1, 0]
    # We'd need the calibration curves; for now show ECE
    ece = [xgb_results["ece"], gru_results["ece"]]
    brier = [xgb_results["brier"], gru_results["brier"]]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, ece, width, label="ECE (10 bins)", color="#E74C3C", alpha=0.8)
    ax.bar(x + width / 2, brier, width, label="Brier Score", color="#8E44AD", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Calibration & Brier Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 5. Performance by Event Bucket
    ax = axes[1, 1]
    buckets = ["1-3", "4-6", "7-10", "11+"]
    xgb_auc = [xgb_results["bucket_stats"].get(b, {}).get("roc_auc", 0) for b in buckets]
    gru_auc = [gru_results["bucket_stats"].get(b, {}).get("roc_auc", 0) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35
    ax.bar(x - width / 2, xgb_auc, width, label="XGBoost", color="#E67E22", alpha=0.8)
    ax.bar(x + width / 2, gru_auc, width, label="GRU", color="#3498DB", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("ROC AUC")
    ax.set_title("ROC AUC by Events-So-Far Bucket")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.5, 0.8)

    # 6. Ball Position Baseline Comparison
    ax = axes[1, 2]
    # From repair retrain results
    baselines = ["Ball XY LR", "Events LR", "XGB Start", "XGBoost", "GRU"]
    auc_vals = [0.642, 0.561, 0.590, 0.665, 0.664]
    colors = ["#95A5A6", "#95A5A6", "#95A5A6", "#E67E22", "#3498DB"]
    ax.barh(baselines, auc_vals, color=colors, alpha=0.8)
    ax.set_xlabel("ROC AUC (all snapshots)")
    ax.set_title("Model vs Baselines (Held-Out Test)")
    ax.set_xlim(0.5, 0.7)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Practical Usefulness Evaluation — Event-Time Danger Forecasting",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate practical usefulness")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "repro")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "reports" / "figures" / "practical")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    test_arrays, pool = load_test_data(ROOT)
    print(f"Loaded {len(test_arrays)} test possessions")

    # Get validation thresholds
    meta = json.loads((ROOT / "models" / "eventtime_meta.json").read_text(encoding="utf-8"))
    xgb_thr = meta["xgb_threshold"]
    gru_thr = meta["gru_threshold"]

    print(f"Evaluating XGBoost at threshold {xgb_thr:.3f}...")
    xgb_results = evaluate_at_threshold(test_arrays, ROOT, "xgb", xgb_thr)

    print(f"Evaluating GRU at threshold {gru_thr:.3f}...")
    gru_results = evaluate_at_threshold(test_arrays, ROOT, "gru", gru_thr)

    # Save results
    results = {
        "xgb": xgb_results,
        "gru": gru_results,
        "validation_thresholds": {"xgb": xgb_thr, "gru": gru_thr},
        "test_possessions": len(test_arrays),
    }

    with open(args.output_dir / "practical_usefulness.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Creating visualization...")
    plot_practical_usefulness(xgb_results, gru_results, args.fig_dir / "practical_usefulness.png")

    print(f"\nResults saved to {args.output_dir / 'practical_usefulness.json'}")
    print(f"Figure saved to {args.fig_dir / 'practical_usefulness.png'}")

    # Print summary
    print("\n=== SUMMARY ===")
    for model, res in [("XGBoost", xgb_results), ("GRU", gru_results)]:
        print(f"\n{model}:")
        print(f"  Threshold: {res['threshold']:.3f}")
        print(f"  Precision: {res['precision']:.3f}")
        print(f"  Recall: {res['recall']:.3f}")
        print(f"  F1: {res['f1']:.3f}")
        print(f"  False Alert Rate: {res['false_alert_rate']:.3f}")
        print(f"  ECE: {res['ece']:.3f}")
        print(f"  Brier: {res['brier']:.3f}")
        if res["lead_time"]:
            print(f"  Median Lead Time: {res['lead_time']['median']:.1f} events")


if __name__ == "__main__":
    main()
