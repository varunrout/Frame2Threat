#!/usr/bin/env python
"""
Interpretability Analysis for Event-Time Models
================================================
Generates SHAP-based feature importance for XGBoost and trajectory analysis for GRU.
Outputs:
- reports/figures/interpretability/shap_summary.png
- reports/repro/interpretability_xgb.json
- reports/repro/interpretability_gru.json
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
import shap

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.app import possession_data as pg
from src.features.event_time import FEATURE_NAMES
from src.models.event_time_artifacts import load_event_time_xgb


def load_test_data(root: Path):
    """Load test split data for interpretability analysis."""
    missing = pg.missing_artefacts(root)
    if missing:
        raise SystemExit("Missing artefacts: " + "; ".join(missing))

    poss = pd.read_parquet(root / pg.REQUIRED["possession table"])
    manifest = pd.read_csv(root / pg.REQUIRED["split manifest"])
    arrays = pg.load_arrays(root / pg.REQUIRED["event-time arrays"])

    ids = pg.held_out_match_ids(manifest, "test")
    pool = poss[poss["match_id"].isin(ids) & (poss["n_events"] >= 4)]

    # Sample a subset for SHAP computation (too slow on all)
    sampled = pool.sample(n=min(50, len(pool)), random_state=42)

    test_arrays = []
    for _, row in sampled.iterrows():
        arr = arrays.get((int(row.match_id), int(row.possession_id)))
        if arr is not None:
            test_arrays.append(arr)

    return test_arrays, sampled


def compute_xgb_shap(root: Path, test_arrays, max_samples: int = 100):
    """Compute SHAP values for XGBoost model."""
    meta = json.loads((root / "models" / "eventtime_meta.json").read_text(encoding="utf-8"))
    art = load_event_time_xgb(root, meta)
    model = art["model"]
    features = art["features"]

    # Build feature matrix from test arrays
    from src.features.event_time import snapshot_features

    X_rows = []
    for arr in test_arrays[:max_samples]:
        feats = snapshot_features(arr)
        X_rows.append(feats)

    if not X_rows:
        return None, None

    X = np.vstack(X_rows)
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)[features]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # For binary classification, shap_values is list of two arrays; take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_df


def compute_gru_trajectory_analysis(test_arrays, root: Path):
    """Analyze GRU prediction trajectories."""
    from src.app.live_scoring import score_arrays as sa

    trajectories = []
    for arr in test_arrays:
        res = sa(arr, root, model="gru")
        probs = [s["p"] for s in res["steps"] if s["p"] is not None]
        if probs:
            trajectories.append(
                {
                    "match_id": arr.match_id,
                    "possession_id": arr.possession_id,
                    "n_events": len(probs),
                    "probs": probs,
                    "threshold": res["threshold"],
                    "first_alert": next(
                        (i for i, s in enumerate(res["steps"]) if s["tipping"]), None
                    ),
                    "realised_at": res.get("realised_at"),
                }
            )

    return trajectories


def plot_shap_summary(shap_values, X_df, output_path: Path):
    """Create SHAP summary plot."""
    # Bar plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.title("XGBoost Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also create beeswarm plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("XGBoost SHAP Beeswarm Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = output_path.parent / "shap_beeswarm.png"
    fig.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gru_trajectories(trajectories, output_path: Path):
    """Plot GRU probability trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, traj in enumerate(trajectories[:4]):
        ax = axes[idx]
        probs = traj["probs"]
        ax.plot(range(1, len(probs) + 1), probs, color="#3498DB", lw=2, marker="s", ms=3)
        ax.axhline(
            traj["threshold"],
            color="#3498DB",
            ls="--",
            lw=1.5,
            alpha=0.7,
            label=f"Threshold ({traj['threshold']:.3f})",
        )

        if traj["first_alert"] is not None:
            ax.axvline(
                traj["first_alert"] + 1,
                color="#E74C3C",
                ls=":",
                lw=2,
                alpha=0.8,
                label=f"First alert (event {traj['first_alert']+1})",
            )

        if traj["realised_at"] is not None:
            ax.axvline(
                traj["realised_at"] + 1,
                color="#2C3E50",
                ls="-",
                lw=2.5,
                alpha=0.9,
                label=f"Outcome (event {traj['realised_at']+1})",
            )
            ax.axvspan(traj["realised_at"] + 1, len(probs) + 0.5, alpha=0.1, color="gray")

        ax.set_xlim(0.5, len(probs) + 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Event")
        ax.set_ylabel("P(danger)")
        ax.set_title(
            f"Match {traj['match_id']}, Poss {traj['possession_id']} ({len(probs)} events)"
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.suptitle("GRU Probability Trajectories (Test Possessions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(test_arrays, root: Path, output_path: Path):
    """Plot XGBoost vs GRU comparison on same possessions."""
    from src.app.live_scoring import score_arrays as sa2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, arr in enumerate(test_arrays[:4]):
        ax = axes[idx]
        xgb_res = sa2(arr, root, model="xgb")
        gru_res = sa2(arr, root, model="gru")

        xgb_probs = [s["p"] for s in xgb_res["steps"] if s["p"] is not None]
        gru_probs = [s["p"] for s in gru_res["steps"] if s["p"] is not None]

        ax.plot(
            range(1, len(xgb_probs) + 1),
            xgb_probs,
            color="#E67E22",
            lw=2,
            marker="o",
            ms=3,
            label="XGBoost",
        )
        ax.plot(
            range(1, len(gru_probs) + 1),
            gru_probs,
            color="#3498DB",
            lw=2,
            marker="s",
            ms=3,
            label="GRU",
        )

        ax.axhline(xgb_res["threshold"], color="#E67E22", ls="--", lw=1, alpha=0.7)
        ax.axhline(gru_res["threshold"], color="#3498DB", ls="--", lw=1, alpha=0.7)

        xgb_tip = next((s["t"] for s in xgb_res["steps"] if s["tipping"]), None)
        gru_tip = next((s["t"] for s in gru_res["steps"] if s["tipping"]), None)

        if xgb_tip is not None:
            ax.axvline(xgb_tip + 1, color="#E67E22", ls=":", lw=1.5, alpha=0.8)
        if gru_tip is not None:
            ax.axvline(gru_tip + 1, color="#3498DB", ls=":", lw=1.5, alpha=0.8)

        realised_at = xgb_res.get("realised_at")
        if realised_at is not None:
            ax.axvline(realised_at + 1, color="#2C3E50", ls="-", lw=2, alpha=0.9)
            ax.axvspan(realised_at + 1, len(arr.type_id) + 0.5, alpha=0.1, color="gray")

        ax.set_xlim(0.5, len(arr.type_id) + 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Event")
        ax.set_ylabel("P(danger)")
        ax.set_title(f"Match {arr.match_id}, Poss {arr.possession_id}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "XGBoost vs GRU: Probability Trajectories Comparison", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate interpretability analysis")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "figures" / "interpretability"
    )
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    test_arrays, sampled = load_test_data(ROOT)
    print(f"Loaded {len(test_arrays)} test possessions")

    print("Computing XGBoost SHAP values...")
    shap_values, X_df = compute_xgb_shap(ROOT, test_arrays, args.max_samples)

    if shap_values is not None:
        print("Creating SHAP summary plot...")
        plot_shap_summary(shap_values, X_df, output_dir / "shap_summary.png")

        # Save SHAP values
        shap_importance = pd.DataFrame(
            {
                "feature": X_df.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
        print(f"Saved SHAP importance to {output_dir / 'shap_importance.csv'}")

    print("Analyzing GRU trajectories...")
    gru_trajectories = compute_gru_trajectory_analysis(test_arrays, ROOT)

    print("Creating GRU trajectory plots...")
    plot_gru_trajectories(gru_trajectories, output_dir / "gru_trajectories.png")

    print("Creating model comparison plots...")
    plot_model_comparison(test_arrays, ROOT, output_dir / "model_comparison.png")

    # Save trajectory data
    with open(output_dir / "gru_trajectories.json", "w") as f:
        json.dump(gru_trajectories, f, indent=2)

    print(f"\nDone! Interpretability outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
