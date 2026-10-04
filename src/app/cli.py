"""
cli.py
======
Command-line interface for Frame2Threat early-warning scoring pipeline.

Entry point registered in pyproject.toml as ``frame2threat``.

Subcommands
-----------
frame2threat score-live <json-file> [--model gru|xgb] [--threshold T]
    Score a single possession event-by-event with the event-time (in-play) model,
    printing a danger trajectory.  The JSON file holds a list of event dicts (the
    possession_sequences.parquet ``event_sequence`` schema, optionally with ``own_team``
    and ``elapsed_seconds`` per event).  Scoring at step t uses events 0..t only.

SUPERSEDED (fraction-based v3; its prefix length encodes the possession length, see
reports/feature_timing_audit.md -- kept runnable for reproducibility only):

frame2threat score-batch <parquet> [-o output.csv] [--fracs 0.25,0.50,0.75,1.00]
frame2threat train-early
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers — lazy-loaded so the CLI stays fast for --help
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Walk up from this file to find the project root (contains src/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir() and (p / "configs").is_dir():
            return p
        p = p.parent
    # Fallback: current working directory
    return Path.cwd()


def _load_config(root: Path) -> dict:
    import yaml

    cfg_path = root / "configs" / "model_possession.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_gru(root: Path, cfg: dict):
    """Load the GRU model (returns model, config dict)."""
    sys.path.insert(0, str(root))
    from src.evaluation.possession_attribution import load_gru_model

    gru_path = root / cfg["early_warning"]["gru_model"]
    return load_gru_model(str(gru_path))


def _load_xgb_cumulative(root: Path, cfg: dict, pct: int):
    """Load a cumulative XGBoost model for a given percentage."""
    import joblib

    pattern = cfg["early_warning"]["xgb_model_pattern"]
    model_path = root / pattern.format(pct=pct)
    if model_path.exists():
        return joblib.load(model_path)
    return None


def _load_xgb_start(root: Path, cfg: dict):
    """Load the start-only XGBoost model."""
    import joblib

    model_path = root / cfg["early_warning"]["start_model"]
    if model_path.exists():
        return joblib.load(model_path)
    return None


# ---------------------------------------------------------------------------
# score-batch
# ---------------------------------------------------------------------------


def cmd_score_batch(args: argparse.Namespace) -> None:
    """Score all possessions in a parquet file at multiple observation fractions."""
    import numpy as np
    import pandas as pd
    import torch

    root = _project_root()
    sys.path.insert(0, str(root))
    cfg = _load_config(root)
    ew = cfg["early_warning"]

    from src.data.parse_possessions import load_possession_sequences
    from src.labels.possession_labels import attach_possession_labels
    from src.features.early_features import (
        build_start_features,
        build_cumulative_tabular_features,
    )
    from src.evaluation.possession_attribution import cumulative_danger_scores, _parse_event_seq

    # Parse fractions
    if args.fracs:
        fracs = [float(f) for f in args.fracs.split(",")]
    else:
        fracs = ew["fractions"]

    threshold = ew["alert_threshold"]
    min_events = ew["min_events"]

    # Load data
    print(f"Loading {args.parquet} …")
    poss_df = load_possession_sequences(args.parquet)
    if "poss_tempo" not in poss_df.columns:
        poss_df = attach_possession_labels(poss_df)
    print(f"  {len(poss_df):,} possessions loaded")

    # Filter possessions with enough events
    n_events_series = poss_df["n_events"].astype(int)
    mask = n_events_series >= min_events
    poss_df = poss_df.loc[mask].reset_index(drop=True)
    print(f"  {len(poss_df):,} after min_events={min_events} filter")

    # Load models
    print("Loading models …")
    gru_model, _ = _load_gru(root, cfg)
    xgb_start = _load_xgb_start(root, cfg)

    xgb_models = {}
    for frac in fracs:
        pct = int(frac * 100)
        if pct == 100:
            import joblib

            main_path = root / "models" / "xgboost_poss_dangerous.joblib"
            if main_path.exists():
                xgb_models[pct] = joblib.load(main_path)
        else:
            m = _load_xgb_cumulative(root, cfg, pct)
            if m is not None:
                xgb_models[pct] = m

    N = len(poss_df)

    # --- Vectorised XGBoost scoring (one batch per fraction) ----------------
    print(f"Building tabular features for {len(fracs)} fraction(s) …")
    xgb_probs: dict[int, np.ndarray] = {}  # pct → array(N,)
    for frac in fracs:
        pct = int(frac * 100)
        if pct not in xgb_models:
            continue
        X_tab = build_cumulative_tabular_features(poss_df, frac=frac).fillna(0)
        xgb_probs[pct] = xgb_models[pct].predict_proba(X_tab)[:, 1]
        print(f"  frac={frac:.0%} XGB features+predict done")

    # Start-only XGBoost (one batch)
    start_probs: np.ndarray | None = None
    if xgb_start is not None:
        X_start = build_start_features(poss_df).fillna(0)
        start_probs = xgb_start.predict_proba(X_start)[:, 1]
        print("  start-only XGB features+predict done")

    # --- GRU scoring (per-possession, but fast on GPU/CPU) ------------------
    print("Computing GRU cumulative scores …")
    gru_cum: list[np.ndarray] = []
    for i in range(N):
        if i > 0 and i % 2000 == 0:
            print(f"  GRU {i:,}/{N:,}", end="\r")
        events = _parse_event_seq(poss_df.iloc[i])
        with torch.no_grad():
            gru_cum.append(cumulative_danger_scores(events, gru_model))
    print(f"  GRU {N:,}/{N:,} done")

    # --- Assemble results ---------------------------------------------------
    print("Assembling output …")
    rows = []
    n_events_arr = poss_df["n_events"].values.astype(int)
    match_ids = poss_df.get("match_id", pd.Series([""] * N)).values
    poss_ids = poss_df.get("possession_id", pd.Series([""] * N)).values
    teams = poss_df.get("team_name", pd.Series([""] * N)).values
    labels = poss_df.get("poss_dangerous", pd.Series([-1] * N)).values.astype(int)

    for i in range(N):
        cum_scores = gru_cum[i]
        n_ev = n_events_arr[i]

        for frac in fracs:
            pct = int(frac * 100)
            t = max(1, int(np.ceil(n_ev * frac)))
            t = min(t, len(cum_scores))
            gru_prob = float(cum_scores[t - 1])

            xgb_prob = float(xgb_probs[pct][i]) if pct in xgb_probs else None

            sp = None
            if start_probs is not None and frac == fracs[0]:
                sp = float(start_probs[i])

            alert = gru_prob >= threshold
            if xgb_prob is not None:
                alert = alert or (xgb_prob >= threshold)

            rows.append(
                {
                    "match_id": match_ids[i],
                    "possession_id": poss_ids[i],
                    "team_name": teams[i],
                    "n_events": n_ev,
                    "fraction": frac,
                    "events_observed": t,
                    "gru_prob": round(gru_prob, 4),
                    "xgb_prob": round(xgb_prob, 4) if xgb_prob is not None else None,
                    "start_prob": round(sp, 4) if sp is not None else None,
                    "alert": int(alert),
                    "true_label": labels[i],
                }
            )

    result = pd.DataFrame(rows)

    # Output
    out_path = args.output or "early_warning_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result):,} rows → {out_path}")

    # Summary stats
    if "true_label" in result.columns and (result["true_label"] >= 0).any():
        labelled = result[result["true_label"] >= 0]
        for frac in fracs:
            subset = labelled[labelled["fraction"] == frac]
            if len(subset) > 0 and subset["true_label"].nunique() > 1:
                from sklearn.metrics import roc_auc_score

                gru_auc = roc_auc_score(subset["true_label"], subset["gru_prob"])
                line = f"  frac={frac:.0%}  GRU AUC={gru_auc:.4f}"
                if subset["xgb_prob"].notna().any():
                    xgb_auc = roc_auc_score(
                        subset.loc[subset["xgb_prob"].notna(), "true_label"],
                        subset.loc[subset["xgb_prob"].notna(), "xgb_prob"],
                    )
                    line += f"  XGB AUC={xgb_auc:.4f}"
                print(line)


# ---------------------------------------------------------------------------
# score-live
# ---------------------------------------------------------------------------


def cmd_score_live(args: argparse.Namespace) -> None:
    """Score a possession event by event with the event-time (in-play) model."""
    root = _project_root()
    sys.path.insert(0, str(root))

    from src.app.live_scoring import score_stream

    with open(args.json_file, encoding="utf-8") as f:
        events = json.load(f)
    meta_in: dict = {}
    if isinstance(events, dict):
        meta_in = {k: v for k, v in events.items() if k in ("origin", "period", "minute")}
        events = events.get("event_sequence", events.get("events", []))
    if not events:
        print("No events found in the JSON file.")
        return

    res = score_stream(
        events,
        root,
        model=args.model,
        threshold=args.threshold,
        origin=meta_in.get("origin", "Regular Play"),
        period=int(meta_in.get("period", 1)),
        minute=float(meta_in.get("minute", 45.0)),
    )
    print(f"Possession: {len(events)} events | model: {res['model']} (event-time) | "
          f"alert threshold: {res['threshold']:.3f}")
    print("P(danger) = P(shot or box entry by the possessing team in a later event | events so far)")
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    print()
    header = f"{'Step':>4}  {'Event Type':<16} {'Team':<5} {'P(danger)':>10}  {'Alert':<5}"
    print(header)
    print("-" * len(header))
    for s in res["steps"]:
        team = "own" if s["own"] else "opp"
        if s["p"] is None:
            note = "  << OUTCOME REALISED (shot/box entry)" if s["realised"] else ""
            print(f"{s['t'] + 1:4d}  {s['event']:<16} {team:<5} {'-':>10}  {'':<5}{note}")
            continue
        mark = " << TIPPING POINT" if s["tipping"] else (" <<" if s["alert"] else "")
        print(f"{s['t'] + 1:4d}  {s['event']:<16} {team:<5} {s['p']:10.4f}  "
              f"{'YES' if s['alert'] else '   '}{mark}")
    print()
    tip = next((s for s in res["steps"] if s["tipping"]), None)
    scored = [s for s in res["steps"] if s["p"] is not None]
    if scored:
        print(f"Last scored step ({scored[-1]['t'] + 1}): P(danger) = {scored[-1]['p']:.4f}")
    if tip:
        print(f"First alert at step {tip['t'] + 1} ({tip['event']}), "
              f"{len(events) - tip['t'] - 1} event(s) before the end of the input.")
    else:
        print("No alert triggered.")


# ---------------------------------------------------------------------------
# train-early
# ---------------------------------------------------------------------------


def cmd_train_early(args: argparse.Namespace) -> None:
    """Run the early-model training script."""
    import subprocess

    root = _project_root()
    script = root / "src" / "models" / "train_early_models.py"
    print(f"Running {script} …")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame2threat",
        description="Frame2Threat — early-warning danger scoring pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # score-batch
    p_batch = sub.add_parser(
        "score-batch",
        help="[SUPERSEDED fraction-based v3] Score possessions at fixed observation fractions",
    )
    p_batch.add_argument(
        "parquet",
        type=str,
        help="Path to possession_sequences.parquet (or any parquet with event_sequence col)",
    )
    p_batch.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: early_warning_scores.csv)",
    )
    p_batch.add_argument(
        "--fracs",
        type=str,
        default=None,
        help="Comma-separated observation fractions (default: from config)",
    )
    p_batch.set_defaults(func=cmd_score_batch)

    # score-live
    p_live = sub.add_parser(
        "score-live",
        help="Score a single possession event-by-event (JSON input)",
    )
    p_live.add_argument(
        "json_file",
        type=str,
        help="Path to JSON file containing a list of event dicts",
    )
    p_live.add_argument(
        "--model", choices=["gru", "xgb"], default="gru", help="Event-time model (default: gru)"
    )
    p_live.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Alert threshold (default: value chosen on validation, models/eventtime_meta.json)",
    )
    p_live.set_defaults(func=cmd_score_live)

    # train-early
    p_train = sub.add_parser(
        "train-early",
        help="[SUPERSEDED fraction-based v3] Train cumulative-fraction XGBoost models",
    )
    p_train.set_defaults(func=cmd_train_early)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
