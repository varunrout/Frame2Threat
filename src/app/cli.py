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

frame2threat score-batch [events.parquet] [-o out.csv] [--limit N] [--seed S]
    Score every event of every possession with the event-time models: per event, the XGBoost
    and GRU probabilities, the alert flags (thresholds chosen on validation, models/eventtime_meta.json)
    and the label where defined.  Input is a flat events parquet (events_parsed.parquet schema).

frame2threat train-early
    DEPRECATED: the fraction-based v3 trainer.  Prints a message and exits; the withdrawn models
    live in models/superseded/.  Train the event-time models with scripts/run_event_time.py.
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


# ---------------------------------------------------------------------------
# score-batch
# ---------------------------------------------------------------------------


def cmd_score_batch(args: argparse.Namespace) -> None:
    """Score every event of every possession with the event-time (in-play) models."""
    import pandas as pd

    root = _project_root()
    sys.path.insert(0, str(root))
    from src.app.batch_scoring import score_events

    path = Path(args.events)
    if not path.is_absolute():
        path = root / path
    events = pd.read_parquet(path)
    result = score_events(events, root, limit=args.limit, seed=args.seed)
    out_path = args.output or "event_time_scores.csv"
    result.to_csv(out_path, index=False)
    n_poss = result.groupby(["match_id", "possession_id"]).ngroups
    print(f"Saved {len(result):,} event rows for {n_poss:,} possessions -> {out_path}")
    print("Columns: p_xgb / p_gru = P(shot or box entry by the possessing team in a later event | events so far);")
    print("alert_* uses the validation-chosen thresholds in models/eventtime_meta.json;")
    print("label_danger_after is defined only where eligible (no shot / box entry yet, not the last event).")

    lab = result[result["eligible"]]
    if lab["label_danger_after"].nunique() > 1:
        from sklearn.metrics import roc_auc_score

        y = lab["label_danger_after"].astype(int)
        print(f"Eligible snapshots: {len(lab):,}  prevalence {y.mean():.3f}")
        print(f"  XGBoost ROC AUC {roc_auc_score(y, lab['p_xgb']):.4f}   GRU ROC AUC {roc_auc_score(y, lab['p_gru']):.4f}")
        print(
            "  (the models were trained on the train matches; use only held-out matches for evaluation)"
        )


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
    """Deprecated: the fraction-based v3 models were withdrawn."""
    print(
        "DEPRECATED: 'train-early' trained the fraction-based v3 models, which were withdrawn "
        "(prefix length encodes the possession length and many prefixes already contain the outcome; "
        "see reports/feature_timing_audit.md).  The models and the old scoring path are kept under "
        "models/superseded/ for reference only."
        + "\n"
        + "Train the event-time models with:  python scripts/run_event_time.py",
        file=sys.stderr,
    )
    sys.exit(2)


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
        help="Score every event of every possession with the event-time models",
    )
    p_batch.add_argument(
        "events",
        nargs="?",
        default="data/processed/events_parsed.parquet",
        help="Flat events parquet (events_parsed.parquet schema; default: data/processed/events_parsed.parquet)",
    )
    p_batch.add_argument(
        "-o", "--output", type=str, default=None, help="Output CSV (default: event_time_scores.csv)"
    )
    p_batch.add_argument("--limit", type=int, default=None, help="Score a random sample of N possessions")
    p_batch.add_argument("--seed", type=int, default=42, help="Seed for --limit sampling")
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
        help="[DEPRECATED] withdrawn fraction-based v3 trainer; prints a message and exits",
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
