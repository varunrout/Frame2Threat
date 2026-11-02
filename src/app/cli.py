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

frame2threat sample-possession [-o sample.json] [--seed S] [--dangerous 0|1]
    Write one real held-out (test-split) possession as a score-live input, so the CLI can be
    demonstrated on genuine StatsBomb data without hand-writing JSON.

frame2threat train-early
    DEPRECATED: the fraction-based v3 trainer.  Prints a message and exits; the withdrawn models
    live in models/superseded/.  Train the event-time models with scripts/run_event_time.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers — lazy-loaded so the CLI stays fast for --help
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Walk up from this file to find the project root (contains src/)."""
    if os.environ.get("FRAME2THREAT_ROOT"):
        return Path(os.environ["FRAME2THREAT_ROOT"]).resolve()
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


from src.app.validation import InputError  # noqa: E402


def _write_table(df, out_path: str) -> None:
    """Write ``df`` as CSV or Parquet according to the file extension."""
    out = Path(out_path)
    suffix = out.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise InputError(f"unsupported output extension '{out.suffix}': use .csv or .parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# sample-possession
# ---------------------------------------------------------------------------


def cmd_sample_possession(args: argparse.Namespace) -> None:
    """Write one real held-out (test-split) possession as a ``score-live`` JSON input."""
    import pandas as pd

    root = _project_root()
    sys.path.insert(0, str(root))
    from src.app.validation import require_file

    poss_path = require_file(
        root / "data" / "processed" / "possession_sequences.parquet",
        "possession table (run the data pipeline first)",
    )
    split_path = require_file(root / "data" / "processed" / "split_manifest.csv", "split manifest")
    poss = pd.read_parquet(poss_path)
    manifest = pd.read_csv(split_path)
    test_ids = set(manifest.loc[manifest["split"] == "test", "match_id"])
    pool = poss[
        poss["match_id"].isin(test_ids) & poss["n_events"].between(args.min_events, args.max_events)
    ]
    if args.dangerous is not None:
        pool = pool[pool["poss_dangerous"] == int(args.dangerous)]
    if pool.empty:
        raise InputError("no held-out possession satisfies the requested filters")
    row = pool.sample(1, random_state=args.seed).iloc[0]
    from src.app.live_scoring import arrays_to_payload
    from src.features.event_time import build_possession_arrays

    events_path = require_file(root / "data/processed/events_parsed.parquet", "parsed events")
    events = pd.read_parquet(
        events_path,
        filters=[
            ("match_id", "==", int(row["match_id"])),
            ("possession_id", "==", int(row["possession_id"])),
        ],
    )
    arr = build_possession_arrays(events)[0]
    payload = arrays_to_payload(arr)
    payload["retrospective_label_poss_dangerous"] = int(row["poss_dangerous"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"Wrote held-out possession (match {payload['match_id']}, possession "
        f"{payload['possession_id']}, {len(payload['event_sequence'])} events) -> {out}"
    )


# ---------------------------------------------------------------------------
# score-batch
# ---------------------------------------------------------------------------


def cmd_score_batch(args: argparse.Namespace) -> None:
    """Score every event of every possession with the event-time (in-play) models."""
    import pandas as pd

    root = _project_root()
    sys.path.insert(0, str(root))
    from src.app.batch_scoring import score_events

    from src.app.validation import require_artefacts, require_file, validate_events_table

    path = Path(args.events)
    if not path.is_absolute() and not path.exists():
        path = root / path
    require_file(path, "events parquet")
    require_artefacts(root, "both")
    try:
        events = pd.read_parquet(path)
    except Exception as exc:  # unreadable / not a parquet file
        raise InputError(f"cannot read {path} as parquet: {exc}") from exc
    validate_events_table(events)
    print(
        "MODE: event-time (prospective) - step t uses events 0..t only; not a retrospective score."
    )
    result = score_events(events, root, limit=args.limit, seed=args.seed)
    out_path = args.output or "event_time_scores.csv"
    _write_table(result, out_path)
    n_poss = result.groupby(["match_id", "possession_id"]).ngroups
    print(f"Saved {len(result):,} event rows for {n_poss:,} possessions -> {out_path}")
    print(
        "Columns: p_xgb / p_gru = P(shot or box entry by the possessing team in a later event | events so far);"
    )
    print("alert_* uses the validation-chosen thresholds in models/eventtime_meta.json;")
    print(
        "label_danger_after is defined only where eligible (no shot / box entry yet, not the last event)."
    )

    lab = result[result["eligible"]]
    if lab["label_danger_after"].nunique() > 1:
        from sklearn.metrics import roc_auc_score

        y = lab["label_danger_after"].astype(int)
        print(f"Eligible snapshots: {len(lab):,}  prevalence {y.mean():.3f}")
        print(
            f"  XGBoost ROC AUC {roc_auc_score(y, lab['p_xgb']):.4f}   GRU ROC AUC {roc_auc_score(y, lab['p_gru']):.4f}"
        )
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

    from src.app.validation import require_artefacts, require_file, validate_event_list

    path = require_file(Path(args.json_file), "input JSON")
    try:
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc
    meta_in: dict = {}
    if isinstance(events, dict):
        meta_in = {k: v for k, v in events.items() if k in ("origin", "period", "minute")}
        events = events.get("event_sequence", events.get("events", []))
    validate_event_list(events)
    require_artefacts(root, args.model)
    try:
        period, minute = int(meta_in.get("period", 1)), float(meta_in.get("minute", 45.0))
    except (TypeError, ValueError):
        raise InputError("'period' must be an integer and 'minute' a number") from None

    res = score_stream(
        events,
        root,
        model=args.model,
        threshold=args.threshold,
        origin=meta_in.get("origin", "Regular Play"),
        period=period,
        minute=minute,
    )
    print(
        "MODE: event-time (prospective) - step t uses events 0..t only; not a retrospective score."
    )
    print(
        f"Possession: {len(events)} events | model: {res['model']} (event-time) | "
        f"alert threshold: {res['threshold']:.3f}"
    )
    print(
        "P(danger) = P(shot or box entry by the possessing team in a later event | events so far)"
    )
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
        print(
            f"{s['t'] + 1:4d}  {s['event']:<16} {team:<5} {s['p']:10.4f}  "
            f"{'YES' if s['alert'] else '   '}{mark}"
        )
    print()
    tip = next((s for s in res["steps"] if s["tipping"]), None)
    scored = [s for s in res["steps"] if s["p"] is not None]
    if scored:
        print(f"Last scored step ({scored[-1]['t'] + 1}): P(danger) = {scored[-1]['p']:.4f}")
    if tip:
        print(
            f"First alert at step {tip['t'] + 1} ({tip['event']}), "
            f"{len(events) - tip['t'] - 1} event(s) before the end of the input."
        )
    else:
        print("No alert triggered.")
    if args.output:
        import pandas as pd

        table = pd.DataFrame(res["steps"]).rename(columns={"p": "p_danger"})
        table.insert(0, "model", res["model"])
        table["threshold"] = res["threshold"]
        _write_table(table, args.output)
        print(f"Saved {len(table)} rows -> {args.output}")


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
    p_batch.add_argument(
        "--limit", type=int, default=None, help="Score a random sample of N possessions"
    )
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
    p_live.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Also write per-event scores (.csv or .parquet)",
    )
    p_live.set_defaults(func=cmd_score_live)

    # sample-possession
    p_sample = sub.add_parser(
        "sample-possession",
        help="Write a real held-out test-split possession as a score-live input JSON",
    )
    p_sample.add_argument("-o", "--output", default="sample_possession.json")
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--min-events", type=int, default=8)
    p_sample.add_argument("--max-events", type=int, default=40)
    p_sample.add_argument(
        "--dangerous", type=int, choices=[0, 1], default=None, help="Restrict to (non-)dangerous"
    )
    p_sample.set_defaults(func=cmd_sample_possession)

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

    try:
        args.func(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
