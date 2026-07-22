"""One-command v1 pass-level reproduction for Frame2Threat.

This script rebuilds the event-only v1 pass model path from existing pipeline
modules: ingest/parse, pass-instance construction, downstream labelling,
match-level splits, event-feature engineering, XGBoost training, and metrics.

It intentionally reproduces the event-only result first.  The event+360
ablation remains a follow-up because geometry labels/features require broader
360 coverage handling and are already documented as a marginal finding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest import get_360_frames, get_events
from src.data.inventory import build_inventory
from src.data.join_pass_frames import build_pass_instances
from src.data.parse_360 import get_frame_summary, parse_360_frames
from src.data.parse_events import parse_events
from src.data.splits import create_match_level_splits, materialise_split_parquets
from src.evaluation.metrics import classification_metrics
from src.features.event_features import build_event_features
from src.labels.dangerous_progression import compute_downstream_labels
from src.models.tabular import TabularClassifier

LOGGER = logging.getLogger("frame2threat.reproduce_v1")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_inventory(data_cfg: dict[str, Any]) -> pd.DataFrame:
    inventory_path = REPO_ROOT / data_cfg["statsbomb"]["interim_dir"] / "inventory.parquet"
    if inventory_path.exists():
        LOGGER.info("Loading cached inventory from %s", inventory_path)
        return pd.read_parquet(inventory_path)

    LOGGER.info("No cached inventory found; building inventory from StatsBomb")
    return build_inventory(data_cfg["statsbomb"]["competitions"])


def _select_matches(
    inventory: pd.DataFrame,
    *,
    max_matches: int,
    require_360: bool,
) -> pd.DataFrame:
    if inventory.empty:
        raise RuntimeError("Inventory is empty; cannot select matches")

    matches = inventory[inventory["has_events"].astype(bool)].copy()
    if require_360 and "has_360" in matches.columns:
        matches = matches[matches["has_360"].astype(bool)]

    if matches.empty:
        raise RuntimeError("No matches available after event/360 filters")

    matches = matches.sort_values(["competition_id", "season_id", "match_id"])
    return matches.head(max_matches).reset_index(drop=True)


def _build_from_matches(
    selected_matches: pd.DataFrame,
    labels_cfg: dict[str, Any],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for i, row in selected_matches.iterrows():
        match_id = int(row["match_id"])
        competition_id = int(row["competition_id"])
        season_id = int(row["season_id"])
        LOGGER.info(
            "[%d/%d] Building pass instances for match_id=%s",
            i + 1,
            len(selected_matches),
            match_id,
        )

        raw_events = get_events(match_id)
        events = parse_events(raw_events)

        frames_summary = pd.DataFrame()
        if bool(row.get("has_360", False)):
            raw_frames = get_360_frames(match_id)
            if raw_frames is not None and not raw_frames.empty:
                frames = parse_360_frames(raw_frames)
                frames_summary = get_frame_summary(frames)

        passes = build_pass_instances(
            events,
            frames_summary,
            competition_id=competition_id,
            season_id=season_id,
        )
        passes = compute_downstream_labels(events, passes, labels_cfg)
        if not passes.empty:
            parts.append(passes)

    if not parts:
        raise RuntimeError("No pass instances were built")

    return pd.concat(parts, ignore_index=True)


def _load_processed_passes(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading processed pass instances from %s", path)
    passes = pd.read_parquet(path)
    if "dangerous_progression_k" not in passes.columns:
        raise RuntimeError(f"{path} does not contain dangerous_progression_k; rebuild labels first")
    return passes


def _make_synthetic_passes(n_matches: int = 6, rows_per_match: int = 160) -> pd.DataFrame:
    """Create a small pass_instances-like table for CI smoke reproduction."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for match_idx in range(n_matches):
        match_id = 900000 + match_idx
        for i in range(rows_per_match):
            start_x = float(rng.uniform(20, 88))
            start_y = float(rng.uniform(8, 72))
            x_gain = float(rng.normal(12, 18))
            end_x = float(max(0, min(120, start_x + x_gain)))
            end_y = float(max(0, min(80, start_y + rng.normal(0, 12))))
            pass_length = float(((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5)
            dangerous = bool((end_x >= 86 and x_gain > 10) or rng.random() < 0.08)
            rows.append(
                {
                    "match_id": match_id,
                    "competition_id": 0,
                    "season_id": 0,
                    "event_uuid": f"synthetic-{match_id}-{i}",
                    "possession_id": i // 3,
                    "team_name": "Synthetic",
                    "player_name": "Synthetic Player",
                    "pass_recipient_name": "Synthetic Recipient",
                    "minute": int(i % 90),
                    "second": int((i * 7) % 60),
                    "period": 1 if i < rows_per_match / 2 else 2,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "pass_length": pass_length,
                    "pass_angle": float(np.arctan2(end_y - start_y, end_x - start_x)),
                    "pass_body_part": "Right Foot",
                    "pass_height": "Ground Pass",
                    "pass_type": None,
                    "pass_outcome_name": None,
                    "under_pressure": bool(rng.random() < 0.25),
                    "pass_switch": bool(rng.random() < 0.05),
                    "pass_cross": bool(rng.random() < 0.08),
                    "has_360": False,
                    "n_visible_players": None,
                    "n_visible_teammates": None,
                    "n_visible_opponents": None,
                    "dangerous_progression_k": dangerous,
                    "final_third_entry_k": bool(end_x >= 80),
                    "box_entry_k": bool(end_x >= 102 and 18 <= end_y <= 62),
                    "shot_within_k": bool(dangerous and rng.random() < 0.35),
                }
            )
    return pd.DataFrame(rows)


def _limit_matches(passes: pd.DataFrame, max_matches: int) -> pd.DataFrame:
    match_ids = sorted(passes["match_id"].dropna().astype(int).unique())[:max_matches]
    if len(match_ids) < 3:
        raise RuntimeError("Need at least three matches for train/val/test reproduction")
    return passes[passes["match_id"].isin(match_ids)].reset_index(drop=True)


def _limit_rows_preserving_matches(passes: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Limit rows for smoke mode while keeping all selected matches represented."""
    n_matches = max(1, int(passes["match_id"].nunique()))
    rows_per_match = max(1, max_rows // n_matches)
    return (
        passes.sort_values(["match_id", "possession_id", "event_uuid"])
        .groupby("match_id", group_keys=False)
        .head(rows_per_match)
        .reset_index(drop=True)
    )


def _feature_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = build_event_features(df).fillna(0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def _xgb_config(model_cfg: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    cfg = dict(model_cfg.get("xgboost", {}))
    if smoke:
        cfg.update(
            {
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_jobs": 1,
            }
        )
    return cfg


def reproduce(args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = _load_yaml(REPO_ROOT / "configs" / "data.yaml")
    labels_cfg = _load_yaml(REPO_ROOT / "configs" / "labels.yaml")
    model_cfg = _load_yaml(REPO_ROOT / "configs" / "model_baseline.yaml")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    max_matches = args.max_matches or (6 if args.smoke else 99)

    if args.synthetic:
        passes = _make_synthetic_passes(n_matches=max_matches)
    elif args.use_processed:
        passes = _load_processed_passes(REPO_ROOT / "data" / "processed" / "pass_instances.parquet")
        passes = _limit_matches(passes, max_matches)
    else:
        inventory = _load_inventory(data_cfg)
        selected = _select_matches(
            inventory,
            max_matches=max_matches,
            require_360=bool(args.require_360),
        )
        passes = _build_from_matches(selected, labels_cfg)

    if args.smoke and len(passes) > args.smoke_rows:
        passes = _limit_rows_preserving_matches(passes, args.smoke_rows)

    pass_instances_path = output_dir / "pass_instances.parquet"
    passes.to_parquet(pass_instances_path, index=False)

    train_df, val_df, test_df = create_match_level_splits(
        passes,
        seed=int(data_cfg["splits"]["random_seed"]),
        manifest_path=output_dir / "split_manifest.csv",
    )
    split_paths = materialise_split_parquets(
        passes,
        output_dir=output_dir,
        manifest_path=output_dir / "split_manifest.csv",
    )

    X_train = _feature_matrix(train_df)
    X_val = _feature_matrix(val_df, list(X_train.columns))
    X_test = _feature_matrix(test_df, list(X_train.columns))

    y_train = train_df["dangerous_progression_k"].astype(int)
    y_val = val_df["dangerous_progression_k"].astype(int)
    y_test = test_df["dangerous_progression_k"].astype(int)

    clf = TabularClassifier(
        model_type="xgboost",
        task="dangerous_progression_k",
        config=_xgb_config(model_cfg, smoke=args.smoke),
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    model_path = output_dir / "v1_event_only_model.joblib"
    clf.save(model_path)

    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_prob)

    scored = test_df[
        [
            "match_id",
            "event_uuid",
            "possession_id",
            "team_name",
            "player_name",
            "pass_recipient_name",
            "dangerous_progression_k",
        ]
    ].copy()
    scored["event_only_prob"] = y_prob.round(6)
    scored.to_csv(output_dir / "v1_event_only_scored_passes.csv", index=False)

    summary = {
        "mode": "smoke" if args.smoke else "full",
        "source": (
            "synthetic"
            if args.synthetic
            else (
                "data/processed/pass_instances.parquet"
                if args.use_processed
                else "statsbomb_open_data"
            )
        ),
        "task": "dangerous_progression_k",
        "n_matches": int(passes["match_id"].nunique()),
        "n_passes": int(len(passes)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(X_train.shape[1]),
        "metrics": metrics,
        "outputs": {
            "pass_instances": str(pass_instances_path),
            "split_manifest": str(output_dir / "split_manifest.csv"),
            "train_parquet": str(split_paths["train"]),
            "val_parquet": str(split_paths["val"]),
            "test_parquet": str(split_paths["test"]),
            "model": str(model_path),
            "scored_passes": str(output_dir / "v1_event_only_scored_passes.csv"),
            "summary": str(output_dir / "v1_event_only_summary.json"),
        },
    }

    (output_dir / "v1_event_only_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("v1 event-only ROC AUC: %.4f", metrics["roc_auc"])
    LOGGER.info("Wrote outputs to %s", output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the v1 event-only pass-level model pipeline."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use fewer matches, fewer rows, and a smaller XGBoost model.",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Use data/processed/pass_instances.parquet instead of fetching/parsing.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic pass instances for CI smoke testing only.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Maximum number of matches to include. Defaults to 6 for smoke, 99 otherwise.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=1500,
        help="Maximum rows after match filtering in smoke mode.",
    )
    parser.add_argument(
        "--require-360",
        action="store_true",
        help="When fetching, select only matches with 360 coverage.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/repro/v1",
        help="Directory for generated manifest, scored passes, and summary JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    summary = reproduce(args)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
