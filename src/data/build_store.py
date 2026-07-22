"""Build the Frame2Threat parquet + SQLite data store.

The store keeps large analytical tables as partitioned parquet files and small
query/provenance metadata in SQLite. It is intentionally lightweight so it can
run in smoke mode during CI and scale to StatsBomb Open Data in normal use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.ingest import get_360_frames, get_events
from src.data.inventory import build_inventory
from src.data.join_pass_frames import build_pass_instances
from src.data.parse_360 import get_frame_summary, parse_360_frames
from src.data.parse_events import parse_events
from src.data.splits import materialise_split_parquets
from src.labels.dangerous_progression import compute_downstream_labels

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StorePaths:
    """Concrete paths for a built store."""

    root: Path
    sqlite: Path
    pass_instances_root: Path
    processed_root: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _make_synthetic_passes(n_matches: int = 6, rows_per_match: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for match_idx in range(n_matches):
        match_id = 910000 + match_idx
        competition_id = 900 + (match_idx % 2)
        season_id = 1 + (match_idx % 2)
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
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "event_uuid": f"store-synthetic-{match_id}-{i}",
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


def _synthetic_inventory(passes: pd.DataFrame) -> pd.DataFrame:
    rows = (
        passes[["competition_id", "season_id", "match_id"]]
        .drop_duplicates()
        .sort_values(["competition_id", "season_id", "match_id"])
        .reset_index(drop=True)
    )
    rows["competition_name"] = "Synthetic"
    rows["season_name"] = "Synthetic"
    rows["has_events"] = True
    rows["has_lineups"] = False
    rows["has_360"] = False
    return rows


def _select_matches(
    inventory: pd.DataFrame,
    *,
    max_matches: int,
    require_360: bool,
) -> pd.DataFrame:
    matches = inventory[inventory["has_events"].astype(bool)].copy()
    if require_360 and "has_360" in matches.columns:
        matches = matches[matches["has_360"].astype(bool)]
    if matches.empty:
        raise RuntimeError("No matches available after event/360 filters")
    return (
        matches.sort_values(["competition_id", "season_id", "match_id"])
        .head(max_matches)
        .reset_index(drop=True)
    )


def _build_from_matches(
    selected_matches: pd.DataFrame,
    labels_cfg: dict[str, Any],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for i, row in selected_matches.iterrows():
        match_id = int(row["match_id"])
        LOGGER.info("[%d/%d] Building match_id=%s", i + 1, len(selected_matches), match_id)
        events = parse_events(get_events(match_id))
        frames_summary = pd.DataFrame()
        if bool(row.get("has_360", False)):
            raw_frames = get_360_frames(match_id)
            if raw_frames is not None and not raw_frames.empty:
                frames_summary = get_frame_summary(parse_360_frames(raw_frames))
        passes = build_pass_instances(
            events,
            frames_summary,
            competition_id=int(row["competition_id"]),
            season_id=int(row["season_id"]),
        )
        passes = compute_downstream_labels(events, passes, labels_cfg)
        if not passes.empty:
            parts.append(passes)
    if not parts:
        raise RuntimeError("No pass instances were built")
    return pd.concat(parts, ignore_index=True)


def _normalise_store_dir(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_partitioned_passes(passes: pd.DataFrame, root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    grouped = passes.groupby(["competition_id", "season_id"], dropna=False)
    for (competition_id, season_id), part in grouped:
        partition_dir = (
            root / f"competition_id={int(competition_id)}" / f"season_id={int(season_id)}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / "pass_instances.parquet"
        part.to_parquet(path, index=False)
        rows.append(
            {
                "table_name": "pass_instances",
                "competition_id": int(competition_id),
                "season_id": int(season_id),
                "path": str(path),
                "n_rows": int(len(part)),
                "n_matches": int(part["match_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _write_sqlite(
    sqlite_path: Path,
    *,
    inventory: pd.DataFrame,
    partitions: pd.DataFrame,
    run_row: dict[str, Any],
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        inventory.to_sql("inventory", conn, if_exists="replace", index=False)
        partitions.to_sql("parquet_partitions", conn, if_exists="replace", index=False)
        pd.DataFrame([run_row]).to_sql("runs", conn, if_exists="append", index=False)


def read_store_pass_instances(
    store_dir: Path | str = Path("data/store"),
    *,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    """Read pass instances from the partitioned parquet store."""
    store_dir = _normalise_store_dir(store_dir)
    sqlite_path = store_dir / "metadata.sqlite"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Store metadata not found: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as conn:
        partitions = pd.read_sql_query("SELECT * FROM parquet_partitions", conn)

    if competition_id is not None:
        partitions = partitions[partitions["competition_id"] == int(competition_id)]
    if season_id is not None:
        partitions = partitions[partitions["season_id"] == int(season_id)]
    if partitions.empty:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_parquet(path) for path in partitions["path"].tolist()],
        ignore_index=True,
    )


def build_store(args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = _load_yaml(REPO_ROOT / "configs" / "data.yaml")
    labels_cfg = _load_yaml(REPO_ROOT / "configs" / "labels.yaml")
    store_dir = _normalise_store_dir(args.output_dir)
    paths = StorePaths(
        root=store_dir,
        sqlite=store_dir / "metadata.sqlite",
        pass_instances_root=store_dir / "pass_instances",
        processed_root=store_dir / "processed",
    )

    if args.synthetic:
        passes = _make_synthetic_passes(n_matches=args.max_matches)
        inventory = _synthetic_inventory(passes)
        source = "synthetic"
    elif args.from_processed:
        processed_path = REPO_ROOT / "data" / "processed" / "pass_instances.parquet"
        passes = pd.read_parquet(processed_path)
        inventory = (
            passes[["competition_id", "season_id", "match_id"]]
            .drop_duplicates()
            .sort_values(["competition_id", "season_id", "match_id"])
            .reset_index(drop=True)
        )
        source = str(processed_path)
    else:
        inventory = build_inventory(data_cfg["statsbomb"]["competitions"])
        selected = _select_matches(
            inventory,
            max_matches=args.max_matches,
            require_360=args.require_360,
        )
        passes = _build_from_matches(selected, labels_cfg)
        source = "statsbomb_open_data"

    if passes.empty:
        raise RuntimeError("No pass instances available for store build")

    partitions = _write_partitioned_passes(passes, paths.pass_instances_root)
    split_paths = materialise_split_parquets(
        passes,
        output_dir=paths.processed_root,
        manifest_path=paths.processed_root / "split_manifest.csv",
    )
    run_row = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_passes": int(len(passes)),
        "n_matches": int(passes["match_id"].nunique()),
        "n_partitions": int(len(partitions)),
        "args_json": json.dumps(vars(args), sort_keys=True),
    }
    _write_sqlite(paths.sqlite, inventory=inventory, partitions=partitions, run_row=run_row)

    summary = {
        "store_dir": str(paths.root),
        "sqlite": str(paths.sqlite),
        "pass_instances_root": str(paths.pass_instances_root),
        "processed_root": str(paths.processed_root),
        "n_passes": run_row["n_passes"],
        "n_matches": run_row["n_matches"],
        "n_partitions": run_row["n_partitions"],
        "outputs": {
            "split_manifest": str(split_paths["manifest"]),
            "train_parquet": str(split_paths["train"]),
            "val_parquet": str(split_paths["val"]),
            "test_parquet": str(split_paths["test"]),
        },
    }
    LOGGER.info("Built store at %s", paths.root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Frame2Threat data store.")
    parser.add_argument("--output-dir", default="data/store", help="Store output directory.")
    parser.add_argument("--max-matches", type=int, default=6, help="Maximum matches to ingest.")
    parser.add_argument("--require-360", action="store_true", help="Only ingest matches with 360.")
    parser.add_argument("--from-processed", action="store_true", help="Use processed pass parquet.")
    parser.add_argument("--synthetic", action="store_true", help="Build a no-network smoke store.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(json.dumps(build_store(args), indent=2))


if __name__ == "__main__":
    sys.exit(main())
