"""Rebuild maintained event-time inputs from the fixed research match manifest.

Uses cached StatsBomb raw events when available, otherwise downloads public data.
Writes to a new directory and refuses to overwrite existing processed tables. Also writes the
event-time arrays cache (``event_time_arrays.pkl``) that the dashboard and CLI read, so a freshly
prepared directory can be inspected without retraining.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import get_events  # noqa: E402
from src.data.parse_events import parse_events  # noqa: E402
from src.features.event_time import build_possession_arrays  # noqa: E402
from src.data.parse_possessions import (
    build_possession_sequences,
    save_possession_sequences,
)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/research_split_manifest.csv")
    ap.add_argument(
        "--small", action="store_true", help="Three matches per split; smoke research only"
    )
    args = ap.parse_args()
    out = args.output_dir
    if any(
        (out / n).exists()
        for n in ("events_parsed.parquet", "possession_sequences.parquet", "split_manifest.csv")
    ):
        raise SystemExit("output already contains research inputs; choose a new --output-dir")
    manifest = pd.read_csv(args.manifest)
    if manifest.match_id.duplicated().any() or set(manifest.split) != {"train", "val", "test"}:
        raise ValueError("manifest must assign each match to exactly one split")
    if args.small:
        manifest = manifest.groupby("split", sort=True).head(3)
    parts = []
    source = json.loads((ROOT / "configs/research_data.json").read_text(encoding="utf-8"))
    for mid in manifest.match_id:
        frame = parse_events(get_events(int(mid), revision=source["revision"]))
        if set(frame.match_id) != {mid}:
            raise ValueError(f"raw data match identity mismatch for {mid}")
        parts.append(frame)
        print(f"Parsed match {mid}: {len(frame)} events", flush=True)
    events = pd.concat(parts, ignore_index=True)
    if events.event_uuid.duplicated().any():
        raise ValueError("duplicate event UUIDs")
    poss = build_possession_sequences(events)
    out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out / "events_parsed.parquet", index=False)
    save_possession_sequences(poss, out / "possession_sequences.parquet")
    manifest.to_csv(out / "split_manifest.csv", index=False)
    (out / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    # The trainer only reads match identities from these files.
    for split, group in manifest.groupby("split"):
        group[["match_id"]].to_parquet(out / f"{split}.parquet", index=False)
    # Same inputs and builder as scripts/run_event_time.py:get_arrays, read back from disk so the
    # arrays are identical to what training would cache.
    keys = pd.read_parquet(
        out / "possession_sequences.parquet", columns=["match_id", "possession_id"]
    )
    arrays = build_possession_arrays(pd.read_parquet(out / "events_parsed.parquet"), keys)
    (out / "event_time_arrays.pkl").write_bytes(pickle.dumps(arrays))
    print(f"Wrote {len(events)} events and {len(poss)} possessions to {out}")


if __name__ == "__main__":
    main()
